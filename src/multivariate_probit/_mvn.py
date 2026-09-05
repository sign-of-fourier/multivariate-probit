"""Normal-CDF machinery: the only place multivariate integration happens.

Everything here works on the *latent* scale. The inner models produce a
real-valued index ``eta`` on (-inf, inf) and the Gaussian CDF is the only
squashing function applied to it.

The evaluator is deterministic and vectorised over observations, and it allows
the correlation matrix to vary by row -- which is what makes the sign trick
below (one expression for all 2^d outcome patterns) practical.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.stats import norm

__all__ = ["bvn_cdf", "mvn_orthant", "orthant_prob", "pattern_prob", "signed_corr_stack", "RHO_MAX"]

# Correlations are kept strictly inside the unit interval: the quadrature below
# degenerates at |rho| = 1, and a boundary correlation is never a defensible
# estimate from finite data anyway.
RHO_MAX = 0.999
_EPS = 1e-12


@lru_cache(maxsize=8)
def _gl_nodes(n_quad):
    """Gauss-Legendre nodes/weights on [-1, 1], cached per requested order."""
    return np.polynomial.legendre.leggauss(n_quad)


def bvn_cdf(a, b, rho, n_quad=24):
    """P(Z1 <= a, Z2 <= b) for a standard bivariate normal with correlation rho.

    Vectorised over ``a``, ``b`` and ``rho``, which broadcast against each
    other -- so correlations may vary by row.

    Uses the Drezner-Wesolowsky form of Plackett's identity,

        Phi2(a, b, r) = Phi(a) Phi(b)
                        + 1/(2 pi) * int_0^asin(r)
                              exp(-(a^2 + b^2 - 2 a b sin t) / (2 cos^2 t)) dt

    whose integrand is smooth and bounded, so a small fixed Gauss-Legendre rule
    is accurate to ~1e-12 over |r| <= 0.999.
    """
    a, b, rho = np.broadcast_arrays(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(rho, dtype=float)
    )
    rho = np.clip(rho, -RHO_MAX, RHO_MAX)
    nodes, weights = _gl_nodes(n_quad)

    # Map the quadrature nodes from [-1, 1] onto [0, asin(rho)].
    half = np.arcsin(rho)[..., None] / 2.0
    sin_t = np.sin(half * (nodes + 1.0))
    cos2 = 1.0 - sin_t**2

    aa, bb = a[..., None], b[..., None]
    integrand = np.exp(-(aa**2 + bb**2 - 2.0 * aa * bb * sin_t) / (2.0 * cos2))
    integral = np.sum(integrand * weights, axis=-1) * half[..., 0]

    return np.clip(norm.cdf(a) * norm.cdf(b) + integral / (2.0 * np.pi), 0.0, 1.0)


def mvn_orthant(A, corr_stack, n_quad=24):
    """P(Z_1 <= A[:, 0], ..., Z_d <= A[:, d-1]) row by row, Z ~ N(0, corr_stack).

    Parameters
    ----------
    A : array of shape (n, d)
        Upper limits, one row per observation.
    corr_stack : array of shape (d, d, n)
        Correlation matrix per row (see :func:`signed_corr_stack`).
    n_quad : int
        Gauss-Legendre order.

    Genz's (1992) recursive conditioning: peel off the last variable under a
    fixed one-dimensional quadrature, reduce to a (d-1)-dimensional problem
    with the conditional correlation matrix, and recurse. Bottoms out at the
    closed-form :func:`bvn_cdf`, so cost is roughly ``n_quad**(d - 2)``
    bivariate evaluations -- comfortable for d = 3-5, slow well before d = 10.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    d = A.shape[1]
    if d == 1:
        return norm.cdf(A[:, 0])
    if d == 2:
        return bvn_cdf(A[:, 0], A[:, 1], corr_stack[0, 1], n_quad=n_quad)

    rho_last = corr_stack[:-1, -1, :]                      # (d-1, n)
    denom = np.sqrt(np.clip(1.0 - rho_last**2, 1e-10, None))
    outer = rho_last[:, None, :] * rho_last[None, :, :]
    cond = (corr_stack[:-1, :-1, :] - outer) / (denom[:, None, :] * denom[None, :, :])
    diag = np.arange(d - 1)
    cond[diag, diag, :] = 1.0
    np.clip(cond, -RHO_MAX, RHO_MAX, out=cond)

    phi_last = np.clip(norm.cdf(A[:, -1]), _EPS, 1.0 - _EPS)
    nodes, weights = _gl_nodes(n_quad)
    acc = np.zeros(A.shape[0])
    for node, weight in zip(nodes, weights):
        # Substituting u = Phi(z) / Phi(A_last) maps the conditioning integral
        # onto [0, 1], where the fixed rule applies.
        u = np.clip(0.5 * (node + 1.0) * phi_last, _EPS, 1.0 - _EPS)
        z = norm.ppf(u)
        reduced = (A[:, :-1] - (z[None, :] * rho_last).T) / denom.T
        acc += 0.5 * weight * mvn_orthant(reduced, cond, n_quad=n_quad)
    return np.clip(phi_last * acc, 0.0, 1.0)


def signed_corr_stack(corr, signs):
    """Stack ``corr_stack[i, j, row] = signs[row, i] * signs[row, j] * corr[i, j]``.

    ``signs`` holds +-1. Flipping the sign of a latent variable flips the sign
    of its correlations, which is what turns every outcome pattern into a plain
    lower orthant.
    """
    corr = np.asarray(corr, dtype=float)
    signs = np.atleast_2d(np.asarray(signs, dtype=float))
    d = corr.shape[0]
    stack = np.empty((d, d, signs.shape[0]))
    for i in range(d):
        stack[i, i, :] = 1.0
        for j in range(i + 1, d):
            value = signs[:, i] * signs[:, j] * corr[i, j]
            stack[i, j, :] = value
            stack[j, i, :] = value
    return stack


def orthant_prob(upper, corr, n_quad=24):
    """P(Z_j <= upper[i, j] for all j) for a single shared correlation matrix."""
    upper = np.atleast_2d(np.asarray(upper, dtype=float))
    stack = signed_corr_stack(corr, np.ones_like(upper))
    return mvn_orthant(upper, stack, n_quad=n_quad)


def pattern_prob(eta, Y, corr, n_quad=24):
    """``P(Y = y | x)`` for the multivariate probit, row by row.

    With ``s = 2y - 1`` the event ``{Y = y}`` is the orthant
    ``{s_j Z_j <= s_j eta_j for all j}``, so one signed expression covers all
    ``2**d`` patterns -- no separate upper/lower-tail bookkeeping.
    """
    eta = np.atleast_2d(np.asarray(eta, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if Y.shape[0] == 1 and eta.shape[0] > 1:
        Y = np.repeat(Y, eta.shape[0], axis=0)
    if eta.shape != Y.shape:
        raise ValueError(f"eta {eta.shape} and Y {Y.shape} must have the same shape")

    signs = 2.0 * Y - 1.0
    return mvn_orthant(signs * eta, signed_corr_stack(corr, signs), n_quad=n_quad)
