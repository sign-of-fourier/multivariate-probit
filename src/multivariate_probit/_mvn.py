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
from scipy.special import log_ndtr, ndtr, ndtri
from scipy.stats import norm

__all__ = [
    "bvn_cdf",
    "mvn_orthant",
    "lower_orthant",
    "orthant_prob",
    "pattern_prob",
    "signed_corr_stack",
    "EVALUATORS",
    "RHO_MAX",
]

# Correlations are kept strictly inside the unit interval: the quadrature below
# degenerates at |rho| = 1, and a boundary correlation is never a defensible
# estimate from finite data anyway.
RHO_MAX = 0.999
_EPS = 1e-12

# The backends that can evaluate a lower-orthant probability. The choice is a
# speed/accuracy trade-off left to the caller: "quadrature" is the
# deterministic evaluator below, "scipy" is scipy.stats.multivariate_normal.cdf
# (randomised quasi-Monte Carlo, one row at a time), "orthant" is the
# optional compiled package bundled under ``multivariate_probit.orthant``, and
# "ghk" is the GHK simulator below with fixed scrambled Sobol draws
# (deterministic, vectorised, any d).
EVALUATORS = ("quadrature", "scipy", "orthant", "ghk")


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


def _scipy_orthant(A, corr_stack):
    from scipy.stats import multivariate_normal

    d = A.shape[1]
    out = np.empty(A.shape[0])
    for i in range(A.shape[0]):
        out[i] = multivariate_normal(mean=np.zeros(d), cov=corr_stack[:, :, i]).cdf(A[i])
    return out


@lru_cache(maxsize=8)
def _sobol(dim, n_draws, seed):
    """Scrambled Sobol points in (0, 1)^dim, cached so repeated calls (an
    optimizer's objective) reuse the same draws."""
    from scipy.stats import qmc

    return qmc.Sobol(dim, scramble=True, seed=seed).random(n_draws)


def _ghk_orthant(A, corr_stack, n_draws=512, seed=0, max_cells=2_000_000):
    """GHK (Geweke-Hajivassiliou-Keane) estimate of the lower orthant, row by row.

    Per row the variables are reordered by ascending limit, the correlation is
    Cholesky-factored, and each draw samples the variables in sequence from
    their truncated conditionals; the estimate is the mean over draws of the
    product of the conditional probabilities. The draws are one scrambled
    Sobol set shared by all rows (seed fixed), so the result is deterministic
    and smooth in A and the correlations. Error falls with ``n_draws``.
    """
    n, d = A.shape
    if d == 1:
        return norm.cdf(A[:, 0])
    S = np.moveaxis(corr_stack, 2, 0)
    order = np.argsort(A, axis=1)
    A = np.take_along_axis(A, order, axis=1)
    S = S[np.arange(n)[:, None, None], order[:, :, None], order[:, None, :]]
    L = np.linalg.cholesky(S)
    U = _sobol(d - 1, int(n_draws), seed)
    out = np.empty(n)
    step = max(1, max_cells // (n_draws * d))
    for lo in range(0, n, step):
        a, Lc = A[lo:lo + step], L[lo:lo + step]
        logp = np.zeros((len(a), n_draws))
        z = np.zeros((len(a), n_draws, d - 1))
        for j in range(d):
            mu = z[:, :, :j] @ Lc[:, j, :j, None]
            b = (a[:, j, None] - mu[:, :, 0]) / Lc[:, j, j, None]
            logp += log_ndtr(b)
            if j < d - 1:
                z[:, :, j] = ndtri(np.clip(U[:, j] * ndtr(b), 1e-300, 1.0 - 1e-16))
        out[lo:lo + step] = np.exp(logp).mean(axis=1)
    return out


def _compiled_orthant(A, corr_stack, resolution):
    try:
        from . import orthant
    except ImportError as exc:
        raise ImportError(
            "evaluator='orthant' is unavailable: the compiled orthant package "
            "could not be loaded on this platform (it is built for CPython 3.11 "
            f"and 3.12 on x86-64 Linux). Choose 'quadrature' or 'scipy'. Cause: {exc}"
        ) from exc
    stack = np.ascontiguousarray(np.moveaxis(corr_stack, 2, 0))
    try:
        prob = orthant.cdf(np.ascontiguousarray(A), stack, resolution=resolution)
    except ValueError as exc:
        raise ValueError(f"evaluator='orthant' (tier {orthant.tier!r}): {exc}") from exc
    return np.asarray(prob, dtype=float)


def lower_orthant(A, corr_stack, evaluator="quadrature", n_quad=24, resolution="high", n_draws=512):
    """:func:`mvn_orthant` with a choice of backend.

    ``n_quad`` is used only by ``"quadrature"``, ``resolution`` only by
    ``"orthant"`` and ``n_draws`` only by ``"ghk"``. A backend that cannot run as asked raises; nothing falls
    back to another backend. Results are clipped to [0, 1].
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    if evaluator == "quadrature":
        return mvn_orthant(A, corr_stack, n_quad=n_quad)
    if evaluator == "scipy":
        prob = _scipy_orthant(A, corr_stack)
    elif evaluator == "orthant":
        prob = _compiled_orthant(A, corr_stack, resolution)
    elif evaluator == "ghk":
        prob = _ghk_orthant(A, corr_stack, n_draws=n_draws)
    else:
        raise ValueError(f"evaluator must be one of {EVALUATORS}; got {evaluator!r}")
    return np.clip(prob, 0.0, 1.0)


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


def orthant_prob(upper, corr, n_quad=24, evaluator="quadrature", resolution="high", n_draws=512):
    """P(Z_j <= upper[i, j] for all j) for a single shared correlation matrix."""
    upper = np.atleast_2d(np.asarray(upper, dtype=float))
    stack = signed_corr_stack(corr, np.ones_like(upper))
    return lower_orthant(upper, stack, evaluator, n_quad=n_quad, resolution=resolution, n_draws=n_draws)


def pattern_prob(eta, Y, corr, n_quad=24, evaluator="quadrature", resolution="high", n_draws=512):
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
    return lower_orthant(
        signs * eta,
        signed_corr_stack(corr, signs),
        evaluator,
        n_quad=n_quad,
        resolution=resolution,
        n_draws=n_draws,
    )
