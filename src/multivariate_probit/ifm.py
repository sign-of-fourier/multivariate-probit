"""Stage two of IFM: the dependence parameters, given fixed margins."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from ._corr import nearest_correlation
from ._mvn import RHO_MAX, bvn_cdf, pattern_prob

__all__ = [
    "joint_correlation",
    "pairwise_correlation",
    "pair_log_likelihood",
    "joint_log_likelihood",
    "corr_from_free",
    "free_from_corr",
]

_LL_EPS = 1e-12


def pair_log_likelihood(rho, eta_j, eta_k, y_j, y_k, weights=None):
    """Bivariate probit log-likelihood in ``rho`` with the margins held fixed.

    With ``s = 2y - 1``, the probability of an observed pair is
    ``Phi2(s_j eta_j, s_k eta_k, s_j s_k rho)`` -- one expression covering all
    four cells of the 2x2 table.
    """
    s_j = 2.0 * np.asarray(y_j, dtype=float) - 1.0
    s_k = 2.0 * np.asarray(y_k, dtype=float) - 1.0
    prob = bvn_cdf(s_j * eta_j, s_k * eta_k, s_j * s_k * rho)
    ll = np.log(np.clip(prob, _LL_EPS, None))
    if weights is None:
        return float(np.sum(ll))
    return float(np.sum(np.asarray(weights, dtype=float) * ll))


def _fit_pair(eta_j, eta_k, y_j, y_k, weights, tol):
    finite = np.isfinite(eta_j) & np.isfinite(eta_k)
    if not np.all(finite):
        eta_j, eta_k, y_j, y_k = eta_j[finite], eta_k[finite], y_j[finite], y_k[finite]
        weights = None if weights is None else np.asarray(weights)[finite]

    # A margin with no variation carries no information about dependence.
    if eta_j.size == 0 or np.ptp(y_j) == 0 or np.ptp(y_k) == 0:
        return 0.0

    def neg_ll(rho):
        return -pair_log_likelihood(rho, eta_j, eta_k, y_j, y_k, weights)

    result = minimize_scalar(
        neg_ll, bounds=(-RHO_MAX, RHO_MAX), method="bounded", options={"xatol": tol}
    )
    return float(np.clip(result.x, -RHO_MAX, RHO_MAX))


def pairwise_correlation(eta, Y, weights=None, tol=1e-6, project=True):
    """Estimate the latent correlation matrix one pair at a time.

    Parameters
    ----------
    eta : array of shape (n, d)
        Latent indices from the stage-one margins. Out-of-fold indices are
        strongly preferred for flexible inner models -- see
        :class:`~multivariate_probit.model.MultivariateProbit`.
    Y : array of shape (n, d)
        Observed binary outcomes.
    weights : array of shape (n,), optional
    tol : float
        Tolerance of the 1-D search for each rho.
    project : bool
        Project the assembled matrix onto the nearest positive-definite
        correlation matrix. Pairwise estimates need not be jointly coherent, so
        this is on by default.

    Returns
    -------
    corr : array of shape (d, d)
    """
    eta = np.atleast_2d(np.asarray(eta, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if eta.shape != Y.shape:
        raise ValueError(f"eta {eta.shape} and Y {Y.shape} must have the same shape")
    d = eta.shape[1]

    corr = np.eye(d)
    for j in range(d):
        for k in range(j + 1, d):
            rho = _fit_pair(eta[:, j], eta[:, k], Y[:, j], Y[:, k], weights, tol)
            corr[j, k] = corr[k, j] = rho

    if project and d > 2:
        corr = nearest_correlation(corr)
    return corr


# ---------------------------------------------------------------- joint MLE


def corr_from_free(free, d):
    """Build a symmetric unit-diagonal matrix from its d(d-1)/2 free entries."""
    corr = np.eye(d)
    upper = np.triu_indices(d, k=1)
    corr[upper] = free
    corr[(upper[1], upper[0])] = free
    return corr


def free_from_corr(corr):
    """The free (strictly upper-triangular) entries of a correlation matrix."""
    corr = np.asarray(corr, dtype=float)
    return corr[np.triu_indices(corr.shape[0], k=1)]


def joint_log_likelihood(corr, eta, Y, weights=None, n_quad=24):
    """Full d-variate log-likelihood in ``corr``, with the margins held fixed."""
    prob = np.clip(pattern_prob(eta, Y, corr, n_quad=n_quad), _LL_EPS, None)
    ll = np.log(prob)
    if weights is None:
        return float(np.sum(ll))
    return float(np.sum(np.asarray(weights, dtype=float) * ll))


def joint_correlation(
    eta,
    Y,
    weights=None,
    n_quad=24,
    optimizer="Nelder-Mead",
    init=None,
    min_eigenvalue=1e-8,
    options=None,
):
    """Estimate the latent correlation matrix by full-information ML in ``R``.

    This is the second IFM stage done exactly: with the margins frozen, the
    d-variate orthant likelihood is maximised over the d(d-1)/2 free
    correlations. The search is derivative-free -- the orthant probability has
    no convenient closed-form gradient here -- and each evaluation costs an
    orthant integral per observation, so the price grows quickly with ``d``.
    For many outcomes prefer :func:`pairwise_correlation`.

    Returns
    -------
    corr : ndarray of shape (d, d)
    result : scipy.optimize.OptimizeResult
    """
    eta = np.atleast_2d(np.asarray(eta, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if eta.shape != Y.shape:
        raise ValueError(f"eta {eta.shape} and Y {Y.shape} must have the same shape")
    d = eta.shape[1]

    if d == 1:
        return np.ones((1, 1)), None

    if init is None:
        # The Pearson correlation of the latent indices is always PSD and lands
        # close enough to start; pairwise IFM is a better but pricier start.
        with np.errstate(invalid="ignore"):
            start = np.corrcoef(eta, rowvar=False)
        start = np.nan_to_num(start, nan=0.0)
    else:
        start = np.asarray(init, dtype=float)

    def neg_ll(free):
        corr = corr_from_free(np.clip(free, -RHO_MAX, RHO_MAX), d)
        if np.linalg.eigvalsh(corr).min() <= min_eigenvalue:
            return 1e10  # not positive definite -- push the optimizer away
        return -joint_log_likelihood(corr, eta, Y, weights=weights, n_quad=n_quad)

    result = minimize(neg_ll, free_from_corr(start), method=optimizer, options=options)
    corr = corr_from_free(np.clip(result.x, -RHO_MAX, RHO_MAX), d)
    if np.linalg.eigvalsh(corr).min() <= min_eigenvalue:
        corr = nearest_correlation(corr)
    return corr, result
