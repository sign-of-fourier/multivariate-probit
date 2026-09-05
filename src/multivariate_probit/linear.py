"""A native linear probit margin -- the default inner model."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["ProbitRegressor"]

_P_EPS = 1e-10


class ProbitRegressor:
    """Binary probit regression fitted by Newton-Raphson (IRLS).

    This is the one inner model that already lives on the probit scale, so no
    calibration is needed: ``latent`` is literally ``X @ coef_ + intercept_``.

    Parameters
    ----------
    alpha : float, default=1e-6
        L2 penalty on the slopes (never on the intercept). The default is small
        enough to be statistically negligible and large enough to keep
        separated or collinear designs from blowing up.
    fit_intercept : bool, default=True
    max_iter : int, default=100
    tol : float, default=1e-8
        Convergence threshold on the max absolute coefficient update.
    """

    def __init__(self, alpha=1e-6, fit_intercept=True, max_iter=100, tol=1e-8):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def _design(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0], 1)), X])
        return X

    def fit(self, X, y):
        Z = self._design(X)
        y = np.asarray(y, dtype=float).ravel()
        n, p = Z.shape

        penalty = np.full(p, float(self.alpha))
        if self.fit_intercept:
            penalty[0] = 0.0
        ridge = np.diag(penalty)

        beta = np.zeros(p)
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            eta = Z @ beta
            prob = np.clip(norm.cdf(eta), _P_EPS, 1.0 - _P_EPS)
            dens = np.maximum(norm.pdf(eta), _P_EPS)

            # IRLS weights and working response for the probit link.
            w = dens**2 / (prob * (1.0 - prob))
            working = eta + (y - prob) / dens

            lhs = Z.T @ (Z * w[:, None]) + ridge
            rhs = Z.T @ (w * working)
            try:
                new_beta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                new_beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

            step = np.max(np.abs(new_beta - beta))
            beta = new_beta
            if step < self.tol:
                break

        self.n_iter_ = n_iter
        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta
        self.classes_ = np.array([0, 1])
        return self

    def latent(self, X):
        """The probit index eta on (-inf, inf)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return X @ self.coef_ + self.intercept_

    decision_function = latent

    def predict_proba(self, X):
        p = norm.cdf(self.latent(X))
        return np.column_stack([1.0 - p, p])

    def predict(self, X, threshold=0.5):
        return (norm.cdf(self.latent(X)) >= threshold).astype(int)

    def get_params(self, deep=True):
        return {
            "alpha": self.alpha,
            "fit_intercept": self.fit_intercept,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self):
        return f"ProbitRegressor(alpha={self.alpha!r}, fit_intercept={self.fit_intercept!r})"
