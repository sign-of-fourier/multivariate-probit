"""The object returned by :meth:`MultivariateProbit.predict_proba`."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ._mvn import mvn_orthant, pattern_prob, signed_corr_stack

__all__ = ["MultivariateProbitProba"]


class MultivariateProbitProba:
    """Marginal probabilities, plus joint queries against the fitted copula.

    Behaves like the ``(n_samples, n_outcomes)`` array of marginal
    probabilities in most contexts -- indexing, ``np.asarray(...)``, ``.shape``
    -- and additionally answers joint questions, which are the reason the
    correlation matrix was estimated in the first place.
    """

    def __init__(self, eta, corr, n_quad=24):
        self.eta = np.atleast_2d(np.asarray(eta, dtype=float))
        self.marginal = norm.cdf(self.eta)
        self.corr = np.asarray(corr, dtype=float)
        self.n_quad = n_quad

    # ------------------------------------------------------ array behaviour

    def __array__(self, dtype=None):
        return np.asarray(self.marginal, dtype=dtype)

    def __getitem__(self, idx):
        return self.marginal[idx]

    def __len__(self):
        return len(self.marginal)

    @property
    def shape(self):
        return self.marginal.shape

    def __repr__(self):
        n, d = self.shape
        return f"MultivariateProbitProba(n_samples={n}, n_outcomes={d})"

    # --------------------------------------------------------- joint queries

    def _subset(self, outcomes):
        idx = np.arange(self.shape[1]) if outcomes is None else np.asarray(outcomes)
        return self.eta[:, idx], self.corr[np.ix_(idx, idx)]

    def joint(self, pattern):
        """``P(Y = pattern | x)`` for a fully specified 0/1 pattern, per row."""
        pattern = np.asarray(pattern, dtype=float)
        if pattern.ndim == 1:
            pattern = pattern[None, :]
        return pattern_prob(self.eta, pattern, self.corr, n_quad=self.n_quad)

    def all(self, outcomes=None):
        """``P(every selected outcome = 1 | x)``. Defaults to all outcomes."""
        eta, corr = self._subset(outcomes)
        return mvn_orthant(eta, signed_corr_stack(corr, np.ones_like(eta)), n_quad=self.n_quad)

    def any(self, outcomes=None):
        """``P(at least one selected outcome = 1 | x)``. Defaults to all outcomes."""
        return 1.0 - self.none(outcomes)

    def none(self, outcomes=None):
        """``P(no selected outcome = 1 | x)``. Defaults to all outcomes."""
        eta, corr = self._subset(outcomes)
        return mvn_orthant(-eta, signed_corr_stack(corr, np.ones_like(eta)), n_quad=self.n_quad)
