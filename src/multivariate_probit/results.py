"""The object returned by :meth:`MultivariateProbit.predict_proba`."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import norm

from ._mvn import lower_orthant, pattern_prob, signed_corr_stack

__all__ = ["MultivariateProbitProba"]


class MultivariateProbitProba:
    """Marginal probabilities, plus joint queries against the fitted copula.

    Behaves like the ``(n_samples, n_outcomes)`` array of marginal
    probabilities in most contexts -- indexing, ``np.asarray(...)``, ``.shape``
    -- and additionally answers joint questions, which are the reason the
    correlation matrix was estimated in the first place.
    """

    def __init__(self, eta, corr, n_quad=24, evaluator="quadrature", resolution="high"):
        self.eta = np.atleast_2d(np.asarray(eta, dtype=float))
        self.marginal = norm.cdf(self.eta)
        self.corr = np.asarray(corr, dtype=float)
        self.n_quad = n_quad
        self.evaluator = evaluator
        self.resolution = resolution

    def _orthant(self, upper, corr):
        stack = signed_corr_stack(corr, np.ones_like(upper))
        return lower_orthant(
            upper, stack, self.evaluator, n_quad=self.n_quad, resolution=self.resolution
        )

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
        return pattern_prob(
            self.eta,
            pattern,
            self.corr,
            n_quad=self.n_quad,
            evaluator=self.evaluator,
            resolution=self.resolution,
        )

    def all(self, outcomes=None):
        """``P(every selected outcome = 1 | x)``. Defaults to all outcomes."""
        eta, corr = self._subset(outcomes)
        return self._orthant(eta, corr)

    def any(self, outcomes=None):
        """``P(at least one selected outcome = 1 | x)``. Defaults to all outcomes."""
        return 1.0 - self.none(outcomes)

    def none(self, outcomes=None):
        """``P(no selected outcome = 1 | x)``. Defaults to all outcomes."""
        eta, corr = self._subset(outcomes)
        return self._orthant(-eta, corr)

    def _event_prob(self, event):
        idx = list(event)
        values = np.column_stack(
            [np.broadcast_to(np.asarray(event[j], dtype=float), (self.shape[0],)) for j in idx]
        )
        if not np.isin(values, (0.0, 1.0)).all():
            raise ValueError("outcome values must be 0 or 1")
        return pattern_prob(
            self.eta[:, idx],
            values,
            self.corr[np.ix_(idx, idx)],
            n_quad=self.n_quad,
            evaluator=self.evaluator,
            resolution=self.resolution,
        )

    def conditional(self, event, given):
        """``P(event | given, x)``, per row.

        ``event`` and ``given`` map outcome index to a 0/1 value, either one
        value for every row or an ``(n,)`` array of per-row values -- so the
        outcomes actually observed for each row can be conditioned on directly.
        An int ``event`` is shorthand for ``{event: 1}``. The two must not share
        an outcome.

        Computed as ``P(event and given) / P(given)``, with ``P(given)`` taken
        as the sum over all ``2 ** len(event)`` values of the event outcomes.
        That costs ``2 ** len(event)`` orthant evaluations, but makes the
        conditional probabilities of one event sum to 1 exactly, which a
        separate evaluation of ``P(given)`` does not when it is small. A row
        whose ``given`` has probability 0 returns NaN.
        """
        if isinstance(event, (int, np.integer)):
            event = {int(event): 1}
        if not event or not given:
            raise ValueError("event and given must each name at least one outcome")
        overlap = set(event) & set(given)
        if overlap:
            raise ValueError(f"event and given share outcomes {sorted(overlap)}")
        keys = list(event)
        target = np.column_stack(
            [np.broadcast_to(np.asarray(event[j]), (self.shape[0],)) for j in keys]
        )
        if not np.isin(target, (0, 1)).all():
            raise ValueError("outcome values must be 0 or 1")
        both = np.zeros(self.shape[0])
        cond = np.zeros(self.shape[0])
        for values in itertools.product((0, 1), repeat=len(keys)):
            prob = self._event_prob({**given, **dict(zip(keys, values))})
            cond += prob
            both += np.where((target == values).all(axis=1), prob, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(cond > 0.0, both / cond, np.nan)
        # Nested-event inequalities can break at ~1e-5 for d >= 3.
        return np.clip(out, 0.0, 1.0)
