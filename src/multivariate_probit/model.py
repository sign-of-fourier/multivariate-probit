"""The estimator: a multivariate probit fitted by IFM."""

from __future__ import annotations

import inspect

import numpy as np
from scipy.stats import norm

from ._corr import is_positive_definite
from ._mvn import pattern_prob
from .ifm import joint_correlation, pair_log_likelihood, pairwise_correlation
from .inner import as_inner
from .results import MultivariateProbitProba

__all__ = ["MultivariateProbit"]

_LL_EPS = 1e-12


def _kfold_indices(n, n_splits, rng):
    idx = np.arange(n)
    rng.shuffle(idx)
    for fold in np.array_split(idx, n_splits):
        mask = np.zeros(n, dtype=bool)
        mask[fold] = True
        yield ~mask, mask


def _require_positive_definite(corr):
    """Reject a correlation matrix that would silently corrupt joint queries.

    Only reachable with ``project_correlation=False``: pairwise estimates are
    fitted one pair at a time and need not be jointly coherent. An indefinite
    matrix makes ``sample`` fail at the Cholesky and -- worse -- makes
    ``joint_proba`` / ``.all()`` / ``.any()`` return numbers that are not
    probabilities of anything, with no error at all.
    """
    if not is_positive_definite(corr):
        raise ValueError(
            "the fitted correlation matrix is not positive definite (smallest "
            f"eigenvalue {np.linalg.eigvalsh(corr).min():.4g}). Pairwise estimates "
            "are fitted independently and need not be jointly coherent, so this is "
            "expected with project_correlation=False. Set project_correlation=True "
            "to project onto the nearest valid correlation matrix, or call "
            "pairwise_correlation(..., project=False) directly if you want the raw "
            "estimates for inspection."
        )
    return corr


def _supports_sample_weight(estimator):
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False


class MultivariateProbit:
    """Multivariate probit for correlated binary outcomes, fitted by IFM.

    The model is a Gaussian squashing function wrapped around an arbitrary
    inner model. For outcome ``j``::

        Y_j = 1[eta_j(x) + e_j > 0],    e ~ N(0, R)

    where ``eta_j`` is any real-valued function of the features -- linear by
    default, a gradient-boosted ensemble if you ask for one -- and ``R`` is a
    correlation matrix carrying the dependence between outcomes. Marginally,
    ``P(Y_j = 1 | x) = Phi(eta_j(x))``.

    Fitting is two-stage IFM (Inference Functions for Margins):

    1. Fit each margin independently, ignoring the other outcomes. The fitted
       margins are cross-fitted (out-of-fold) so that stage two never sees an
       in-sample, over-confident prediction.
    2. Hold the margins fixed and estimate ``R`` by maximum likelihood --
       either the full d-variate orthant likelihood (``dependence="joint"``)
       or the sum of bivariate ones (``dependence="pairwise"``).

    IFM was chosen over full joint MLE (FIML) and over a direct joint
    classifier: it leaves the inner model a black box, and its cost advantage
    over FIML widens with the number of outcomes, since the orthant probability
    IFM pays for once is paid by FIML on every boosting round of every margin.

    See ``docs/ifm.md`` for the derivation and the trade-offs.

    Parameters
    ----------
    inner : str, estimator, callable or list, default="linear"
        The inner model. A preset name (``"linear"``, ``"xgboost"``, ``"rf"``),
        an unfitted estimator instance, a factory, or a list of length ``d``
        giving a different inner model per outcome. Instances are deep-copied,
        so one instance can safely seed every margin.
    inner_params : dict, optional
        Keyword arguments forwarded to the preset factory. Ignored when
        ``inner`` is already an instance.
    dependence : {"joint", "pairwise"}, default="joint"
        How stage two estimates ``R``. ``"joint"`` maximises the full
        d-variate likelihood -- exact, and the default. ``"pairwise"``
        maximises each pair's bivariate likelihood separately (a composite
        likelihood): consistent, far cheaper, and the practical choice once
        ``d`` is large enough that orthant integration bites.
    cv : int or None, default=5
        Number of folds used to cross-fit the latent indices that stage two
        consumes. In-sample indices from a flexible learner have already
        absorbed part of the noise, which attenuates the estimated
        correlations toward zero. ``None`` skips cross-fitting -- reasonable
        for the linear default, risky for anything that can overfit.
    n_quad : int, default=24
        Gauss-Legendre order for the orthant-probability evaluator. Lower it
        if fitting with many outcomes gets slow.
    optimizer : str, default="Nelder-Mead"
        Passed to ``scipy.optimize.minimize`` for ``dependence="joint"``.
        Derivative-free by design: the orthant likelihood has no convenient
        closed-form gradient here.
    project_correlation : bool, default=True
        Project a pairwise estimate onto the nearest positive-definite
        correlation matrix (pairwise fits need not be jointly coherent).
    random_state : int or Generator, optional
        Controls the cross-fitting split and :meth:`sample`.

    Attributes
    ----------
    inner_models_ : list
        The fitted margins, one per outcome.
    correlation_ : ndarray of shape (d, d)
    eta_ : ndarray of shape (n, d)
        The (cross-fitted) latent indices stage two was fitted on.
    nll_ : float
        Negative log-likelihood at the end of the dependence fit.
    n_outcomes_ : int
    n_features_in_ : int
    """

    def __init__(
        self,
        inner="linear",
        inner_params=None,
        dependence="joint",
        cv=5,
        n_quad=24,
        optimizer="Nelder-Mead",
        project_correlation=True,
        random_state=None,
    ):
        self.inner = inner
        self.inner_params = inner_params
        self.dependence = dependence
        self.cv = cv
        self.n_quad = n_quad
        self.optimizer = optimizer
        self.project_correlation = project_correlation
        self.random_state = random_state

    # ------------------------------------------------------------------ fit

    def fit(self, X, Y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.shape[0] != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but Y has {Y.shape[0]}")
        if not np.isin(np.unique(Y), (0, 1)).all():
            raise ValueError("Y must contain only 0/1 values")
        Y = Y.astype(float)

        n, d = Y.shape
        self.n_features_in_ = X.shape[1]
        self.n_outcomes_ = d
        rng = np.random.default_rng(self.random_state)
        params = dict(self.inner_params or {})

        specs = self.inner if isinstance(self.inner, (list, tuple)) else [self.inner] * d
        if len(specs) != d:
            raise ValueError(f"inner has length {len(specs)} but Y has {d} columns")

        # ---- stage 1: independent margins
        self.inner_models_ = []
        for j, spec in enumerate(specs):
            model = as_inner(spec, **params)
            self._fit_margin(model, X, Y[:, j], sample_weight)
            self.inner_models_.append(model)

        # ---- latent indices for stage 2
        if self.cv is None:
            eta = self.decision_function(X)
        else:
            eta = self._oof_decision_function(X, Y, specs, params, sample_weight, rng)
        self.eta_ = eta

        # ---- stage 2: dependence
        if self.dependence == "joint":
            self.correlation_, self.optimize_result_ = joint_correlation(
                eta,
                Y,
                weights=sample_weight,
                n_quad=self.n_quad,
                optimizer=self.optimizer,
            )
            self.nll_ = None if self.optimize_result_ is None else float(self.optimize_result_.fun)
        elif self.dependence == "pairwise":
            self.correlation_ = pairwise_correlation(
                eta, Y, weights=sample_weight, project=self.project_correlation
            )
            self.optimize_result_ = None
            # For the composite fit, report the objective that was actually
            # optimised: the summed pairwise log-likelihood, not the joint one.
            self.nll_ = -sum(
                pair_log_likelihood(
                    self.correlation_[j, k], eta[:, j], eta[:, k], Y[:, j], Y[:, k], sample_weight
                )
                for j in range(d)
                for k in range(j + 1, d)
            )
        else:
            raise ValueError(
                f"dependence must be 'joint' or 'pairwise'; got {self.dependence!r}"
            )

        _require_positive_definite(self.correlation_)
        return self

    def _fit_margin(self, model, X, y, sample_weight):
        if sample_weight is not None and _supports_sample_weight(model):
            model.fit(X, y, sample_weight=sample_weight)
        else:
            model.fit(X, y)
        return model

    def _oof_decision_function(self, X, Y, specs, params, sample_weight, rng):
        n, d = Y.shape
        n_splits = int(self.cv)
        if not 2 <= n_splits <= n:
            raise ValueError(f"cv must be between 2 and n_samples ({n}); got {self.cv}")

        eta = np.empty((n, d))
        for train, test in _kfold_indices(n, n_splits, rng):
            weight = None if sample_weight is None else np.asarray(sample_weight)[train]
            for j, spec in enumerate(specs):
                fold_model = as_inner(spec, **params)
                self._fit_margin(fold_model, X[train], Y[train, j], weight)
                eta[test, j] = fold_model.latent(X[test])
        return eta

    # ------------------------------------------------------------- predict

    def _check_fitted(self):
        if not hasattr(self, "inner_models_"):
            raise RuntimeError("this MultivariateProbit is not fitted yet; call fit first")

    def _check_X(self, X):
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features but the model was fitted with {self.n_features_in_}"
            )
        return X

    def decision_function(self, X):
        """Latent indices ``eta``, shape (n, d), on (-inf, inf)."""
        X = self._check_X(X)
        return np.column_stack([m.latent(X) for m in self.inner_models_])

    def transform(self, X):
        """Alias of :meth:`decision_function`: features to latent scores."""
        return self.decision_function(X)

    def fit_transform(self, X, Y, **fit_params):
        return self.fit(X, Y, **fit_params).transform(X)

    def predict_proba(self, X):
        """Marginal probabilities, plus joint queries.

        Returns a :class:`~multivariate_probit.results.MultivariateProbitProba`,
        which behaves like the ``(n, d)`` array of marginal probabilities and
        also answers ``.joint(pattern)``, ``.all()``, ``.any()`` and
        ``.none()``.
        """
        return MultivariateProbitProba(
            self.decision_function(X), self.correlation_, n_quad=self.n_quad
        )

    def predict_marginal_proba(self, X):
        """Marginal probabilities ``P(Y_j = 1 | x)`` as a plain (n, d) array."""
        return norm.cdf(self.decision_function(X))

    def predict(self, X, threshold=0.5):
        """Per-outcome 0/1 predictions at a marginal threshold, shape (n, d).

        For a joint decision -- thresholding ``P(all outcomes = 1)``, say --
        use ``predict_proba(X).all()`` / ``.any()`` / ``.joint(...)``.
        """
        return (self.predict_marginal_proba(X) >= threshold).astype(int)

    def joint_proba(self, X, Y):
        """``P(Y = y | x)`` for the given outcome pattern(s), shape (n,).

        ``Y`` may be one pattern of length ``d`` (broadcast over all rows) or an
        array of shape (n, d).
        """
        eta = self.decision_function(X)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[None, :]
        return pattern_prob(eta, Y, self.correlation_, n_quad=self.n_quad)

    def joint_log_proba(self, X, Y):
        return np.log(np.clip(self.joint_proba(X, Y), _LL_EPS, None))

    def score(self, X, Y, sample_weight=None):
        """Mean joint log-likelihood -- higher is better."""
        ll = self.joint_log_proba(X, Y)
        if sample_weight is None:
            return float(np.mean(ll))
        w = np.asarray(sample_weight, dtype=float)
        return float(np.sum(w * ll) / np.sum(w))

    def sample(self, X, n_samples=1, random_state=None):
        """Draw outcome patterns from the fitted model.

        Returns shape (n, d) when ``n_samples == 1``, else (n_samples, n, d).
        """
        X = self._check_X(X)
        eta = self.decision_function(X)
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        chol = np.linalg.cholesky(self.correlation_)
        n, d = eta.shape
        noise = rng.standard_normal((n_samples, n, d)) @ chol.T
        draws = (eta[None, :, :] + noise > 0).astype(int)
        return draws[0] if n_samples == 1 else draws

    # -------------------------------------------------------------- params

    def get_params(self, deep=True):
        return {
            "inner": self.inner,
            "inner_params": self.inner_params,
            "dependence": self.dependence,
            "cv": self.cv,
            "n_quad": self.n_quad,
            "optimizer": self.optimizer,
            "project_correlation": self.project_correlation,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"invalid parameter {key!r} for MultivariateProbit")
            setattr(self, key, value)
        return self

    def __repr__(self):
        return (
            f"MultivariateProbit(inner={self.inner!r}, dependence={self.dependence!r}, "
            f"cv={self.cv!r})"
        )
