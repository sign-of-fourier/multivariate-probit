"""Inner models and the named-preset registry.

The multivariate probit here is a *squashing function around an arbitrary inner
model*: each margin j owns a model that maps features to a real-valued index
``eta_j(x)`` on (-inf, inf), and the Gaussian CDF turns that index into a
marginal probability. Nothing about the fit cares how ``eta_j`` was produced,
so the inner model stays a black box.

Two shapes are accepted, and nothing else:

* Anything exposing ``fit(X, y)`` and ``latent(X) -> (n,)`` is used directly
  (:class:`~multivariate_probit.linear.ProbitRegressor` is the canonical one).
* Any classifier exposing ``fit(X, y)`` and ``predict_proba(X)`` is wrapped by
  :class:`ProbitCalibrated`, whose ``latent`` inverts the link with the probit
  quantile function.

The probability is the contract. Inverting the link is exact for a calibrated
``p_hat``, whatever produced it, so a classifier may be arbitrarily bad and
still be a legal inner model -- ``MultivariateProbit.calibration_`` exists to
report that. An estimator emitting only an uncalibrated score
(``decision_function``) is not legal: its scale is arbitrary, ``Phi`` applied
to it means nothing, and there is no way to detect the mismatch from the score
alone. Wrap such an estimator in a calibrator first.

The registry is deliberately thin. IFM has no per-family estimation logic --
"linear", "xgboost" and "rf" differ only in which pre-wired estimator instance
they hand back -- so a preset is a factory function and nothing more.
"""

from __future__ import annotations

import copy

import numpy as np
from scipy.stats import norm

from .linear import ProbitRegressor

__all__ = [
    "ProbitCalibrated",
    "make_inner",
    "register_inner",
    "available_inners",
    "as_inner",
]

_P_EPS = 1e-6


def _no_proba_message(estimator):
    name = type(estimator).__name__
    extra = ""
    if hasattr(estimator, "decision_function"):
        extra = (
            f" {name} exposes decision_function, but that score is on an arbitrary "
            "scale, so Phi applied to it is not a probability and nothing can "
            "detect the mismatch."
        )
    return (
        f"{name} does not expose predict_proba, so it cannot serve as an inner "
        f"model.{extra} Wrap it in a calibrator that does -- scikit-learn's "
        "CalibratedClassifierCV, or SVC(probability=True) -- or give it a "
        "latent(X) method already on the probit scale."
    )


class ProbitCalibrated:
    """Adapt a probability-emitting classifier to the latent probit scale.

    ``latent(X)`` returns ``Phi^-1(p_hat)``, clipped away from 0 and 1 so a
    saturated tree ensemble cannot emit infinite indices. This inverts the
    link; it does not calibrate anything, and a miscalibrated ``p_hat`` gives a
    miscalibrated index.

    The wrapped estimator must expose ``predict_proba``.
    """

    def __init__(self, estimator, clip=_P_EPS):
        self.estimator = estimator
        self.clip = clip

    def fit(self, X, y):
        self.estimator_ = copy.deepcopy(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def latent(self, X):
        est = getattr(self, "estimator_", self.estimator)
        if hasattr(est, "predict_proba"):
            proba = np.asarray(est.predict_proba(X), dtype=float)
            p = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.ravel()
            return norm.ppf(np.clip(p, self.clip, 1.0 - self.clip))
        raise TypeError(_no_proba_message(est))

    def predict_proba(self, X):
        p = norm.cdf(self.latent(X))
        return np.column_stack([1.0 - p, p])

    def __repr__(self):
        return f"ProbitCalibrated({self.estimator!r})"


def _linear(**kwargs):
    return ProbitRegressor(**kwargs)


def _xgboost(**kwargs):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - exercised only without xgboost
        raise ImportError(
            "The 'xgboost' preset requires xgboost. Install it with "
            "`pip install multivariate-probit[xgboost]`."
        ) from exc

    # Defaults tuned for probability calibration rather than ranking: shallow
    # trees, plenty of shrinkage, and a logistic objective whose output maps
    # cleanly back through Phi^-1.
    params = dict(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
    )
    params.update(kwargs)
    return ProbitCalibrated(XGBClassifier(**params))


def _random_forest(**kwargs):
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:  # pragma: no cover - exercised only without sklearn
        raise ImportError(
            "The 'rf' preset requires scikit-learn. Install it with "
            "`pip install multivariate-probit[sklearn]`."
        ) from exc

    params = dict(
        n_estimators=500,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=1,
    )
    params.update(kwargs)
    return ProbitCalibrated(RandomForestClassifier(**params))


_REGISTRY = {
    "linear": _linear,
    "probit": _linear,
    "xgboost": _xgboost,
    "xgb": _xgboost,
    "rf": _random_forest,
    "random_forest": _random_forest,
}


def register_inner(name, factory, overwrite=False):
    """Register a named preset. ``factory(**kwargs)`` returns a fresh estimator."""
    if not callable(factory):
        raise TypeError("factory must be callable")
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"preset {name!r} already registered; pass overwrite=True to replace it")
    _REGISTRY[name] = factory
    return factory


def available_inners():
    """Sorted names of the registered presets."""
    return sorted(_REGISTRY)


def make_inner(name, **kwargs):
    """Instantiate a preset by name, forwarding ``kwargs`` to the estimator."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown inner model {name!r}; available presets: {available_inners()}"
        ) from None
    return factory(**kwargs)


def as_inner(spec, **kwargs):
    """Coerce ``spec`` into an unfitted inner model.

    ``spec`` may be a preset name, a callable factory, or an estimator instance
    (which is deep-copied, then wrapped in :class:`ProbitCalibrated` unless it
    already exposes ``latent``).

    An instance with neither ``latent`` nor ``predict_proba`` is rejected here,
    so the failure arrives at coercion rather than midway through a fold.
    """
    if isinstance(spec, str):
        return make_inner(spec, **kwargs)
    if isinstance(spec, type):
        return as_inner(spec(**kwargs))
    if callable(spec) and not hasattr(spec, "fit"):
        return as_inner(spec(**kwargs))
    if not hasattr(spec, "fit"):
        raise TypeError(f"{spec!r} is not a valid inner model: no fit method")
    est = copy.deepcopy(spec)
    if hasattr(est, "latent"):
        return est
    if not hasattr(est, "predict_proba"):
        raise TypeError(_no_proba_message(est))
    return ProbitCalibrated(est)
