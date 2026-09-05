"""Multivariate probit models for correlated binary outcomes.

A multivariate probit is a Gaussian squashing function wrapped around an
arbitrary inner model. This package fits one by Inference Functions for Margins
(IFM): margins first, dependence second. The inner model is pluggable -- linear
by default, gradient boosting or random forests via named presets, anything
scikit-learn-shaped via the adapter.

    >>> from multivariate_probit import MultivariateProbit
    >>> model = MultivariateProbit(inner="linear").fit(X, Y)
    >>> proba = model.predict_proba(X)
    >>> proba.marginal              # P(Y_j = 1 | x)
    >>> proba.all()                 # P(every outcome = 1 | x)
    >>> proba.joint([1, 0, 1])      # P(Y = (1, 0, 1) | x)
    >>> model.correlation_
"""

from ._mvn import bvn_cdf, mvn_orthant, orthant_prob, pattern_prob
from .ifm import (
    joint_correlation,
    joint_log_likelihood,
    pair_log_likelihood,
    pairwise_correlation,
)
from .inner import (
    ProbitCalibrated,
    as_inner,
    available_inners,
    make_inner,
    register_inner,
)
from .linear import ProbitRegressor
from .model import MultivariateProbit
from .results import MultivariateProbitProba

__version__ = "0.1.0"

__all__ = [
    "MultivariateProbit",
    "MultivariateProbitProba",
    "ProbitRegressor",
    "ProbitCalibrated",
    "make_inner",
    "register_inner",
    "available_inners",
    "as_inner",
    "joint_correlation",
    "joint_log_likelihood",
    "pairwise_correlation",
    "pair_log_likelihood",
    "bvn_cdf",
    "mvn_orthant",
    "orthant_prob",
    "pattern_prob",
    "__version__",
]
