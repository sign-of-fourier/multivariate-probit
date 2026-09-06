"""The inner-model contract: `latent`, or `predict_proba`, and nothing else.

A classifier may be arbitrarily bad and still be legal -- `calibration_` is
there to report that. What is not legal is an uncalibrated score: `Phi` applied
to a `decision_function` output is not a probability, and no diagnostic can
detect the mismatch from the score alone. So it is rejected at coercion.
"""

import numpy as np
import pytest
from scipy.stats import norm

from multivariate_probit import MultivariateProbit, ProbitCalibrated, as_inner


class ScoreOnly:
    """The shape the estimator used to accept silently."""

    def fit(self, X, y):
        return self

    def decision_function(self, X):
        return np.asarray(X, dtype=float)[:, 0] * 7.0


class ProbaOnly:
    def fit(self, X, y):
        self.p_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.p_)
        return np.column_stack([1.0 - p, p])


class Opaque:
    def fit(self, X, y):
        return self


def test_a_score_only_estimator_is_rejected_at_coercion():
    with pytest.raises(TypeError, match="does not expose predict_proba"):
        as_inner(ScoreOnly())


def test_the_rejection_names_the_scale_problem():
    with pytest.raises(TypeError, match="arbitrary scale"):
        as_inner(ScoreOnly())


def test_an_estimator_with_neither_is_rejected():
    with pytest.raises(TypeError, match="does not expose predict_proba"):
        as_inner(Opaque())


def test_a_probability_emitting_estimator_is_wrapped():
    inner = as_inner(ProbaOnly())
    assert isinstance(inner, ProbitCalibrated)

    X = np.zeros((4, 2))
    inner.fit(X, np.array([1.0, 1.0, 0.0, 0.0]))
    assert inner.latent(X) == pytest.approx(np.full(4, norm.ppf(0.5)))


def test_a_latent_bearing_model_passes_through_untouched():
    inner = as_inner(as_inner("linear"))
    assert not isinstance(inner, ProbitCalibrated)
    assert hasattr(inner, "latent")


def test_probit_calibrated_still_refuses_at_latent():
    """Direct construction is not policed, so `latent` must refuse too."""
    wrapper = ProbitCalibrated(ScoreOnly()).fit(np.zeros((3, 1)), np.zeros(3))
    with pytest.raises(TypeError, match="does not expose predict_proba"):
        wrapper.latent(np.zeros((3, 1)))


def test_the_failure_arrives_before_any_margin_is_fitted():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    Y = (rng.normal(size=(50, 2)) > 0).astype(int)

    model = MultivariateProbit(inner=ScoreOnly())
    with pytest.raises(TypeError, match="does not expose predict_proba"):
        model.fit(X, Y)
    assert not hasattr(model, "correlation_")


def test_svc_is_accepted_only_with_probability_estimates():
    """The user-facing case: sklearn's SVC is legal in exactly one of its modes."""
    pytest.importorskip("sklearn")
    from sklearn.svm import SVC

    with pytest.raises(TypeError, match="CalibratedClassifierCV"):
        as_inner(SVC(probability=False))

    assert isinstance(as_inner(SVC(probability=True)), ProbitCalibrated)
