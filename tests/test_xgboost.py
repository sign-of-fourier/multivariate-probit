"""XGBoost inner model: the same IFM machinery, a non-linear margin.

These tests are skipped when xgboost is not installed
(``pip install multivariate-probit[xgboost]``).
"""

import numpy as np
import pytest

from multivariate_probit import (
    MultivariateProbit,
    ProbitCalibrated,
    available_inners,
    make_inner,
    register_inner,
)

xgboost = pytest.importorskip("xgboost")

FAST_XGB = dict(n_estimators=60, max_depth=3, learning_rate=0.2, n_jobs=1, random_state=0)


def make_nonlinear_data(n=4000, seed=0, corr=0.5):
    """Margins that no linear probit can fit: interactions and a step."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2.0, 2.0, size=(n, 4))
    eta = np.column_stack(
        [
            1.5 * X[:, 0] * X[:, 1] - 0.5,
            2.0 * (np.abs(X[:, 2]) > 1.0) - 1.0 + 0.5 * X[:, 3],
        ]
    )
    errors = rng.multivariate_normal(np.zeros(2), [[1.0, corr], [corr, 1.0]], size=n)
    Y = (eta + errors > 0).astype(int)
    return X, Y


def test_preset_is_registered_and_wired():
    assert "xgboost" in available_inners()
    inner = make_inner("xgboost", **FAST_XGB)
    assert isinstance(inner, ProbitCalibrated)
    assert isinstance(inner.estimator, xgboost.XGBClassifier)
    assert inner.estimator.get_params()["n_estimators"] == 60


def test_calibrated_latent_is_finite_and_monotone():
    X, Y = make_nonlinear_data(n=1500)
    inner = make_inner("xgboost", **FAST_XGB).fit(X, Y[:, 0])

    eta = inner.latent(X)
    assert eta.shape == (X.shape[0],)
    assert np.all(np.isfinite(eta))
    # Phi is monotone, so the latent index must order the fitted probabilities.
    order_eta = np.argsort(eta)
    order_p = np.argsort(inner.predict_proba(X)[:, 1])
    assert np.array_equal(order_eta, order_p)


def test_xgboost_margins_beat_linear_on_nonlinear_data():
    X, Y = make_nonlinear_data(n=6000, seed=1)
    n_train = 4000
    X_tr, Y_tr, X_te, Y_te = X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]

    common = dict(dependence="pairwise", cv=3, random_state=0)
    boosted = MultivariateProbit(inner="xgboost", inner_params=FAST_XGB, **common).fit(X_tr, Y_tr)
    linear = MultivariateProbit(inner="linear", **common).fit(X_tr, Y_tr)

    assert boosted.score(X_te, Y_te) > linear.score(X_te, Y_te)
    assert boosted.predict_proba(X_te).shape == Y_te.shape
    # Margin error attenuates rho toward zero, so this is a floor, not a
    # recovery test: the dependence must be detected with the right sign.
    assert 0.2 < boosted.correlation_[0, 1] < 0.6


def test_cross_fitting_removes_upward_bias_in_the_correlation():
    """In-sample indices from a saturated ensemble bias rho toward +1.

    Two overfit margins inflate their scores on the same rows -- the ones where
    both labels are 1 -- and the correlation fit reads that spurious co-movement
    as dependence, pegging rho at its upper bound whatever the truth is.
    Cross-fitting removes it. See docs/ifm.md.
    """
    X, Y = make_nonlinear_data(n=3000, seed=2, corr=0.6)
    overfit_params = dict(FAST_XGB, n_estimators=300, max_depth=8, min_child_weight=1)

    common = dict(inner="xgboost", inner_params=overfit_params, dependence="pairwise")
    in_sample = MultivariateProbit(cv=None, **common).fit(X, Y)
    crossfit = MultivariateProbit(cv=4, random_state=0, **common).fit(X, Y)

    assert in_sample.correlation_[0, 1] > 0.95           # pegged, regardless of truth
    assert abs(crossfit.correlation_[0, 1] - 0.6) < abs(in_sample.correlation_[0, 1] - 0.6)


def test_mixed_inner_models_per_outcome():
    X, Y = make_nonlinear_data(n=2000, seed=3)
    inner = ["linear", make_inner("xgboost", **FAST_XGB)]
    model = MultivariateProbit(inner=inner, dependence="pairwise", cv=None).fit(X, Y)

    assert not isinstance(model.inner_models_[0], ProbitCalibrated)
    assert isinstance(model.inner_models_[1], ProbitCalibrated)
    assert model.joint_proba(X, [1, 1]).shape == (X.shape[0],)
    assert model.predict_proba(X).all() == pytest.approx(model.joint_proba(X, [1, 1]))


def test_custom_preset_registration():
    def _shallow_xgb(**kwargs):
        return make_inner("xgboost", **dict(FAST_XGB, max_depth=1, **kwargs))

    register_inner("xgb_stumps", _shallow_xgb, overwrite=True)
    assert "xgb_stumps" in available_inners()

    X, Y = make_nonlinear_data(n=1000, seed=4)
    model = MultivariateProbit(inner="xgb_stumps", dependence="pairwise", cv=None).fit(X, Y)
    assert model.inner_models_[0].estimator.get_params()["max_depth"] == 1

    with pytest.raises(ValueError):
        register_inner("xgb_stumps", _shallow_xgb)


def test_joint_dependence_with_boosted_margins():
    """The default full-likelihood stage two, on a boosted margin."""
    X, Y = make_nonlinear_data(n=1200, seed=5, corr=0.5)
    model = MultivariateProbit(
        inner="xgboost", inner_params=FAST_XGB, dependence="joint", cv=3, random_state=0
    ).fit(X, Y)

    assert 0.15 < model.correlation_[0, 1] < 0.65
    assert model.nll_ > 0 and np.isfinite(model.nll_)
    assert model.optimize_result_ is not None
    assert model.eta_.shape == Y.shape
