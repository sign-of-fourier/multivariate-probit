"""Random-forest inner model: the "rf" preset through the same IFM machinery.

Skipped when scikit-learn is not installed
(``pip install multivariate-probit[sklearn]``).
"""

import numpy as np
import pytest

from multivariate_probit import MultivariateProbit, ProbitCalibrated, make_inner

ensemble = pytest.importorskip("sklearn.ensemble")

from test_xgboost import make_nonlinear_data  # noqa: E402

FAST_RF = dict(n_estimators=100, random_state=0)


def test_preset_is_registered_and_wired():
    for name in ("rf", "random_forest"):
        inner = make_inner(name, **FAST_RF)
        assert isinstance(inner, ProbitCalibrated)
        assert isinstance(inner.estimator, ensemble.RandomForestClassifier)
    params = make_inner("rf", **FAST_RF).estimator.get_params()
    assert params["n_estimators"] == 100
    assert params["min_samples_leaf"] == 5


def test_leaves_rarely_saturate():
    """A pure leaf emits p = 0 or 1, which pins the index at the clip.

    ``min_samples_leaf=5`` averages over enough rows that this stays rare.
    """
    X, Y = make_nonlinear_data(n=3000, seed=1)
    inner = make_inner("rf", **FAST_RF).fit(X[:2000], Y[:2000, 0])
    p = inner.estimator_.predict_proba(X[2000:])[:, 1]
    assert np.mean((p == 0.0) | (p == 1.0)) < 0.01
    assert np.all(np.isfinite(inner.latent(X[2000:])))


def test_rf_margins_beat_linear_on_nonlinear_data():
    X, Y = make_nonlinear_data(n=3000, seed=1)
    X_tr, Y_tr, X_te, Y_te = X[:2000], Y[:2000], X[2000:], Y[2000:]

    common = dict(dependence="pairwise", cv=3, random_state=0)
    forest = MultivariateProbit(inner="rf", inner_params=FAST_RF, **common).fit(X_tr, Y_tr)
    with pytest.warns(UserWarning, match="calibration slope"):
        linear = MultivariateProbit(inner="linear", **common).fit(X_tr, Y_tr)

    assert forest.score(X_te, Y_te) > linear.score(X_te, Y_te)
    # Cross-fitted forest margins are roughly on scale; a saturated or
    # in-sample index would read far from 1.
    slopes = forest.calibration_[:, 1]
    assert np.all((slopes > 0.7) & (slopes < 1.5))
    # Attenuated toward zero by margin error, as with any flexible margin.
    assert 0.15 < forest.correlation_[0, 1] < 0.6


def test_rf_mixes_with_other_presets():
    X, Y = make_nonlinear_data(n=1500, seed=2)
    inner = [make_inner("rf", **FAST_RF), "linear"]
    model = MultivariateProbit(inner=inner, dependence="pairwise", cv=3, random_state=0)
    model.fit(X, Y)
    assert isinstance(model.inner_models_[0], ProbitCalibrated)
    assert not isinstance(model.inner_models_[1], ProbitCalibrated)
    assert model.predict_proba(X).shape == Y.shape
