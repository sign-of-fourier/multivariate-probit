"""``fitter="modal"``: the hosted full-likelihood fit.

Offline: the HTTP client is faked, so these check the request, the polling
protocol and how a result becomes a fitted model. The live round trip runs
only with ``MVP_LIVE_MODAL=1``.
"""

import io
import json
import os
import urllib.error

import numpy as np
import pytest

from multivariate_probit import MultivariateProbit, ProbitRegressor
from multivariate_probit import _remote

D, P = 3, 2
COEF = np.array([[0.5, -1.0], [1.0, 0.2], [-0.5, 0.3]])
INTERCEPT = np.array([0.2, -0.3, 0.1])
CORR = np.array([[1.0, 0.5, -0.3], [0.5, 1.0, 0.4], [-0.3, 0.4, 1.0]])
K = D * P + D + D * (D - 1) // 2


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, P))
    E = rng.multivariate_normal(np.zeros(D), CORR, size=n)
    return X, (X @ COEF.T + INTERCEPT + E > 0).astype(int)


def _result(converged=True):
    se = np.arange(1, K + 1) / 100.0
    return {
        "coef": COEF, "intercept": INTERCEPT, "corr": CORR, "nll": np.array(123.4),
        "n_iter": np.array(17), "converged": np.array(converged), "grad_max": np.array(3e-6),
        "n_draws": np.array(512), "seed": np.array(0),
        "se_hessian": se, "se_sandwich": 2 * se,
        "cov_hessian": np.diag(se**2), "cov_sandwich": np.diag(4 * se**2),
    }


@pytest.fixture
def fake_fit(monkeypatch):
    calls = []

    def fit_remote(X, Y, sample_weight=None, **params):
        calls.append({"X": X, "Y": Y, "sample_weight": sample_weight, **params})
        return _result()

    monkeypatch.setattr(_remote, "fit_remote", fit_remote)
    return calls


def test_result_becomes_a_fitted_linear_model(fake_fit):
    X, Y = _data()
    model = MultivariateProbit(fitter="modal", random_state=7).fit(X, Y)
    assert all(isinstance(m, ProbitRegressor) for m in model.inner_models_)
    assert np.allclose(model.decision_function(X), X @ COEF.T + INTERCEPT)
    assert np.allclose(model.correlation_, CORR)
    assert model.nll_ == pytest.approx(123.4)
    assert model.fit_result_["converged"] and model.fit_result_["n_iter"] == 17
    assert model.eta_.shape == (len(X), D) and model.calibration_.shape == (D, 2)
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, D) and proba.all().shape == (5,)
    assert fake_fit[0]["seed"] == 7 and fake_fit[0]["n_draws"] == 512


def test_standard_errors_are_unpacked_in_parameter_order(fake_fit):
    X, Y = _data()
    model = MultivariateProbit(fitter="modal").fit(X, Y)
    se = np.arange(1, K + 1) / 100.0
    assert np.allclose(model.stderr_["coef"], se[: D * P].reshape(D, P))
    assert np.allclose(model.stderr_["intercept"], se[D * P : D * P + D])
    rho = model.stderr_["correlation"]
    assert rho[0, 1] == rho[1, 0] == se[-3] and rho[1, 2] == se[-1] and rho[0, 0] == 0
    assert np.allclose(model.stderr_robust_["intercept"], 2 * se[D * P : D * P + D])
    assert model.param_names_[0] == "coef[0,0]" and model.param_names_[-1] == "rho[1,2]"
    assert model.cov_params_.shape == (K, K) == model.cov_params_robust_.shape


def test_alpha_weights_and_seed_default_reach_the_service(fake_fit):
    X, Y = _data()
    w = np.linspace(0.5, 2.0, len(X))
    MultivariateProbit(fitter="modal", inner_params={"alpha": 0.1}, n_draws=128).fit(
        X, Y, sample_weight=w
    )
    call = fake_fit[0]
    assert call["alpha"] == 0.1 and call["n_draws"] == 128 and call["seed"] == 0
    assert np.array_equal(call["sample_weight"], w)


def test_non_linear_inner_is_rejected():
    X, Y = _data(50)
    with pytest.raises(ValueError, match="linear margins only"):
        MultivariateProbit(inner="rf", fitter="modal").fit(X, Y)
    with pytest.raises(ValueError, match="inner_params"):
        MultivariateProbit(fitter="modal", inner_params={"max_iter": 5}).fit(X, Y)


def test_unknown_fitter_is_rejected():
    X, Y = _data(50)
    with pytest.raises(ValueError, match="fitter must be one of"):
        MultivariateProbit(fitter="gpu").fit(X, Y)


def test_unconverged_fit_warns(monkeypatch):
    monkeypatch.setattr(_remote, "fit_remote", lambda *a, **k: _result(converged=False))
    X, Y = _data()
    with pytest.warns(UserWarning, match="iteration limit"):
        MultivariateProbit(fitter="modal").fit(X, Y)


def test_params_round_trip():
    model = MultivariateProbit(fitter="modal", n_draws=256)
    params = model.get_params()
    assert params["fitter"] == "modal" and params["n_draws"] == 256
    assert MultivariateProbit(**params).get_params() == params


# ------------------------------------------------------------ HTTP protocol


class _Resp:
    def __init__(self, status, body):
        self.status, self._body = status, body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _npz(arrays):
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def test_client_submits_npz_then_polls_until_done(monkeypatch):
    seen = []
    pending = [_Resp(202, b'{"status": "pending"}')] * 2

    def urlopen(request, timeout=None):
        if not isinstance(request, str):  # the POST
            with np.load(io.BytesIO(request.data)) as npz:
                seen.append((npz["X"].shape, json.loads(str(npz["params"]))))
            return _Resp(200, b'{"call_id": "fc-123"}')
        assert "call_id=fc-123" in request and "format=npz" in request
        return pending.pop() if pending else _Resp(200, _npz({"coef": np.ones((2, 1))}))

    monkeypatch.setattr(_remote.urllib.request, "urlopen", urlopen)
    out = _remote.fit_remote(np.zeros((4, 1)), np.zeros((4, 2)), poll_interval=0, n_draws=64)
    assert out["coef"].shape == (2, 1)
    assert seen == [((4, 1), {"n_draws": 64})]


@pytest.mark.parametrize("code, exc", [(400, ValueError), (500, RuntimeError)])
def test_client_maps_submit_errors(monkeypatch, code, exc):
    def urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            _remote.FIT_URL, code, "x", {}, io.BytesIO(b'{"error": "bad Y"}')
        )

    monkeypatch.setattr(_remote.urllib.request, "urlopen", urlopen)
    with pytest.raises(exc, match="bad Y"):
        _remote.fit_remote(np.zeros((4, 1)), np.zeros((4, 2)))


def test_client_reports_a_failed_fit(monkeypatch):
    def urlopen(request, timeout=None):
        if not isinstance(request, str):
            return _Resp(200, b'{"call_id": "fc-9"}')
        raise urllib.error.HTTPError(request, 422, "x", {}, io.BytesIO(b'{"error": "boom"}'))

    monkeypatch.setattr(_remote.urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError, match="fc-9.*boom"):
        _remote.fit_remote(np.zeros((4, 1)), np.zeros((4, 2)))


@pytest.mark.skipif(os.environ.get("MVP_LIVE_MODAL") != "1", reason="set MVP_LIVE_MODAL=1")
def test_live_round_trip():
    X, Y = _data(3000)
    model = MultivariateProbit(fitter="modal").fit(X, Y)
    assert np.abs(np.vstack([m.coef_ for m in model.inner_models_]) - COEF).max() < 0.15
    assert np.abs(model.correlation_ - CORR).max() < 0.15
    assert np.all(model.stderr_["coef"] > 0)
