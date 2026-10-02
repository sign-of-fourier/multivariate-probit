"""The calibration slope: what it sees, and what it is blind to.

The slope is a probit of each outcome on its own fitted index. It detects
estimation error in the margin (Gap B), the mechanism that attenuates
``correlation_``. It is blind to omitted signal (Gap A), which moves the
estimand without biasing it. Both directions are pinned here, because a
diagnostic that is quiet for the wrong reason is worse than none.
"""

import warnings

import numpy as np
import pytest

from multivariate_probit import MultivariateProbit
from multivariate_probit.model import _calibration_slope

TRUE_RHO = 0.5


def make_data(n=20000, seed=0, hidden_var=0.0):
    """Bivariate probit with linear margins, plus optional per-margin omitted signal.

    ``hidden_var`` is the variance of a determinant of the outcome that the
    fitted margin never sees. Drawn independently per outcome, so it adds no
    shared dependence and Gap A acts through the ``1 + v`` denominator alone.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    beta = np.array([[1.0, -0.5], [-0.8, 0.9], [0.3, 0.4]])
    eta = X @ beta
    if hidden_var > 0.0:
        eta = eta + rng.normal(scale=np.sqrt(hidden_var), size=eta.shape)
    corr = np.array([[1.0, TRUE_RHO], [TRUE_RHO, 1.0]])
    errors = rng.multivariate_normal(np.zeros(2), corr, size=n)
    return X, (eta + errors > 0).astype(int)


def test_a_correct_margin_calibrates_to_one():
    X, Y = make_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a correct fit must not warn
        model = MultivariateProbit(inner="linear", dependence="pairwise").fit(X, Y)

    assert model.calibration_.shape == (2, 2)
    assert model.calibration_[:, 0] == pytest.approx(np.zeros(2), abs=0.06)
    assert model.calibration_[:, 1] == pytest.approx(np.ones(2), abs=0.06)


def test_omitted_signal_moves_the_correlation_but_not_the_slope():
    """Gap A: the estimand moves, the diagnostic stays silent.

    With independent hidden variables of variance ``v``, the recoverable
    correlation is ``rho / (1 + v)``. The margin remains calibrated with
    respect to the features it can see, so the slope stays at 1 throughout --
    which is exactly why a quiet slope is not evidence that Sigma is the
    structural correlation.
    """
    for hidden_var in (0.5, 1.5):
        X, Y = make_data(hidden_var=hidden_var)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model = MultivariateProbit(inner="linear", dependence="pairwise").fit(X, Y)

        assert model.calibration_[:, 1] == pytest.approx(np.ones(2), abs=0.06)
        assert model.correlation_[0, 1] == pytest.approx(TRUE_RHO / (1 + hidden_var), abs=0.03)


def noisy_margin_data(n=600, n_noise=40, seed=5):
    """Real signal in three features, buried in forty pure-noise ones.

    The margin is correctly specified but badly estimated -- Gap B, with no
    omitted signal at all.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3 + n_noise))
    beta = np.zeros((3 + n_noise, 2))
    beta[:3] = [[1.0, -0.5], [-0.8, 0.9], [0.3, 0.4]]
    corr = np.array([[1.0, TRUE_RHO], [TRUE_RHO, 1.0]])
    errors = rng.multivariate_normal(np.zeros(2), corr, size=n)
    return X, (X @ beta + errors > 0).astype(int)


def test_estimation_error_pushes_the_slope_below_one():
    """Gap B: a correctly specified margin, poorly estimated.

    The cross-fitted index carries the right signal attenuated by estimation
    error, and the slope reports exactly that. Note it stays inside the warning
    band -- a moderate Gap B is visible in ``calibration_`` without being loud
    enough to warn about.
    """
    X, Y = noisy_margin_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = MultivariateProbit(inner="linear", dependence="pairwise", random_state=0).fit(X, Y)

    slopes = model.calibration_[:, 1]
    assert ((slopes > 0.4) & (slopes < 0.9)).all()


def test_an_in_sample_linear_margin_is_exactly_calibrated_by_construction():
    """The one case where the diagnostic says nothing at all.

    A probit margin fitted on the rows it is scored on satisfies
    ``X'(y - p) = 0``, which contains ``eta'(y - p) = 0`` and ``1'(y - p) = 0``
    -- precisely the score equations of the calibrating probit at slope 1,
    intercept 0. So ``cv=None`` with the ``linear`` preset returns 1.00
    whatever the fit is worth. The diagnostic needs either cross-fitting or an
    inner model that is not a probit MLE.
    """
    X, Y = noisy_margin_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = MultivariateProbit(inner="linear", dependence="pairwise", cv=None).fit(X, Y)

    assert model.calibration_[:, 0] == pytest.approx(np.zeros(2), abs=1e-6)
    assert model.calibration_[:, 1] == pytest.approx(np.ones(2), abs=1e-6)


def test_an_in_sample_flexible_margin_reads_far_above_one():
    """The memorisation signature cross-fitting exists to remove.

    No such score identity holds for a boosted ensemble, so an in-sample index
    that has absorbed the noise of its own rows shows up loudly.
    """
    pytest.importorskip("xgboost")
    X, Y = noisy_margin_data()

    with pytest.warns(UserWarning, match="calibration slope outside"):
        model = MultivariateProbit(
            inner="xgboost",
            dependence="pairwise",
            cv=None,
            inner_params=dict(n_estimators=60, max_depth=4, n_jobs=1, random_state=0),
        ).fit(X, Y)

    assert (model.calibration_[:, 1] > 2.0).all()


def test_degenerate_inputs_report_nan_without_warning():
    y = np.array([0, 1, 0, 1, 1, 0])
    assert np.isnan(_calibration_slope(np.zeros(6), y)).all()
    assert np.isnan(_calibration_slope(np.full(6, np.nan), y)).all()
    assert np.isnan(_calibration_slope(np.arange(6.0), np.ones(6))).all()


def test_nan_slopes_do_not_warn():
    X, Y = make_data(n=400, seed=11)
    Y[:, 1] = 1  # single-class outcome: the slope is not identified
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = MultivariateProbit(inner="linear", dependence="pairwise").fit(X, Y)
    assert np.isnan(model.calibration_[1]).all()
