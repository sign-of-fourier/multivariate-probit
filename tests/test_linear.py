"""Linear (default) inner model: does IFM recover the data-generating process?"""

import itertools

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from multivariate_probit import MultivariateProbit, ProbitRegressor, bvn_cdf

TRUE_CORR = np.array([[1.0, 0.6], [0.6, 1.0]])
TRUE_BETA = np.array([[1.0, -0.5], [-0.8, 0.9], [0.3, 0.4]])
TRUE_INTERCEPT = np.array([-0.2, 0.35])


def make_data(n=20000, seed=0):
    """Draw from a correctly specified bivariate probit with linear margins."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, TRUE_BETA.shape[0]))
    eta = X @ TRUE_BETA + TRUE_INTERCEPT
    errors = rng.multivariate_normal(np.zeros(2), TRUE_CORR, size=n)
    Y = (eta + errors > 0).astype(int)
    return X, Y


def test_bvn_cdf_is_accurate():
    # Exact at a = b = 0, where Phi2 = 1/4 + asin(rho) / (2 pi).
    for rho in (-0.95, -0.25, 0.0, 0.5, 0.99):
        assert bvn_cdf(0.0, 0.0, rho) == pytest.approx(0.25 + np.arcsin(rho) / (2 * np.pi))

    # Elsewhere, agree with SciPy's (quasi-Monte-Carlo, hence looser) integrator.
    for rho in (-0.9, -0.25, 0.0, 0.5, 0.95):
        ref = multivariate_normal(mean=np.zeros(2), cov=[[1, rho], [rho, 1]])
        for a, b in ((1.5, -0.7), (-2.0, 2.0), (2.5, 1.0)):
            assert bvn_cdf(a, b, rho) == pytest.approx(ref.cdf([a, b]), abs=1e-4)


def test_probit_regressor_recovers_coefficients():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20000, 3))
    beta, intercept = np.array([0.8, -1.2, 0.4]), -0.3
    y = (X @ beta + intercept + rng.normal(size=X.shape[0]) > 0).astype(int)

    model = ProbitRegressor().fit(X, y)

    assert model.coef_ == pytest.approx(beta, abs=0.05)
    assert model.intercept_ == pytest.approx(intercept, abs=0.05)
    assert model.predict_proba(X).sum(axis=1) == pytest.approx(np.ones(X.shape[0]))


@pytest.mark.parametrize("dependence", ["joint", "pairwise"])
def test_ifm_recovers_margins_and_correlation(dependence):
    X, Y = make_data()
    model = MultivariateProbit(inner="linear", dependence=dependence, cv=None).fit(X, Y)

    assert model.n_outcomes_ == 2
    assert model.correlation_.shape == (2, 2)
    assert model.correlation_[0, 1] == pytest.approx(TRUE_CORR[0, 1], abs=0.05)

    for j, margin in enumerate(model.inner_models_):
        assert margin.coef_ == pytest.approx(TRUE_BETA[:, j], abs=0.06)
        assert margin.intercept_ == pytest.approx(TRUE_INTERCEPT[j], abs=0.06)


def test_cross_fitting_agrees_with_in_sample_for_linear_margins():
    """Cross-fitting is insurance against overfit margins; a probit barely needs it."""
    X, Y = make_data(n=6000, seed=5)
    plain = MultivariateProbit(dependence="pairwise", cv=None).fit(X, Y)
    crossfit = MultivariateProbit(dependence="pairwise", cv=5, random_state=0).fit(X, Y)

    assert crossfit.correlation_[0, 1] == pytest.approx(plain.correlation_[0, 1], abs=0.02)


def test_marginal_probabilities_are_calibrated():
    X, Y = make_data(seed=1)
    model = MultivariateProbit(dependence="pairwise", cv=None).fit(X, Y)

    proba = model.predict_proba(X)
    assert proba.shape == Y.shape
    assert np.asarray(proba).shape == Y.shape  # behaves like the marginal array
    assert np.all((proba.marginal >= 0) & (proba.marginal <= 1))
    assert proba.marginal.mean(axis=0) == pytest.approx(Y.mean(axis=0), abs=0.01)
    assert np.array_equal(model.predict(X), (proba.marginal >= 0.5).astype(int))
    assert np.allclose(norm.cdf(model.transform(X)), proba.marginal)


def test_joint_queries_are_coherent_and_beat_independence():
    X, Y = make_data(n=4000, seed=2)
    model = MultivariateProbit(dependence="pairwise", cv=None).fit(X, Y)
    proba = model.predict_proba(X)

    total = sum(proba.joint(p) for p in itertools.product([0, 1], repeat=2))
    assert total == pytest.approx(np.ones(X.shape[0]), abs=1e-9)

    both = proba.all()
    assert both == pytest.approx(proba.joint([1, 1]))
    assert proba.any() == pytest.approx(1.0 - proba.joint([0, 0]))
    assert proba.none() == pytest.approx(proba.joint([0, 0]))

    # The empirical (1, 1) rate should match the model's average joint estimate,
    # which an independence assumption badly understates at rho = 0.6.
    assert both.mean() == pytest.approx(np.mean((Y == 1).all(axis=1)), abs=0.01)
    assert both.mean() > np.prod(proba.marginal, axis=1).mean()

    # Modelling the dependence must beat asserting independence, on the very
    # likelihood the second IFM stage optimises.
    indep_ll = np.mean(
        np.sum(np.log(np.where(Y == 1, proba.marginal, 1 - proba.marginal)), axis=1)
    )
    assert model.score(X, Y) > indep_ll


def test_three_outcomes_and_sampling():
    rng = np.random.default_rng(3)
    corr = np.array([[1.0, 0.5, -0.3], [0.5, 1.0, 0.2], [-0.3, 0.2, 1.0]])
    X = rng.normal(size=(6000, 2))
    beta = rng.normal(size=(2, 3))
    Y = (X @ beta + rng.multivariate_normal(np.zeros(3), corr, size=X.shape[0]) > 0).astype(int)

    model = MultivariateProbit(dependence="pairwise", cv=None, random_state=0).fit(X, Y)

    assert model.correlation_ == pytest.approx(corr, abs=0.06)
    np.linalg.cholesky(model.correlation_)  # projected result must be PD

    proba = model.predict_proba(X)
    total = sum(proba.joint(p) for p in itertools.product([0, 1], repeat=3))
    assert total == pytest.approx(np.ones(X.shape[0]), abs=1e-9)  # exact by construction
    # Monotone in the event, up to the ~1e-5 quadrature error of the d >= 3
    # orthant recursion.
    assert np.all(proba.all() <= proba.all(outcomes=[0, 1]) + 1e-4)

    draws = model.sample(X, n_samples=5)
    assert draws.shape == (5, X.shape[0], 3)
    assert set(np.unique(draws)) <= {0, 1}
    assert draws.reshape(-1, 3).mean(axis=0) == pytest.approx(Y.mean(axis=0), abs=0.03)


def test_joint_and_pairwise_dependence_agree_on_three_outcomes():
    rng = np.random.default_rng(11)
    corr = np.array([[1.0, 0.55, -0.25], [0.55, 1.0, 0.15], [-0.25, 0.15, 1.0]])
    X = rng.normal(size=(1500, 2))
    beta = rng.normal(size=(2, 3))
    Y = (X @ beta + rng.multivariate_normal(np.zeros(3), corr, size=X.shape[0]) > 0).astype(int)

    joint = MultivariateProbit(dependence="joint", cv=None).fit(X, Y)
    pairwise = MultivariateProbit(dependence="pairwise", cv=None).fit(X, Y)

    assert joint.correlation_ == pytest.approx(pairwise.correlation_, abs=0.06)
    assert joint.nll_ < pairwise.nll_ is not None  # different objectives, both recorded


def test_input_validation():
    X, Y = make_data(n=200, seed=4)

    with pytest.raises(RuntimeError):
        MultivariateProbit().predict_proba(X)
    with pytest.raises(ValueError):
        MultivariateProbit().fit(X, np.full_like(Y, 2))
    with pytest.raises(ValueError):
        MultivariateProbit(inner="does-not-exist").fit(X, Y)
    with pytest.raises(ValueError):
        MultivariateProbit(dependence="magic", cv=None).fit(X, Y)

    model = MultivariateProbit(dependence="pairwise", cv=None).fit(X, Y)
    with pytest.raises(ValueError):
        model.predict_proba(X[:, :1])
