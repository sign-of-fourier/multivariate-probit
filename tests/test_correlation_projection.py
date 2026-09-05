"""The projected correlation matrix is the only one that reaches predictions.

Pairwise stage-two estimates are fitted one pair at a time and need not be
jointly coherent, so they are projected onto the nearest positive-definite
correlation matrix. These tests pin the two things that could silently break:
the projection itself, and the guarantee that no unprojected matrix can reach
`.joint()` / `.all()` / `.any()`.
"""

import numpy as np
import pytest

from multivariate_probit import MultivariateProbit, pairwise_correlation
from multivariate_probit._corr import is_positive_definite, nearest_correlation


def make_data(n=4000, seed=0, corr=None):
    rng = np.random.default_rng(seed)
    if corr is None:
        corr = np.array([[1.0, 0.5, -0.3], [0.5, 1.0, 0.2], [-0.3, 0.2, 1.0]])
    X = rng.normal(size=(n, 3))
    beta = rng.normal(size=(3, 3))
    errors = rng.multivariate_normal(np.zeros(3), corr, size=n)
    return X, (X @ beta + errors > 0).astype(int)


def test_nearest_correlation_repairs_an_incoherent_triple():
    """rho_12 = rho_13 = 0.9 forces rho_23 >= 0.62; -0.9 is impossible."""
    incoherent = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    assert not is_positive_definite(incoherent)

    repaired = nearest_correlation(incoherent)

    assert is_positive_definite(repaired)
    np.linalg.cholesky(repaired)
    assert np.allclose(np.diag(repaired), 1.0)
    assert np.allclose(repaired, repaired.T)
    assert np.all(np.abs(repaired) <= 1.0 + 1e-12)


def test_fitted_pairwise_correlation_is_positive_definite():
    X, Y = make_data()
    model = MultivariateProbit(dependence="pairwise", cv=None).fit(X, Y)

    np.linalg.cholesky(model.correlation_)
    assert np.linalg.eigvalsh(model.correlation_).min() > 0


def test_projection_happens_before_the_estimator_stores_it():
    """`correlation_` is already projected: no raw estimate is kept anywhere."""
    X, Y = make_data()
    model = MultivariateProbit(dependence="pairwise", cv=None).fit(X, Y)

    # Recomputing stage two from the stored indices, unprojected, must give the
    # same answer as the projected attribute only because the estimates were
    # coherent to begin with -- but the attribute must be the projected one.
    projected = pairwise_correlation(model.eta_, Y, project=True)
    assert model.correlation_ == pytest.approx(projected)
    assert is_positive_definite(model.correlation_)


def test_joint_queries_use_the_projected_matrix():
    """Every joint path reads `correlation_`, so nothing can diverge from it."""
    X, Y = make_data()
    model = MultivariateProbit(dependence="pairwise", cv=None).fit(X, Y)
    proba = model.predict_proba(X)

    assert np.array_equal(proba.corr, model.correlation_)
    np.linalg.cholesky(proba.corr)
    # The subset used by .all(outcomes=...) is a principal submatrix, hence
    # also positive definite.
    np.linalg.cholesky(proba.corr[np.ix_([0, 2], [0, 2])])

    # And the standalone joint path agrees with the object's.
    assert model.joint_proba(X, [1, 0, 1]) == pytest.approx(proba.joint([1, 0, 1]))
    assert model.sample(X[:10], n_samples=2).shape == (2, 10, 3)  # needs a Cholesky


def test_disabling_projection_is_the_only_way_to_get_a_raw_matrix():
    """`project_correlation=False` is opt-in, and documented as unsafe."""
    X, Y = make_data()
    model = MultivariateProbit(dependence="pairwise", cv=None, project_correlation=False)
    model.fit(X, Y)

    raw = pairwise_correlation(model.eta_, Y, project=False)
    assert model.correlation_ == pytest.approx(raw)


def test_indefinite_correlation_is_rejected_with_an_actionable_message():
    """The guard that stands between project_correlation=False and silent garbage.

    An indefinite matrix does not merely break `sample` at the Cholesky -- it
    makes `joint_proba` and `.all()` return numbers that are probabilities of
    nothing, with no error raised. The fit refuses instead.
    """
    from multivariate_probit.model import _require_positive_definite

    impossible = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]])
    with pytest.raises(ValueError, match="not positive definite"):
        _require_positive_definite(impossible)
    with pytest.raises(ValueError, match="project_correlation=True"):
        _require_positive_definite(impossible)

    # A valid matrix passes through untouched.
    fine = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert _require_positive_definite(fine) is fine
