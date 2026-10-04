"""`MultivariateProbitProba.conditional`: P(event | given outcomes, x)."""

import numpy as np
import pytest
from scipy.stats import norm

from multivariate_probit import MultivariateProbitProba
from multivariate_probit._mvn import bvn_cdf

RHO = 0.6


def make_proba(n=50, d=2, rho=RHO, seed=0):
    rng = np.random.default_rng(seed)
    corr = np.full((d, d), rho)
    np.fill_diagonal(corr, 1.0)
    return MultivariateProbitProba(rng.normal(size=(n, d)), corr)


def test_matches_the_bivariate_closed_form():
    proba = make_proba()
    a, b = proba.eta[:, 0], proba.eta[:, 1]
    truth = bvn_cdf(a, b, RHO) / norm.cdf(a)
    assert proba.conditional(1, given={0: 1}) == pytest.approx(truth, abs=1e-9)
    # Y_0 = 0 flips the sign of eta_0 and of rho.
    truth0 = (norm.cdf(b) - bvn_cdf(a, b, RHO)) / norm.cdf(-a)
    assert proba.conditional({1: 1}, given={0: 0}) == pytest.approx(truth0, abs=1e-9)


def test_positive_dependence_moves_the_probability_the_right_way():
    proba = make_proba()
    marginal = proba.marginal[:, 1]
    assert np.all(proba.conditional(1, given={0: 1}) > marginal)
    assert np.all(proba.conditional(1, given={0: 0}) < marginal)


def test_independence_returns_the_marginal():
    proba = make_proba(d=3, rho=0.0)
    out = proba.conditional(2, given={0: 1, 1: 0})
    assert out == pytest.approx(proba.marginal[:, 2], abs=1e-6)


def test_event_values_sum_to_one():
    proba = make_proba(d=3)
    given = {0: 1}
    total = sum(
        proba.conditional({1: a, 2: b}, given=given) for a in (0, 1) for b in (0, 1)
    )
    assert total == pytest.approx(np.ones(len(proba)), abs=1e-5)


def test_per_row_given_values():
    """Condition each row on its own observed outcomes."""
    proba = make_proba(d=3)
    observed = np.random.default_rng(1).integers(0, 2, size=len(proba))
    out = proba.conditional(2, given={0: observed})
    on = proba.conditional(2, given={0: 1})
    off = proba.conditional(2, given={0: 0})
    assert out == pytest.approx(np.where(observed == 1, on, off))


@pytest.mark.parametrize(
    "event, given, message",
    [
        ({1: 1}, {1: 0}, "share outcomes"),
        ({1: 1}, {}, "at least one"),
        ({1: 2}, {0: 1}, "0 or 1"),
    ],
)
def test_bad_queries_are_rejected(event, given, message):
    with pytest.raises(ValueError, match=message):
        make_proba(d=3).conditional(event, given)
