# Implementation notes

What is written from scratch in this library, what comes from SciPy and NumPy,
and why each choice was made. For the algorithm itself see [ifm.md](ifm.md);
for the public interface see [api.md](api.md).

## Summary

There are three maximum-likelihood fits in the package, and none of them is
delegated to a statistics library. Each is a likelihood written here, driven
either by a SciPy optimizer or by a Newton loop written here:

| Fit | Objective | Driver |
| --- | --- | --- |
| Marginal probit | binary probit log-likelihood | own Newton-Raphson (IRLS) |
| Σ, `dependence="joint"` | d-variate orthant log-likelihood | `scipy.optimize.minimize`, Nelder-Mead |
| ρ_jk, `dependence="pairwise"` | bivariate probit log-likelihood | `scipy.optimize.minimize_scalar`, bounded Brent |

## What is hand-rolled

### The probit margin

`ProbitRegressor` (`linear.py`) is a short Newton-Raphson / IRLS loop. At each
step it computes `p = Φ(η)` and `φ = φ(η)`, forms the probit IRLS weights and
working response

```
w = φ² / ( p (1 - p) )
z = η + (y - p) / φ
```

and solves the weighted normal equations `(X'WX + λI) β = X'Wz`. It calls only
`scipy.stats.norm.cdf` / `.pdf` and `numpy.linalg.solve`. A small ridge is
applied to the slopes but never to the intercept, so separated or collinear
designs do not blow up. Convergence is on the maximum absolute coefficient
update; the iteration count is kept in `n_iter_`.

This is the one inner model that needs no calibration step: it produces
η_j(x) = x'β_j directly on the probit scale, with no round trip through a
probability.

### The normal CDFs

`_mvn.py` implements Φ_2 by Gauss-Legendre quadrature of the
Drezner-Wesolowsky form of Plackett's identity, and Φ_d for d ≥ 3 by Genz's
recursive conditioning down onto it. Both are vectorised over observations and
accept a `(d, d, n)` stack so the correlation matrix may vary by row.

`scipy.stats.multivariate_normal.cdf` was deliberately not used:

- It is quasi-Monte-Carlo, so it is **stochastic**. Noise inside an optimizer
  objective is corrosive — Nelder-Mead cannot distinguish a genuine likelihood
  improvement from integration jitter.
- It takes **one** covariance matrix, not a per-row stack, so the sign trick
  would become a Python loop over 2^d patterns rather than one vectorised call.
- It is about **1e-5** accurate where the closed-form bivariate case reaches
  about 1e-12.

SciPy's version does appear in the test suite, as an independent cross-check of
the hand-rolled evaluator.

### Cross-fitting

The k-fold split is a few lines in `model.py` rather than
`sklearn.model_selection`. That keeps scikit-learn out of the core dependency
set: it is needed only by the `rf` preset, and the `xgboost` preset, not by the
estimation path.

### Nearest correlation matrix

`_corr.py` implements Higham's alternating projections with Dykstra's
correction, plus a final eigenvalue floor and rescaling to unit diagonal. Used
only to repair pairwise estimates that are not jointly coherent.

## What comes from SciPy and NumPy

Optimizers and primitives only:

- `scipy.optimize.minimize` (Nelder-Mead) — the joint Σ fit
- `scipy.optimize.minimize_scalar` (bounded Brent) — each pairwise ρ
- `scipy.stats.norm` — Φ, φ, Φ⁻¹
- `numpy.polynomial.legendre.leggauss` — quadrature nodes and weights
- `numpy.linalg` — `solve`, `eigh`, `cholesky`

## Dependencies that were rejected

**statsmodels `Probit`.** It would work. But it is a heavy dependency for one
IRLS loop, and it would put a hard third-party requirement in the core install
path when the point is to keep that path at numpy + scipy.

It does offer something the hand-rolled loop does not: standard errors and
inference for the margin coefficients. That is worth knowing, but it does not
solve the harder problem — even correct margin standard errors would be the
wrong ones for step 4, since they ignore that ρ conditions on estimated
margins. See the inference note in [ifm.md](ifm.md#inference).

**scikit-learn `LogisticRegression` or its GLM classes.** Not an option at all:
`LogisticRegression` is logit-only, and the GLM classes do not offer a binomial
probit link.

**GHK simulation** for the orthant probability. Rejected for the same reason as
SciPy's QMC integrator: a stochastic likelihood inside a derivative-free
optimizer. The deterministic recursion is used instead, at the cost of scaling
poorly past d ≈ 7.

## Numerical accuracy

| Quantity | Accuracy | Notes |
| --- | --- | --- |
| Φ_2 | ~1e-12 | over abs(r) ≤ 0.999, 24-point quadrature |
| Φ_d, d ≥ 3 | ~1e-5 | deterministic, so repeatable |
| Pattern probabilities | sum to 1 exactly | algebraic consequence of the sign trick |

The last row is worth stating precisely: the 2^d pattern probabilities sum to 1
to machine precision because the sign trick makes that an identity, not a
numerical accident. But strict inequalities between *nested* events (for
example `P(all three) ≤ P(first two)`) can be violated at the 1e-5 scale for
d ≥ 3. Code that depends on such an inequality should carry a tolerance.

Correlations are clipped to abs(ρ) ≤ 0.999 throughout: the quadrature
degenerates at the boundary, and a boundary correlation is not a defensible
estimate from finite data.

## Verification

Claims above are pinned by the test suite:

- `test_linear.py::test_bvn_cdf_is_accurate` — Φ_2 against the exact identity
  `Φ_2(0, 0, r) = 1/4 + asin(r)/(2π)`, then against SciPy's integrator at the
  looser tolerance SciPy itself warrants.
- `test_linear.py::test_probit_regressor_recovers_coefficients` — IRLS recovers
  known coefficients to about 0.01 on n = 20,000 (asserted at 0.05).
- `test_linear.py::test_ifm_recovers_margins_and_correlation` — the full
  two-stage fit recovers both margins and ρ = 0.6, under each `dependence`
  setting.
- `test_linear.py::test_joint_queries_are_coherent_and_beat_independence` —
  pattern probabilities sum to 1; the fitted model beats an independence
  assumption on joint log-likelihood.
- `test_xgboost.py::test_cross_fitting_removes_upward_bias_in_the_correlation`
  — the bias mechanism documented in [ifm.md](ifm.md#why-cross-fitting-is-required).
