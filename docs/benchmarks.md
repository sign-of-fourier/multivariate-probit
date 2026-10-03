# Benchmarks against other tools

**Task.** Score `P(Y = y | x)`, the probability of the pattern each held-out
row actually has, from one fitted model. Synthetic data: 6 features, linear
margins, equicorrelated Σ at ρ = 0.4, 3,000 training rows, 100 scored rows per
d. The model is `MultivariateProbit(inner="linear", dependence="pairwise")`.
Every column scores that same fitted model (η̂, Σ̂), except statsmodels.

**Reference.** SciPy `multivariate_normal.cdf` with `abseps=1e-7`,
`releps=1e-6`, one row at a time. Error is the absolute difference from it in
probability: median over rows, maximum in parentheses. Time is wall-clock per
row after one warm-up call, on a two-core x86-64 Linux machine.

Reproduce with [benchmarks/benchmarks.ipynb](../benchmarks/benchmarks.ipynb)
(logic in `benchmarks/bench.py`; raw numbers in `benchmarks/results.csv`).
The reference values are cached in `benchmarks/ref_*.npz`, so a rerun skips
the slow part. The `orthant` column needs a key in `$ORTHANT_KEY` or
`~/.orthant/key`.

| d | median p | multivariate-probit `orthant` (`resolution="high"`) | multivariate-probit `quadrature` | SciPy, default tolerances | pybhatlib (Bhat OVUS) | statsmodels |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.348 | 8.9e-6 s / 0.0001 (0.0002) | 6.0e-5 s / 0.0000 (0.0001) | 4.2e-3 s / 0.0000 (0.0000) | 6.4e-5 s / 0.0001 (0.0009) | 4.4e-7 s / 0.0284 (0.0989) |
| 5 | 0.154 | 1.6e-5 s / 0.0001 (0.0003) | 3.4e-2 s / 0.0000 (0.0001) | 7.1e-3 s / 0.0000 (0.0000) | 9.4e-5 s / 0.0004 (0.0019) | 5.8e-7 s / 0.0245 (0.1288) |
| 8 | 0.101 | 2.3e-5 s / 0.0001 (0.0005) | impractical past d = 7 | 1.3e-2 s / 0.0000 (0.0000) | 1.4e-4 s / 0.0003 (0.0023) | 8.4e-7 s / 0.0144 (0.1152) |
| 12 | 0.022 | 3.7e-5 s / 0.0001 (0.0007) | impractical past d = 7 | 1.9e-2 s / 0.0000 (0.0000) | 2.0e-4 s / 0.0002 (0.0035) | 1.1e-6 s / 0.0064 (0.0782) |
| 16 | 0.0024 | 6.0e-5 s / 0.0000 (0.0006) | impractical past d = 7 | 2.6e-2 s / 0.0000 (0.0000) | 2.5e-4 s / 0.0000 (0.0012) | 1.7e-6 s / 0.0015 (0.0534) |

R `mvProbit` was not run (see below).

The reference itself took 0.24 to 1.7 s per row. At d = 16 most pattern
probabilities are small (median 0.0024), so the maximum error says more than
the median there.

## Reading the table

- **statsmodels** fits one `Probit` per outcome and has no correlation matrix,
  so its pattern probability is a product of marginals. Its error is the cost
  of ignoring Σ, not of numerical integration, and it does not shrink with
  more data. Its time covers prediction only.
- **`orthant`** scores all rows in one compiled call. On this Σ it was the
  fastest of the integrating methods at every d, 4 to 7 times faster than
  pybhatlib and 430 to 570 times faster than SciPy, with the smallest maximum
  error of the approximations.
- **pybhatlib** scores with Bhat's analytic approximation (`mvncd_batch`,
  `method="ovus"`), one row at a time in Python. It is 65 to 105 times faster
  than SciPy, with a median error of 0.0004 or less.
- **This Σ is the easy case.** [studies/ghk.md](studies/ghk.md) measured
  `orthant` on a near-singular, pairwise-shaped Σ, where its median error rose
  to 0.0024-0.0068. pybhatlib has not been measured there.
- **R `mvProbit`** was not run: R is not installed on the benchmark machine,
  and the notebook has no R column yet.

## What this does not establish

- One Σ shape (equicorrelated, ρ = 0.4) and 100 rows per d. A near-singular
  pairwise-shaped Σ is harder for every approximation; see
  [studies/ghk.md](studies/ghk.md) and [studies/comparators.md](studies/comparators.md).
- Scoring only. Fitting time against full-likelihood estimation is in
  [studies/comparators.md](studies/comparators.md).
