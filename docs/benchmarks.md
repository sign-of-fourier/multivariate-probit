# Benchmarks against other tools

**Task.** Score `P(Y = y | x)`, the probability of the pattern each held-out
row actually has, from one fitted model. Synthetic data: 6 features, linear
margins, 3,000 training rows, 100 scored rows per d. The model is
`MultivariateProbit(inner="linear", dependence="pairwise", random_state=0)`.
Every column scores that same fitted model (η̂, Σ̂), except statsmodels.

There are two shapes of true Σ:

- **Equicorrelated**, ρ = 0.4: the easy case.
- **Pairwise-shaped**: a rank-2 factor plus noise, projected to the nearest
  correlation matrix (smallest eigenvalue about 1e-8), the same construction
  as [studies/ghk.md](studies/ghk.md). A fitted pairwise Σ̂ lands on that PSD
  boundary, which is hard for every approximation.

**Reference.** SciPy `multivariate_normal.cdf` with `abseps=1e-7`,
`releps=1e-6`, one row at a time. Error is the absolute difference from it in
probability: median over rows, maximum in parentheses. Time is wall-clock per
row after one warm-up call, on a two-core x86-64 Linux machine.

Reproduce with [benchmarks/benchmarks.ipynb](../benchmarks/benchmarks.ipynb)
(logic in `benchmarks/bench.py`; raw numbers in `benchmarks/results.csv`).
The reference values are cached in `benchmarks/ref_*.npz` together with the
problem they were computed for, so a rerun skips the slow part and a changed
problem recomputes. The `orthant` column needs a key in `$ORTHANT_KEY` or
`~/.orthant/key`.

### Equicorrelated Σ (ρ = 0.4)

| d | median p | multivariate-probit `orthant` (`resolution="high"`) | multivariate-probit `quadrature` | SciPy, default tolerances | pybhatlib (Bhat OVUS) | statsmodels |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.348 | 9.5e-6 s / 0.0001 (0.0002) | 6.2e-5 s / 0.0000 (0.0001) | 4.2e-3 s / 0.0000 (0.0000) | 6.5e-5 s / 0.0001 (0.0009) | 5.4e-7 s / 0.0284 (0.0992) |
| 5 | 0.154 | 1.3e-5 s / 0.0001 (0.0003) | 3.3e-2 s / 0.0000 (0.0001) | 6.6e-3 s / 0.0000 (0.0000) | 9.4e-5 s / 0.0004 (0.0019) | 6.4e-7 s / 0.0246 (0.1287) |
| 8 | 0.101 | 2.6e-5 s / 0.0001 (0.0005) | impractical past d = 7 | 1.3e-2 s / 0.0000 (0.0000) | 1.4e-4 s / 0.0003 (0.0023) | 1.2e-6 s / 0.0145 (0.1154) |
| 12 | 0.022 | 4.1e-5 s / 0.0001 (0.0007) | impractical past d = 7 | 1.9e-2 s / 0.0000 (0.0000) | 1.9e-4 s / 0.0002 (0.0036) | 1.2e-6 s / 0.0063 (0.0776) |
| 16 | 0.0024 | 6.6e-5 s / 0.0000 (0.0006) | impractical past d = 7 | 2.6e-2 s / 0.0000 (0.0000) | 2.6e-4 s / 0.0000 (0.0011) | 2.3e-6 s / 0.0015 (0.0536) |

### Pairwise-shaped Σ (near-singular)

| d | median p | multivariate-probit `orthant` (`resolution="high"`) | multivariate-probit `quadrature` | SciPy, default tolerances | pybhatlib (Bhat OVUS) | statsmodels |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.431 | 9.2e-6 s / 0.0008 (0.0029) | 6.0e-5 s / 0.0000 (0.0001) | 6.5e-3 s / 0.0000 (0.0000) | 6.4e-5 s / 0.0004 (0.0047) | 4.2e-7 s / 0.0371 (0.1600) |
| 5 | 0.171 | 1.3e-5 s / 0.0061 (0.0338) | 3.5e-2 s / 0.0000 (0.0001) | 1.1e-1 s / 0.0000 (0.0000) | 9.5e-5 s / 0.0015 (0.0276) | 5.7e-7 s / 0.0523 (0.2549) |
| 8 | 0.097 | 2.5e-5 s / 0.0048 (0.0379) | impractical past d = 7 | 8.8e-1 s / 0.0000 (0.0000) | 1.4e-4 s / 0.0015 (0.0103) | 9.3e-7 s / 0.0249 (0.1426) |
| 12 | 0.025 | 3.9e-5 s / 0.0034 (0.0431) | impractical past d = 7 | 1.2e+0 s / 0.0000 (0.0000) | 2.0e-4 s / 0.0007 (0.0139) | 1.2e-6 s / 0.0143 (0.1855) |
| 16 | 0.013 | 5.4e-5 s / 0.0037 (0.0453) | impractical past d = 7 | 1.8e+0 s / 0.0000 (0.0000) | 2.5e-4 s / 0.0010 (0.0151) | 1.5e-6 s / 0.0107 (0.1848) |

R `mvProbit` was not run (see below).

The reference itself took 0.24 to 1.6 s per row on the equicorrelated Σ and
0.24 to 20 s per row on the pairwise-shaped one. At d = 16 on the
equicorrelated Σ most pattern probabilities are small (median 0.0024), so the
maximum error says more than the median there.

## Reading the tables

- **statsmodels** fits one `Probit` per outcome and has no correlation matrix,
  so its pattern probability is a product of marginals. Its error is the cost
  of ignoring Σ, not of numerical integration, and it does not shrink with
  more data. Its time covers prediction only.
- **`orthant`** scores all rows in one compiled call. It was the fastest of
  the integrating methods at every d on both Σ shapes: 4 to 7 times faster
  than pybhatlib, and 390 to 510 times faster than SciPy on the
  equicorrelated Σ. On the equicorrelated Σ it also had the smallest maximum
  error of the approximations. On the pairwise-shaped Σ it was the *least*
  accurate integrating method: median error 0.0008 to 0.0061, maximum up to
  0.045.
- **pybhatlib** scores with Bhat's analytic approximation (`mvncd_batch`,
  `method="ovus"`), one row at a time in Python. It is 65 to 100 times faster
  than SciPy on the equicorrelated Σ. On the pairwise-shaped Σ its median error
  (0.0004 to 0.0015) was 2 to 5 times smaller than `orthant`'s, and so was its
  maximum from d = 5 up. pybhatlib also has a PyTorch backend that runs on
  your own GPU, but in pybhatlib 0.4.0 `mvncd` converts its inputs to NumPy
  and `mvncd_batch` loops over rows in Python, so the GPU does not speed up
  this scoring.
- **SciPy** stays accurate on the pairwise-shaped Σ but slows sharply, to
  0.9 to 1.8 s per row from d = 8. That is 65 to 70 times its time on the
  equicorrelated Σ at the same d.
- **`quadrature`** matched the reference on both shapes up to d = 5.
- **R `mvProbit`** was not run: R is not installed on the benchmark machine,
  and the notebook has no R column yet.

## What this does not establish

- Two Σ shapes, one seed, and 100 rows per d. The pairwise-shaped Σ is one
  construction of a near-singular matrix; see [studies/ghk.md](studies/ghk.md)
  and [studies/comparators.md](studies/comparators.md) for more.
- Scoring only. Fitting time against full-likelihood estimation is in
  [studies/comparators.md](studies/comparators.md).
