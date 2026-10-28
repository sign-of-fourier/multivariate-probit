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

| d | median p | multivariate-probit `orthant` (`resolution="high"`) | multivariate-probit `orthant` (`resolution="highest"`) | multivariate-probit `quadrature` | SciPy, default tolerances | pybhatlib (Bhat OVUS) | statsmodels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.348 | 1.5e-05 s / 0.0001 (0.0002) | 4.3e-05 s / 0.0000 (0.0000) | 7.8e-05 s / 0.0000 (0.0001) | 5.8e-03 s / 0.0000 (0.0000) | 8.5e-05 s / 0.0001 (0.0009) | 6.6e-07 s / 0.0284 (0.0992) |
| 5 | 0.154 | 2.4e-05 s / 0.0001 (0.0003) | 7.3e-05 s / 0.0000 (0.0000) | 4.3e-02 s / 0.0000 (0.0001) | 1.0e-02 s / 0.0000 (0.0000) | 1.2e-04 s / 0.0004 (0.0019) | 5.7e-07 s / 0.0246 (0.1287) |
| 8 | 0.101 | 4.3e-05 s / 0.0001 (0.0005) | 1.0e-04 s / 0.0000 (0.0000) | impractical past d = 7 | 1.6e-02 s / 0.0000 (0.0000) | 1.8e-04 s / 0.0003 (0.0023) | 9.9e-07 s / 0.0145 (0.1154) |
| 12 | 0.022 | 6.3e-05 s / 0.0001 (0.0007) | 1.5e-04 s / 0.0000 (0.0000) | impractical past d = 7 | 2.4e-02 s / 0.0000 (0.0000) | 2.4e-04 s / 0.0002 (0.0036) | 1.5e-06 s / 0.0063 (0.0776) |
| 16 | 0.0024 | 9.9e-05 s / 0.0000 (0.0006) | 2.2e-04 s / 0.0000 (0.0000) | impractical past d = 7 | 3.3e-02 s / 0.0000 (0.0000) | 3.2e-04 s / 0.0000 (0.0011) | 1.5e-06 s / 0.0015 (0.0536) |

### Pairwise-shaped Σ (near-singular)

| d | median p | multivariate-probit `orthant` (`resolution="high"`) | multivariate-probit `orthant` (`resolution="highest"`) | multivariate-probit `quadrature` | SciPy, default tolerances | pybhatlib (Bhat OVUS) | statsmodels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.431 | 2.0e-05 s / 0.0006 (0.0029) | 4.4e-05 s / 0.0001 (0.0007) | 7.3e-05 s / 0.0000 (0.0001) | 7.3e-03 s / 0.0000 (0.0000) | 8.1e-05 s / 0.0004 (0.0047) | 4.6e-07 s / 0.0371 (0.1600) |
| 5 | 0.171 | 2.4e-05 s / 0.0061 (0.0337) | 6.5e-05 s / 0.0031 (0.0325) | 4.6e-02 s / 0.0000 (0.0001) | 1.4e-01 s / 0.0000 (0.0000) | 1.2e-04 s / 0.0015 (0.0276) | 5.6e-07 s / 0.0523 (0.2549) |
| 8 | 0.097 | 4.0e-05 s / 0.0048 (0.0379) | 9.1e-05 s / 0.0022 (0.0134) | impractical past d = 7 | 1.1e+00 s / 0.0000 (0.0000) | 1.8e-04 s / 0.0015 (0.0103) | 8.1e-07 s / 0.0249 (0.1426) |
| 12 | 0.025 | 6.4e-05 s / 0.0034 (0.0431) | 1.6e-04 s / 0.0024 (0.0215) | impractical past d = 7 | 1.6e+00 s / 0.0000 (0.0001) | 2.5e-04 s / 0.0007 (0.0139) | 1.3e-06 s / 0.0143 (0.1855) |
| 16 | 0.013 | 8.4e-05 s / 0.0038 (0.0451) | 2.1e-04 s / 0.0029 (0.0452) | impractical past d = 7 | 1.8e+00 s / 0.0000 (0.0000) | 3.2e-04 s / 0.0010 (0.0151) | 1.8e-06 s / 0.0107 (0.1848) |

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
- **`orthant`** scores all rows in one compiled call.
  - **`resolution="high"`** was the fastest of the integrating methods at every
    d on both Σ shapes: 3 to 6 times faster than pybhatlib, and 330 to 440 times
    faster than SciPy on the equicorrelated Σ. Median error 0.0001 or less on the
    equicorrelated Σ (maximum up to 0.0007, against pybhatlib's 0.0009 to
    0.0036). On the pairwise-shaped Σ, median error 0.0006 to 0.0061, maximum up
    to 0.045, the largest of the integrating methods.
  - **`resolution="highest"`** costs 2.2 to 3.1 times `"high"`'s time and is
    still 1.4 to 2 times faster than pybhatlib. On the equicorrelated Σ its
    median and maximum error are below 0.00005 at every d. On the
    pairwise-shaped Σ its median error is 0.0001 to 0.0031 (`"high"`: 0.0006 to
    0.0061) and its maximum up to 0.045; pybhatlib's median there is 0.0004 to
    0.0015.
- **pybhatlib** scores with Bhat's analytic approximation (`mvncd_batch`,
  `method="ovus"`), one row at a time in Python. It is 70 to 105 times faster
  than SciPy on the equicorrelated Σ. On the pairwise-shaped Σ its median error
  is 0.0004 to 0.0015 and its maximum 0.0047 to 0.028. pybhatlib also has a PyTorch backend that runs on
  your own GPU, but in pybhatlib 0.4.0 `mvncd` converts its inputs to NumPy
  and `mvncd_batch` loops over rows in Python, so the GPU does not speed up
  this scoring.
- **SciPy** stays accurate on the pairwise-shaped Σ but slows sharply, to
  1.1 to 1.8 s per row from d = 8. That is 54 to 69 times its time on the
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
