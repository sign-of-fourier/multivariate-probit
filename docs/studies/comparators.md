# Joint, pairwise, statsmodels and the orthant .so

**Question.** Against the obvious alternatives, what does this package buy in
accuracy, and what does it cost in time? Two costs are kept separate: *fitting*
(estimating η and Σ) and *scoring* (evaluating `P(Y = y | x)` for a fitted
model). Scoring is the one that grows with d.

Throughout, the error of a probability is reported as a difference, never as a
percentage.

## Summary

| | accuracy | speed |
| --- | --- | --- |
| **joint** | `P(any)` error 0.021 | fit **2-4 min** at d = 4, n = 250; scores through `_mvn` |
| **pairwise** | `P(any)` error 0.021 on the same fits | fit **0.07 s**; scores through `_mvn` |
| **statsmodels** | `P(any)` biased **+0.072** at ρ = +0.6, **-0.036** at ρ = -0.3 | fit 0.007 s; scores in closed form, ~2e-6 s per row |
| **orthant .so** | error 0.001 or less on a well-conditioned Σ; 0.005-0.017 typical, up to 0.1, on a pairwise-shaped Σ | ~1e-5 s per row, flat in d up to 20 |

## Setup

**Accuracy and fit time.** Synthetic, with known truth: d = 4, three
standard-normal predictors, coefficients `N(0, 0.7²)`, intercepts `N(0, 0.5²)`,
equicorrelated Σ at ρ = +0.6 and ρ = -0.3. 250 training rows, 2,000 test rows,
five seeds per ρ. The query is `P(any)`, 1 minus the all-negative orthant,
which is where ignoring Σ does the most damage. Truth and every estimate are
scored with `_mvn` at `n_quad=24`, which is exact at this d.

- **joint**: `MultivariateProbit(dependence="joint")`. Fitted on four data
  sets only (three at +0.6, one at -0.3), because each fit takes 2-4 minutes.
- **pairwise**: `MultivariateProbit(dependence="pairwise")`, all ten data sets.
- **statsmodels**: one `sm.Probit` per outcome, so `P(any)` is
  `1 - prod_j (1 - p_j)`. Its margins match `ProbitRegressor`; only Σ is
  missing.

**Scoring time against d.** Time per row of `P(Y = y | x)` for a random
pattern on a well-conditioned random Σ, d = 3 to 20. `_mvn` at `n_quad=12`,
the settled order for a joint log score, batched over 50 rows at d ≤ 6 and
timed on one row at d = 7. The .so batched over 50 rows. SciPy one row at a
time.

**The .so against SciPy.** The binary measured here is the trial build that
preceded the `orthant` package now bundled as `evaluator="orthant"`; the
shipped build is measured in [ghk.md](ghk.md). Each row
scores the probability of the pattern that actually occurred: η drawn
`N(0, 1)`, the pattern drawn from the model itself, so the probabilities have
realistic sizes. SciPy (`multivariate_normal.cdf`, default tolerances,
`allow_singular=True`) is the reference; at d ≤ 6 it agrees with `_mvn` to
within 0.0007. Two kinds of Σ:

- **well-conditioned**: `A A^T + d I`, rescaled to unit diagonal.
- **pairwise-shaped**: a rank-2 factor plus `0.3 I`, rescaled, with symmetric
  `N(0, 0.25²)` noise on the off-diagonal, then projected through
  `nearest_correlation`. This lands on the PSD boundary (smallest eigenvalue
  1e-8), as every pairwise fit does.

The .so ran at its defaults and again with its accuracy settings raised.

## Results

### statsmodels, joint and pairwise

Mean absolute error of `P(any)` against truth:

| | ρ = +0.6 | ρ = -0.3 |
| --- | --- | --- |
| joint, per fit | 0.036, 0.019, 0.019 | 0.009 |
| pairwise, same data sets | 0.037, 0.019, 0.018 | 0.011 |
| pairwise, mean of 5 seeds | 0.020 | 0.006 |
| statsmodels, mean of 5 seeds | 0.073 | 0.036 |

Fit time: joint 123-228 s, pairwise 0.06-0.08 s, statsmodels 0.006-0.007 s.

### Scoring time against d

Seconds per row, well-conditioned Σ:

| d | `_mvn` (`n_quad=12`) | SciPy | .so | independence |
| --- | --- | --- | --- | --- |
| 3 | 8e-5 | 5e-3 | 6e-5 | 2e-6 |
| 4 | 9e-4 | 3e-3 | 2e-5 | 2e-6 |
| 5 | 1e-2 | 3e-3 | 9e-6 | 2e-6 |
| 6 | 0.12 | 5e-3 | 1e-5 | 2e-6 |
| 7 | 58 (one unbatched row) | 6e-3 | 2e-5 | 2e-6 |
| 10 | — | 1e-2 | 1e-5 | 2e-6 |
| 14 | — | 2e-2 | 1e-5 | 3e-6 |
| 20 | — | 5e-2 | 3e-5 | 1e-5 |

### The .so against SciPy

| d | Σ | rows | median p | .so bias | median error | max error | rows at 0 | SciPy s/row | .so s/row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | well | 200 | 0.167 | -0.001 | 0.002 | 0.008 | 0 | 0.005 | 1e-5 |
| 6 | well | 200 | 0.063 | -0.000 | 0.001 | 0.008 | 0 | 0.006 | 6e-6 |
| 10 | well | 60 | 0.009 | +0.000 | 0.000 | 0.004 | 0 | 0.011 | 1e-5 |
| 14 | well | 30 | 0.002 | +0.000 | 0.000 | 0.001 | 0 | 0.022 | 1e-5 |
| 4 | pairwise-shaped | 60 | 0.176 | -0.007 | 0.008 | 0.078 | 0 | 0.61 | 5e-6 |
| 6 | pairwise-shaped | 60 | 0.088 | -0.008 | 0.017 | 0.100 | 1 | 1.25 | 7e-6 |
| 10 | pairwise-shaped | 12 | 0.024 | -0.003 | 0.007 | 0.027 | 0 | 2.47 | 1e-5 |
| 14 | pairwise-shaped | 8 | 0.005 | +0.003 | 0.005 | 0.010 | 0 | 0.53 | 1e-5 |

Raising its accuracy settings changed no entry beyond the fourth decimal. The
.so clips its output to
[0, 1], so the rows reported at 0 are exact zeros, not negatives.

### What SciPy's time depends on

One pairwise-shaped Σ at d = 14, and its leading 10 × 10 block for d = 10. 20
rows each, with the upper limits shifted so that p spans several orders of
magnitude:

| p | d = 10 | d = 14 |
| --- | --- | --- |
| below ~0.003 | 0.01 s | 0.03 s |
| 0.02-0.1 | 1-10 s | 7-21 s |
| above 0.1 | 0.6-11 s | 21-22 s, every row |

The Spearman correlation of time with p is +0.77 at d = 10 and +0.87 at d = 14.

## Conclusions

**statsmodels is fast and wrong on every joint query.** Independence biases
`P(any)` in the direction of ρ, by 0.04-0.12 at ρ = +0.6: 3-6 times the error
of either route that fits Σ. For per-outcome probabilities it is equivalent;
for anything joint it is not a competitor.

**Joint and pairwise are equally accurate here, and pairwise fits 2,000-3,500
times faster.** Their `P(any)` errors agree to within 0.003 on every data set
where both were fitted, consistent with [uci-credit.md](uci-credit.md). There
is no accuracy case for `dependence="joint"` at this size, and its fitting
cost rises with d as `n_quad ** (d - 2)` per likelihood call.

**Beyond d ≈ 6 the limit is scoring, not fitting.** A pairwise fit is a
one-dimensional search per pair and stays cheap. The package's own evaluator
grows by a factor of about `n_quad` per extra outcome: 0.12 s per row at
d = 6, 58 s for a single unbatched row at d = 7, and out of reach beyond that.

**The .so is accurate on a well-conditioned Σ and not on the Σ a pairwise fit
produces.** On a well-conditioned Σ its error is 0.001 or less at every d
tested, with no bias, and it is 500-2,000 times faster than SciPy. On a
pairwise-shaped Σ the typical error is 0.005-0.017, the worst is 0.1, and some
rows come back as exact zeros. At d = 14 the typical error (0.005) is as large
as the typical probability (0.005), which rules it out for a log score. Its
settings do not change this.

**SciPy's cost is set by the probability and by Σ, not by d.** Its stopping
rule is an absolute error of 1e-5. A probability below about 0.003 meets that
immediately, whatever its precision, so it returns in milliseconds; at d = 14
that includes values of 0.0, 5e-324 and 1e-152, which carry no useful
precision. Larger probabilities run until they converge or hit the default
point cap of `1e6 * d`, about 21 s per row at d = 14. On a well-conditioned Σ
the same routine takes milliseconds at every d. The yeast log score
([log-score-evaluator.md](log-score-evaluator.md)) sat on a pairwise Σ with a
median p of 0.019, right at the cliff, which is how it averaged 5.8 s per row
and 88 minutes for 917 rows.

Reporting an error as a difference is the right convention. A *tolerance*
for a log score still has to be relative (`releps`), because the absolute
default is exactly what lets the smallest rows through with no precision.

## What this does not establish

- **d = 4, one Σ shape, one query for the accuracy comparison.** Joint was
  fitted four times, so its figure has no real spread.
- **The pairwise-shaped Σ is synthetic.** It reproduces the PSD-boundary
  property of a pairwise fit, not any particular data set. At d = 10 and 14
  those rows number only 12 and 8.
- **SciPy's timing curve is one Σ.** The d = 10 block is less degenerate
  (smallest eigenvalue 1.5e-4 against 1e-8), so the d = 10 against d = 14
  comparison is not matched on conditioning.
- **`_mvn` at d = 7 was timed on one unbatched row.** Batching amortises
  Python overhead, so the batched cost is probably near 12 × the d = 6 figure.

## Open threads

- **Moving a pairwise Σ off the boundary** would help both SciPy and the .so.
  The projection's eigenvalue floor (`eps=1e-8`) is what puts it there; a
  larger floor was not tested.
- **A SciPy fallback for the .so's zero rows** costs only a handful of rows and
  removes the rows with no logarithm, but it leaves the .so's error on every
  other row.
