# GHK against the compiled orthant backend

**Question.** [implementation.md](../implementation.md) rejected GHK simulation
on design grounds -- a random likelihood inside a derivative-free optimizer --
without measuring it. For *scoring* a fitted model, where that objection does
not apply, how does GHK compare with the compiled `orthant` backend in
accuracy and speed?

## Summary

| | accuracy | speed (s per row) |
| --- | --- | --- |
| **GHK, 100 draws** | median error 0.0003-0.0060 | 5e-5 to 4e-4 |
| **GHK, 1000 draws** | median error 0.0001-0.0019 | 3e-4 to 2e-3 |
| **`orthant`, `resolution="high"`** | median error 0.0000-0.0068 | 2e-5 to 9e-5, deterministic |
| **`orthant`, `resolution="low"`** | median error up to 0.023 | about the same as high |
| **SciPy** (the reference) | -- | 3e-3 to 2.1 |

At `resolution="high"` the compiled backend is 3-6 times faster than GHK at
100 draws with comparable accuracy, better on a well-conditioned Σ and about
equal on a pairwise-shaped one. GHK at 1000 draws is more accurate on a
pairwise-shaped Σ and 15-20 times slower. The compiled backend returns the same
value on every call; GHK does not.

## Setup

The probability of the pattern that actually occurred, `P(Y = y | x)`, at
d = 4, 6, 10, 14, 20. η is drawn `N(0, 1)` and the pattern is drawn from the
model, as in [comparators.md](comparators.md), whose two Σ shapes are reused:
**well-conditioned** (`A A^T + d I`, rescaled) and **pairwise-shaped** (rank-2
factor plus noise, projected through `nearest_correlation`, so on the PSD
boundary). 30 rows at d ≤ 10, 15 rows at d ≥ 14; one Σ per cell; seed 0.

- **Reference**: `scipy.stats.multivariate_normal.cdf` at its defaults
  (`abseps = releps = 1e-5`), one row at a time.
- **GHK**: plain Geweke-Hajivassiliou-Keane, pseudo-random uniforms, no
  antithetics or quasi-random points, numpy, vectorised over rows and draws.
  100 and 1000 draws per row.
- **`orthant`**: `multivariate_probit.orthant.cdf` with a key, both
  resolutions, all rows in one call. Version shipped in 0.2.1.

Errors are absolute differences from the reference. Times are wall-clock on a
two-core x86-64 Linux machine, divided by the row count.

## Results

Median absolute error, with the maximum in parentheses:

| d | Σ | median p | GHK 100 | GHK 1000 | `orthant` low | `orthant` high |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | well | 0.188 | 0.0013 (0.0085) | 0.0007 (0.0044) | 0.0052 (0.0211) | 0.0003 (0.0009) |
| 4 | pairwise | 0.154 | 0.0049 (0.0272) | 0.0019 (0.0084) | 0.0205 (0.0859) | 0.0024 (0.0093) |
| 6 | well | 0.053 | 0.0009 (0.0112) | 0.0003 (0.0023) | 0.0076 (0.0412) | 0.0009 (0.0035) |
| 6 | pairwise | 0.086 | 0.0053 (0.0199) | 0.0018 (0.0135) | 0.0233 (0.1093) | 0.0068 (0.0261) |
| 10 | well | 0.014 | 0.0003 (0.0074) | 0.0001 (0.0017) | 0.0020 (0.0257) | 0.0002 (0.0010) |
| 10 | pairwise | 0.032 | 0.0060 (0.0351) | 0.0015 (0.0147) | 0.0186 (0.0859) | 0.0058 (0.0226) |
| 14 | well | 0.0007 | 0.0000 (0.0005) | 0.0000 (0.0001) | 0.0002 (0.0063) | 0.0000 (0.0009) |
| 14 | pairwise | 0.0053 | 0.0014 (0.0101) | 0.0001 (0.0050) | 0.0027 (0.0284) | 0.0023 (0.0216) |
| 20 | well | 0.0001 | 0.0000 (0.0001) | 0.0000 (0.0000) | 0.0000 (0.0002) | 0.0000 (0.0001) |
| 20 | pairwise | 0.0019 | 0.0010 (0.0041) | 0.0001 (0.0008) | 0.0017 (0.0061) | 0.0009 (0.0054) |

Seconds per row:

| d | Σ | GHK 100 | GHK 1000 | `orthant` low | `orthant` high | SciPy |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | well | 8.5e-5 | 3.4e-4 | 3.1e-5 | 2.5e-5 | 3.3e-3 |
| 4 | pairwise | 5.1e-5 | 3.0e-4 | 1.2e-5 | 1.6e-5 | 6.4e-3 |
| 6 | well | 7.2e-5 | 4.3e-4 | 1.5e-5 | 2.3e-5 | 3.9e-3 |
| 6 | pairwise | 8.2e-5 | 4.4e-4 | 1.6e-5 | 2.3e-5 | 1.1 |
| 10 | well | 1.2e-4 | 6.9e-4 | 2.5e-5 | 3.5e-5 | 8.8e-3 |
| 10 | pairwise | 1.4e-4 | 7.7e-4 | 2.3e-5 | 3.1e-5 | 2.1 |
| 14 | well | 2.3e-4 | 1.1e-3 | 4.6e-5 | 5.8e-5 | 1.8e-2 |
| 14 | pairwise | 2.6e-4 | 1.3e-3 | 3.9e-5 | 5.8e-5 | 0.82 |
| 20 | well | 3.4e-4 | 1.6e-3 | 9.7e-5 | 9.3e-5 | 3.8e-2 |
| 20 | pairwise | 3.6e-4 | 1.7e-3 | 6.5e-5 | 7.9e-5 | 0.45 |

## Conclusions

- **For scoring, the compiled backend is the fast route at GHK-100 accuracy.**
  Against GHK at 100 draws it is 3-6 times faster, more accurate on a
  well-conditioned Σ and roughly level on a pairwise-shaped one, and it is
  deterministic.
- **GHK at 1000 draws buys accuracy on a pairwise-shaped Σ** -- median error
  0.0015-0.0019 against 0.0058-0.0068 for the compiled backend at d = 6 and 10
  -- at 15-20 times the time. A caller who needs that accuracy at large d
  should use GHK or SciPy; the compiled backend is the speed option.
- **`resolution="low"` is not an accuracy setting.** It is 3-20 times less
  accurate than `high` and no faster at these batch sizes.
- **SciPy is the slowest by two to five orders of magnitude** on a
  pairwise-shaped Σ, where it runs to its point budget, as
  [comparators.md](comparators.md) found.

## What this does not establish

- One Σ per cell and 15-30 rows: the medians are indicative, not tight.
- The reference is itself approximate, to about 1e-5 absolute; errors at that
  scale (the d = 14 and 20 well-conditioned rows) are not resolved.
- GHK here is plain. Quasi-random points or antithetic draws would lower its
  error at a fixed draw count; that variant was not run.
- Batch sizes are small. Per-row times at N in the thousands, where both GHK
  and the compiled backend amortise fixed costs, were not measured here.
- GHK as the evaluator *inside* a `dependence="joint"` fit remains untested;
  the design objection in [implementation.md](../implementation.md) stands.

## Open threads

- Quasi-random GHK at 100-300 draws is the obvious next comparator.
- The hosted `quantecarlo.orthant_cdf` route at large N is not timed here.
