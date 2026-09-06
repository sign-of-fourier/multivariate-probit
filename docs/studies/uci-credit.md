# Pairwise versus joint Sigma on UCI credit-card default

**Question.** `dependence="pairwise"` replaces the full d-variate likelihood
with a sum of bivariate ones. It is consistent in theory and orders of
magnitude cheaper. Does it give up anything that matters in practice?

**Why it came up.** Everything else in this library was developed against
synthetic draws where the true Σ is known. The composite-likelihood shortcut is
the design decision most likely to be wrong in a way synthetic data would not
reveal, because both objectives are correct under the model and only differ
when the model is imperfect.

## Setup

UCI *Default of Credit Card Clients* (Yeh & Lien, 2009), 30,000 clients. Three
binary outcomes: any repayment delay in September, July and April 2005 — that
is, `PAY_0`, `PAY_3`, `PAY_6` greater than zero. Five demographic predictors:
`LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`. Outcome prevalence 22.8 /
14.0 / 10.2 percent. An 80/20 split and `cv=5`.

The dataset is not redistributed here; it is available from the UCI Machine
Learning Repository.

Both modes were run on identical margins — same inner model, same seed, same
folds — so the only thing varying is stage two.

## Results

Fitted correlations (ρ_12, ρ_13, ρ_23):

| inner | joint | pairwise | max entrywise difference |
| --- | --- | --- | --- |
| linear | 0.669, 0.523, 0.682 | 0.673, 0.535, 0.690 | 0.012 |
| xgboost | 0.655, 0.510, 0.667 | 0.661, 0.524, 0.676 | 0.014 |

Pairwise came out slightly higher in every entry — a small systematic offset,
not noise.

Cost of stage two alone, on the same cross-fitted η (30,000 rows, d = 3):

| inner | pairwise | joint | ratio |
| --- | --- | --- | --- |
| linear | 0.53 s | 60.3 s | 114x |
| xgboost | 0.52 s | 45.6 s | 88x |

Held-out mean joint log-likelihood:

| inner | joint | pairwise | difference |
| --- | --- | --- | --- |
| linear | -1.09124 | -1.09131 | 0.00007 nats/row |
| xgboost | -1.08451 | -1.08461 | 0.00010 nats/row |

Per-row `P(all three late)` differed between modes by 0.001 on average, 0.0026
at worst. Marginal AUC, log-loss and Brier are identical across modes by
construction, since the margins are stage one.

An independent earlier tetrachoric/Gaussian-copula fit on the same data found
correlations in the 0.51-0.66 range, with AUC 0.72, log-loss 0.173 and Brier
0.043 for the joint event "late in all three months". Both modes reproduce it:
correlations 0.510-0.655, and for that same joint event AUC 0.7224, log-loss
0.1827, Brier 0.0458.

### Margin calibration and the sampling noise floor

Calibration slopes on the same fits were 0.977 / 0.991 / 0.992 for the linear
margins and 0.927 / 0.949 / 0.956 for the boosted ones. All sit inside the
warning band, so `calibration_` is silent on this dataset.

A 40-replicate bootstrap refitting both stages on the training split
(n = 24,000, linear margins, pairwise) gives:

| quantity | sampling SD |
| --- | --- |
| ρ_12, ρ_13, ρ_23 | 0.008, 0.011, 0.010 |
| the three calibration slopes | about 0.0035 |

## Conclusions

At d = 3 with moderate positive correlations, `"pairwise"` gives up about 1e-4
nats/row of held-out likelihood and saves two orders of magnitude of fitting
time. That is a strong result for the composite objective.

The correlations should be read with their conditioning set in mind. The three
outcomes are repayment delay for the **same client** in three different months,
and the predictors are five demographics. Persistent unobserved creditworthiness
is exactly a shared omitted determinant, so much of the 0.51-0.68 is that shared
heterogeneity. That is what Σ conditional on demographics *means*, and it is why
the independent tetrachoric fit agrees — it conditions on the same information.
A model given repayment history would report smaller correlations without either
fit being wrong. See [margin-gaps.md](margin-gaps.md).

The bootstrap settles a separate question about the warning band on
`calibration_`. The slope is estimated several times more precisely than the
correlation it diagnoses, so at this sample size a departure from 1 can be
overwhelmingly significant and still immaterial: the linear margins sit 2 to 6
standard errors from a slope of 1.0 while reproducing published correlations. A
trigger keyed to statistical significance would fire on essentially every real
fit, which is why the band is a loose fixed interval and not a standard-error
rule.

## What this does not establish

- **Σ is never validated against a known truth.** There is none here. This
  compares two objectives on identical margins.
- **One regime only.** Σ was well conditioned throughout — smallest eigenvalue
  0.27 or above — so the projection step never engaged. Nothing here speaks to
  large d, near-singular Σ, or strongly mixed-sign correlations, which are the
  settings where a composite likelihood is most likely to diverge from the full
  one.
- **Cross-fitting is assumed, not retested.** It was used throughout, so these
  runs take the case for it as given; the in-sample bias measurement remains
  synthetic.
- The bootstrap is 40 replicates on one dataset with linear margins and the
  pairwise objective. Enough to size a noise floor, not a standard-error
  facility.

## Open threads

- The systematic offset — pairwise higher in every entry, both inner models —
  is unexplained. It is small enough not to matter at d = 3 and might not stay
  that way.
- The untested regimes above need a dataset that reaches them. Multi-label
  benchmark data is the obvious candidate.
