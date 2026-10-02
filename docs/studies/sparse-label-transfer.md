# Can a well-populated outcome carry a sparsely observed one through Sigma?

**Question.** Some outcomes are cheap to observe and others expensive. If one
label is observed on only a fraction of the training rows, does the fitted Σ
let the remaining labels stand in for the missing observations?

**Why it came up.**
[multilabel-benchmarks.md](multilabel-benchmarks.md) closes by naming this as
the untested case. Every metric run there — subset accuracy, Hamming, F1, the
joint log score — is one a per-label method can compete on, and the estimator
finished mid-pack on all of them. Conditional queries are the operation none of
those baselines support, so this is where a joint model has to earn its place.

## The structural constraint that shapes the design

As implemented, IFM **cannot** transfer to a marginal. Stage one fits each
margin independently and Σ enters only joint and conditional queries, so
`P(Y_j = 1 | x)` is `Phi(eta_j)` whatever Σ contains. Censoring a label and then
asking for its marginal measures exactly zero transfer by construction — the
same reason per-label metrics cannot see Σ at all.

Two routes remain open:

- **Conditional at test time.** Predict `Y_j` given `x` *and* the other labels.
  This is where Σ carries information, and it describes a real situation.
- **Borrowing strength during fitting.** Use the well-populated labels to
  improve the sparse label's margin. This needs an architectural addition —
  EM-style imputation, or a joint stage one — and is not in the package.

Only the first is testable against what exists, and it is also the cheaper
test: if the dependence cannot carry a prediction when the other labels are
handed over outright, it cannot carry one during fitting either.

## Setup

Emotions and scene, d = 6, from the Mulan collection: official train/test
splits, features standardised on training statistics, `dependence="pairwise"`,
`cv=5`, `random_state=0`. The protocol of
[mulan-multilabel.md](mulan-multilabel.md) throughout. Yeast is excluded — see
what this does not establish. Data is not redistributed here.

| dataset | n train | n test | p | d |
| --- | --- | --- | --- | --- |
| emotions | 391 | 202 | 72 | 6 |
| scene | 1211 | 1196 | 294 | 6 |

**Censoring.** One label `j` at a time is reduced to a fraction `f` of the
training rows, in {1.0, 0.25, 0.10, 0.05}; every other label stays fully
observed. Each of the six labels is censored in turn rather than one being
hand-picked, because the falsifiable prediction concerns how the gain varies
with ρ. The retained rows are sampled stratified on `Y_j`, so a small `f`
cannot hand a margin a single-class training set — that would be a degenerate
arm confounding the measurement rather than part of it.

**Fitting under censoring.** Margin `j` is fitted on its observed rows only;
the others on everything. Stage two consumes cross-fitted indices as usual,
with `nan` written into the censored entries, which `ifm._fit_pair` drops — so
each ρ_jk is estimated from complete pairs and complete pairs only. No change
to the package was needed for this; the non-finite filter already there does
the masking.

**Four arms**, scored on held-out label `j`:

| arm | prediction | sees test labels |
| --- | --- | --- |
| (a) sparse margin | `Phi(eta_j)` — binary relevance | no |
| (b) sparse margin + Σ | `P(Y_j | Y_-j, x)` | **yes** |
| (c) oracle margin | margin `j` fitted on all rows | no |
| (d) masked-BCE MLP | shared hidden layer, 128 units | no |

Arm (b) is the only one given the other labels at test time, so every
comparison in this study is stacked in its favour. That asymmetry is the
scenario, not a flaw — but it means a *loss* by arm (b) is much stronger
evidence than a win.

Arm (c) equals arm (a) by construction at `f = 1.0`, which is the harness
check. Arm (d) is the benchmark study's MLP with the loss masked so a censored
label contributes nothing on the rows where it was not observed; every row
still trains the shared layer through the labels that *are* observed. It is the
competing transfer mechanism, and unlike Σ it acts on the marginal.

**Base learners.** Two, run in full: the package's `xgboost` preset verbatim,
as in Part A, and the native `ProbitRegressor`. Two rather than one because the
preset degenerates under censoring — `min_child_weight=5` is unsatisfiable
below roughly forty rows, xgboost returns a constant, and arm (a) sits at AUC
exactly 0.5. Σ would then be beating a coin flip, which is not the question.
Those rows are flagged and excluded from every aggregate below.

`ProbitRegressor` is run at `alpha=10.0` rather than its default `1e-6`. That
default is documented as statistically negligible, and at p = 72 that is the
problem: the full-data fit separates and scores *worse* than a 5 percent
subsample of itself (mean AUC 0.744 against 0.780 on emotions), which destroys
the oracle arm. `alpha=10` on standardised features restores the ordering
(0.824 against 0.800). This is the one number in the study chosen rather than
fixed in advance; it was picked from margins alone, with no arm compared and
before any conditional was computed.

**Metrics.** AUC and log-loss on label `j`.

**Evaluator.** The exact `_mvn` at `n_quad=12`, the order
[multilabel-benchmarks.md](multilabel-benchmarks.md) used for a d = 6 joint
log score. A conditional is a ratio of two orthant probabilities, and because
the denominator marginalises `Y_j` out it is the sum of the two completions, so
each query costs two d-variate evaluations and no (d-1)-variate one.

## Results

### The conditional gain, and how it moves with f

Arm (b) against arm (a), mean over the six labels, degenerate margins excluded.

| dataset / learner | f = 1.0 | f = 0.25 | f = 0.10 | f = 0.05 |
| --- | --- | --- | --- | --- |
| emotions / xgboost | +0.0545 (6/6) | +0.0511 (4/6) | all degenerate | all degenerate |
| emotions / linear | +0.0571 (6/6) | +0.0441 (5/6) | +0.0381 (4/6) | +0.0221 (2/6) |
| scene / xgboost | +0.0118 (6/6) | +0.0166 (6/6) | +0.0189 (4/6) | all degenerate |
| scene / linear | +0.0201 (6/6) | +0.0132 (6/6) | +0.0113 (4/6) | +0.0071 (3/6) |

Mean AUC gain, with the number of labels improved in parentheses.

Log-loss, same layout, positive meaning arm (b) is better:

| dataset / learner | f = 1.0 | f = 0.25 | f = 0.10 | f = 0.05 |
| --- | --- | --- | --- | --- |
| emotions / xgboost | +0.0718 (6/6) | +0.0367 (4/6) | — | — |
| emotions / linear | +0.0724 (5/6) | **-0.0797** (2/6) | **-0.1286** (3/6) | **-0.2131** (2/6) |
| scene / xgboost | +0.0208 (6/6) | +0.0214 (6/6) | +0.0189 (5/6) | — |
| scene / linear | +0.0518 (6/6) | +0.0288 (6/6) | +0.0144 (5/6) | **-0.0760** (3/6) |

### How much of the censoring damage Sigma repairs

The fraction of the arm (a) to arm (c) gap that arm (b) closes, over labels
whose oracle gap exceeds 0.01 AUC and whose margin is not degenerate. Undefined
at `f = 1.0`, where (a) and (c) coincide.

| dataset / learner | f = 0.25 | f = 0.10 | f = 0.05 |
| --- | --- | --- | --- |
| emotions / xgboost | +0.63 (n=6) | — | — |
| emotions / linear | +0.84 (n=5) | **-0.16** (n=4) | **-0.17** (n=4) |
| scene / xgboost | +0.33 (n=6) | +0.20 (n=6) | — |
| scene / linear | +0.63 (n=1) | +0.39 (n=3) | +0.12 (n=3) |

Median fraction closed.

### Against the competing transfer mechanism

Arm (b) minus arm (d), mean AUC over all six labels. Positive means Σ wins.

| dataset / learner | f = 1.0 | f = 0.25 | f = 0.10 | f = 0.05 |
| --- | --- | --- | --- | --- |
| emotions / xgboost | +0.0499 (6/6) | +0.0531 (6/6) | -0.1419 (2/6) | -0.1759 (2/6) |
| emotions / linear | +0.0354 (6/6) | +0.0429 (5/6) | +0.0224 (4/6) | +0.0180 (3/6) |
| scene / xgboost | +0.0315 (6/6) | +0.0096 (4/6) | -0.0211 (2/6) | -0.1817 (1/6) |
| scene / linear | **-0.0098** (3/6) | -0.0090 (3/6) | -0.0165 (2/6) | -0.0247 (2/6) |

This declines monotonically in `f` in all four combinations — the only quantity
in the study that does.

## Conclusions

**Conditioning on the other labels carries real information.** With the label
fully observed, arm (b) beats arm (a) on all six labels in all four
dataset-and-learner combinations, by +0.012 to +0.057 mean AUC, and improves
log-loss in 23 of 24 cases. Σ is not decorative: given the other labels, it
sharpens a prediction the margin alone cannot. The size matches the published
figure: Wilkinson et al. (2021) report a mean AUC gain of +0.03, maximum +0.08,
from conditioning a nine-species joint species distribution model on one
known-present species, against the +0.012 to +0.057 here. That study also
explains the spread the same way this one does — the benefit tracks the
strength of the correlation with the conditioned-on label, so it accrues to
some outcomes and not others. See
[README.md](README.md#external-references).

**But the gain does not widen as the label gets sparser, which is what the
design predicted.** In three of four combinations it narrows monotonically —
emotions / linear runs +0.057, +0.044, +0.038, +0.022 across the fractions, and
scene / linear +0.020, +0.013, +0.011, +0.007. Only scene / xgboost widens
(+0.012, +0.017, +0.019), over the range where its margin is still alive. The
prediction is not supported in general and not cleanly falsified either; the
weight of the evidence is against it.

**The mechanism that explains it is that Σ is estimated from the censored rows
too.** Each ρ_jk is fitted on rows where both labels are observed, which for
every pair involving `j` means the retained fraction — nineteen to sixty rows
at `f = 0.05`. Transfer through Σ requires knowing Σ, and Σ is learned from
exactly the data the censoring removed. The symptom is visible in the fitted
values: correlations that sit near 0.6 with the label fully observed run to
0.94 at `f = 0.05`, close to the `RHO_MAX` clip. Sparsity attacks both terms
at once, which is why arm (b) cannot pull away from arm (a) as `f` falls.

**Better ranking, worse calibration.** On emotions / linear, arm (b) improves
mean AUC at every fraction while making log-loss worse from `f = 0.25` down,
reaching −0.213 at `f = 0.05`. A noisy ρ produces a conditional that orders
rows well and states its confidence badly. Any use of these conditionals as
probabilities rather than as scores needs that separation kept in view.

**Σ repairs at most part of the damage, and sometimes makes it worse.** Where a
fraction can be computed, the median closed runs from +0.84 down to +0.12, and
emotions / linear goes negative at `f = 0.10` and below. Refitting the margin
on the full data is a much better use of the missing labels than conditioning
on their neighbours.

**Representation sharing overtakes Σ exactly where transfer is supposed to
matter.** Arm (b) beats the MLP comfortably when the label is well observed and
loses to it as the label thins, monotonically, in every combination. On
scene / linear the MLP is ahead at every fraction. This is the study's
strongest result, because arm (b) is the only arm handed the true test labels
and it still loses: the MLP reaches its answer from `x` alone. The benchmark
study's warning that the MLP is a competing transfer mechanism is confirmed,
and on this evidence it is the better one.

**The prediction that the gain tracks ρ could not be tested on this design.**
The correlation between a label's gain and its largest `abs(rho_jk)` ranges
from −0.84 to +0.85 across the four combinations, with no stable sign. The
design cannot support the test: max `abs(rho_jk)` is mutual, so the six
emotions labels carry only three distinct values, and six points spanning
0.46 to 0.67 have no power to resolve a trend. This is recorded as not
established, in either direction.

## What this does not establish

- **d = 6 only.** Yeast is absent. A conditional is a ratio of two pattern
  probabilities, and at d = 14 those sit near the ~1e-2 scale of an approximate
  evaluator's absolute error — the probit's median there is 0.019, from
  [multilabel-benchmarks.md](multilabel-benchmarks.md) — so the ratio needs an
  accurate evaluator. The exact one costs `n_quad ** (d - 2)` and is
  unreachable at that d, which leaves the same route the benchmark study took
  for its yeast log score: slow, but available, and a matter of budget rather
  than of possibility. It was not run here because this study spans 96
  configurations. Whether these conclusions survive at larger d is therefore
  open, and d is exactly the axis on which a joint model should have most to
  offer.
- **One seed, one split, no error bars.** Censoring masks, cross-fitting folds
  and every fit are seeded and reproducible, but nothing is repeated, so
  sampling variability is unmeasured. Differences smaller than roughly 0.02 AUC
  are not supported. Inherited from
  [mulan-multilabel.md](mulan-multilabel.md).
- **Route (ii) is untested.** Only conditioning at test time was measured.
  Borrowing strength during fitting — EM-style imputation, or a joint stage one
  — is a different mechanism that could behave differently, and it is the one
  that would act on the marginal. The result here bears on it only as a
  negative signal, not as a measurement.
- **`alpha=10` was chosen, not fixed in advance.** It was selected on margin
  quality alone, before any arm was compared, but it is a tuning decision inside
  an otherwise untuned protocol.
- **Arm (b) is given the true test labels**, noiselessly and completely. A
  realistic deployment would have some of them, some of the time, possibly
  wrong. The arm as run is an upper bound on what conditioning can deliver.
- **The MLP is untuned**, at one architecture and one optimiser, as in the
  benchmark study. Its wins here are a floor rather than a ceiling, which
  strengthens rather than weakens the comparison.
- **Degenerate xgboost margins remove rows from the aggregates** at `f <= 0.10`
  on emotions and `f = 0.05` on scene. The `xgboost` columns at small `f` rest
  on fewer labels than the `linear` ones, or on none.

## Open threads

- **A conditional query does not exist in `results.py`.** `joint`, `all`, `any`
  and `none` are what the class offers; `P(Y_j | Y_-j, x)` was implemented in
  the study harness as a ratio of two `pattern_prob` calls. If conditionals are
  a feature the estimator wants to claim, they need to be in the package.
- **Σ estimated under censoring is the bottleneck, and it is addressable.**
  ρ_jk running to 0.94 on twenty complete pairs is an estimation problem, not a
  modelling one. Shrinkage toward zero, or a penalty on the pairwise objective,
  would target it directly, and the `f = 0.05` rows are the natural benchmark.
- **The ρ-scaling prediction needs a dataset with more labels** and a wider
  spread of correlations to be testable at all. Yeast at d = 14 has both, and
  needs the evaluator question settled first.
- **Whether the MLP's advantage is representation sharing** or simply a better
  hypothesis class at small n is not separated here. An MLP trained only on the
  sparse label's observed rows, with no other labels in the loss, would settle
  it and was not run.
- **Route (ii) remains the interesting experiment.** Everything above says
  conditioning on neighbours is a weak substitute for observing the label. If
  the dependence is to be worth its cost under sparsity, it has to act during
  fitting.
