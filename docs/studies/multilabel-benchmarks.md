# How multivariate probit compares to the standard multi-label methods

**Question.** [mulan-multilabel.md](mulan-multilabel.md) established that
modelling Σ beats binary relevance on subset accuracy in 12 of 12
configurations. Binary relevance is also the weakest baseline available. Does
the estimator hold up against methods that model dependence properly, and
against the algorithms actually run on these datasets?

**Why it came up.** That study lists "no comparison against other multi-label
algorithms" under what it does not establish, and names classifier chains as
an open thread. The Σ = I arm is binary relevance, so the 12/12 result compares
the estimator against *no* dependence model rather than against a competing
one. Nothing established that Σ is a good way to model dependence, only that
some dependence beats none.

## Setup

The three Mulan datasets and protocol of
[mulan-multilabel.md](mulan-multilabel.md): official train/test splits,
features standardised on training statistics, `dependence="pairwise"`, `cv=5`,
`random_state=0`. Data is not redistributed here.

| dataset | n train | n test | p | d |
| --- | --- | --- | --- | --- |
| emotions | 391 | 202 | 72 | 6 |
| scene | 1211 | 1196 | 294 | 6 |
| yeast | 1500 | 917 | 103 | 14 |

Two comparisons are run, because they answer different questions.

**Table 1 holds the base learner fixed** at the package's `xgboost` preset for
every arm, so the only thing varying is how dependence is modelled. Binary
relevance and the probit additionally share one fitted `MultivariateProbit`
object -- Σ is swapped and swapped back, margins never refit -- so their
marginal probabilities are identical by construction.

**Table 2 adds methods from outside that family**, which is what a benchmark
comparison requires:

- **ML-kNN** (Zhang & Zhou, 2007), `k = 10`, smoothing `s = 1`. Instance-based,
  no base learner at all. Per-label Bayesian posteriors, so it models no
  dependence.
- **MLP**, one hidden layer of 128 units, one sigmoid output per label, binary
  cross-entropy, `max_iter=800`. Labels share a hidden representation but the
  outputs are conditionally independent given `x`, so it has no joint model.
- **ECC**, an ensemble of 10 classifier chains over random label orders. A
  single chain depends on an arbitrary order; the ensemble averages that away.
  Reported two ways: the standard rule, averaging each label's confidence
  across chains and thresholding, and an exact argmax over the mixture
  `(1/K) sum_k P_k(y | x)`, which is the fair match to what the probit does.

Published Mulan numbers are deliberately not quoted. Preprocessing is matched
across the rows below but would not match another paper's, and every method
here runs at fixed defaults with no tuning.

**Metrics.** Subset accuracy -- the fraction of rows whose entire predicted
label vector is exactly correct -- plus Hamming accuracy, micro and macro F1,
and the **median** log `P(Y = y | x)` of the pattern that actually occurred.
The median rather than the mean: at d = 14 a handful of rows are given
near-zero probability, and a mean of logs is dominated by them.

**Evaluators.** At d = 6 the exact `mvn_orthant` throughout (`n_quad=12` for
the joint score, `n_quad=8` for the pattern argmax). At d = 14 the exact
evaluator is unreachable -- it costs `n_quad ** (d - 2)` -- so the subset
accuracy argmax uses an external approximate orthant evaluator, and the joint
score uses `scipy.stats.multivariate_normal.cdf`. That external evaluator is
third-party, is not redistributable, and is not a dependency of this package;
it appears here only as a measurement instrument. The split is deliberate: its
error is absolute, roughly 1e-2 on a general covariance, which is negligible
against an argmax reading probabilities of 0.08-0.68 and fatal against a log
score reading probabilities near 2e-3. That split was later measured directly
in [log-score-evaluator.md](log-score-evaluator.md): on the yeast column here
the approximate evaluator lands 0.38 nats low and changes no ranking, so the
exact pass bought precision no conclusion below rests on. It remains the right
route for a *reported* figure, and the approximate one is unusable for a mean
or sum at this d, where six rows return a non-positive probability.

**A check on that instrument.** Under independence the pattern argmax
separates, so the best joint pattern is per-label thresholding at 0.5 -- a
closed form needing no integration. The approximate evaluator's Σ = I argmax
agreed with it on **100.00%** of rows on all three datasets, yeast included.
This is an exact reference at d = 14 where no other exists. It confirms the
harness and confirms the evaluator handles a factorising structure exactly; it
does *not* validate it on a general Σ at that dimension.

## Results

### Table 1 -- dependence mechanism, base learner fixed

Subset accuracy. Chains here use a single fixed label order.

| arm | emotions | scene | yeast |
| --- | --- | --- | --- |
| binary relevance | 0.2525 | 0.6221 | 0.1930 |
| classifier chains, greedy | 0.2772 | 0.6789 | 0.2356 |
| classifier chains, argmax | **0.2871** | **0.6973** | n/a |
| multivariate probit | 0.2723 | 0.6446 | 0.2356 |

### Table 2 -- benchmark comparison

Subset accuracy:

| arm | emotions | scene | yeast |
| --- | --- | --- | --- |
| binary relevance | 0.2525 | 0.6221 | 0.1930 |
| ML-kNN (k = 10) | 0.2574 | 0.5936 | 0.1658 |
| MLP (128, BCE) | **0.2871** | 0.5936 | 0.1341 |
| ECC vote (K = 10) | 0.2772 | 0.6697 | 0.2345 |
| ECC argmax (K = 10) | **0.2871** | **0.6848** | n/a |
| multivariate probit | 0.2723 | 0.6446 | **0.2356** |

Median log `P(Y = y | x)`:

| arm | emotions | scene | yeast |
| --- | --- | --- | --- |
| binary relevance | -2.2887 | -0.6184 | -5.7375 |
| ML-kNN | -2.4269 | -0.9082 | -5.8347 |
| MLP | -3.5890 | **-0.2332** | -20.3174 |
| ECC (K = 10) | **-2.0560** | -0.4942 | **-3.2725** |
| multivariate probit | -2.1115 | -0.5688 | -3.9710 |

Micro F1 / macro F1:

| arm | emotions | scene | yeast |
| --- | --- | --- | --- |
| binary relevance | 0.6667 / 0.6601 | 0.7592 / 0.7622 | 0.6475 / 0.4026 |
| ML-kNN | 0.6390 / 0.6139 | 0.7152 / 0.7196 | 0.6254 / 0.3438 |
| MLP | **0.6813 / 0.6761** | 0.7286 / 0.7319 | 0.5972 / **0.4280** |
| ECC vote | 0.6658 / 0.6596 | **0.7710 / 0.7757** | **0.6542** / 0.4067 |
| multivariate probit | 0.6702 / 0.6625 | 0.7652 / 0.7690 | 0.6531 / 0.4100 |

### Cost of one joint probability

The metric tables hide a structural difference. `P(Y = y | x)` for a single row:

| method | how it is computed | cost |
| --- | --- | --- |
| binary relevance, ML-kNN, MLP | product of d numbers | microseconds |
| classifier chains | d classifier calls per chain, times K | milliseconds |
| multivariate probit | numerical integration over a d-variate orthant | ~0.04 s at d = 6 (exact, vectorised); 5.8 s at d = 14 by the exact route, ~3e-5 s by the approximate one |

## Conclusions

**Ensemble chains are the stronger predictor at d = 6, on every metric that
matters here.** ECC argmax leads subset accuracy on emotions (0.2871 against
0.2723) and scene (0.6848 against 0.6446), and leads the median log score on
emotions and yeast. The probit is second or third throughout. Against a
competing dependence model rather than against none, it does not win.

**At d = 14 the probit edges ECC on subset accuracy, 0.2356 against 0.2345 --
but that difference is inside the noise.** 28.4 percent of its rows have a
first-to-second gap below the evaluator's error floor. The ranking is
directional; the fourth decimal is not real. ECC's better median log score at
the same d, computed exactly, is the more trustworthy comparison and it goes
the other way -- and it survives the approximate evaluator too, by a margin of
1.8x its error ([log-score-evaluator.md](log-score-evaluator.md)).

**The 12/12 result from [mulan-multilabel.md](mulan-multilabel.md) stands and
means less than it appeared to.** The probit beats binary relevance on subset
accuracy in every configuration here too. But binary relevance is the floor,
and a method that models dependence through the chain rule beats both.

**ML-kNN and MLP are erratic, and MLP's joint is badly calibrated.** Both beat
binary relevance on emotions and fall below it on scene and yeast. MLP reaches
the best median log score anywhere on scene (-0.2332) and the worst anywhere on
yeast (-20.3174), which is what a product of 14 overconfident marginals
produces. Neither was tuned, so these are a floor rather than a ceiling.

**Subset accuracy does not reward a joint model.** Any method holding per-label
probabilities produces its prediction by thresholding at 0.5, and under
independence that *is* the pattern argmax -- verified exactly above. The metric
therefore never asks for a readable Σ, for order-independence, or for arbitrary
joint and conditional queries, which are the estimator's distinguishing
features. Being mid-pack on it is consistent with those features being
valuable, and is also consistent with them not being.

**The cost profile is the reverse of its competitors'.** The probit is cheap to
fit -- stage two is pairwise, 8 to 56 seconds across these datasets -- and
expensive to query, by three to four orders of magnitude per joint probability.
ECC is the opposite: 90 to 104 seconds to train ten chains, then near-free
joints indefinitely. For workloads that score many rows repeatedly, that trade
runs against the estimator.

## What this does not establish

- **One seed, one split, no error bars.** No repetition and no estimate of
  sampling variability, so small differences between adjacent rows are not
  supported. This inherits directly from
  [mulan-multilabel.md](mulan-multilabel.md).
- **No tuning of any method.** Every arm runs at fixed defaults. ML-kNN's `k`,
  the MLP's architecture, `K` for the ensemble and the boosting parameters were
  all fixed in advance and never searched. Rankings could move under tuning.
- **The baselines are run here, not quoted.** Preprocessing is matched across
  these rows, which makes them comparable to each other and not to published
  results on the same datasets.
- **The yeast subset-accuracy column rests on an approximate evaluator** with
  no exact reference on a general Σ at that dimension. The independence check
  above validates the harness and the factorising case only. An unbiased
  sampler later reproduced the accuracy figure to under three rows
  ([log-score-evaluator.md](log-score-evaluator.md)), which supports the column
  without supplying the exact reference it lacks — and the same comparison
  shows the evaluator reading modal probabilities 23 percent low on a general Σ
  while agreeing on Σ = I.
- **ECC has no argmax row at d = 14.** Its exact pattern argmax is exponential
  in d, so the chain family is scored there by the standard vote rule while the
  probit gets an exact-in-principle argmax. The comparison at that dimension is
  therefore not rule-for-rule.
- **No conditional queries are tested anywhere.** `P(Y_j | Y_k, x)` is the
  operation the estimator supports and none of these baselines do, and nothing
  here measures it.
- **Cost figures are single-machine wall clock**, taken from these runs rather
  than from a controlled benchmark.

## Open threads

- **Conditional queries are the untested case for the estimator.** Every metric
  here is one a per-label method can compete on. Whether a well-populated
  outcome can carry a sparsely observed one through Σ is the experiment that
  would test what the estimator uniquely provides.
- **The MLP is a competing transfer mechanism.** Its shared hidden
  representation improves each label's marginal from the other labels' data,
  which is the same benefit Σ is supposed to deliver, arrived at differently
  and far more cheaply. Any claim for Σ-based transfer should be measured
  against it, not only against binary relevance.
- **The per-query cost may be reducible.** Nothing here explores whether joint
  probabilities at large d admit a cheaper route than a full orthant integral.
  Partly answered since: a pattern argmax does not need one at all
  ([log-score-evaluator.md](log-score-evaluator.md)).
- **Subset accuracy may be the wrong scoreboard for the intended user.** The
  metric is settled above as one that never asks for a readable Σ. The applied
  multivariate probit literature does not use it: Wilkinson et al. (2021) score
  joint against independent log-likelihood and species-level AUC, and treat the
  partitioning of environmental effect from residual correlation as the point
  of the model rather than a means to a predicted vector. Finishing mid-pack on
  a metric that audience does not compute is weaker evidence against the
  estimator than it appears here. Which audience is being served — predictive
  multi-label, or Σ as an estimand — is undecided, and it determines whether
  the gap that matters is accuracy or the absence of standard errors.
- **Predicted label counts are never checked against observed.** Wilkinson et
  al. (2021) report a JSDM overpredicting species richness by roughly one
  species per site across nine species, and attribute it to a largely positive
  correlation matrix combined with high marginal probabilities. The yeast Σ
  here reaches ρ = +0.99. Comparing `sum_j Y_hat` against `sum_j Y` per row is
  a cheap diagnostic and has not been run on any dataset in this archive.
- **Why chains win at d = 6.** The chain rule factorisation is exact for any
  joint, while the Gaussian copula is an assumption. Whether the gap reflects
  that assumption failing, or merely the greedy-versus-argmax difference in
  prediction rules, is not separated here.
- **A beam search over patterns** would give the chain family an approximate
  argmax at d = 14 and make that row rule-for-rule.
