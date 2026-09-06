# Does modelling Sigma beat binary relevance? Mulan multi-label benchmarks

**Question.** Stage two exists to model dependence between outcomes. Setting
Σ = I removes it and leaves exactly binary relevance — the canonical multi-label
baseline. On identical margins, is the fitted Σ worth anything held out?

**Why it came up.** [uci-credit.md](uci-credit.md) validated the *objective*
(pairwise against joint) but never asked whether dependence pays, and it names
three regimes it does not reach: large d, near-singular Σ, and strongly
mixed-sign correlations. Multi-label benchmark data reaches all three.

There is a second reason. Classical multivariate probit is linear-only, and on
data shaped like this a linear margin often fails outright — which is why a
pluggable inner model is this package's reason to exist. That claim had never
been tested against anything but synthetic draws.

## Setup

Three datasets from the [Mulan
repository](https://mulan.sourceforge.net/datasets-mlc.html), using their
official train/test splits. Not redistributed here.

| dataset | n train | n test | p | d | label density |
| --- | --- | --- | --- | --- | --- |
| emotions | 391 | 202 | 72 | 6 | 0.311 |
| scene | 1211 | 1196 | 294 | 6 | 0.179 |
| yeast | 1500 | 917 | 103 | 14 | 0.303 |

Features standardised on training statistics. `dependence="pairwise"`, `cv=5`,
`random_state=0` throughout.

**Four margins, all fixed in advance and never tuned on test:** `linear` at
`alpha` = 1e-6 (the package default, and the faithful classical MVP choice,
which is unpenalised maximum likelihood), 1.0, and 10.0; and `xgboost` at preset
defaults.

**The two arms share one fitted model.** Σ is swapped on the fitted object and
swapped back; margins are never refit, so marginal probabilities are identical
between arms and any difference is attributable to stage two alone.

**Metrics.** Held-out mean joint log-likelihood, and subset accuracy — the
`argmax` over all `2^d` patterns, which is what multi-label work reports and
what `.joint()` targets. Joint scoring uses `n_quad=12` and the pattern argmax
`n_quad=8`, applied identically to both arms; the differences at stake are far
above that quadrature error.

At d = 14 neither is available: the orthant evaluator costs `n_quad ** (d - 2)`,
so a joint probability cannot be computed at any quadrature order, and the
16,384-pattern argmax is out of reach. Yeast is therefore scored by held-out
**composite pairwise log-likelihood** — summed over all 91 pairs, so its scale
is not comparable to the other two rows.

## Results

| dataset | margin | median slope | rho range | eig_min | joint LL, Σ | joint LL, Σ = I | gain | subset acc, Σ | subset acc, Σ = I |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| emotions | a=1e-6 | 0.16 | [-0.90, +0.65] | 0.0000 | -10.4503 | -10.5429 | +0.0926 | 0.1634 | 0.1287 |
| emotions | a=1.0 | 0.29 | [-0.48, +0.29] | 0.1077 | -3.2537 | -3.4542 | +0.2005 | 0.2228 | 0.1931 |
| emotions | a=10 | 0.53 | [-0.63, +0.34] | 0.0000 | -2.6749 | -2.8998 | +0.2250 | 0.2574 | 0.1980 |
| emotions | xgboost | 0.68 | [-0.75, +0.29] | 0.0000 | -2.5522 | -2.7659 | +0.2137 | 0.2723 | 0.2525 |
| scene | a=1e-6 | 0.00 | [-0.24, +1.00] | 0.0000 | -16.0496 | -15.1884 | **-0.8613** | 0.3562 | 0.3545 |
| scene | a=1.0 | 0.19 | [-0.62, +0.22] | 0.0000 | -5.0138 | -5.1245 | +0.1107 | 0.4214 | 0.4147 |
| scene | a=10 | 0.43 | [-0.56, +0.21] | 0.0000 | -2.2780 | -2.3915 | +0.1135 | 0.4916 | 0.4774 |
| scene | xgboost | 0.95 | [-0.57, +0.47] | 0.0000 | -1.1724 | -1.2349 | +0.0625 | 0.6446 | 0.6221 |

Yeast, composite pairwise log-likelihood (different scale — 91 pairs summed):

| margin | median slope | rho range | Σ | Σ = I | gain |
| --- | --- | --- | --- | --- | --- |
| a=1e-6 | 0.39 | [-0.56, +0.99] | -96.9622 | -99.0309 | +2.0686 |
| a=1.0 | 0.41 | [-0.56, +0.99] | -87.4678 | -89.5349 | +2.0671 |
| a=10 | 0.43 | [-0.68, +0.99] | -84.6848 | -86.7151 | +2.0303 |
| xgboost | 0.62 | [-0.58, +0.99] | -79.3459 | -81.2442 | +1.8983 |

All test rows carry at least one label in all three datasets, so subset accuracy
restricted to non-empty rows is identical to the figure above.

## Conclusions

**Subset accuracy improved with Σ in 12 of 12 configurations**, across every
margin and both d = 6 datasets. It is the most robust finding here, and it holds
even where the joint likelihood says Σ is harmful.

**Σ can hurt.** Scene at `alpha=1e-6` loses 0.861 nats/row against independence.
The margin has no out-of-fold signal (median slope 0.00), ρ pegs at +1.000, and
applying that Σ to held-out data is worse than assuming none. This is the first
case anywhere in this project of modelling dependence being actively harmful,
and it is worth stating plainly: a fitted Σ is only as trustworthy as the
margins under it. `calibration_` warned on that configuration.

**The gain declines as the margins improve**, once degenerate arms are excluded.
Scene falls from +0.114 at `alpha=10` to +0.063 with boosted margins; yeast from
+2.07 to +1.90. Emotions is flat across its three non-degenerate arms. This is
the behaviour [margin-gaps.md](margin-gaps.md) predicts: a weak margin leaves
shared signal unexplained, and Σ absorbs it as dependence, so part of what Σ
appears to buy is work the margin failed to do.

**Margins dominate stage two.** On emotions, regularisation alone is worth 7.2
nats/row (-10.45 to -3.25); the best Σ contributes 0.23. On scene the margin
spread is 15 nats against 0.11. Choosing a competent inner model matters far
more than anything the dependence stage does, which supports the pluggable-inner
premise more directly than any synthetic test has.

**Near-singular Σ is the normal case here, not the exception.** The smallest
eigenvalue hit the floor in 11 of 12 runs, so the Higham projection engaged
almost every time. On UCI it never engaged at all.

**Mixed signs are routine and handled.** Every dataset produced substantial
negative correlations alongside positive ones, spanning [-0.90, +1.00] across
runs, with no failure in fitting, projection or prediction.

## What this does not establish

- **No ground-truth Σ.** As with UCI, the true dependence is unknown; only
  held-out predictive performance is measured.
- **No comparison against other multi-label algorithms.** The Σ = I arm is
  binary relevance, but classifier chains, RAkEL and ML-kNN were not run.
  Published Mulan numbers were deliberately not used as a comparison, since
  matching other papers' preprocessing is its own source of error.
- **Yeast is on a different metric.** Composite pairwise log-likelihood is not
  comparable in scale to the joint figures, and it is the objective stage two
  optimised, so it flatters the fitted Σ relative to a genuinely held-out joint
  criterion.
- **One seed, one split.** No repetition, no error bars. The gains are not
  accompanied by any estimate of their sampling variability.
- Subset accuracy uses a coarse quadrature for the argmax. Identical across
  arms, so the comparison is fair, but the absolute figures are approximate.

## Open threads

- **Classifier chains as a third arm.** It runs on identical splits, factorises
  into a proper joint, and would answer "how do typical multi-label algorithms
  do" directly rather than by analogy.
- **Yeast pegs ρ at +0.99 in every arm**, regardless of margin. Either the
  dependence really is near-perfect for some label pair, or something in the
  pairwise fit is running to the bound. Unexplained.
- **The projection engaged in 11 of 12 runs.** Nothing here checks whether the
  projected Σ is closer to the truth than the raw pairwise estimates, or merely
  closer to being a valid correlation matrix. Those are different claims.
- **The composite-versus-joint question from [uci-credit.md](uci-credit.md) is
  untouched here**, because the joint objective is unaffordable at these d. The
  regimes that study named as untested are exactly the ones where the comparison
  cannot currently be made.
