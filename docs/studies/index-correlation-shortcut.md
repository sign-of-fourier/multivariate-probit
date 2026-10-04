# Can Sigma be read off the marginal predictions?

**Question.** Stage two costs an orthant-probability MLE. A recurring proposal
is to skip it: rank-transform each margin's predicted probabilities through
their empirical CDF, push the result through Φ⁻¹, and take an ordinary Pearson
correlation of the resulting indices. Does that recover Σ?

**Why it came up.** [ifm.md](../ifm.md#why-not-naive-correlation) already
rejects three naive correlation shortcuts, but none of them carried a rank
transform, and [ifm.md](../ifm.md) mentions the probability integral transform
only to distinguish it from link-inversion — never to measure it. The proposal
is distinct enough from what was rejected to deserve its own reading, and its
appeal is that it removes the only expensive step in the estimator.

The structural prediction, stated before the run: Φ⁻¹ ∘ ECDF is strictly
increasing, so it is a monotone reparameterisation of the predicted score and
correlates the same two series the first rejected bullet already correlates.
More decisively, `p̂_j(x)` is a function of `x` alone. Σ is the correlation of
the latent errors — the part of `Z_j` that `x` does not explain — so a
statistic built from the margins' predictions after stage one, never touching
`y` again, has no access to it.

## Setup

**Synthetic**, so the truth is known. `n = 5000`, `d = 3`, six standard normal
features of which four carry signal, `Z = η(x) + e` with `e ~ N(0, Σ)` and
`Y = 1[Z > 0]`. Margins are the `xgboost` preset at its defaults with `cv=5`;
stage two is `dependence="joint"` at the default `n_quad=24` (orders inherited,
not searched). Five seeds per cell, reported as the mean over the three pairs
and five fits.

Coefficients are built so `Var(η_j) = 1` and the *index* correlation
`corr(η_j, η_k)` is fixed by construction — 0.6 in the "shared" cells, 0 in
"disjoint", undefined in "no signal" where every `η_j = 0`. Four cells:

| cell | features | true ρ | what it discriminates |
| --- | --- | --- | --- |
| 1 | shared, index corr 0.6 | 0.0 | false positive from shared predictors |
| 2 | shared, index corr 0.6 | 0.5 | the ordinary case |
| 3 | none (η ≡ 0) | 0.5 | false negative when x is uninformative |
| 4 | disjoint, index corr 0.0 | 0.5 | shared-predictor contamination, isolated |

**Real data.** UCI *Default of Credit Card Clients* (Yeh & Lien, 2009), the
same three outcomes and five demographic predictors as
[uci-credit.md](uci-credit.md), all 30,000 rows rather than that study's 80/20
split. Not redistributed here. The true Σ is unknown, so this leg checks
magnitudes against a fit already validated against an independent tetrachoric
estimate, not against ground truth.

Arms, all computed on the *same* cross-fitted `eta_` so the only thing varying
is the statistic:

- **(a)** the proposal — `Φ⁻¹((rank - 0.5) / n)` per margin, then Pearson.
- **(b)** Pearson of the link-inverted index with no rank transform. Present
  only to isolate what the ECDF step contributes.
- **(c)** the shipped estimator, `joint` and `pairwise`.
- **(d)** Pearson of the raw binary outcomes, the known-attenuated baseline.
- **(e)** Pearson of the truncated-normal latent residual
  `E[w_j | y_j, η_j] = s_j φ(η_j) / Φ(s_j η_j)` with `s_j = 2 y_j - 1`. Not
  part of the proposal; added because it is the cheapest statistic that both
  lives on the latent scale and uses `y`.

## Results

Synthetic, mean over three pairs and five seeds:

| arm | cell 1 (ρ = 0) | cell 2 (ρ = 0.5) | cell 3 (ρ = 0.5) | cell 4 (ρ = 0.5) |
| --- | --- | --- | --- | --- |
| (c) joint | **-0.008** | **0.468** | **0.480** | **0.467** |
| (c) pairwise | -0.008 | 0.470 | 0.484 | 0.469 |
| (a) ECDF → Φ⁻¹ → Pearson | 0.581 | 0.591 | 0.302 | 0.017 |
| (b) index, no ECDF | 0.580 | 0.590 | 0.303 | 0.014 |
| (d) raw Y | 0.193 | 0.371 | 0.335 | 0.160 |
| (e) latent residual | -0.004 | 0.261 | 0.333 | 0.236 |
| calibration slope | 0.92 | 0.92 | 0.05 | 0.92 |

Seed-to-seed spread is 0.01-0.03 in every cell, so none of the gaps above is
sampling noise.

UCI credit, per pair (ρ_12, ρ_13, ρ_23):

| arm | linear margins | xgboost margins |
| --- | --- | --- |
| (c) joint | 0.666, 0.524, 0.679 | 0.655, 0.509, 0.664 |
| (a) ECDF → Φ⁻¹ → Pearson | 0.988, 0.979, 0.983 | 0.908, 0.884, 0.933 |
| (b) index, no ECDF | 0.995, 0.990, 0.992 | 0.915, 0.878, 0.940 |
| (d) raw Y | 0.431, 0.310, 0.434 | 0.431, 0.310, 0.434 |
| (e) latent residual | 0.401, 0.281, 0.406 | 0.395, 0.274, 0.398 |

The joint column reproduces [uci-credit.md](uci-credit.md) to within 0.003,
which is the expected effect of fitting on all 30,000 rows instead of 80% of
them, and confirms nothing else drifted.

Arm (e) against a swept truth, disjoint predictors, three seeds:

| true ρ | (c) pairwise | (e) latent residual | (a) ECDF |
| --- | --- | --- | --- |
| 0.00 | 0.019 | 0.010 | 0.034 |
| 0.25 | 0.246 | 0.125 | 0.019 |
| 0.50 | 0.475 | 0.240 | 0.027 |
| 0.75 | 0.721 | 0.367 | 0.022 |

## Conclusions

**The shortcut does not estimate Σ. It estimates the overlap between the
margins' predictors.** Cells 1 and 4 are the demonstration and they fail in
opposite directions. In cell 1 the truth is exactly zero and the proposal
returns 0.581, because the three indices share 60% of their variance by
construction; the estimator returns -0.008. In cell 4 the truth is 0.5 and the
proposal returns 0.017, because the indices are built from disjoint features;
the estimator returns 0.467. Across all four cells the proposal tracks the
index correlation the generator was given — 0.6, 0.6, undefined, 0.0 — and
ignores ρ entirely. Cell 2's 0.591 against a true 0.5 is the trap: it looks
like a near-hit, and it is the same 0.6 reading that cell 1 produces against a
true 0.

**The ECDF step contributes nothing.** Arms (a) and (b) differ by 0.003 on
average and never by more than 0.011, across 20 synthetic fits and both UCI
margins. Φ⁻¹ ∘ ECDF is strictly increasing, so it changes the marginal scale of
each index and not the pair of series being correlated; Pearson correlation of
two near-linearly-related quantities is close to invariant to it. The proposal
is therefore a rank-flavoured restatement of the first bullet already rejected
in [ifm.md](../ifm.md#why-not-naive-correlation), and inherits its verdict for
the same reason.

**Real data hides the failure rather than avoiding it.** On UCI credit all
three margins see the same five demographic predictors, so the indices are
nearly collinear and the proposal reads 0.88-0.99 against a validated
0.51-0.68. The direction is right and the number is unusable, which is the
worst configuration for a shortcut — a synthetic cell announces the failure,
a real dataset merely inflates.

**The threshold reframing is correct and leads to the estimator, not away from
it.** Writing `w_j = Z_j - η_j`, so that `w_j ~ N(0, 1)` and
`Y_j = 1[w_j > -η_j]`, does move the row-varying quantity into the threshold
and leave a standard latent whose correlation is exactly Σ. That is the sign
trick stage two already uses ([ifm.md](../ifm.md), step 4). It also shows why
no transform of the margins can substitute for it: `w_j` is observed only
through one bit per row, and recovering `corr(w)` from those bits and the
thresholds is the orthant integral.

**Cell 3 is a separate warning.** With no signal at all the proposal returns
0.302 rather than 0. Out-of-fold predictions are functions of `x`, but the
models producing them were fitted against correlated labels, so two margins
chasing noise in the same features chase it in partly the same direction. The
number is an overfitting artifact with no population meaning. The calibration
slope reads 0.05 there and the estimator's built-in warning fires, so this cell
is at least loud.

**Arm (e) is a legitimate zero-test and nothing more.** It is the only cheap
arm that returns ~0 when the truth is 0 under shared predictors (-0.004 in
cell 1), and it is monotone in the truth: 0.010, 0.125, 0.240, 0.367 against
0, 0.25, 0.5, 0.75, an attenuation of about 0.49 that is close to constant
*within* one margin configuration. But the factor is not portable — the same
true 0.5 reads 0.261, 0.333 and 0.236 in cells 2, 3 and 4 — so it cannot be
de-attenuated without knowing the very margin quality that produced it. Usable
as a fast "is there any residual dependence here at all" screen before paying
for stage two; not usable as an estimate, and not proposed for the API.

## What this does not establish

- One dimension only, `d = 3`, and one inner model on the synthetic leg. The
  failure mechanism is algebraic and does not depend on `d`, but the
  magnitudes reported here do.
- Only Gaussian features and linear indices. A nonlinear `η` would change how
  closely arm (a) tracks arm (b) — the near-equality of the two is a statement
  about Pearson correlation under a near-linear rank transform, not a theorem.
- The synthetic cells fix the index correlation at 0 or 0.6 exactly. Nothing
  here maps the intermediate range, so "the proposal returns the index
  correlation" is supported at two points, not fitted.
- Arm (e)'s attenuation is measured at one sweep in one configuration. Whether
  its factor has a closed form was not investigated.
- The UCI leg has no ground truth. It shows the proposal disagrees with a
  validated fit, not that the validated fit is right.

## Open threads

- Arm (e)'s attenuation looks close to a product of per-margin factors, which
  is the shape Gap B takes in
  [ifm.md](../ifm.md#gap-b--a-real-bias). If that is what it is, the residual
  correlation and the calibration slope are measuring the same reliability
  from two directions, and the relationship is worth writing down even though
  neither is used to correct Σ.
- Whether arm (e) is cheap enough and reliable enough to become a documented
  pre-flight screen, rather than a study finding, is undecided.
