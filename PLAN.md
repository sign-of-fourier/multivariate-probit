# PLAN: contract enforcement and margin calibration diagnostics

Internal working notes. Not user-facing documentation, and not yet implemented.

Supersedes the previous plan ("latent-scale calibration and diagnostics")
entirely. That plan's central proposal -- an affine `ProbitScaled` wrapper --
is dropped; the reasoning is in "What changed" below.

## What changed

The previous plan was built around a risk table with three inner-model paths,
of which `decision_function` was treated as the worst case and `predict_proba`
as understated. That framing assumed the estimator should accept a model
emitting an arbitrary real-valued score.

It should not. **The contract is that an inner model emits `predict_proba`.**
Under that contract:

- The `decision_function` path is not a risk to be diagnosed. It is outside the
  contract and should be rejected rather than accommodated.
- Inverting the link is exact. `Phi^-1` applied to a *calibrated* `p_hat`
  yields the correct index, whatever produced the probability. Any inner model
  fitted by maximum likelihood on a proper scoring rule already satisfies this
  up to estimation error, so there is usually nothing to correct.
- What remains is that `p_hat` may be *miscalibrated* -- vote shares from a
  forest, or an overfitted model whose out-of-fold probabilities are
  overconfident. That is real, and it is what the previous plan was reaching
  for, but the response is to **report it, not repair it**. Repair is the
  caller's job (calibrate the classifier) or a preset's, not the estimator's.

So the affine wrapper goes, and what survives is a diagnostic and a documented
account of what `correlation_` actually estimates.

## The contract

An inner model is either:

- an object with `fit(X, y)` and `latent(X)` -- already on the probit scale,
  used directly (`ProbitRegressor`); or
- a classifier with `fit(X, y)` and `predict_proba(X)`, wrapped by
  `ProbitCalibrated`, whose `latent` is `Phi^-1(p_hat)` clipped away from 0
  and 1.

Anything else is an error. A classifier may be arbitrarily bad -- that is
supported, and the diagnostic exists to say so -- but it must emit
probabilities.

## The problem, in math

The model is `Y_j = 1[eta*_j(x) + e_j > 0]`, `e ~ N(0, R)`, unit diagonal.
Stage two evaluates `P(Y_j = 1, Y_k = 1 | x) = Phi_2(eta_j, eta_k; rho_jk)`,
which is valid only if the index it is handed satisfies

    P(Y_j = 1 | x) = Phi(eta_j(x))     for all x                        (*)

Write the ideal index for the features the model can see:

    eta_deg_j(x) = Phi^-1( P(Y_j = 1 | x) )

Two gaps separate the fitted index from the truth, and **they are not the same
kind of thing**. Conflating them is the error in the previous plan and in the
measurements taken against it.

### Gap A -- eta_deg_j versus eta*_j: the estimand moves

The model's features and functional form may omit determinants of the outcome.
Project the full-information index onto the fitted one:

    eta*_j = c_j * eta_j + d_j + w_j,    w_j indep of eta_j,  Var(w_j) = v_j

Then `w_j + e_j ~ N(0, 1 + v_j)`, and dividing through gives a model in which
the recoverable correlation is

    rho_recovered = [ rho_jk + Cov(w_j, w_k) ] / sqrt( (1 + v_j)(1 + v_k) )

**This is not an estimation error.** A margin calibrated with respect to `x`
satisfies (*) by construction, and Σ then measures the dependence *conditional
on x*, with omitted shared variation correctly included in it. That is a
well-posed quantity and arguably the one a user wants. It simply is not the
structural correlation of a fully specified model, and the docs do not
currently say which of the two `correlation_` is.

Both terms matter and they pull opposite ways:

- `Cov(w_j, w_k) > 0` -- the margins miss the *same* thing. Shared omitted
  signal enters Σ as dependence. No bound, no fixed sign in general.
- `v_j > 0` -- shrinks whatever the numerator is.

Verified numerically against known `v` (linear DGP, hidden variables, separate
per margin so the covariance term vanishes): predicted `rho/(1+v)` versus
measured gave 0.500/0.501, 0.400/0.386, 0.250/0.248, 0.154/0.155.

### Gap B -- eta_hat_j versus eta_deg_j: a real bias

The fitted index also differs from the ideal one by finite-sample estimation
error and by miscalibration of the classifier. **This is different algebra, not
the same mechanism at a different stage**, and getting that wrong is what
produced the confusion in the previous plan.

Write `eta_hat = eta_deg + u`, with `u` the estimation error, variance `tau^2`,
and `sigma^2 = Var(eta_deg)`. Here `u` is *part of* `eta_hat`, so this is
classical measurement error rather than the Berkson form of Gap A. Under the
normal approximation the reliability ratio is

    lambda = sigma^2 / (sigma^2 + tau^2)   < 1

and `E[eta_deg | eta_hat] = lambda * eta_hat + (1 - lambda) * mean`, with
residual variance `lambda * tau^2`. Substituting as before, the marginally
correct index is `[lambda*eta_hat + (1-lambda)*mean] / sqrt(1 + lambda*tau^2)`,
and (for estimation errors independent across margins)

    rho_recovered = rho_jk / (1 + lambda * tau^2)

This is a genuine downward bias -- `u` is not part of the conditional estimand
-- and it is the effect `limitations.md` already describes. It is **derivable**;
it needs no empirical demonstration.

### Why this matters for the diagnostic

The two gaps have opposite visibility, and that is the whole case for Piece 2:

| | calibration slope | Σ |
| --- | --- | --- |
| Gap A (omitted signal) | exactly 1 | moves, legitimately |
| Gap B (estimation error) | `lambda / sqrt(1 + lambda*tau^2)`, strictly below 1 | biased down |

So the slope is precisely a **Gap B detector**: sensitive to the genuine bias,
blind to the estimand shift. Under Gap A it reads 1.00 while Σ moves a long way
-- measured at 1.00 across `v` from 0 to 2.25 while ρ fell from 0.501 to 0.155.

Two consequences, both of which the documentation must state:

- A quiet diagnostic is **not** evidence that Σ is trustworthy. It rules out
  one of the two mechanisms, and not the one that moves Σ furthest.
- A slope far *above* 1 means something else again: in-sample, the index has
  absorbed the noise of the rows it is being scored on, so the labels look more
  predictable than the index admits. That is the memorisation signature
  cross-fitting exists to remove, which is why `cv=None` on an overfitting
  learner reads 5.24 while the same margin cross-fitted reads 0.45.

## Consequence for the existing UCI study

`docs/ifm.md` reports fitted correlations of 0.51-0.68 on *Default of Credit
Card Clients*: three outcomes that are repayment delay for the **same client**
in three different months, against five demographic predictors.

Persistent unobserved creditworthiness is precisely a `w` shared across all
three outcomes, so `Cov(w_j, w_k)` is large and positive by construction. The
high correlations are largely that shared heterogeneity -- which is what Σ
conditional on demographics *means*, and explains why an independent
tetrachoric fit on the same data agrees closely. This corroborates the section
rather than undermining it, but the section states the number without stating
its conditioning set, and should.

No `rf` fits appear in that study (only `linear` and `xgboost`), so nothing
there needs re-running.

## Design

### Piece 1 -- enforce the contract

`ProbitCalibrated.latent` currently falls back to `decision_function` when
`predict_proba` is absent, silently producing an index on an arbitrary scale.
Make that an error, and check for `predict_proba` in `as_inner` so the failure
arrives at coercion with a clear message rather than midway through a fold.

Net deletion. Removes a row from the risk table by removing the path.

### Piece 2 -- the calibration slope

After stage 1, `eta_` (cross-fitted) and `Y` are both in hand, so a probit of
`Y[:, j]` on `eta_[:, j]` costs one 1-D fit per outcome and no extra inner-model
fits. Slope 1 and intercept 0 mean `p_hat` is calibrated out of fold.

This is not a new invention and should not be presented as one: it is the
**calibration slope** of prognostic-model validation (Cox). Naming it that way
gives it an established interpretation instead of invented thresholds.

What it estimates is stated exactly above: the reliability of the fitted index,
`lambda / sqrt(1 + lambda*tau^2)`. That is the justification for the piece --
not that the number looks reasonable on test data, but that it is a consistent
estimate of the one quantity separating Gap B from Gap A. Readings:

| slope | meaning |
| --- | --- |
| ~1 | index is reliable, or the model omits signal (indistinguishable here) |
| below 1 | estimation error or an overconfident classifier |
| well above 1 | the index has been scored on rows it was fitted on |

Store as a `(d, 2)` array, `calibration_`. Warn outside a loose band, never
raise -- an existing fit that works must keep working. A degenerate index
(constant or non-finite) reports `nan` and does not warn.

On the current test suite this fires three times, all true positives: 5.24 on
an in-sample overfitted XGBoost margin, 0.45 on the same margin cross-fitted,
and 0.48 on a deliberately misspecified linear margin. Those are findings about
those fits, not a reason to widen the band.

### Piece 3 -- documentation

Per one-fact-one-home:

- `ifm.md` -- Gap A and Gap B as distinct mechanisms, the formula, and what
  `correlation_` is conditional on. Add the conditioning-set note to the UCI
  section.
- `limitations.md` -- the existing attenuation entry is about Gap B and stays;
  add Gap A as a separate entry, and add that a quiet `calibration_` does not
  license trusting Σ.
- `api.md` -- the contract (`predict_proba` required), `calibration_` in the
  attributes table.
- `implementation.md` -- nothing. No accuracy figures are being claimed.

## Explicitly not doing

- **No `ProbitScaled`, no affine repair, no isotonic.** Report, do not fix.
  A caller wanting calibrated forest probabilities can wrap the classifier in
  `CalibratedClassifierCV` before handing it over; `rf` already depends on
  scikit-learn if that is ever wanted in the preset.
- **No change to any preset's numerics.** `linear`, `xgboost` and `rf` all fit
  exactly as they do today. No existing test should move.
- **No correction for either gap** -- but for different reasons, and the
  distinction should not be blurred:
  - *Gap A cannot be corrected.* `v_j` is not identified from `(score, label)`
    pairs at any level of flexibility: the marginal law depends on `c_j` and
    `v_j` only through `c_j / sqrt(1 + v_j)`. Two margins, one correctly scaled
    but incomplete and one over-dispersed but complete, produce identical
    marginal data and imply different Σ. No amount of calibration machinery
    recovers the difference, because the information is not in the data being
    fitted.
  - *Gap B could be, in principle.* The calibration slope is an estimate of
    exactly the reliability that attenuates Σ, so inflating `correlation_` by
    it is arithmetically available. It is still not being done: the slope is
    itself estimated, dividing by it amplifies its error into Σ, and the result
    would be a point estimate with no standard error and no way to tell the
    user how much to trust it. Correcting one gap while the other is silently
    present would also make `correlation_` harder to interpret, not easier.
    Report both, correct neither.
- **No rename of `ProbitCalibrated`.** Still misleading -- it inverts a link --
  but it is exported and renaming needs a deprecation alias. Separate change.

## Open questions

1. **Band for the warning.** Roughly [0.5, 2.0] keeps the linear path silent
   and catches the three known cases. The clinical literature often flags below
   0.8, which would be noisier. Loose is probably right for a warning.
2. **Does `rf` need anything?** Its slopes measured 0.9-1.26 at moderate signal
   -- inside any reasonable band. The preset still has no tests at all, which
   is a real gap independent of this work.

Retired: whether Gap B needs an empirical demonstration. It does not. It
follows from the reliability ratio, and the existing suite already exhibits it
(0.45 cross-fitted against 5.24 in-sample on the same margin). No experiment.

## Sequencing

1. Piece 2 alone (diagnostic + tests). No estimation path changes, so the whole
   existing suite must pass unchanged.
2. Piece 1 (contract enforcement + tests). Touches `ProbitCalibrated` and
   `as_inner`; no preset uses the removed path, so again nothing should move.
3. Piece 3 (docs).
4. Housekeeping: re-add `PLAN.md` to the sdist exclude list in `pyproject.toml`
   alongside `CLAUDE.md`.

Tests for `rf` are worth adding under (1) since the diagnostic gives something
concrete to assert about it, but that is separable.
