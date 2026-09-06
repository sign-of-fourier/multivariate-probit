# Known gaps and roadmap

Current version: 0.1.0 (alpha). This is a first, deliberately minimal
implementation. The gaps below are known and stated rather than discovered
later.

## Statistical

**No standard errors.** `correlation_` is a point estimate. Because stage two
conditions on estimated margins, inverse-Hessian standard errors would be
wrong; correct inference needs a Godambe (sandwich) information matrix or a
bootstrap that resamples and refits both stages. Neither is implemented, and
the margin fit does not report coefficient standard errors either. See
[ifm.md](ifm.md#inference).

**Correlations are attenuated when margins are noisy.** Cross-fitting removes
the large upward bias from in-sample margins, but leaves a residual downward
bias: an out-of-fold index still carries prediction error, and noise in a
regressor attenuates the correlation measured through it. Read a fitted
correlation as a floor on the true dependence. See
[ifm.md](ifm.md#two-bias-directions-at-two-different-stages) for the effect and
[Gap B](ifm.md#gap-b--a-real-bias) for the algebra.

**Σ is conditional on what the margins can see.** Determinants of the outcome
that the features or the functional form omit do not vanish; they join the
latent error, and shared omitted variation is counted as dependence. `Σ` is
therefore the correlation conditional on x, not the structural correlation of a
fully specified model. This is not a bias that better estimation removes, and
it cannot be diagnosed: a margin that is correctly scaled but incomplete and
one that is over-dispersed but complete produce identical marginal data while
implying different Σ. Widen the feature set if the structural quantity is what
is wanted. See [Gap A](ifm.md#gap-a--the-estimand-moves).

**A quiet `calibration_` does not license trusting Σ.** The calibration slope
detects estimation error in the margins, and is blind by construction to
omitted signal — it reads exactly 1 while Σ moves a long way. It rules out one
of the two mechanisms, and not the one that moves Σ furthest.

**`calibration_` is uninformative for an in-sample GLM margin.** A probit
margin fitted on the rows it is scored on satisfies `X'(y - p) = 0`, which
contains `eta'(y - p) = 0` and `1'(y - p) = 0` — precisely the score equations
of the calibrating probit at slope 1, intercept 0. So `cv=None` with the
`linear` preset returns 1.00 to machine precision whatever the fit is worth.
The number carries information only under cross-fitting, or for an inner model
that is not itself a probit MLE. It is also computed unweighted, ignoring any
`sample_weight` passed to `fit`.

**Misspecified margins contaminate the dependence.** IFM's one-way information
flow means anything the margins fail to explain surfaces in Σ, where it does
not belong. Check the margins before interpreting the correlations.

**Efficiency.** IFM is consistent but not fully efficient relative to FIML. The
loss is typically small at moderate correlations and grows as abs(ρ) → 1.

**Composite likelihood under `dependence="pairwise"`.** The pairwise objective
discards information in three-way and higher dependence. For a Gaussian copula
Σ is fully determined by its pairs, so this costs little, but the estimator's
asymptotics are the composite ones, not the full-likelihood ones.

## Computational

**Outcome count.** The deterministic orthant evaluator costs roughly
`n_quad ** (d - 2)`. Trivial at d = 3, noticeably heavier by d = 6-7,
impractical well before d = 10. That regime needs a randomized or QMC
evaluator, which is not implemented. `dependence="pairwise"` is the escape
hatch and has no such scaling.

**No analytic gradient for the Σ MLE.** `dependence="joint"` is derivative-free
(Nelder-Mead) only. A closed-form gradient with respect to the correlation
entries would allow a quasi-Newton method and would materially widen the usable
range of d.

**Rare outcomes across folds.** No special handling for an outcome too rare to
appear in every cross-validation fold. A fold with a single-class training
split will fail or produce a degenerate margin. Check outcome prevalence before
raising `cv`.

## Coverage

**The `rf` preset is untested.** It is registered and it runs, but no test
pins its behaviour. `linear` and `xgboost` are covered.

## Not implemented

**FIML.** Full joint estimation is not supported, for the reasons in
[ifm.md](ifm.md#why-not-full-joint-mle-fiml). If it is ever added as a second
`estimation=` strategy, it is also the point at which model-specific modules
stop being avoidable: it needs a joint objective, gradients through the
margins, a parameterisation of Σ optimised jointly, and a simulator for the
likelihood at d ≥ 3. Until then the thin preset registry is the right shape.

**Ordinal and count outcomes.** Binary only. The latent-threshold construction
generalises to ordered categories with per-outcome cut points, but nothing in
the current code does that.

**Sparse input.** Dense arrays only; `X` is coerced with `numpy.asarray`.

**Missing outcomes.** `Y` must be fully observed. Partially observed outcome
vectors would be a natural fit for IFM — each pair contributes wherever both
outcomes are present — but are not handled.
