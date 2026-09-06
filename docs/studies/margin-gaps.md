# What `correlation_` estimates when the margins are imperfect

**Question.** Stage two treats the fitted index as if it satisfied
`P(Y_j = 1 | x) = Φ(η_j(x))` exactly. It does not. What is the fitted Σ an
estimate *of*, and can the difference be detected or corrected?

**Why it came up.** A proposed change to this library rested on the claim that
an arbitrary inner model needs an affine rescaling of its index, without which
Σ is biased. Checking that claim meant deciding what Σ estimates in the first
place, which had never been written down. The claim turned out to be wrong for
a reason that also retired the fix: the scaling only ever mattered for models
emitting an uncalibrated score, and those are now outside the contract. What
survived is this account.

The derivations live in
[ifm.md](../ifm.md#what-sigma-is-conditional-on); this file holds the numbers
that check them.

## Setup

Synthetic bivariate probit with linear margins, so the true ρ is known. Two
configurations:

- **Gap A** — a determinant of the outcome that the fitted margin never sees,
  drawn independently per outcome so it contributes no shared dependence. Its
  variance `v` is the knob. With the covariance term zero, the prediction is
  `ρ_recovered = ρ / (1 + v)`.
- **Gap B** — a correctly specified margin that is badly estimated: real signal
  in three features buried in forty pure-noise ones, n = 600.

Both are exercised by `tests/test_calibration.py`, which pins the readings
below.

## Results

Gap A, predicted against recovered, true ρ = 0.5:

| predicted `ρ / (1 + v)` | recovered |
| --- | --- |
| 0.500 | 0.501 |
| 0.400 | 0.386 |
| 0.250 | 0.248 |
| 0.154 | 0.155 |

Across the same sweep — `v` from 0 to 2.25, ρ falling from 0.501 to 0.155 — the
calibration slope read 1.00 throughout.

Gap B, and the in-sample contrast:

| configuration | calibration slope |
| --- | --- |
| correctly specified, cross-fitted | 1.00 |
| correctly specified, badly estimated | 0.65 - 0.77 |
| over-capacity boosted margin, in-sample | 5.24 |
| the same margin, cross-fitted | 0.45 |
| linear margin, in-sample (`cv=None`) | exactly 1.00 |

## Conclusions

**The two gaps are different kinds of thing, and the diagnostic sees one of
them.** Omitted signal moves the estimand: a margin calibrated with respect to
`x` is correct by construction, and Σ becomes the dependence conditional on `x`
with shared omitted variation properly counted in it. Estimation error is a
genuine downward bias. The calibration slope estimates the reliability of the
index, so it detects the second and reads exactly 1 under the first.

**A quiet slope is not evidence Σ is trustworthy.** The Gap A sweep is the
demonstration: the diagnostic never moved while ρ fell by a factor of three.

**Neither gap is corrected, for different reasons.** Gap A cannot be — the
marginal law depends on the index scale and the omitted variance only through
their ratio, so two margins with identical marginal data imply different Σ.
Gap B could be, since the slope estimates the attenuation factor, but the slope
is itself estimated, dividing by it amplifies its error into Σ, and there are no
standard errors anywhere in this package to say how far to trust the result.

**An in-sample GLM margin makes the diagnostic vacuous.** A probit fitted on the
rows it is scored on satisfies `X'(y - p) = 0`, which contains the score
equations of the calibrating probit at slope 1, intercept 0. `cv=None` with the
`linear` preset therefore returns 1.00 to machine precision whatever the fit is
worth. This was not anticipated when the check was designed.

## What this does not establish

- All of it is synthetic. Real-data readings are in
  [uci-credit.md](uci-credit.md), where the slopes are 0.93-0.99 and the check
  is silent.
- The Gap A arithmetic is checked only with *independent* omitted variables, so
  the `Cov(w_j, w_k)` term in the numerator is never exercised. That term has no
  bound and no fixed sign, and it is the one that matters most on real data,
  where margins tend to miss the same things.
- No sign is established for general miscalibration. Stage two has no free
  parameter but ρ, so margin error necessarily lands in Σ — but which way it
  lands depends on the configuration of the true indices. A fully grown decision
  tree, slope 0.11, inflated ρ to 0.914 against a true 0.5; a noisy index
  attenuates instead.

## Open threads

- Does calibrating the margins convert Gap B into Gap A? The argument is that a
  calibrated margin is honest with respect to the information its own score
  carries, which would make `correlation_` interpretable rather than merely
  closer to something. Untested. The `rf` preset is the natural vehicle, since
  vote shares are the textbook uncalibrated probability and the preset has no
  tests at all.
- The `Cov(w_j, w_k)` term deserves its own configuration.
