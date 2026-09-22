# What does the approximate evaluator cost at d = 14?

**Question.** [multilabel-benchmarks.md](multilabel-benchmarks.md) scores the
joint log-likelihood at `d = 14` with `scipy.stats.multivariate_normal.cdf`
rather than the third-party approximate orthant evaluator, on the grounds that
the approximate evaluator's error is absolute and a log score needs relative
precision. That route is slow. Is the caution warranted, or would the cheap
evaluator have produced the same conclusions?

**Why it came up.** The argument for the split was made analytically and never
measured head to head. It also predates the decision to aggregate the log score
by **median** rather than by mean, and a median discards precisely the
near-zero rows the argument is about. The two facts pull in opposite
directions, so the cost was being paid on an untested premise.

## What a median does and does not fix

Two distinct failure modes get conflated here, and separating them is the whole
question.

A **sum** of logs gives unbounded leverage to rows whose probability approaches
zero, so a handful of rows own the total. The median removes that leverage
completely, and a probability floor is redundant once it does.

An **evaluator's error** is not leverage. It applies to every row, including
the one at the centre, and the median reports that row's magnitude verbatim.
Each row is a different quantity measured once, so there is no repetition for
error to cancel across. Robustness to heavy tails is not robustness to a biased
instrument.

The ordering of rows is a third thing again, and it is what survives an
absolute error best — the reason the same evaluator is sound for a pattern
argmax.

## Setup

Yeast from the Mulan collection, `d = 14`, `n_test = 917`: official
train/test split, features standardised on training statistics, the package's
`xgboost` preset, `dependence="pairwise"`, `cv=5`, `random_state=0`. The
protocol of [multilabel-benchmarks.md](multilabel-benchmarks.md) throughout.
Data is not redistributed here.

One fitted model supplies η and Σ to both evaluators, so nothing varies but the
integration route. Σ arrives from the Higham projection on the PSD boundary
(`eig_min = 1e-8`), which is the ordinary condition for a pairwise fit.

- **Exact route:** `scipy.stats.multivariate_normal.cdf`, `allow_singular=True`,
  `maxpts=1e6 * d`, `abseps` and `releps` at their defaults. Setting `abseps`
  tight is the error recorded in the project's working notes and is not
  repeated.
- **Approximate route:** the third-party orthant binary, same sign trick, same
  per-row covariance stack. That evaluator is not redistributable, is not a
  dependency of this package, and appears here only as a measurement
  instrument.

The SciPy pass reproduces the published figure to four decimals (`-3.9706`
against the reported `-3.9710`), which confirms the protocol matched.

## Results

| | SciPy (exact) | orthant (approximate) |
| --- | --- | --- |
| median log `P(Y = y \| x)` | **-3.9706** | **-4.3518** |
| median `P(Y = y \| x)` | 0.01886 | 0.01288 |
| mean log `P(Y = y \| x)` | -4.4803 | -9.2240 |
| rows returning `p <= 0` | 0 | 6 |
| wall clock, 917 rows | **5302 s** | **~0.03 s** |

Per-row absolute error against the exact route: median 0.0057, p90 0.0460, max
0.0923. The nominal 1e-2 describes the centre of that distribution, not its
tail. And 40.1 percent of rows carry a true probability below 1e-2, so the
error band covers a large share of the sample rather than a fringe of it.

Substituting the approximate figure into the yeast column changes no ordering:

| arm | median log `P(Y = y \| x)` |
| --- | --- |
| ECC (K = 10) | -3.2725 |
| multivariate probit, SciPy | -3.9710 |
| multivariate probit, orthant | *-4.3518* |
| binary relevance | -5.7375 |
| ML-kNN | -5.8347 |
| MLP | -20.3174 |

## Conclusions

**Every conclusion the benchmark study draws from that column survives the
cheap evaluator.** ECC still leads the probit, the probit still leads binary
relevance and ML-kNN, MLP is still last. The 88-minute pass bought a
first-decimal correction that no stated finding rested on. The cost/benefit
recorded in that study was not favourable, and its cost figure understated the
expense besides — the exact route measures 5.8 s per row, against the ~1 s the
table claimed.

**The error is nevertheless large, one-sided, and applied to a single arm.**
0.38 nats is 55 percent of the probit-to-ECC gap. Every competing method in
that table computes its joint in closed form — a product of d marginals, or a
mixture of chain-rule products — so the probit is the only arm needing an
integral at all. A systematic downward shift on exactly one competitor is not
noise that washes out of a comparison. Nothing flipped here because the gap
happened to be 1.8x the error; at a gap of 0.3 nats it would have flipped, with
nothing in the output to signal it.

**Any mean- or sum-aggregated log score is destroyed outright.** Six rows
return a non-positive probability, which has no logarithm at all. The gap
between the two mean figures above is 4.7 nats, entirely an artifact of those
six rows meeting a floor. This is independent of precision and cannot be tuned
away.

**The usable rule is narrower than either prior position.** The approximate
evaluator is adequate for a *median-aggregated log score consumed as a ranking*
when the gaps between arms are known in advance to be several times the error.
It remains unusable for any reported figure, for any mean or sum, and for any
comparison whose margin is not established independently.

## A second instrument: the argmax under sampling

[multilabel-benchmarks.md](multilabel-benchmarks.md) lists the yeast subset
accuracy as resting on the approximate evaluator "with no exact reference on a
general Σ at that dimension". Enumeration is not the only route to a modal
pattern. Drawing from the fitted latent is unbiased, needs no integral, and
never materialises the `2 ** d` candidates, so it is an independent instrument
for that column.

Σ is global under IFM — stage two fits one matrix and only η shifts per row —
so one matrix square root serves every row. It is taken from an
eigendecomposition rather than a Cholesky, since a pairwise fit lands on the
PSD boundary. `S = 20000` draws per row, `sign(z)` bit-packed to a pattern
index, modal bin reported; five seeds.

| | sampling (5 seeds) | orthant | published |
| --- | --- | --- | --- |
| subset accuracy, Σ | **0.2386** (sd 0.0014) | 0.2356 | 0.2356 |
| subset accuracy, independent | **0.1932** (sd 0.0014) | 0.1930 | 0.1930 |
| median modal probability, Σ | **0.1161** (sd 0.0006) | 0.0895 | — |
| median modal probability, independent | **0.0566** (sd 0.0003) | 0.0561 | — |
| wall clock, 917 rows | 6 s | 236 s | — |

**The published subset accuracy holds.** 0.2386 against 0.2356 is under three
rows in 917, and the Σ-over-independence gain is +0.0454 by sampling against
+0.0426 by enumeration. Nothing in the benchmark table needs revising.

**The modal probability does not.** Under independence the two instruments
agree to 1 percent. Under a general Σ the approximate evaluator reads 23
percent low, at roughly 40 standard deviations of the sampling noise. This is
the same sign and rough magnitude as the log-score result above, where it read
the median observed-pattern probability 32 percent low against SciPy — two
different quantities, two independent references, one direction.

The obvious alternative explanation is a winner's curse: taking the maximum of
16,384 noisy bins inflates the winner. The independence arm rules it out. Its
first-to-second gaps are *tighter* (0.0097 against 0.0342), so it should carry
more selection bias, and it shows none. The discrepancy tracks Σ, not the
selection.

**Sampling also supplies an error bar the enumeration route lacks.** The
benchmark study reports "28.4 percent of rows have a first-to-second gap below
the evaluator's error floor", where the floor is set by hand at 1e-2. The
multinomial standard error replaces that with a measured quantity: 16.9 percent
of rows are unresolved at three standard errors.

**None of this transfers to the log score.** A modal pattern is where the mass
is; an observed pattern often is not. Against the exact probabilities, crude
sign-binning returns *zero* hits — and so no logarithm at all — for 15.1
percent of rows at 1e3 draws, 6.0 percent at 1e4, 2.4 percent at 1e5 and 1.1
percent at 1e6. Ten percent relative precision on the observed pattern needs
5.2e3 draws at the median row, 3.6e5 at the 90th percentile and 1.3e8 at the
99th. The concentration that makes the mode cheap is the same fact that puts
the tail out of reach.

## What this does not establish

- **One dataset, one arm, one seed.** Yeast under the `xgboost` preset only.
  The three linear-margin arms in the same family were not re-scored, and no
  other `d` was tested.
- **No error bars.** Inherited from
  [multilabel-benchmarks.md](multilabel-benchmarks.md), which is one split with
  no repetition.
- **The ranking survived by margin, not by mechanism.** Nothing here shows the
  ordering is robust in general, only that this particular set of gaps exceeded
  this particular error.
- **The exact route is itself a reference, not ground truth.** SciPy's
  evaluator is stochastic. It agrees with the published figure to four
  decimals, which bounds run-to-run variation well below the effect measured,
  but it was not independently validated at `d = 14`.
- **The approximate evaluator's error was characterised only on this Σ.** It
  arrives from a pairwise fit on the PSD boundary; nothing here describes its
  behaviour on a well-conditioned Σ.
- **The sampling arm is one budget at one dimension.** `S = 20000` on yeast
  only. Its own bias as an estimator of the argmax — it recovers the true modal
  pattern only with probability, not certainty — is bounded by the seed spread
  reported and not otherwise measured.
- **Sampling and enumeration disagree on 11.3 percent of predicted patterns**
  under Σ, while agreeing on the accuracy those patterns produce. Which
  instrument is right row by row is not established.

## Open threads

- **The downward shift is corroborated and still unexplained.** Two references
  now agree that the approximate route reads low on a general Σ at `d = 14` —
  32 percent on the observed pattern, 23 percent on the modal one — and that it
  does not on Σ = I. Whether that is a property of the instrument, of
  near-singular Σ, or of the dimension is not established, and a bias with a
  known sign could in principle be corrected. Characterising it against a
  well-conditioned Σ is the next step.
- **The composite pairwise log-likelihood needs no evaluator at all.**
  `ifm.pair_log_likelihood` is exact, closed-form and flat in `d`. It is not a
  joint and must never be reported as one, but as a dependence score at large
  `d` it sidesteps this entire question and was never compared against either
  route.
- **Sampling is the only argmax route that survives past this dimension.**
  Enumeration is `2 ** d` candidates per row — 1.05 million at `d = 20` — while
  a sampler is flat in the number of candidates and pays only in draws. Nothing
  here tests `d = 20`, but the argmax at large `d` is the case where the two
  routes stop being comparable.
- **A bounded proper scoring rule would change the calculus.** A Brier score
  over the pattern distribution is linear in the probability, so an absolute
  error of 1e-2 costs 1e-2 rather than corrupting a relative quantity — and its
  cost profile (many evaluations, low precision each) is what the approximate
  evaluator is actually suited to. Untested.
