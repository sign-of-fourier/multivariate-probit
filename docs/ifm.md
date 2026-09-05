# IFM: cross-fit Inference Functions for Margins

This is the estimation method `multivariate_probit` uses to fit the model

```
Z ~ N(η(x), Σ),    Y_j = 1[ Z_j > 0 ],    j = 1 … d
```

See [../README.md](../README.md) for the model itself and the public API. This
file is the algorithm, in detail: what IFM is, the four steps, and the
alternatives that were tested and rejected.

**Notation.** η_j(x) is outcome j's latent index, an unbounded real-valued
score (attribute `eta_`). Σ is the latent correlation matrix — a *correlation*
matrix, with unit diagonal, not a covariance matrix (attribute
`correlation_`). Φ and φ are the standard normal CDF and density; Φ_d is the
d-variate normal CDF. `d` counts outcomes, `n` counts observations.

## What IFM is

IFM (Inference Functions for Margins; Joe & Xu, 1996) is a two-step estimation
strategy for exactly this kind of model: several outcomes, each with its own
marginal distribution, tied together by a copula.

Step one fits every marginal completely independently — outcome j's model never
sees outcome k's labels. Step two fits the dependence parameters (here, Σ)
treating those already-fitted margins as fixed, never revisiting them.
Information flows one way, margins to dependence, never back.

This is legitimate because in a multivariate probit the marginal distribution
of Y_j **does not depend on Σ**: `P(Y_j = 1 | x) = Φ(η_j(x))` regardless of how
the outcomes covary. Ignoring the correlation in step one costs efficiency, not
consistency.

## The algorithm

### Step 1 - fit each marginal independently

One classifier per outcome, each trained only on its own label. This is d
ordinary binary classification problems, and the model can be anything that
fits `(X, y)` — see [api.md](api.md) for the contract.

### Step 2 - move to the latent scale via link-inversion

```
η_j(x) = Φ⁻¹( p̂_j(x) )
```

This is *link-inversion*, not the probability integral transform (PIT). PIT
would require an empirical CDF built from a population of realized draws
(`u = F(x)`, rank-based). Link-inversion is pointwise: since the model assumes
`P(Y_j = 1 | x) = Φ(η_j(x))` by construction, solving for η_j(x) given the
model's own predicted probability recovers the implied latent index directly,
using nothing about the rest of the population. `p̂_j(x)` is the model's
*predicted probability* — the target `y_j` never enters this step.

Because Φ⁻¹ is strictly monotone, link-inversion preserves the ranking the base
learner produced; it changes only the scale on which dependence is measured.
Probabilities are clipped away from 0 and 1 first, so a saturated ensemble
cannot emit an infinite index.

One inner model skips this step entirely: the native linear probit already
produces η_j(x) = x'β_j on the correct scale, with no round trip through a
probability.

### Step 3 - cross-fit the marginals before Sigma ever sees them

Not part of the classical IFM definition, but non-negotiable once the margins
are flexible ML models rather than simple parametric ones (see
[Why cross-fitting is required](#why-cross-fitting-is-required)).

Every η_j(x) used to fit Σ comes from an out-of-fold prediction — a model that
never saw that row during training — via ordinary k-fold cross-validation,
rotated so every row gets exactly one out-of-fold prediction per outcome. The
final marginal models used for `transform` / `predict_proba` / `predict` are
refit on the full training set afterward, same as any meta-estimator that
cross-validates internally but deploys a model fit on all the data.

The k-fold split is implemented internally rather than pulled from
scikit-learn, which keeps the core install at numpy + scipy.

### Step 4 - fit Sigma by maximum likelihood, holding the margins fixed

The likelihood of one row's observed pattern is a multivariate normal CDF
evaluated at a sign-flipped version of that row's latent scores — the **sign
trick**. Let `s_j = 2 y_j - 1` (so `s_j = +1` if `y_j = 1`, else `-1`). Then:

```
P(Y = y | x) = Φ_d( s_1 η_1(x), …, s_d η_d(x) ;  Σ_eff )
```

where Σ_eff has off-diagonal entries `s_j s_k ρ_jk` (diagonal stays 1). This is
pure bookkeeping: flipping a variable's sign, and the sign of every correlation
touching it, algebraically converts a "less than 0" event back into the same
Φ_d form as a "greater than 0" event. One expression covers all 2^d outcome
patterns, with no separate upper/lower-tail handling.

The log-likelihood over n rows is then

```
sum_i  log Φ_d( s_i1 η_1(x_i), …, s_id η_d(x_i) ;  Σ_eff,i )
```

— ordinary i.i.d. maximum likelihood, where the only unusual ingredient is that
each row's contribution requires a multivariate normal CDF rather than a plain
Bernoulli probability. It is maximised over the free correlation parameters
only; every η_j(x) is fixed data at this point, not a parameter, by IFM's
definition.

## Two objectives for Sigma

The library offers two ways to carry out step 4.

### dependence="joint" (default) - the full likelihood

Maximise the d-variate likelihood above over the d(d-1)/2 free correlations,
using `scipy.optimize.minimize` (Nelder-Mead by default), discarding any
candidate matrix that is not positive definite during the search. This is
derivative-free by design: the orthant-probability likelihood has no convenient
closed-form gradient with respect to the correlation entries in this
implementation, and derivative-free optimization was sufficient in practice.

Each evaluation costs one orthant integral per observation, so this is the
exact answer at a price that grows sharply with d.

### dependence="pairwise" - composite likelihood

For each pair (j, k), maximise the bivariate probit log-likelihood in the
single parameter ρ_jk:

```
ρ_jk = argmax_r  sum_i  log Φ_2( s_ij η_j(x_i),  s_ik η_k(x_i),  s_ij s_ik r )
```

Each is a smooth one-dimensional problem on (-1, 1), solved by bounded Brent
search, and no d-dimensional integral appears anywhere. Summing the bivariate
log-likelihoods rather than using the true d-variate one makes this a
**composite likelihood**: consistent, orders of magnitude cheaper, and in
practice within a few hundredths of the joint estimate. It is the right choice
once d is large enough that orthant integration bites.

## Evaluating the orthant probability

Both objectives, and every joint prediction, bottom out in the same evaluator.

Φ_2 is computed in closed form from the Drezner-Wesolowsky form of Plackett's
identity,

```
Φ_2(a, b, r) = Φ(a) Φ(b)
             + 1/(2π) ∫₀^{asin r} exp( -(a² + b² - 2ab sin t) / (2 cos² t) ) dt
```

with 24-point Gauss-Legendre quadrature — fully vectorised over observations
and accurate to about 1e-12 over |r| ≤ 0.999.

For d ≥ 3, Φ_d is built from it by Genz's (1992) recursive conditioning: peel
off the last variable under a fixed one-dimensional quadrature, reduce to a
(d-1)-dimensional problem with the conditional correlation matrix, and recurse
until the closed-form bivariate case is reached. Two properties matter:

- **It is deterministic.** No GHK simulation, so no Monte-Carlo noise inside an
  optimiser objective, and repeated calls return identical values. Accuracy at
  the default quadrature order is about 1e-5 for d ≥ 3.
- **It admits row-varying correlations.** The evaluator takes a `(d, d, n)`
  stack, which is what makes the sign trick a single vectorised call rather
  than a loop over 2^d patterns.

## Positive definiteness

The joint fit constrains Σ to be positive definite directly, by rejecting
candidates whose smallest eigenvalue falls below a floor.

Pairwise estimates carry no such constraint — each ρ_jk is fitted in isolation,
so the assembled matrix can fail to be positive definite when d ≥ 3. It is
projected onto the nearest correlation matrix by Higham's (2002) alternating
projections, with a small eigenvalue floor so the result is strictly positive
definite and safe to Cholesky-factor for sampling. In practice the projection
is a no-op unless the pairwise estimates are genuinely contradictory or the
sample is small.

## Why cross-fitting is required

If a marginal model is fit on all n rows and then predicts on those same n
rows, its prediction is not really `P(Y_j = 1 | x)` — it is contaminated by
having partially memorized `y_j` for that row. Flexible models
(gradient-boosted trees especially) push in-sample predictions toward the
observed label itself, not toward the true conditional probability, given
enough capacity.

This breaks step 4 specifically, not just "overfitting is generically bad".
Suppose outcomes j and k are genuinely correlated in the true Σ. Rows where
`y_j = 1` tend to also have `y_k = 1`. If model j overfits, it inflates
`p̂_j(x)` specifically on rows where `y_j = 1` — and if model k overfits the
same way, it inflates `p̂_k(x)` on rows where `y_k = 1`. Since those are largely
the same rows (that is what correlation means), the two inflated scores end up
moving together *more* than the true latent scores do, and the Σ MLE reads that
spurious co-movement as extra correlation.

The effect is not subtle. With deliberately over-capacity boosted margins
(depth 8, 300 rounds, no minimum child weight), the in-sample estimate pegs at
the +0.999 bound irrespective of the truth:

| true ρ | n | in-sample | cross-fit (k=4) |
| --- | --- | --- | --- |
| 0.6 | 1500 | +0.999 | +0.376 |
| 0.6 | 6000 | +0.999 | +0.340 |
| 0.3 | 6000 | +0.999 | +0.163 |

Five-fold cross-fitting removes it.

## Two bias directions, at two different stages

The table above shows both effects at once, and they pull opposite ways. Keep
them distinct:

- **In-sample margins bias Σ upward**, toward +1, by the mechanism just
  described. Cross-fitting is what removes this, and it is the reason `cv=5` is
  the default.
- **Cross-fitted margins leave a residual downward bias.** An out-of-fold index
  still carries genuine prediction error, and noise in a regressor attenuates
  the correlation measured through it — classical errors-in-variables. In the
  table, cross-fit estimates land at 0.376 and 0.340 against a true 0.6.

So cross-fitting trades a large upward bias for a smaller downward one. Treat a
fitted correlation as a **floor** on the true dependence when the margins are
only moderately predictive; the gap closes as the margins get sharper. Nothing
in the estimator corrects for the second effect, and correcting it would
require knowing the margins' prediction error on the latent scale.

## Why not full joint MLE (FIML)

The other classical estimator for the same model is FIML: optimize the marginal
parameters and Σ together, in one pass, letting information flow both
directions. FIML is, in principle, more statistically efficient — it can, in
theory, use the dependence structure to sharpen the marginals too.

In practice (Joe, 2005; confirmed again in testing across synthetic and real
data during this library's development) that efficiency gain is small, while
the cost is not. FIML requires evaluating the same expensive
orthant-probability likelihood *inside every boosting round of every marginal
model being jointly fit*, whereas IFM pays that cost exactly once, in a
standalone optimization over the correlation parameters after the margins are
already trained. That asymmetry gets worse, not better, as d grows.

There is also a structural objection specific to this library. Joint
optimization means propagating gradients from the joint likelihood into η_j.
That is fine for x'β_j. It is a non-starter for a gradient-boosted ensemble,
which is fitted by its own greedy stagewise algorithm against its own objective
and exposes no parameter vector for an outer optimizer to move. Any method
demanding a joint gradient rules out most of the estimators worth plugging in —
and a swappable inner model is the entire point.

Testing never turned up a real dataset where FIML won, including one
constructed specifically to favor it (a marginal with labels made artificially
sparse). FIML is not currently supported.

## Why not naive correlation

Before settling on IFM, the shortcut of computing a Pearson correlation of
something simpler than the properly de-attenuated latent correlation was tested
and rejected, each variant for a specific, empirically confirmed reason:

- **Correlating the marginals' raw predicted scores directly.** This reflects
  shared predictors between the marginals as much as it reflects genuine
  residual dependence, and can overstate or understate the true correlation
  depending on the dataset — it went both ways on different data.
- **Correlating the raw binary outcomes.** In the right spirit, but attenuated:
  binary thresholding compresses a continuous latent correlation toward zero,
  so raw-Y correlation systematically understates the latent Σ the model
  actually assumes.
- **Pearson correlation of GEE-style residuals** (removing each marginal's own
  predictors first). This correctly strips out the shared-predictor
  contribution, but stays on the same attenuated binary scale as raw-Y
  correlation — empirically, it landed *below* raw-Y correlation, not between
  raw-Y and the correct value, since removing a genuinely positive
  shared-predictor contribution pulls the number down further.

All three are linear correlation measures applied to a relationship that is
nonlinear by construction (a threshold on a latent Gaussian). Only the
maximum-likelihood approach above, working through Φ, correctly inverts that
nonlinearity.

A **direct joint classifier** — predicting the 2^d outcome patterns as classes
— was also tested and rejected during development. It abandons the latent
structure entirely: the class count grows exponentially, rare patterns get no
support, and nothing constrains the result to be consistent with the marginals.

## Validation on real data

Everything above was developed against synthetic draws, where the true Σ is
known. The claim that matters most in practice — that the composite pairwise
objective is a legitimate substitute for the full likelihood — was checked on
the UCI *Default of Credit Card Clients* data (Yeh & Lien, 2009): 30,000
clients, three binary outcomes (any repayment delay in September, July and
April 2005, i.e. `PAY_0`, `PAY_3`, `PAY_6` > 0), five demographic predictors
(`LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`), an 80/20 split and
`cv=5`. Outcome prevalence was 22.8 / 14.0 / 10.2 percent. The dataset is not
redistributed with this library.

Both modes were run on identical margins — same inner model, same seed, same
folds — so the only thing varying is stage two.

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

**Conclusion.** At d = 3 with moderate positive correlations, `"pairwise"`
gives up about 1e-4 nats/row of held-out likelihood and saves two orders of
magnitude of fitting time. That is a strong result for the composite objective,
but it is evidence from one regime only: Σ was well conditioned throughout
(smallest eigenvalue 0.27 or above), so the projection step never engaged, and
nothing here speaks to large d, near-singular Σ, or strongly mixed-sign
correlations — the settings where a composite likelihood is most likely to
diverge from the full one.

Note also what this study does *not* establish. Cross-fitting was used
throughout, so these runs assume the case made above rather than retesting it;
the in-sample bias measurement remains synthetic.

## Computational cost

The multivariate normal CDF is evaluated by the deterministic recursion
described above. Cost scales roughly as `n_quad ** (d - 2)`: trivial at d = 3,
noticeably heavier by d = 6-7, and impractical with this approach well before
d = 10 — that regime needs a randomized or QMC evaluator, which this library
does not implement. Lower `n_quad`, or switch to `dependence="pairwise"`, when
it starts to hurt.

IFM's advantage over FIML widens as d grows for the same reason: IFM pays this
cost once per Σ fit, optimizing d(d-1)/2 correlations, while FIML would pay the
identical per-call cost inside every boosting round, for every marginal,
throughout training.

## Inference

IFM is consistent but not fully efficient, and the usual standard errors do not
apply. Because step 4 conditions on estimated margins, inverse-Hessian standard
errors for ρ are wrong — they ignore stage-one estimation uncertainty. Correct
inference needs a Godambe (sandwich) information matrix, or a bootstrap that
resamples and refits *both* stages. Neither is implemented; `correlation_` is a
point estimate. See [limitations.md](limitations.md).

## References

- Joe, H., & Xu, J. J. (1996). *The estimation method of inference functions
  for margins for multivariate models.* Technical Report No. 166, Department of
  Statistics, University of British Columbia.
- Joe, H. (2005). Asymptotic efficiency of the two-stage estimation method for
  copula-based models. *Journal of Multivariate Analysis*, 94(2), 401-419.
- Plackett, R. L. (1954). A reduction formula for normal multivariate
  integrals. *Biometrika*, 41(3/4), 351-360.
- Drezner, Z., & Wesolowsky, G. O. (1990). On the computation of the bivariate
  normal integral. *Journal of Statistical Computation and Simulation*, 35(1-2),
  101-107.
- Genz, A. (1992). Numerical computation of multivariate normal probabilities.
  *Journal of Computational and Graphical Statistics*, 1(2), 141-150.
- Gassmann, H. I. (2003). Multivariate normal probabilities: Implementing an old
  idea of Plackett's. *Journal of Computational and Graphical Statistics*,
  12(3), 731-752.
- Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques
  for the predictive accuracy of probability of default of credit card clients.
  *Expert Systems with Applications*, 36(2), 2473-2480.
- Higham, N. J. (2002). Computing the nearest correlation matrix - a problem
  from finance. *IMA Journal of Numerical Analysis*, 22(3), 329-343.
- Varin, C., Reid, N., & Firth, D. (2011). An overview of composite likelihood
  methods. *Statistica Sinica*, 21(1), 5-42.
