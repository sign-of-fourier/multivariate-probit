# multivariate-probit

**Tie out correlated binary outcomes coherently, without assuming independence.**

Fit one model per yes/no outcome, any classifier you like, and this package
estimates how the outcomes move together. From one fitted model, P(any),
P(all), every pattern and every conditional agree with each other: the 2^d
pattern probabilities sum to exactly 1.

```bash
pip install multivariate-probit            # numpy + scipy only
```

```python
from multivariate_probit import MultivariateProbit

model = MultivariateProbit(inner="linear", dependence="pairwise").fit(X, Y)  # Y: (n, d) of 0/1

proba = model.predict_proba(X_new)
proba.marginal                          # P(Y_j = 1 | x), shape (n, d)
proba.any()                             # P(at least one outcome)
proba.all([0, 2])                       # P(outcomes 0 and 2 both)
proba.joint([1, 0, 1])                  # P(exactly this pattern)
proba.conditional(2, given={0: 1})      # P(Y_2 = 1 | Y_0 = 1, x)
model.correlation_                      # the fitted correlation matrix Σ, (d, d)
```

Background and a worked example:
[quantecarlo.com/multivariate-probit](https://quantecarlo.com/multivariate-probit).
Using this on a real problem?
[Tell us in a use-case issue](https://github.com/sign-of-fourier/multivariate-probit/issues/new?template=use-case.md).

## You might be looking for this if…

- You need **P(at least one)** across several correlated binary labels, and
  multiplying per-label probabilities gives the wrong answer.
- Your **multilabel** or **multi-output** model treats labels as independent,
  and you need the **joint probability** of a label combination.
- You've used R `mvProbit` or Stata `mvprobit` and want **multivariate probit in
  Python** for prediction at scale (no standard errors yet; see below).
- You want to **combine several fraud or risk model scores** into one
  probability that "any of these happened."
- You need **co-occurrence** or **label dependence** expressed as one
  correlation matrix, a **tetrachoric correlation** that conditions on features.
- You want a **Gaussian copula for binary outcomes** on top of XGBoost or
  scikit-learn classifiers.
- You need a **multi-outcome model that a validator can read**: per-outcome
  coefficients plus an explicit correlation matrix (see
  [Explainability](#explainability-scoped)).
- You need **P(outcome A | outcome B already happened)** from the same model
  that gives the marginals.

## The model

A **multivariate probit**: each outcome is a threshold on a latent Gaussian
score, and the scores are correlated.

```
Y_j = 1[ η_j(x) + e_j > 0 ],    e ~ N(0, Σ),    j = 1 … d
```

`η_j` comes from any per-outcome model. Σ is a correlation matrix; with
intercept-only margins, ρ_jk is the classical tetrachoric correlation, and with
features it is the tetrachoric correlation conditional on x. Fitting is
two-stage **Inference Functions for Margins** (IFM): each margin is fitted
alone and cross-fitted, then Σ is estimated by maximum likelihood with the
margins held fixed ([docs/ifm.md](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/ifm.md)).

## Why not write it yourself

It is a short model to write down. These are the places a first version goes
wrong; each is handled here and documented.

- **In-sample margins drive every fitted correlation toward +1.** An overfit
  margin is most confident on the rows where the labels co-occur, and stage two
  reads that as dependence. Margins are cross-fitted by default (`cv=5`);
  `test_cross_fitting_removes_upward_bias_in_the_correlation` pins it.
  [Mechanism](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/ifm.md#why-cross-fitting-is-required).
- **Full-likelihood Σ does not scale with d.** Each likelihood call costs
  `n_quad ** (d - 2)` bivariate evaluations per row; at d = 4 and 250 rows a
  joint fit took 2-4 minutes. The pairwise estimator uses only closed-form
  bivariate integrals and took about 0.07 s on the same data, with the same
  accuracy ([studies/comparators.md](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/studies/comparators.md)).
- **SciPy's multivariate normal CDF is one call per row.** Scoring a test set
  is a Python loop, measured at 3e-3 to 2 s per row, slowest on the
  near-singular Σ that pairwise fits produce
  ([studies/ghk.md](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/studies/ghk.md)).
- **Near-duplicate labels push ρ to ±1, where Σ is singular.** Each ρ is
  bounded to abs(ρ) ≤ 0.999, and a pairwise Σ is projected onto the nearest
  valid correlation matrix before any joint query
  ([docs/ifm.md](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/ifm.md#positive-definiteness)).
- **Pairwise estimates need not form a valid matrix.** Three correlations
  estimated separately can be jointly impossible; the projection repairs that
  (`test_nearest_correlation_repairs_an_incoherent_triple`).

## Explainability, scoped

- **Linear margins** (`inner="linear"`, the default): per-outcome probit
  coefficients (`model.inner_models_[j].coef_`, `.intercept_`) plus an explicit
  correlation matrix. That is an interpretable joint model end to end.
- **XGBoost, random forest or other flexible margins**: the dependence is still
  explicit and readable in `correlation_`, but each margin is as much a black
  box as the classifier inside it. SHAP is the tool for explaining a single
  black-box model; this package does not make one explainable.
- **No standard errors yet.** Coefficients and Σ are point estimates
  ([limitations](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/limitations.md)).

## Bring your own model, or use ours

Stage two consumes only each margin's predicted probability, mapped to the
latent scale, so the per-outcome model is a black box: anything with
`predict_proba`, or anything with `latent(X)`.

| Preset | Estimator | Install |
| --- | --- | --- |
| `"linear"` (default), `"probit"` | native probit via IRLS | — |
| `"xgboost"`, `"xgb"` | `XGBClassifier`, tuned for calibration | `[xgboost]` |
| `"rf"`, `"random_forest"` | `RandomForestClassifier` | `[sklearn]` |

```python
MultivariateProbit(inner=XGBClassifier(max_depth=4))       # any sklearn-style classifier
MultivariateProbit(inner=["linear", "xgboost", "rf"])      # one per outcome
register_inner("mine", make_my_model)                      # your own preset
```

The API is scikit-learn style (`fit`, `predict`, `predict_proba`, `score`,
`get_params`, `set_params`) without requiring scikit-learn, and works with
`clone`, `Pipeline` and `cross_val_score`. Contract and
calibration check: [docs/api.md](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/api.md#inner-models).

## Scale

Scoring a fitted model, `P(Y = y | x)`, is a d-dimensional integral per row.
The backend is chosen with `evaluator=`:

| `evaluator` | Speed | Accuracy | Runs |
| --- | --- | --- | --- |
| `"quadrature"` (default) | 8e-5 s per row at d = 3, 0.12 s at d = 6, impractical past 7 | about 1e-5, deterministic | locally |
| `"scipy"` | 3e-3 to 2 s per row, any d | reference grade, randomised | locally |
| `"orthant"` | 2e-5 to 9e-5 s per row, d = 4 to 20 | median error ≤ 0.007, deterministic | locally, compiled, keyed above d = 3 |

**Speed, with a small accuracy trade-off:** `"orthant"` is two to five orders
of magnitude faster than SciPy, at roughly the accuracy of 100-draw GHK
simulation ([studies/ghk.md](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/studies/ghk.md)). It ships for CPython 3.11
and 3.12 on x86-64 Linux (`pip install multivariate-probit[orthant]`); without
a key it runs d ≤ 3 at `resolution="low"`. Key:
[quantecarlo.com/orthant_key](https://quantecarlo.com/orthant_key).

Benchmarks against other tools: [docs/benchmarks.md](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/benchmarks.md).

## Where your data goes

Nowhere. Fitting and every `evaluator` run on your machine; the package makes
no network calls, and the `"orthant"` key is checked offline.

The one hosted path is a separate package you call yourself:
[`quantecarlo.orthant_cdf`](https://github.com/sign-of-fourier/quantecarlo#orthant-probabilities-orthant_cdf),
a GPU service for test sets too large for one machine. It sends the `(N, d)`
array of latent scores (`model.transform(X)`), the correlation matrix and, for
a pattern query, that pattern as a sign vector. It never receives X. To keep
everything local, don't install or call `quantecarlo`.

## When to use what

| Tool | The right choice when |
| --- | --- |
| scikit-learn `MultiOutputClassifier` | Only per-label probabilities matter, never combinations. |
| scikit-learn `ClassifierChain` (or an ensemble of chains) | The goal is predicting the exact label set; ensembles of chains had the best subset accuracy at d = 6 in [this benchmark](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/studies/multilabel-benchmarks.md). |
| Label powerset (one class per label combination) | Few labels, and every combination you care about is common in the training data. |
| One stacked model on a combined target | Only one fixed combination matters and it has enough positives to train on. |
| `statsmodels` `Probit`, one per outcome | You need classical inference on each outcome's coefficients and only marginal probabilities. |
| [pybhatlib](https://github.com/UMN-Choi-Lab/pybhatlib) | Discrete-choice econometrics (multinomial probit, multivariate ordered response probit) by maximum likelihood with Bhat's analytic MVNCD approximation and analytic gradients. |
| R `mvProbit`, Stata `mvprobit` | A classical full-likelihood multivariate probit with linear indices and small d, inside R or Stata. |
| **multivariate-probit** | Coherent joint queries (any, all, pattern, conditional) on top of flexible per-outcome models, with scoring that scales to large N. |

## Use cases

Problems that fit this shape, each with the joint query that answers it and a
runnable example on synthetic data:

- [Fusing fraud and identity vendor scores](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/use-cases/fraud-vendor-fusion.md): `.any()`, posterior over fraud types
- [Claims fast-track eligibility](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/use-cases/claims-fast-track.md): P(no SIU ∧ no attorney ∧ no total loss)
- [HCC codes with hierarchies and interactions](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/use-cases/hcc-codes.md): `.joint()` over code patterns
- [Content moderation and ad integrity](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/use-cases/content-moderation.md): P(any policy violation)
- [Cyber: phishing and exploited vulnerabilities](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/use-cases/cyber.md): `.all()`, `.any()`
- [Property imagery attributes](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/use-cases/property-imagery.md): roof, pool, solar

## Documentation

- [docs/ifm.md](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/ifm.md): the estimation algorithm, and why IFM
- [docs/api.md](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/api.md): parameters, evaluators, the inner-model contract
- [docs/implementation.md](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/implementation.md): what is hand-rolled, numerical accuracy
- [docs/limitations.md](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/limitations.md): known gaps
- [docs/benchmarks.md](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/benchmarks.md): comparison against other tools
- [docs/studies/](https://github.com/sign-of-fourier/multivariate-probit/blob/main/docs/studies/): every measurement behind the claims above

## Citing

See [CITATION.cff](https://github.com/sign-of-fourier/multivariate-probit/blob/main/CITATION.cff).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](https://github.com/sign-of-fourier/multivariate-probit/blob/main/LICENSE).
