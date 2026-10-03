# multivariate-probit

Correlated yes/no outcomes, modelled together. Bring any classifier for each
outcome; the package estimates how the outcomes move together and answers every
joint question from one fitted model.

```bash
pip install multivariate-probit
```

**Speed.** Fitting the dependence takes about 0.07 s where a full-likelihood
fit takes minutes. Scoring runs at 2e-5 to 9e-5 s per row up to d = 20 with
the compiled backend, and about a second per million rows on the hosted GPU
service, against 3e-3 to 2 s per row for SciPy. The cost is a median error of
up to about 0.007 on a probability, roughly 100-draw GHK accuracy
([studies/ghk.md](docs/studies/ghk.md)).

Background and worked examples: **[quantecarlo.com/multivariate-probit](https://quantecarlo.com/multivariate-probit)**.

## 1. Explainable when the target breaks into parts

Many targets are a combination of simpler outcomes: a conversion is a click,
a sign-up and a purchase; a claim is fraud on any of several checks. Model the
parts and the combination falls out:

```
Y_j = 1[ η_j(x) + e_j > 0 ],    e ~ N(0, Σ),    j = 1 … d
```

- Each part has its own margin, `P(Y_j = 1 | x) = Φ(η_j(x))`, from a model you
  can inspect on its own.
- How the parts move together is Σ: d(d-1)/2 correlations, each one readable.
- Every question about the parts is an exact consequence of those two pieces:
  a full pattern, all, any, none, or one part given the others. The 2^d pattern
  probabilities sum to 1 exactly.

```python
proba = model.predict_proba(X)
proba.all([0, 1, 2])                  # converted: clicked AND signed up AND bought
proba.any([3, 4, 5])                  # flagged by at least one check
proba.conditional(2, given={0: 1})    # P(buys | clicked, x)
```

The other multivariate-probit options in Python do not get there.
`statsmodels` fits one probit per outcome, so it has margins but no Σ, and its
joint answers are biased: `P(any)` off by +0.072 at ρ = +0.6 where this package
is off by 0.020 ([studies/comparators.md](docs/studies/comparators.md)).
[multinomial_probit](https://github.com/david-cortes/multinomial_probit) (David
Cortes, archived 2024) is a full-likelihood probit for one categorical outcome;
applied here it has to treat every one of the 2^d patterns as its own class,
and on current SciPy its fit returned NaN coefficients or ran past 30 s on
every test run. Fitting a full probit likelihood directly is hard in general;
the next section is how this package avoids it.

## 2. IFM makes the fit tractable

Fitting margins and Σ jointly means evaluating a d-dimensional normal integral
inside every step of every margin's optimizer, and it rules out any margin that
is not fitted by gradient, such as a boosted ensemble. Inference Functions for
Margins splits the fit in two:

1. Fit each margin on its own, cross-fitted so Σ never sees in-sample
   predictions (in-sample margins drive every correlation toward +1).
2. Holding the margins fixed, estimate Σ by maximum likelihood, either over
   the full d-variate likelihood (`dependence="joint"`) or pair by pair
   (`dependence="pairwise"`, closed-form bivariate integrals).

The pairwise fit matched the joint fit's accuracy in testing at about 1/3000 of
the time (0.07 s against 2-4 minutes at d = 4). Algorithm and the alternatives
rejected: **[docs/ifm.md](docs/ifm.md)**.

## 3. Scales, with a small accuracy trade-off

Scoring a fitted model, `P(Y = y | x)` per row, is a d-dimensional integral per
row. Three evaluators, chosen with `evaluator=`, plus a hosted service:

| Route | Speed | Accuracy |
| --- | --- | --- |
| `"quadrature"` (default) | 8e-5 s per row at d = 3, 0.1 s at d = 6, impractical past 7 | about 1e-5, deterministic |
| `"scipy"` | 3e-3 to 2 s per row, any d | reference grade, random |
| `"orthant"` (compiled, keyed) | 2e-5 to 9e-5 s per row, flat to d = 20 | median error ≤ 0.007, deterministic |
| [`quantecarlo.orthant_cdf`](https://github.com/sign-of-fourier/quantecarlo#orthant-probabilities-orthant_cdf) (hosted GPU) | about 1 s per million rows at d = 20 | same as `"orthant"` |

```bash
pip install multivariate-probit[orthant]   # CPython 3.11/3.12, x86-64 Linux
```

Without a key, `"orthant"` runs d ≤ 3 at `resolution="low"`. A key unlocks
every d: [quantecarlo.com/orthant_key](https://quantecarlo.com/orthant_key).
Measurements: [studies/ghk.md](docs/studies/ghk.md).

## 4. Bring your own model, or use ours

Stage two consumes only each margin's latent index η_j(x), so the inner model
is a black box: anything with `predict_proba`, or anything with `latent(X)`.

| Preset | Estimator | Requires |
| --- | --- | --- |
| `"linear"` (default), `"probit"` | native probit via IRLS | — |
| `"xgboost"`, `"xgb"` | `XGBClassifier`, tuned for calibration | `[xgboost]` |
| `"rf"`, `"random_forest"` | `RandomForestClassifier` | `[sklearn]` |

```python
MultivariateProbit(inner="rf")                             # a preset
MultivariateProbit(inner=XGBClassifier(max_depth=4))       # any sklearn-shaped classifier
MultivariateProbit(inner=["linear", "xgboost", "rf"])      # one per outcome
register_inner("mine", make_my_model)                      # your own preset
```

The contract, and why an uncalibrated score is rejected:
**[docs/api.md](docs/api.md#inner-models)**.

## Quickstart

```python
from multivariate_probit import MultivariateProbit

model = MultivariateProbit(inner="xgboost", dependence="pairwise").fit(X, Y)  # Y is (n, d), 0/1

proba = model.predict_proba(X)
proba.marginal                  # P(Y_j = 1 | x), shape (n, d)
proba.joint([1, 0, 1])          # P(Y = pattern | x), shape (n,)
proba.all(), proba.any()        # P(every / at least one outcome = 1 | x)
proba.conditional(2, given={0: 1, 1: 0})

model.correlation_              # the fitted Σ, shape (d, d)
model.calibration_              # per-margin calibration slope, a scale check
model.sample(X, n_samples=100)  # simulated outcome patterns
model.score(X, Y)               # mean joint log-likelihood
```

`predict_proba` returns an object that behaves like the marginal-probability
array and also answers the joint questions. Marginal predictions do not involve
Σ; everything joint does.

## Documentation

- **[docs/ifm.md](docs/ifm.md)**: the estimation algorithm, and why IFM
- **[docs/api.md](docs/api.md)**: parameters, evaluators, inner-model contract
- **[docs/implementation.md](docs/implementation.md)**: what is hand-rolled, numerical accuracy
- **[docs/limitations.md](docs/limitations.md)**: known gaps (no standard errors yet; Σ is a point estimate)
- **[docs/studies/](docs/studies/)**: every measurement behind the claims above

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
