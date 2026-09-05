# multivariate-probit

Multivariate probit models for correlated binary outcomes, with a pluggable
inner model.

## The model

Each outcome is a threshold on a latent Gaussian variable, and the outcomes are
tied together by the correlation of those latents:

```
Y_j = 1[ η_j(x) + e_j > 0 ],    e ~ N(0, Σ),    j = 1 … d
```

`η_j` is an arbitrary real-valued function of the features — linear by default,
or XGBoost, a random forest, or anything else that fits `(X, y)`. Σ is a
correlation matrix (unit diagonal) carrying the dependence between outcomes.
Marginally, `P(Y_j = 1 | x) = Φ(η_j(x))`.

Fitting is cross-fit two-stage IFM: every margin is fitted independently, then
Σ is estimated by maximum likelihood with those margins held fixed. That
separation is what lets the inner model be a black box. The algorithm, and the
alternatives that were tested and rejected, are in
**[docs/ifm.md](docs/ifm.md)**.

## Install

```bash
pip install multivariate-probit           # core: numpy + scipy
pip install multivariate-probit[xgboost]  # adds the "xgboost" preset
pip install multivariate-probit[all]      # every preset
```

## Quickstart

```python
from multivariate_probit import MultivariateProbit

model = MultivariateProbit(inner="linear").fit(X, Y)   # Y is (n, d), 0/1

proba = model.predict_proba(X)
proba.marginal                  # P(Y_j = 1 | x), shape (n, d)
proba.joint([1, 0, 1])          # P(Y = pattern | x), shape (n,)
proba.all()                     # P(every outcome = 1 | x)
proba.any(outcomes=[0, 2])      # P(at least one of these | x)

model.correlation_              # the fitted Σ, shape (d, d)
model.transform(X)              # latent scores η, shape (n, d)
model.sample(X, n_samples=100)  # simulated outcome patterns
model.score(X, Y)               # mean joint log-likelihood
```

`predict_proba` returns an object that behaves like the marginal-probability
array (`np.asarray(proba)`, indexing, `.shape`) and additionally answers the
joint questions Σ was estimated for. Marginal predictions do not involve Σ at
all; everything joint does.

## Everything here is a squashing function over a latent index

The inner model never sees a probability, and never sees another outcome's
labels. It produces an unbounded score η_j(x) on (-∞, ∞); Φ is the only
squashing function applied to it. Any estimator that emits a real-valued score,
or a probability that can be pushed back through Φ⁻¹, is a legal margin.

That is the whole abstraction, and it is why the inner model is swappable
without touching the estimation code.

## Inner models

| Preset | Estimator | Requires |
| --- | --- | --- |
| `"linear"` (default), `"probit"` | native probit via IRLS | — |
| `"xgboost"`, `"xgb"` | `XGBClassifier`, tuned for calibration | `xgboost` |
| `"rf"`, `"random_forest"` | `RandomForestClassifier` | `scikit-learn` |

```python
MultivariateProbit(inner="xgboost")                        # a preset
MultivariateProbit(inner=XGBClassifier(max_depth=4))       # any sklearn-shaped model
MultivariateProbit(inner=["linear", "xgboost", "linear"])  # one per outcome
```

`available_inners()` lists the presets; `register_inner(name, factory)` adds
your own. See **[docs/api.md](docs/api.md)** for the inner-model contract.

## Two knobs that cost time

- **`dependence`** — `"joint"` (default) maximises the full d-variate
  likelihood for Σ. `"pairwise"` maximises each pair's bivariate likelihood
  instead: consistent, orders of magnitude cheaper, and the right choice once
  you have more than a handful of outcomes.
- **`cv`** — `5` by default, cross-fitting the margins so Σ is never estimated
  from in-sample predictions. This is not optional hygiene: in-sample margins
  drive every fitted correlation to +1. `cv=None` skips it, which is defensible
  for the linear default and reckless for anything that can overfit.

## Status

Alpha. The linear and XGBoost paths are covered by tests; the `rf` preset is
wired but untested. Standard errors are not computed — `correlation_` is a
point estimate. Known gaps are listed in
**[docs/limitations.md](docs/limitations.md)**.

## Documentation

- **[docs/ifm.md](docs/ifm.md)** — the estimation algorithm, and why IFM over
  the alternatives
- **[docs/implementation.md](docs/implementation.md)** — what is hand-rolled,
  what comes from SciPy, and why
- **[docs/api.md](docs/api.md)** — parameters, attributes, methods, extension
  points
- **[docs/limitations.md](docs/limitations.md)** — known gaps and roadmap

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
