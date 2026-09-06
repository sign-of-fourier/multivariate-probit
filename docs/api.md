# API reference

For the model and a quickstart see [../README.md](../README.md); for the
estimation algorithm see [ifm.md](ifm.md).

## MultivariateProbit

```python
MultivariateProbit(
    inner="linear",
    inner_params=None,
    dependence="joint",
    cv=5,
    n_quad=24,
    optimizer="Nelder-Mead",
    project_correlation=True,
    random_state=None,
)
```

### Parameters

| Name | Default | Meaning |
| --- | --- | --- |
| `inner` | `"linear"` | Preset name, estimator instance, factory, or a list of length d giving one per outcome. Instances are deep-copied, so one instance can seed every margin. |
| `inner_params` | `None` | Keyword arguments forwarded to the preset factory. Ignored when `inner` is already an instance. |
| `dependence` | `"joint"` | `"joint"` maximises the full d-variate likelihood for Σ; `"pairwise"` maximises each pair's bivariate likelihood (composite likelihood, far cheaper). |
| `cv` | `5` | Folds used to cross-fit the latent indices that stage two consumes. `None` skips cross-fitting — see the warning below. |
| `n_quad` | `24` | Gauss-Legendre order for the orthant evaluator. Lower it if fitting with many outcomes gets slow. |
| `optimizer` | `"Nelder-Mead"` | Passed to `scipy.optimize.minimize` for `dependence="joint"`. Derivative-free by design. |
| `project_correlation` | `True` | Project a pairwise estimate onto the nearest positive-definite correlation matrix. No effect when `dependence="joint"`. |
| `random_state` | `None` | Controls the cross-fitting split and `sample`. |

> **Reading `calibration_`.** This is the Cox calibration slope, a deliberately
> simple scale check: a probit of each outcome on its own fitted index. A slope
> near 1 is expected. It is *not* a calibration assessment — a slope of 1 rules
> out a first-order scale error and nothing more, and a departure from 1 has
> several possible causes: miscalibrated probabilities from the classifier, a
> correctly calibrated but noisy index, or an index scored on the rows it was
> fitted on. For a real assessment of a classifier's probabilities use a
> reliability curve; calibrating the margins is the caller's job, not the
> estimator's, and `CalibratedClassifierCV` is the usual way to do it.
>
> Slopes outside roughly [0.5, 2.0] emit a `UserWarning` and never raise. The
> band is a loose convenience, not a test with a calibrated error rate — a
> slope inside it is not a clean bill of health, and one outside it is a
> suggestion to look, not a verdict on the fit. A slope that is not identified
> — a constant or non-finite index, or a single-class outcome — is `nan` and
> stays silent. A slope near 1 is not evidence that Σ is trustworthy: the check
> is blind to omitted signal by construction. See
> [ifm.md](ifm.md#the-calibration-slope-sees-exactly-one-of-them) and
> [limitations.md](limitations.md).

> **`cv=None` is not a neutral speed-up.** In-sample margins drive every fitted
> correlation toward +1. It is defensible for the linear default and reckless
> for anything that can overfit. See
> [ifm.md](ifm.md#why-cross-fitting-is-required).

### Attributes

| Name | Shape | Meaning |
| --- | --- | --- |
| `inner_models_` | list, length d | The fitted margins, one per outcome. |
| `correlation_` | (d, d) | The fitted Σ. |
| `eta_` | (n, d) | The (cross-fitted) latent indices stage two was fitted on. |
| `calibration_` | (d, 2) | Intercept and slope of a probit of each outcome on its own fitted index — the calibration slope. See the note below. |
| `nll_` | float | Negative log-likelihood at the end of the dependence fit. Joint and pairwise fits optimise different objectives, so the values are not comparable across settings. |
| `optimize_result_` | OptimizeResult or None | The SciPy result for `dependence="joint"`. |
| `n_outcomes_`, `n_features_in_` | int | |

### Methods

| Method | Returns | Notes |
| --- | --- | --- |
| `fit(X, Y, sample_weight=None)` | self | `Y` is (n, d) and strictly 0/1. Weights are forwarded to margins that accept them. |
| `decision_function(X)` / `transform(X)` | (n, d) | Latent indices η on (-∞, ∞). |
| `fit_transform(X, Y)` | (n, d) | |
| `predict_proba(X)` | `MultivariateProbitProba` | See below. |
| `predict_marginal_proba(X)` | (n, d) | Marginal probabilities as a plain array. |
| `predict(X, threshold=0.5)` | (n, d) | Per-outcome 0/1 at a marginal threshold. For a joint decision use `predict_proba(X).all()`. |
| `joint_proba(X, Y)` | (n,) | `P(Y = y | x)`. `Y` may be one pattern of length d, broadcast over rows, or an (n, d) array. |
| `joint_log_proba(X, Y)` | (n,) | |
| `score(X, Y, sample_weight=None)` | float | Mean joint log-likelihood. Higher is better. |
| `sample(X, n_samples=1, random_state=None)` | (n, d) or (n_samples, n, d) | Draws patterns from the fitted model. |
| `get_params` / `set_params` | | scikit-learn-style, without requiring scikit-learn. |

## MultivariateProbitProba

Returned by `predict_proba`. Behaves like the (n, d) array of marginal
probabilities — `np.asarray(proba)`, `proba[i, j]`, `proba.shape`, `len(proba)`
— and additionally answers the joint questions Σ was estimated for.

| Member | Returns | Meaning |
| --- | --- | --- |
| `.marginal` | (n, d) | `P(Y_j = 1 | x)`. Does not involve Σ. |
| `.eta` | (n, d) | The latent indices behind it. |
| `.corr` | (d, d) | The Σ used for joint queries. |
| `.joint(pattern)` | (n,) | `P(Y = pattern | x)` for a fully specified 0/1 pattern. |
| `.all(outcomes=None)` | (n,) | `P(every selected outcome = 1 | x)`. |
| `.any(outcomes=None)` | (n,) | `P(at least one selected outcome = 1 | x)`. |
| `.none(outcomes=None)` | (n,) | `P(no selected outcome = 1 | x)`. |

`outcomes` takes a list of column indices, so "do these three co-occur" is a
one-liner.

## Inner models

### The contract

An inner model is anything with `fit(X, y)` and `latent(X) -> (n,)`, where
`latent` returns a real-valued index on the probit scale.

Estimators that do not expose `latent` must expose `predict_proba`, and are
wrapped automatically by `ProbitCalibrated`, whose `latent(X)` is `Φ⁻¹(p̂)`
clipped away from 0 and 1. Anything else is a `TypeError` at coercion, before
any margin is fitted: an uncalibrated `decision_function` score is on an
arbitrary scale, so `Φ` applied to it is not a probability and no diagnostic
can detect the mismatch. Wrap such an estimator in a calibrator first
(`CalibratedClassifierCV`, or `SVC(probability=True)`).

The probability itself need not be any good. Inverting the link is exact for a
calibrated `p̂`, and a miscalibrated one is reported by `calibration_` rather
than repaired.

### Presets

| Name | Aliases | Estimator | Requires |
| --- | --- | --- | --- |
| `linear` | `probit` | `ProbitRegressor` (native IRLS) | — |
| `xgboost` | `xgb` | `XGBClassifier`, defaults tuned for calibration rather than ranking | `xgboost` |
| `rf` | `random_forest` | `RandomForestClassifier` | `scikit-learn` |

The registry is deliberately thin. Under IFM there is no per-family estimation
logic — the presets differ only in which pre-wired instance stage one fits — so
a preset is a factory function and nothing more.

### Registry functions

| Function | Purpose |
| --- | --- |
| `available_inners()` | Sorted preset names. |
| `make_inner(name, **kwargs)` | Instantiate a preset, forwarding kwargs to the estimator. |
| `register_inner(name, factory, overwrite=False)` | Add a preset. `factory(**kwargs)` returns a fresh estimator. |
| `as_inner(spec, **kwargs)` | Coerce a name, factory, class, or instance into an unfitted inner model. |

### Mixing models per outcome

```python
MultivariateProbit(inner=["linear", "xgboost", "linear"])
MultivariateProbit(inner=["linear", make_inner("xgboost", max_depth=2)])
```

## ProbitRegressor

The default margin, usable on its own as a binary classifier.

```python
ProbitRegressor(alpha=1e-6, fit_intercept=True, max_iter=100, tol=1e-8)
```

`alpha` is an L2 penalty on the slopes only, never the intercept. Exposes
`coef_`, `intercept_`, `n_iter_`, `classes_`, and `fit` / `latent` /
`decision_function` / `predict_proba` / `predict`.

## Lower-level functions

Exported for callers who want the pieces without the estimator.

| Function | Purpose |
| --- | --- |
| `bvn_cdf(a, b, rho)` | Φ_2, vectorised, correlations may vary by row. |
| `mvn_orthant(A, corr_stack)` | Φ_d with a per-row correlation stack. |
| `orthant_prob(upper, corr)` | Φ_d with one shared correlation matrix. |
| `pattern_prob(eta, Y, corr)` | `P(Y = y | x)` via the sign trick. |
| `pairwise_correlation(eta, Y, ...)` | Stage two, composite likelihood. |
| `joint_correlation(eta, Y, ...)` | Stage two, full likelihood. Returns `(corr, result)`. |
| `pair_log_likelihood(rho, ...)` | Bivariate probit log-likelihood in ρ. |
| `joint_log_likelihood(corr, eta, Y, ...)` | d-variate log-likelihood in Σ. |
