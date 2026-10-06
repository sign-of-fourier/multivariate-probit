# Full maximum likelihood on a GPU (`fitter="modal"`)

An alternative to [IFM](ifm.md), not a stage of it. Every margin's
coefficients and the correlation matrix Σ are estimated together, by
maximising the joint likelihood

```
L(B, Σ) = Σ_i w_i log P(Y = y_i | x_i),    η_ij = b_j0 + x_i' b_j
```

on a hosted GPU service. Only linear margins: the joint likelihood has to be
differentiated through every margin, which rules out boosted trees and forests.

```python
model = MultivariateProbit(fitter="modal").fit(X, Y)
model.correlation_            # Σ
model.stderr_["coef"]         # (d, n_features) standard errors
model.stderr_robust_          # sandwich standard errors
model.predict_proba(X_new)    # runs locally, on any evaluator
```

## When to use it instead of IFM

- You need **standard errors** for the coefficients and for Σ. IFM does not
  report them ([limitations.md](limitations.md)); this path does.
- You want the **efficient estimator**. IFM is consistent but discards the
  information that the other outcomes carry about each margin; full ML does not.
- The margins are linear anyway. With flexible margins, IFM is the only option.

IFM stays the default: it runs locally, takes any inner model, and its cost
does not grow with the number of margin parameters.

## How it works

- The probability of each observed pattern is a d-dimensional orthant integral,
  simulated by GHK with randomised quasi-Monte Carlo: one scrambled Sobol set
  of `n_draws` points, shifted per row. The draws are **fixed for the whole
  fit**, so the simulated log-likelihood is a smooth deterministic function of
  the parameters and a quasi-Newton optimiser (L-BFGS) with exact gradients
  applies.
- Σ is parameterised so that every parameter value is a valid correlation
  matrix: no projection or penalty is ever needed.
- The fit starts from the margins fitted with Σ = I (exact, no simulation),
  then optimises everything jointly.
- X is standardised on the service; coefficients come back on the original
  scale. `inner_params={"alpha": ...}` sets the same small ridge on the slopes
  as `ProbitRegressor` (default 1e-6).

## Standard errors

At the optimum the service computes the exact Hessian of the simulated
log-likelihood. Two covariance matrices come back, mapped to the reported
parameters (coefficients on the original X scale, intercepts, and the
correlations ρ_jk):

- `cov_params_` / `stderr_`: inverse Hessian, the classical ML standard errors.
- `cov_params_robust_` / `stderr_robust_`: the sandwich `H⁻¹ (Σ_i w_i² g_i g_i') H⁻¹`.
  Use it with `sample_weight`, or when the model may be misspecified.

`param_names_` gives the order of both matrices. The standard errors treat the
Sobol draws as fixed: they do not include simulation error, which is small
next to sampling error at the default `n_draws=512`.

## Where the data goes

X, Y and `sample_weight` are uploaded to the service. The response holds
parameters only; the service keeps the result for collection, and nothing
after `fit` makes a network call — prediction and scoring run locally with the
chosen `evaluator`.

## Cost

A fit is submitted, then polled until done. On a T4: about 50 s end to end for
n = 50,000, 10 features and d = 6, standard errors included; small fits take
15 to 20 s, plus up to a minute when the service is starting cold.
