# CLAUDE.md

Working notes for this repo. Not user-facing documentation.

## Layout

```
src/multivariate_probit/
    model.py      MultivariateProbit -- the estimator, both IFM stages wired together
    results.py    MultivariateProbitProba -- marginal array + joint queries
    inner.py      ProbitCalibrated adapter, preset registry, as_inner coercion
    linear.py     ProbitRegressor -- native IRLS probit, the default margin
    ifm.py        stage 2: joint_correlation (full MLE), pairwise_correlation (composite)
    _mvn.py       bvn_cdf, mvn_orthant (Genz recursion), sign trick, signed_corr_stack
    _corr.py      Higham nearest-correlation projection
tests/            test_linear.py, test_xgboost.py, test_calibration.py,
                  test_contract.py, test_correlation_projection.py,
                  conftest.py (adds src/ to path)
docs/             ifm.md, implementation.md, api.md, limitations.md
docs/studies/     research archive, one .md per study; data never committed
```

## Commands

```bash
pip install -e ".[dev]"
pytest -q                  # 38 tests, about 13s (3 warnings, all true positives)
python -m pyflakes src tests
```

## Invariants

- **Core dependencies are numpy + scipy only.** scikit-learn and xgboost are
  optional extras used by presets, never by the estimation path. Do not reach
  for `sklearn.model_selection` — `model.py` has its own `_kfold_indices`.
- **The inner model is a black box.** Anything with `fit` + `latent`, or
  anything sklearn-shaped that `ProbitCalibrated` can wrap. No estimation code
  branches on model family, and there is no per-family subpackage. FIML would
  be the trigger to add one; nothing before that.
- **Do not swap in `scipy.stats.multivariate_normal.cdf`.** It is stochastic,
  single-matrix, and ~1e-5. The hand-rolled evaluator is deterministic and
  takes a per-row correlation stack. SciPy's belongs in tests only, as a
  cross-check.
- **Σ is a correlation matrix**, unit diagonal, clipped to abs(ρ) ≤ 0.999.
- Pattern probabilities sum to 1 exactly (sign-trick identity), but nested-event
  inequalities can break at ~1e-5 for d ≥ 3. Tests need tolerances there.

## Bias directions - easy to get backwards

In-sample margins bias Σ **upward** (toward +1); cross-fitting fixes that.
Cross-fitted margins still leave a **downward** attenuation from prediction
error. Both are real, at different stages. An earlier draft of the docs stated
this backwards; the measurement is in `docs/ifm.md`, and
`test_cross_fitting_removes_upward_bias_in_the_correlation` pins it.

## Documentation rules

- **One fact, one home.** README links, it does not duplicate. Bias mechanism
  lives in `ifm.md`; accuracy figures live in `implementation.md`; parameter
  tables live in `api.md`; gaps live in `limitations.md`.
- **Mechanism in `ifm.md`, measurements in `docs/studies/`.** A derivation
  keeps only the numbers it needs to be legible and links to the study for
  the rest. A study states its question first, gives data provenance rather
  than data, records the exact configuration, and ends with what it does not
  establish and what is still open. Datasets are never committed.
- **Impersonal voice.** No "we", no "our". Empirical claims keep their
  provenance through phrasing ("testing across synthetic and real data
  confirmed"), not through a pronoun.
- **Notation:** Σ (correlation matrix), ρ, η, Φ, φ, d outcomes, n observations.
  Greek in prose and math blocks; ASCII in code spans, code fences, headings,
  and identifiers (`correlation_`, `n_quad`, `eta_`). Never use ∑ (U+2211) or
  µ (U+00B5) — Greek block only. Avoid Unicode subscripts; write `ρ_jk`. The
  only superscripts in use are ⁻¹ and ².
- Do not overload Σ as a summation sign; write `sum_i`.

## Known rough edges

- `rf` preset has no tests. (The Higham projection path is covered by
  `tests/test_correlation_projection.py`.)
- No standard errors anywhere; `correlation_` is a point estimate.
- `dependence="joint"` costs `n_quad ** (d - 2)` per likelihood call.
