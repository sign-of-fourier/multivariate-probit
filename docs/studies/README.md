# Studies

Empirical work behind the estimator, one file per study. This is the project's
research archive: what was measured, why the question came up, what it settled,
and what it left open.

The division of labour with the rest of the docs:

- [ifm.md](../ifm.md) holds *mechanism* — the algorithm and its derivations.
- These files hold *measurements*. A derivation keeps only the numbers it needs
  to be legible; the study it came from keeps the rest.
- [limitations.md](../limitations.md) holds *gaps*, linking to whichever study
  established one.

Each study states its question first, gives data provenance rather than data,
records the exact configuration, and ends with what it does **not** establish
and what is still open. Datasets are never committed.

| Study | Question | Status |
| --- | --- | --- |
| [uci-credit.md](uci-credit.md) | Is the composite pairwise objective a legitimate substitute for the full likelihood? | settled for d = 3, moderate positive Σ |
| [margin-gaps.md](margin-gaps.md) | What does `correlation_` estimate when the margins are imperfect? | derivations confirmed; correction still declined |
| [mulan-multilabel.md](mulan-multilabel.md) | Does modelling Σ beat binary relevance? | yes on subset accuracy, 12/12; gain shrinks as margins improve |
| [multilabel-benchmarks.md](multilabel-benchmarks.md) | Does modelling Σ beat the standard multi-label methods? | no on subset accuracy; ensemble chains lead at d = 6 |
| [sparse-label-transfer.md](sparse-label-transfer.md) | Can a well-populated outcome carry a sparsely observed one through Σ? | conditioning helps, but the gain shrinks with the label; a shared-representation MLP overtakes it |
| [index-correlation-shortcut.md](index-correlation-shortcut.md) | Can Σ be read off the marginal predictions, skipping stage two? | no; the shortcut tracks predictor overlap, not ρ |
| [log-score-evaluator.md](log-score-evaluator.md) | What does the approximate evaluator cost at d = 14? | no ranking or subset-accuracy change, but it reads low on a general Σ (0.006 on the median observed pattern, 0.027 on the modal one); unusable for a mean |
| [comparators.md](comparators.md) | What do joint, pairwise, statsmodels and the orthant .so trade in accuracy and speed? | statsmodels biased on joint queries; pairwise matches joint at ~1/3000 the fit time; the .so is accurate only off the PSD boundary; SciPy's cost tracks p, not d |

## External references

The closest prior art sits in ecology, where the multivariate probit is the
basis of **joint species distribution models** (JSDMs). Two papers bear
directly on the work archived here.

- **Wilkinson, Golding, Guillera-Arroita, Tingley & McCarthy (2021), "Defining
  and evaluating predictions of joint species distribution models",
  *Methods in Ecology and Evolution* 12:394-404,
  [doi:10.1111/2041-210X.13518](https://doi.org/10.1111/2041-210X.13518).**
  Defines four prediction types — *marginal*, *joint*, *conditional marginal*
  and *conditional joint* — which is the taxonomy `results.py` is missing;
  treats the joint against independent log-likelihood as the standard
  comparison, the same one [mulan-multilabel.md](mulan-multilabel.md) runs;
  recommends drawing from the fitted latent rather than enumerating `2 ** d`
  assemblages, naming d = 20 as the point where enumeration becomes infeasible,
  which [log-score-evaluator.md](log-score-evaluator.md) later measured.
- **"Fast Multivariate Probit Estimation via a Two-Stage Composite Likelihood",
  [arXiv:2004.09623](https://arxiv.org/abs/2004.09623), *Statistics in
  Biosciences* (2022).** The estimator this package implements — stage one
  marginal probits, stage two pairwise composite likelihood — under its
  published name. Notes that inference on the correlation estimates requires
  standard errors adjusted for the stage-one dependence, which is the
  methodology behind the gap recorded in [../limitations.md](../limitations.md).
