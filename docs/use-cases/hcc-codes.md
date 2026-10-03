# HCC codes with hierarchies and interactions

**The problem.** A risk-adjustment score adds a coefficient for each
hierarchical condition category (HCC) a member has. Two rules make it
non-additive: a hierarchy (a more severe HCC in a family drops the less severe
one) and interaction terms (an extra coefficient when two conditions occur
together). Predicting each HCC separately and adding up expected coefficients
gets both rules wrong, because both depend on which codes occur *together*.

**The joint query.** `.joint(pattern)` gives the probability of every code
pattern. The expected score is the sum over patterns of P(pattern) times the
score of that pattern after the hierarchy and interactions are applied.
Enumeration costs 2^d evaluations per row, so keep d to the codes the rules
involve.

**This fits the shape if** the HCCs are labelled 0/1 per member-year.

```python
import numpy as np
from multivariate_probit import MultivariateProbit

rng = np.random.default_rng(0)
n, d = 5000, 3
X = rng.normal(size=(n, 6))                                   # stand-in features
Sigma = np.full((d, d), 0.3); np.fill_diagonal(Sigma, 1.0)     # how the outcomes move together
Y = (X[:, :d] + -1.2 + rng.normal(size=(n, d)) @ np.linalg.cholesky(Sigma).T > 0).astype(int)

model = MultivariateProbit(inner="linear", dependence="pairwise").fit(X, Y)
proba = model.predict_proba(X[:5])
coef = np.array([0.30, 0.45, 0.35])  # illustrative values, not published coefficients
def score(p):  # p = (diabetes, diabetes with complications, CHF)
    p = (0, 1, p[2]) if p[1] else p                  # hierarchy: complicated drops uncomplicated
    return coef @ p + 0.12 * (p[1] and p[2])         # illustrative interaction term
from itertools import product
expected = sum(proba.joint(p) * score(p) for p in product((0, 1), repeat=3))
```
