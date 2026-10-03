# Claims fast-track eligibility

**The problem.** A claim can be fast-tracked when it needs no special
handling: no referral to the special investigations unit (SIU), no attorney
representation, no total loss. Three models give three probabilities, but the
eligibility rule is about all three at once, and the outcomes are correlated:
a claim with an attorney is more likely to be referred.

**The joint query.** P(no SIU ∧ no attorney ∧ no total loss | x) is
`.none()` over the three outcomes. Thresholding it gives the fast-track set
directly.

**This fits the shape if** all three outcomes are recorded on closed claims.

```python
import numpy as np
from multivariate_probit import MultivariateProbit

rng = np.random.default_rng(0)
n, d = 5000, 3
X = rng.normal(size=(n, 6))                                   # stand-in features
Sigma = np.full((d, d), 0.4); np.fill_diagonal(Sigma, 1.0)     # how the outcomes move together
Y = (X[:, :d] + -1.0 + rng.normal(size=(n, d)) @ np.linalg.cholesky(Sigma).T > 0).astype(int)

model = MultivariateProbit(inner="linear", dependence="pairwise").fit(X, Y)
p_clean = model.predict_proba(X).none()              # outcomes: SIU, attorney, total loss
fast_track = p_clean > 0.8
```

Multiplying the three "no" probabilities instead understates P(none) when the
outcomes are positively correlated, so it fast-tracks fewer claims than the
threshold intends.
