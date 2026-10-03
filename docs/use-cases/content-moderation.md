# Content moderation and ad integrity

**The problem.** A post or an ad is scored by several policy classifiers:
hate, harassment, violence, adult content, scams. Enforcement often turns on
whether *any* policy is violated, and the classes are correlated: harassment
and hate co-occur far more often than independence suggests. Combining the
classifier outputs as if independent gets P(any) wrong.

**The joint query.** `.any()` over the policy classes. `.conditional()`
answers "given it was confirmed as harassment, how likely is hate too?",
useful for routing a confirmed item to the right reviewers.

**This fits the shape if** reviewed items carry a 0/1 decision per policy.

```python
import numpy as np
from multivariate_probit import MultivariateProbit

rng = np.random.default_rng(0)
n, d = 5000, 5
X = rng.normal(size=(n, 6))                                   # stand-in features
Sigma = np.full((d, d), 0.4); np.fill_diagonal(Sigma, 1.0)     # how the outcomes move together
Y = (X[:, :d] + -2.0 + rng.normal(size=(n, d)) @ np.linalg.cholesky(Sigma).T > 0).astype(int)

model = MultivariateProbit(inner="linear", dependence="pairwise").fit(X, Y)
proba = model.predict_proba(X[:1000])
review = proba.any() > 0.3                           # any of the five policies
p_hate_given_harassment = proba.conditional(0, given={1: 1})
```

With flexible classifiers per policy, pass them as `inner=`; see
[api.md](../api.md#inner-models).
