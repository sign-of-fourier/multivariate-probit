# Property imagery attributes

**The problem.** Image models tag a property with attributes: roof damage, a
pool, solar panels. Downstream rules care about combinations, for example a
pool with no fence, or roof damage on a property with solar. The attributes
are correlated through things the features only partly capture (age of the
house, neighbourhood), so per-attribute probabilities do not combine by
multiplication.

**The joint query.** `.joint()` for an exact combination, `.all()` for
"these attributes together", and `.conditional()` for "given the roof is
damaged, how likely is solar?".

**This fits the shape if** a labelled sample has every attribute marked 0/1.

```python
import numpy as np
from multivariate_probit import MultivariateProbit

rng = np.random.default_rng(0)
n, d = 5000, 3
X = rng.normal(size=(n, 6))                                   # stand-in features
Sigma = np.full((d, d), 0.3); np.fill_diagonal(Sigma, 1.0)     # how the outcomes move together
Y = (X[:, :d] + -0.5 + rng.normal(size=(n, d)) @ np.linalg.cholesky(Sigma).T > 0).astype(int)

model = MultivariateProbit(inner="linear", dependence="pairwise").fit(X, Y)
proba = model.predict_proba(X[:1000])
p_damage_and_solar = proba.all([0, 2])               # columns: roof damage, pool, solar
p_solar_given_damage = proba.conditional(2, given={0: 1})
```
