# Fusing fraud and identity vendor scores

**The problem.** An application passes through several vendors: document
verification (forged ID), bot detection, account-takeover (ATO) signals, a
synthetic-identity score. Each returns its own score, and investigations label
which kinds of fraud actually happened. The decision needs one number, the
probability that *any* fraud is present, and, for a flagged application,
which kind is most likely. Taking the maximum score, or multiplying
"probability of no fraud" across vendors, ignores that the fraud types travel
together.

**The joint query.** Model the four fraud types as correlated outcomes, with
the vendor scores among the features. `.any()` gives P(any fraud | x). The
posterior over fraud configurations given that something is wrong is
`.joint(pattern) / .any()`.

**This fits the shape if** each fraud type is labelled 0/1 on every
application in the training data.

```python
import numpy as np
from multivariate_probit import MultivariateProbit

rng = np.random.default_rng(0)
n, d = 5000, 4
X = rng.normal(size=(n, 6))                                   # stand-in features
Sigma = np.full((d, d), 0.5); np.fill_diagonal(Sigma, 1.0)     # how the outcomes move together
Y = (X[:, :d] + -1.5 + rng.normal(size=(n, d)) @ np.linalg.cholesky(Sigma).T > 0).astype(int)

model = MultivariateProbit(inner="linear", dependence="pairwise").fit(X, Y)
proba = model.predict_proba(X[:3])
p_any = proba.any()                                  # forged ID, bot, ATO or synthetic
from itertools import product
posterior = {p: proba.joint(p) / p_any for p in product((0, 1), repeat=4) if any(p)}
```

The 15 posterior values per row sum to 1, so "most likely configuration given
a flag" is an argmax over them.
