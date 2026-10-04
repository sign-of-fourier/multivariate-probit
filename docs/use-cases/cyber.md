# Cyber: phishing and exploited vulnerabilities

**The problem.** Two common shapes. An email can be malicious, phishing,
credential-harvesting, and a response playbook may trigger only when it is
both malicious *and* phishing. Separately, a host carries several
vulnerabilities, and the risk question is whether *any* of them gets
exploited, where exploitation of one often signals exposure to the others.

**The joint query.** `.all([0, 1])` for malicious ∧ phishing; `.any()`
across a vulnerability set.

**This fits the shape if** each outcome is labelled 0/1 per email or per host.

```python
import numpy as np
from multivariate_probit import MultivariateProbit

rng = np.random.default_rng(0)
n, d = 5000, 4
X = rng.normal(size=(n, 6))                                   # stand-in features
Sigma = np.full((d, d), 0.5); np.fill_diagonal(Sigma, 1.0)     # how the outcomes move together
Y = (X[:, :d] + -1.0 + rng.normal(size=(n, d)) @ np.linalg.cholesky(Sigma).T > 0).astype(int)

model = MultivariateProbit(inner="linear", dependence="pairwise").fit(X, Y)
proba = model.predict_proba(X[:1000])
p_malicious_and_phishing = proba.all([0, 1])         # columns: malicious, phishing, ...
p_any_exploited = proba.any()                        # or: columns = vulnerabilities on a host
```
