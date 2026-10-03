"""Scoring benchmark: P(Y = y | x) for held-out rows of a fitted model.

Every column scores the same fitted multivariate-probit model (eta, Sigma),
except statsmodels, which fits one Probit per outcome and has no Sigma, so its
pattern probability is a product of marginals. Error is the absolute
difference from a tight-tolerance SciPy reference. Times are wall-clock per row.
"""
import os, sys, time, warnings
import numpy as np
from scipy.stats import multivariate_normal, norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))  # this checkout, not an installed copy

from multivariate_probit import MultivariateProbit
from multivariate_probit._mvn import signed_corr_stack

try:
    from pybhatlib.gradmvn import mvncd_batch
except ImportError:
    mvncd_batch = None
try:
    import statsmodels.api as sm
except ImportError:
    sm = None
try:
    from multivariate_probit import orthant
    ORTHANT_KEYED = orthant.tier != "free"
except ImportError:
    orthant, ORTHANT_KEYED = None, False

REF_TOL = dict(abseps=1e-7, releps=1e-6)


def make_data(d, n_train, n_test, rho, seed):
    rng = np.random.default_rng(seed)
    p = 6
    X = rng.normal(size=(n_train + n_test, p))
    B = rng.normal(0, 0.6, size=(p, d))
    b0 = rng.normal(-0.3, 0.5, size=d)
    Sigma = np.full((d, d), rho)
    np.fill_diagonal(Sigma, 1.0)
    E = rng.normal(size=(len(X), d)) @ np.linalg.cholesky(Sigma).T
    Y = (X @ B + b0 + E > 0).astype(int)
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def signed_problem(eta, Y, C):
    s = 2.0 * Y - 1.0
    return s * eta, signed_corr_stack(C, s)  # lower-orthant limits, (d, d, n) stack


def timed(f, warm=True):
    if warm:
        f()  # first call pays for imports and JIT; time the second
    t = time.perf_counter()
    out = f()
    return np.asarray(out, dtype=float), time.perf_counter() - t


def run_cell(d, n_train=3000, n_test=100, rho=0.4, seed=0):
    Xtr, Ytr, Xte, Yte = make_data(d, n_train, n_test, rho, seed)
    model = MultivariateProbit(inner="linear", dependence="pairwise").fit(Xtr, Ytr)
    eta, C = model.transform(Xte), model.correlation_
    A, stack = signed_problem(eta, Yte, C)
    n = len(A)
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ref_d{d}_n{n_train}_{n_test}_rho{rho}_s{seed}.npz")
    if os.path.exists(cache):  # the reference is the slow part; reuse it across reruns
        z = np.load(cache)
        ref, t_ref = z["ref"], float(z["t_ref"])
    else:
        ref, t_ref = timed(lambda: [
            multivariate_normal.cdf(A[i], np.zeros(d), stack[:, :, i], allow_singular=True, **REF_TOL)
            for i in range(n)
        ], warm=False)
        np.savez(cache, ref=ref, t_ref=t_ref)
    res = {}
    if d <= 6:
        res["mvp quadrature"] = timed(lambda: model.set_params(evaluator="quadrature").joint_proba(Xte, Yte))
    if orthant is not None and (ORTHANT_KEYED or d <= 3):
        r = "high" if ORTHANT_KEYED else "low"
        res[f"mvp orthant ({r})"] = timed(lambda: model.set_params(evaluator="orthant", resolution=r).joint_proba(Xte, Yte))
    res["SciPy default"] = timed(lambda: model.set_params(evaluator="scipy").joint_proba(Xte, Yte))
    if mvncd_batch is not None:
        st = np.moveaxis(stack, 2, 0)
        res["pybhatlib ovus"] = timed(lambda: mvncd_batch(A, st, method="ovus"))
    if sm is not None:
        fits = [sm.Probit(Ytr[:, j], sm.add_constant(Xtr)).fit(disp=0) for j in range(d)]
        Xc = sm.add_constant(Xte)

        def sm_pattern():  # no Sigma: the pattern probability is a product of marginals
            P = np.column_stack([f.predict(Xc) for f in fits])
            return np.prod(np.where(Yte == 1, P, 1 - P), axis=1)
        res["statsmodels"] = timed(sm_pattern)
    model.set_params(evaluator="quadrature", resolution="high")
    rows = []
    for name, (p, t) in res.items():
        e = np.abs(p - ref)
        rows.append(dict(d=d, N=n, method=name, median_err=np.median(e), max_err=e.max(), s_per_row=t / n))
    rows.append(dict(d=d, N=n, method="reference", median_err=0.0, max_err=0.0, s_per_row=t_ref / n))
    return rows, float(np.median(ref))


if __name__ == "__main__":
    import sys
    warnings.filterwarnings("ignore")
    for d in map(int, sys.argv[1:] or [3]):
        rows, mp = run_cell(d)
        print(f"d={d} median p={mp:.4f}")
        for r in rows:
            print(f"  {r['method']:16s} {r['median_err']:.5f} ({r['max_err']:.5f})  {r['s_per_row']:.2e} s/row")
