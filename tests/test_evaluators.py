"""The orthant backends: selection, plumbing, and refusing rather than falling back.

These are wiring tests, not accuracy studies. The compiled ``orthant`` package
decides its tier once, at import, from ``$ORTHANT_KEY`` or ``~/.orthant/key``,
so its tests run in a subprocess with a controlled environment.
"""

import glob
import os
import platform
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from multivariate_probit import MultivariateProbit, pattern_prob

SRC = Path(__file__).resolve().parents[1] / "src"
ORTHANT_DIR = SRC / "multivariate_probit" / "orthant"

orthant_built = pytest.mark.skipif(
    not (
        sys.platform.startswith("linux")
        and platform.machine() == "x86_64"
        and sys.version_info[:2] == (3, 12)
        and glob.glob(str(ORTHANT_DIR / "_ofree.*.so"))
    ),
    reason="the compiled orthant package is built for CPython 3.12 on x86-64 Linux",
)

CORR3 = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, -0.3], [0.2, -0.3, 1.0]])


def _problem(d, n=6, seed=0):
    rng = np.random.default_rng(seed)
    corr = 0.3 * np.ones((d, d)) + 0.7 * np.eye(d)
    return rng.standard_normal((n, d)), rng.integers(0, 2, (n, d)), corr


def test_unknown_evaluator_is_rejected():
    eta, Y, corr = _problem(3)
    with pytest.raises(ValueError, match="evaluator must be one of"):
        pattern_prob(eta, Y, corr, evaluator="mvn")


def test_unknown_evaluator_is_rejected_before_any_fitting():
    X = np.random.default_rng(0).standard_normal((20, 2))
    Y = (X > 0).astype(int)
    with pytest.raises(ValueError, match="evaluator must be one of"):
        MultivariateProbit(evaluator="gh").fit(X, Y)


def test_scipy_agrees_with_quadrature():
    rng = np.random.default_rng(1)
    eta, Y = rng.standard_normal((5, 3)), rng.integers(0, 2, (5, 3))
    quad = pattern_prob(eta, Y, CORR3)
    sp = pattern_prob(eta, Y, CORR3, evaluator="scipy")
    np.testing.assert_allclose(sp, quad, atol=1e-4)


def test_evaluator_reaches_the_joint_queries():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((200, 2))
    Y = (X + rng.standard_normal((200, 2)) > 0).astype(int)
    model = MultivariateProbit(cv=None, evaluator="scipy").fit(X, Y)
    assert model.get_params()["evaluator"] == "scipy"
    proba = model.predict_proba(X[:4])
    assert proba.evaluator == "scipy"
    model.set_params(evaluator="quadrature")
    reference = model.predict_proba(X[:4])
    np.testing.assert_allclose(proba.all(), reference.all(), atol=1e-4)
    np.testing.assert_allclose(proba.joint([1, 0]), reference.joint([1, 0]), atol=1e-4)


def _run_orthant(script, key=None, tmp_path=None):
    env = {k: v for k, v in os.environ.items() if k != "ORTHANT_KEY"}
    env["HOME"] = str(tmp_path)  # hides any ~/.orthant/key
    env["PYTHONPATH"] = str(SRC)
    if key is not None:
        env["ORTHANT_KEY"] = key
    prelude = textwrap.dedent(
        """
        import numpy as np
        from multivariate_probit import pattern_prob
        rng = np.random.default_rng(0)
        def problem(d):
            corr = 0.3 * np.ones((d, d)) + 0.7 * np.eye(d)
            return rng.standard_normal((6, d)), rng.integers(0, 2, (6, d)), corr
        """
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@orthant_built
def test_orthant_without_key_runs_at_low_resolution_up_to_three(tmp_path):
    out = _run_orthant(
        """
        eta, Y, corr = problem(3)
        got = pattern_prob(eta, Y, corr, evaluator="orthant", resolution="low")
        ref = pattern_prob(eta, Y, corr)
        assert np.all((got >= 0) & (got <= 1))
        assert np.abs(got - ref).max() < 1e-2, np.abs(got - ref).max()
        """,
        tmp_path=tmp_path,
    )
    assert out.returncode == 0, out.stderr


@orthant_built
@pytest.mark.parametrize(
    "d, resolution, message",
    [(4, "low", "n=4 requires a license"), (3, "high", "resolution='high' requires a license")],
)
def test_orthant_without_key_refuses_rather_than_falling_back(tmp_path, d, resolution, message):
    out = _run_orthant(
        f"""
        eta, Y, corr = problem({d})
        try:
            pattern_prob(eta, Y, corr, evaluator="orthant", resolution={resolution!r})
        except ValueError as exc:
            print(exc)
        else:
            raise SystemExit("no error")
        """,
        tmp_path=tmp_path,
    )
    assert out.returncode == 0, out.stderr
    assert "evaluator='orthant'" in out.stdout
    assert message in out.stdout


@orthant_built
@pytest.mark.skipif(not os.environ.get("ORTHANT_KEY"), reason="needs $ORTHANT_KEY")
def test_orthant_with_key_runs_above_three(tmp_path):
    out = _run_orthant(
        """
        from multivariate_probit import orthant
        assert orthant.tier == "paid", orthant.tier
        eta, Y, corr = problem(5)
        got = pattern_prob(eta, Y, corr, evaluator="orthant")
        ref = pattern_prob(eta, Y, corr)
        assert np.abs(got - ref).max() < 1e-2, np.abs(got - ref).max()
        """,
        key=os.environ["ORTHANT_KEY"],
        tmp_path=tmp_path,
    )
    assert out.returncode == 0, out.stderr
