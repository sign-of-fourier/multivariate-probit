"""Client for the hosted full-likelihood fit (``fitter="modal"``).

The fit runs on a GPU service: the request carries X, Y and the weights; the
response carries parameters only -- coefficients, intercepts, the correlation
matrix and their covariance. Everything after that runs locally. The service
answers a submission with a call id and is then polled, because a fit can
outlast a single HTTP request.

Standard library only (``urllib``), so the package gains no dependency.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

__all__ = ["fit_remote", "FIT_URL", "RESULT_URL"]

FIT_URL = "https://info-29741--bo-gp-service-mvp-fit.modal.run"
RESULT_URL = "https://info-29741--bo-gp-service-mvp-fit-result.modal.run"

_REQUEST_TIMEOUT = 300  # seconds per HTTP request (the upload can be large)


def _error_text(exc):
    try:
        body = json.loads(exc.read())
        return body.get("error", str(body))
    except Exception:
        return str(exc)


def fit_remote(X, Y, sample_weight=None, poll_interval=2.0, timeout=3600, **params):
    """Submit a fit, wait for it, and return the result as a dict of arrays.

    ``params`` are the service's fields: ``n_draws``, ``seed``, ``max_iter``,
    ``tol``, ``alpha``, ``se``, ``dtype``. Raises ``ValueError`` if the service
    rejects the input, ``RuntimeError`` if the fit fails or cannot be
    collected, and ``TimeoutError`` after ``timeout`` seconds.
    """
    arrays = {"X": np.asarray(X, dtype=np.float64), "Y": np.asarray(Y, dtype=np.float64)}
    if sample_weight is not None:
        arrays["sample_weight"] = np.asarray(sample_weight, dtype=np.float64)
    buf = io.BytesIO()
    np.savez(buf, params=np.array(json.dumps(params)), **arrays)
    request = urllib.request.Request(
        FIT_URL,
        data=buf.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT) as resp:
            call_id = json.loads(resp.read())["call_id"]
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            raise ValueError(f"the fitting service rejected the input: {_error_text(exc)}") from None
        raise RuntimeError(f"the fitting service failed ({exc.code}): {_error_text(exc)}") from None

    url = RESULT_URL + "?" + urllib.parse.urlencode({"call_id": call_id, "format": "npz"})
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(url, timeout=_REQUEST_TIMEOUT) as resp:
                status, body = resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"remote fit {call_id} failed ({exc.code}): {_error_text(exc)}"
            ) from None
        if status == 200:
            with np.load(io.BytesIO(body), allow_pickle=False) as npz:
                return {key: npz[key] for key in npz.files}
        if time.monotonic() > deadline:
            raise TimeoutError(f"remote fit {call_id} still running after {timeout} s")
        time.sleep(poll_interval)
