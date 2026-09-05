"""Correlation-matrix utilities for the second IFM stage."""

from __future__ import annotations

import numpy as np

__all__ = ["nearest_correlation", "is_positive_definite"]


def is_positive_definite(mat, tol=0.0):
    try:
        np.linalg.cholesky(mat - tol * np.eye(mat.shape[0]))
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_correlation(mat, eps=1e-8, max_iter=100):
    """Project a symmetric matrix onto the nearest correlation matrix.

    Pairwise IFM estimates each off-diagonal element separately, so the
    assembled matrix is symmetric but not guaranteed positive definite. This is
    Higham's alternating-projections algorithm, with a final eigenvalue floor so
    the result is strictly PD and safe to Cholesky-factor.
    """
    mat = np.asarray(mat, dtype=float)
    x = (mat + mat.T) / 2.0
    np.fill_diagonal(x, 1.0)
    if is_positive_definite(x, tol=eps):
        return x

    dykstra = np.zeros_like(x)
    y = x.copy()
    for _ in range(max_iter):
        r = y - dykstra
        # Projection onto the positive semi-definite cone.
        vals, vecs = np.linalg.eigh((r + r.T) / 2.0)
        s = (vecs * np.maximum(vals, eps)) @ vecs.T
        dykstra = s - r
        # Projection onto the unit-diagonal set.
        y = s.copy()
        np.fill_diagonal(y, 1.0)
        if is_positive_definite(y, tol=eps):
            break

    vals, vecs = np.linalg.eigh((y + y.T) / 2.0)
    y = (vecs * np.maximum(vals, eps)) @ vecs.T
    scale = np.sqrt(np.diag(y))
    y = y / np.outer(scale, scale)
    np.fill_diagonal(y, 1.0)
    return (y + y.T) / 2.0
