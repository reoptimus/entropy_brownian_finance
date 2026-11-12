"""Ledoit-Wolf shrinkage covariance, a lean copy of scikit-learn's estimator.

``sklearn.covariance.ledoit_wolf`` spends most of its time validating its
input; the rolling estimators call it tens of thousands of times per run.
This function computes exactly the same quantity (same formulas as
``sklearn.covariance._shrunk_covariance.ledoit_wolf_shrinkage`` with
``assume_centered=False``) without the overhead. ``tests/test_lw.py`` checks
the two agree to machine precision.
"""
from __future__ import annotations

import numpy as np


def ledoit_wolf_cov(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    X = X - X.mean(axis=0)
    emp = X.T @ X / n
    if p == 1:
        return emp
    X2 = X * X
    tr = X2.sum(axis=0) / n
    mu = tr.sum() / p
    beta_ = float((X2.T @ X2).sum())
    delta_ = float((emp * emp).sum())
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * tr.sum() + p * mu * mu) / p
    beta = min(beta, delta)
    s = 0.0 if beta == 0 else beta / delta
    out = (1.0 - s) * emp
    out.flat[::p + 1] += s * mu
    return out
