"""Transfer-entropy helpers (Gaussian analytic + kNN/KSG)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import BallTree


_EPS = 1e-10


def conditional_variance(y: np.ndarray, X: np.ndarray, epsilon: float = 1e-6) -> float:
    """Return variance of residuals for linear regression y ~ X."""

    y = np.asarray(y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.shape[0] < 2:
        return epsilon
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    coeffs, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    residuals = y - X_aug @ coeffs
    var = float(np.mean(residuals**2))
    return max(var, epsilon)


def gaussian_transfer_entropy(
    x_past: np.ndarray,
    y_past: np.ndarray,
    y_present: np.ndarray,
    epsilon: float = 1e-6,
    base_var: Optional[float] = None,
) -> float:
    """Compute Gaussian TE using conditional variances."""

    y_present = np.asarray(y_present, dtype=float)
    y_past = np.asarray(y_past, dtype=float)
    x_past = np.asarray(x_past, dtype=float)
    if y_present.size < 2:
        return 0.0
    if base_var is None:
        base_var = conditional_variance(y_present, y_past.reshape(-1, 1), epsilon)
    var_joint = conditional_variance(
        y_present,
        np.column_stack([y_past, x_past]),
        epsilon,
    )
    te = 0.5 * np.log((base_var + epsilon) / (var_joint + epsilon))
    return float(max(0.0, te))


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _standardize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    mean = np.mean(arr, axis=0, keepdims=True)
    std = np.std(arr, axis=0, keepdims=True)
    std[std < _EPS] = 1.0
    return (arr - mean) / std


def _ksg_mutual_information(X: np.ndarray, Y: np.ndarray, k: int = 5) -> float:
    """Estimate the mutual information I(X; Y) via the Kraskov kNN method."""

    X = _ensure_2d(X)
    Y = _ensure_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError("X and Y must share the same number of samples")
    if n <= k:
        raise ValueError("Number of samples must exceed k")

    XY = np.hstack([X, Y])
    tree_xy = BallTree(XY, metric="chebyshev")
    distances, _ = tree_xy.query(XY, k=k + 1)
    eps = distances[:, k]
    eps = np.maximum(eps - _EPS, 0.0)

    tree_x = BallTree(X, metric="chebyshev")
    tree_y = BallTree(Y, metric="chebyshev")

    nx = tree_x.query_radius(X, eps, count_only=True) - 1
    ny = tree_y.query_radius(Y, eps, count_only=True) - 1

    nx = np.maximum(nx, 0)
    ny = np.maximum(ny, 0)

    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return float(max(0.0, mi))


def knn_transfer_entropy(
    x_past: np.ndarray,
    y_past: np.ndarray,
    y_present: np.ndarray,
    *,
    k: int = 5,
    standardize: bool = True,
) -> float:
    """Estimate TE(x → y) via a kNN/KSG estimator."""

    x_past = _ensure_2d(x_past)
    y_past = _ensure_2d(y_past)
    y_present = _ensure_2d(y_present)

    if len(x_past) != len(y_past) or len(y_past) != len(y_present):
        raise ValueError("Input arrays must share the same number of samples")
    if len(x_past) <= k:
        raise ValueError("Sample size must exceed k")

    if standardize:
        x_past = _standardize(x_past)
        y_past = _standardize(y_past)
        y_present = _standardize(y_present)

    combined = np.hstack([y_past, x_past])
    mi_joint = _ksg_mutual_information(combined, y_present, k=k)
    mi_cond = _ksg_mutual_information(y_past, y_present, k=k)
    te = max(0.0, mi_joint - mi_cond)
    return float(te)
