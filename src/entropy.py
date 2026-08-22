from __future__ import annotations
import numpy as np
from sklearn.covariance import LedoitWolf


def maxent_entropy_from_cov(cov: np.ndarray, dt: float = 1.0) -> float:
    """Gaussian maximum-entropy value h_max = 1/2 log((2*pi*e)^N det(cov*dt)).

    Raises ValueError if cov*dt is not positive definite, consistent with
    covariance_decomposition below (both quantities are undefined otherwise;
    silently returning NaN would let a degenerate covariance estimate pass
    unnoticed into downstream aggregation).
    """
    n = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov * dt)
    if sign <= 0:
        raise ValueError('Covariance matrix must be positive definite.')
    return 0.5 * (n * np.log(2 * np.pi * np.e) + logdet)


def covariance_decomposition(cov: np.ndarray):
    """Split the Gaussian MaxEnt entropy into volatility and dependence terms.

    Given Sigma = D R D with D = diag(sigma_i), returns h_vol = sum log(sigma_i)
    and h_dep = 1/2 log det(R), which satisfy h_cov = h_vol + h_dep with
    h_cov = 1/2 log det(Sigma) (up to the (N/2) log(2 pi e) additive constant
    carried separately by maxent_entropy_from_cov).
    """
    vol = np.sqrt(np.clip(np.diag(cov), 0, None))
    Dinv = np.diag(1.0 / vol)
    R = Dinv @ cov @ Dinv
    sign_r, logdet_r = np.linalg.slogdet(R)
    sign_s, logdet_s = np.linalg.slogdet(cov)
    if sign_r <= 0 or sign_s <= 0:
        raise ValueError('Covariance/correlation matrix must be positive definite.')
    h_vol = float(np.log(vol).sum())
    h_dep = float(0.5 * logdet_r)
    h_cov = float(0.5 * logdet_s)
    return {
        'h_vol': h_vol,
        'h_dep': h_dep,
        'h_cov': h_cov,
        'logdet_cov': float(logdet_s),
        'logdet_R': float(logdet_r),
        'vol_mean': float(vol.mean()),
        'vol_geomean': float(np.exp(np.log(vol).mean())),
    }


def estimate_covariance(X: np.ndarray, method: str = 'ledoit_wolf') -> np.ndarray:
    if method == 'ledoit_wolf':
        return LedoitWolf().fit(X).covariance_
    if method == 'sample':
        return np.cov(X, rowvar=False, ddof=1)
    raise ValueError(f'Unknown covariance estimator: {method}')
