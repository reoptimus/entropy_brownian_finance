"""Gaussian maximum-entropy quantities and their decomposition.

The module keeps the two original primitives (``maxent_entropy_from_cov`` and
``covariance_decomposition``) and adds the covariance estimators and spectral
diagnostics used by the jump/diffusion analysis:

* :func:`estimate_covariance` -- sample, Ledoit--Wolf or EWMA covariance.
* :func:`ewma_weights` -- exponential weights and their effective sample size.
* :func:`effective_dimension` -- the participation number implied by the
  correlation spectrum, i.e. the "effective number of independent risk modes".
* :func:`gaussian_multi_information` -- the Gaussian total correlation
  :math:`-\\tfrac12\\log\\det R \\ge 0`.

All entropies are in nats.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

__all__ = [
    'maxent_entropy_from_cov',
    'covariance_decomposition',
    'estimate_covariance',
    'ewma_weights',
    'effective_dimension',
    'gaussian_multi_information',
    'inverse_sqrt',
    'gaussian_kl',
]


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


def covariance_decomposition(cov: np.ndarray) -> dict:
    """Split the Gaussian MaxEnt entropy into volatility and dependence terms.

    Given Sigma = D R D with D = diag(sigma_i), returns h_vol = sum log(sigma_i)
    and h_dep = 1/2 log det(R), which satisfy h_cov = h_vol + h_dep with
    h_cov = 1/2 log det(Sigma) (up to the (N/2) log(2 pi e) additive constant
    carried separately by maxent_entropy_from_cov).

    ``h_dep <= 0`` always (Hadamard's inequality), with equality iff R = I;
    ``-h_dep`` is the Gaussian multi-information (total correlation) and is the
    dependence channel of the structural index used throughout the paper.
    """
    vol = np.sqrt(np.clip(np.diag(cov), 0, None))
    if not np.all(vol > 0):
        raise ValueError('Covariance matrix has a non-positive diagonal entry.')
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
        'n_eff_modes': effective_dimension(R),
        'mean_corr': float((R.sum() - R.shape[0]) / (R.shape[0] * (R.shape[0] - 1))),
    }


def ewma_weights(n_obs: int, halflife: float) -> np.ndarray:
    """Exponential weights over ``n_obs`` observations, most recent last.

    Returns weights summing to one. ``halflife`` is expressed in observations.
    """
    if halflife <= 0:
        raise ValueError('halflife must be strictly positive.')
    lam = 0.5 ** (1.0 / halflife)
    age = np.arange(n_obs - 1, -1, -1, dtype=float)
    w = lam ** age
    return w / w.sum()


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size of a weight vector."""
    w = np.asarray(weights, dtype=float)
    return float(w.sum() ** 2 / np.sum(w ** 2))


def _ewma_covariance(X: np.ndarray, halflife: float, shrinkage: float) -> np.ndarray:
    w = ewma_weights(X.shape[0], halflife)
    mu = w @ X
    Xc = X - mu
    cov = (Xc * w[:, None]).T @ Xc
    # Weighted covariance is biased downward; rescale by the Kish correction.
    cov = cov / (1.0 - np.sum(w ** 2))
    if shrinkage > 0:
        target = np.diag(np.diag(cov))
        cov = (1.0 - shrinkage) * cov + shrinkage * target
    return cov


def estimate_covariance(
    X: np.ndarray,
    method: str = 'ledoit_wolf',
    halflife: float = 60.0,
    shrinkage: float = 0.10,
) -> np.ndarray:
    """Covariance estimate for a (T x N) window of returns.

    ``ledoit_wolf`` (default) reproduces the original baseline. ``ewma`` applies
    exponential weights with the given halflife (in observations) plus a linear
    shrinkage toward the diagonal, and is the estimator used for the
    jump/diffusion analysis: a box-car window makes an extreme observation
    *leaving* the window look like an event, which contaminates any jump test on
    the resulting entropy series.
    """
    if method == 'ledoit_wolf':
        return LedoitWolf().fit(X).covariance_
    if method == 'sample':
        return np.cov(X, rowvar=False, ddof=1)
    if method == 'ewma':
        return _ewma_covariance(X, halflife, shrinkage)
    raise ValueError(f'Unknown covariance estimator: {method}')


def inverse_sqrt(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Symmetric inverse square root of a positive-definite matrix."""
    vals, vecs = np.linalg.eigh(cov)
    if vals.min() <= eps:
        raise ValueError('Matrix is not positive definite; cannot whiten.')
    return vecs @ np.diag(vals ** -0.5) @ vecs.T


def effective_dimension(R: np.ndarray) -> float:
    """Participation number exp(H) of the normalised correlation spectrum.

    With eigenvalues ``lambda_k`` of an N x N correlation matrix (which sum to
    N), set ``p_k = lambda_k / N`` and return ``exp(-sum p_k log p_k)``. The
    value equals N for R = I and tends to 1 when a single collective mode
    absorbs all the variance, so it reads directly as the effective number of
    independent risk drivers.
    """
    vals = np.linalg.eigvalsh(R)
    vals = np.clip(vals, 1e-15, None)
    p = vals / vals.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def gaussian_multi_information(R: np.ndarray) -> float:
    """Gaussian total correlation ``-1/2 log det R`` (nats, non-negative)."""
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        raise ValueError('Correlation matrix must be positive definite.')
    return float(-0.5 * logdet)


def gaussian_kl(mu1, cov1, mu0, cov0) -> float:
    """KL(N(mu1, cov1) || N(mu0, cov0)) in nats."""
    mu1 = np.asarray(mu1, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)
    n = cov0.shape[0]
    inv0 = np.linalg.inv(cov0)
    s1, ld1 = np.linalg.slogdet(cov1)
    s0, ld0 = np.linalg.slogdet(cov0)
    if s1 <= 0 or s0 <= 0:
        raise ValueError('Both covariance matrices must be positive definite.')
    d = mu0 - mu1
    return float(0.5 * (np.trace(inv0 @ cov1) - n + ld0 - ld1 + d @ inv0 @ d))
