"""Cross-sectional coupling measures for the phase A benchmark.

The dependence channel ``h_dep = 1/2 log det R`` is the only regime statistic
that clears both calibrated nulls on both panels. It belongs to a known family
of correlation-concentration indicators, so before building anything on it the
paper must answer three questions a central-bank reader will ask first:

1. Does it do better than the standard member of that family, the absorption
   ratio of Kritzman, Li, Page and Rigobon (2011), or the first-eigenvalue share?
2. Does it survive the Forbes-Rigobon (2002) objection, i.e. is the coupling
   still there once each return is divided by its own conditional volatility?
3. Is it driven by a few extreme days? The rank (normal-scores) version is
   invariant to the marginal distributions and bounded in the tails.

All measures are computed on the same rolling window and with the same
Ledoit-Wolf shrinkage as the baseline ``h_dep``, so the comparison isolates the
choice of statistic rather than the choice of estimator.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.covariance import ledoit_wolf

__all__ = [
    'ewma_devolatilize',
    'absorption_ratio',
    'normal_scores',
    'coupling_from_window',
    'COUPLING_COLUMNS',
]

# Column name -> sign of the stress-minus-calm gap if the market couples under
# stress (h_dep falls, the absorption ratio and first-eigenvalue share rise).
COUPLING_COLUMNS = {
    'h_dep': -1,
    'h_dep_devol': -1,
    'h_dep_rank': -1,
    'absorption_ratio': +1,
    'eig1_share': +1,
}


def ewma_devolatilize(X: np.ndarray, lam: float = 0.94, init: int = 21,
                      clip: float = 8.0) -> np.ndarray:
    """Divide each return by its RiskMetrics conditional volatility.

    ``sigma2_t = lam * sigma2_{t-1} + (1 - lam) * r_{t-1}^2`` uses only returns
    strictly before ``t``, so no future information enters. The first ``init``
    observations seed the recursion and are returned as NaN.

    Standardised returns are clipped at ``+/- clip``. Without it a stale price
    series (RRC trades on about one day in ten in early 1990) has a conditional
    volatility near zero, the next move becomes a 200-sigma outlier, and the
    Ledoit-Wolf intensity goes to one, which reads as R = I.
    """
    X = np.asarray(X, dtype=float)
    T, N = X.shape
    out = np.full_like(X, np.nan)
    if T <= init:
        return out
    s2 = X[:init].var(axis=0)
    s2 = np.where(s2 > 0, s2, np.nanmean(s2[s2 > 0]) if np.any(s2 > 0) else 1.0)
    for t in range(init, T):
        s2 = lam * s2 + (1.0 - lam) * X[t - 1] ** 2
        out[t] = X[t] / np.sqrt(np.maximum(s2, 1e-12))
    return np.clip(out, -clip, clip)


def absorption_ratio(cov: np.ndarray, k: int | None = None) -> float:
    """Share of total variance absorbed by the ``k`` largest eigenvectors.

    Kritzman et al. (2011) use ``k = N/5``; that is the default here.
    """
    n = cov.shape[0]
    k = max(1, int(np.ceil(n / 5))) if k is None else int(k)
    vals = np.linalg.eigvalsh(cov)[::-1]
    return float(vals[:k].sum() / vals.sum())


def normal_scores(X: np.ndarray) -> np.ndarray:
    """Column-wise van der Waerden scores ``Phi^{-1}(rank / (n + 1))``."""
    n = X.shape[0]
    ranks = X.argsort(axis=0).argsort(axis=0)
    table = norm.ppf(np.arange(1, n + 1) / (n + 1.0))
    return table[ranks]


def _half_logdet_corr(cov: np.ndarray) -> float:
    d = np.sqrt(np.diag(cov))
    R = cov / np.outer(d, d)
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        raise ValueError('Correlation matrix must be positive definite.')
    return float(0.5 * logdet)


def coupling_from_window(cov_base: np.ndarray, Xw: np.ndarray,
                         Zw: np.ndarray | None) -> dict:
    """Extra coupling measures on one window.

    ``cov_base`` is the baseline Ledoit-Wolf covariance already computed by the
    pipeline, ``Xw`` the raw window, ``Zw`` the devolatilised window (or None
    when the devolatilised series is not yet available at the window's start).
    """
    d = np.sqrt(np.diag(cov_base))
    R = cov_base / np.outer(d, d)
    rec = {
        'absorption_ratio': absorption_ratio(cov_base),
        'eig1_share': float(np.linalg.eigvalsh(R)[-1] / R.shape[0]),
        'h_dep_rank': _half_logdet_corr(ledoit_wolf(normal_scores(Xw))[0]),
    }
    if Zw is not None and np.isfinite(Zw).all():
        rec['h_dep_devol'] = _half_logdet_corr(ledoit_wolf(Zw)[0])
    else:
        rec['h_dep_devol'] = np.nan
    return rec


def rolling_coupling(logret, window: int = 252):
    """Only the coupling measures, on the same rolling window as the pipeline.

    Gives exactly the ``h_dep`` of :func:`src.empirical.rolling_measures`
    (Ledoit-Wolf on the raw window) plus the four benchmark columns, at a
    fraction of the cost, because the shape channel and the flow measures are
    not needed to test regime gaps in coupling.
    """
    import pandas as pd

    X = logret.to_numpy()
    T = X.shape[0]
    Z = ewma_devolatilize(X)
    rows, dates = [], []
    for end in range(window, T + 1):
        Xw = X[end - window:end]
        cov = ledoit_wolf(Xw)[0]
        rec = {'h_dep': _half_logdet_corr(cov)}
        rec.update(coupling_from_window(cov, Xw, Z[end - window:end]))
        rows.append(rec)
        dates.append(logret.index[end - 1])
    return pd.DataFrame(rows, index=pd.Index(dates, name='date'))


def coupling_statistics(cpl, logret, stress_quantile: float = 0.90) -> dict:
    """Stress-minus-calm gap of every coupling measure, raw and standardised.

    Regimes are labelled exactly as in ``scripts/placebo_null.py``: rolling
    21-day volatility of the equal-weight market, past-only expanding quantile,
    burn-in dates in neither regime. The standardised gap divides by the
    measure's standard deviation over labelled dates, so measures in nats and
    in variance shares can be compared.
    """
    from .regimes import past_only_stress

    signal = logret.mean(axis=1).rolling(21).std().reindex(cpl.index)
    stress, valid = past_only_stress(signal, stress_quantile)
    calm = ~stress & valid
    out = {}
    for col in COUPLING_COLUMNS:
        x = cpl[col]
        gap = float(x[stress].mean() - x[calm].mean())
        sd = float(x[stress | calm].std())
        out[f'CPL_{col}_gap'] = gap
        out[f'CPL_{col}_gap_std'] = gap / sd if sd > 0 else float('nan')
    return out
