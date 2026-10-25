"""Exact block decomposition of the Gaussian total correlation.

With the variables split into blocks (market segments, sectors, countries),
the total correlation ``-1/2 log det R`` splits exactly into

* the coupling *inside* each block, ``-1/2 log det R_kk``, and
* the multi-information *between* blocks,
  ``I(B_1; ...; B_K) = -1/2 log det R + 1/2 sum_k log det R_kk >= 0``.

The between-block term is the systemic part: it is zero when the segments move
independently of one another, whatever happens inside each of them. For any
pair of blocks the mutual information
``I(A; B) = 1/2 (log det R_AA + log det R_BB - log det R_{A+B})`` gives a
contagion map in the same unit (nats). Everything is invariant to rescaling any
variable, so returns, rate changes and FX moves can sit in one matrix.

This is the construction proposed for a euro-area "entropic CISS"; the five
CISS segments (money market, bonds, equities, financial intermediaries, FX)
play the role of the blocks.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

from .coupling import ewma_devolatilize, normal_scores

__all__ = ['block_decomposition', 'rolling_block_coupling', 'block_statistics',
           'FF49_BLOCKS', 'SP500_BLOCKS']

# Approximation of the Fama-French five-industry grouping on the 48 FF49
# portfolios, with finance kept as its own block because it is the segment a
# macroprudential reader cares about.
FF49_BLOCKS = {
    'consumer': ['Food', 'Soda', 'Beer', 'Smoke', 'Toys', 'Fun', 'Books', 'Hshld',
                 'Clths', 'Txtls', 'Autos', 'Whlsl', 'Rtail', 'Meals', 'PerSv'],
    'manufacturing': ['Agric', 'Chems', 'Rubbr', 'BldMt', 'Cnstr', 'Steel', 'FabPr',
                      'Mach', 'ElcEq', 'Aero', 'Ships', 'Guns', 'Gold', 'Mines',
                      'Coal', 'Oil', 'Util', 'Paper', 'Boxes', 'Trans'],
    'hitec': ['Telcm', 'BusSv', 'Hardw', 'Softw', 'Chips', 'LabEq'],
    'health': ['Hlth', 'MedEq', 'Drugs'],
    'finance': ['Banks', 'Insur', 'RlEst', 'Fin'],
}

# The twenty S&P 500 stocks by broad sector; GE sits with energy and materials
# so that no block is a single stock.
SP500_BLOCKS = {
    'tech': ['AAPL', 'AMD', 'MSFT'],
    'finance': ['BAC', 'JPM'],
    'energy_industry': ['CVX', 'XOM', 'RRC', 'GE'],
    'health': ['JNJ', 'LLY', 'MRK', 'PFE', 'UNH'],
    'consumer': ['BBY', 'HD', 'KO', 'PEP', 'PG', 'WMT'],
}


def _half_logdet(M: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        raise ValueError('Matrix must be positive definite.')
    return 0.5 * float(logdet)


def block_decomposition(R: np.ndarray, blocks: dict[str, list[int]]) -> dict:
    """Total, within-block, between-block and pairwise coupling, in nats.

    ``R`` is a correlation matrix, ``blocks`` maps a block name to the column
    positions it holds; the blocks must partition the columns.
    """
    idx = sorted(i for v in blocks.values() for i in v)
    if idx != list(range(R.shape[0])):
        raise ValueError('Blocks must partition the columns exactly once.')
    half = {k: _half_logdet(R[np.ix_(v, v)]) for k, v in blocks.items()}
    total = -_half_logdet(R)
    out = {'total': total}
    for k in blocks:
        out[f'intra_{k}'] = -half[k]
    out['inter'] = total + sum(half.values())
    for a, b in combinations(blocks, 2):
        ab = blocks[a] + blocks[b]
        out[f'mi_{a}__{b}'] = half[a] + half[b] - _half_logdet(R[np.ix_(ab, ab)])
    return out


def _corr(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def rolling_block_coupling(logret: pd.DataFrame, blocks: dict[str, list[str]],
                           window: int = 252, version: str = 'rank') -> pd.DataFrame:
    """Rolling block decomposition on the same window as the main pipeline.

    ``version`` is ``'rank'`` (normal scores, the strongest variant in phase A),
    ``'raw'`` (returns, as the paper's ``h_dep``) or ``'devol'`` (returns
    divided by their past-only EWMA volatility). Ledoit-Wolf shrinkage in all
    three cases.
    """
    cols = list(logret.columns)
    pos = {k: [cols.index(c) for c in v] for k, v in blocks.items()}
    X = logret.to_numpy()
    if version == 'devol':
        X = ewma_devolatilize(X)
    elif version not in ('rank', 'raw'):
        raise ValueError(f'Unknown version: {version}')
    rows, dates = [], []
    for end in range(window, len(X) + 1):
        Xw = X[end - window:end]
        if not np.isfinite(Xw).all():
            continue
        if version == 'rank':
            Xw = normal_scores(Xw)
        rows.append(block_decomposition(_corr(ledoit_wolf(Xw)[0]), pos))
        dates.append(logret.index[end - 1])
    return pd.DataFrame(rows, index=pd.Index(dates, name='date'))


def block_statistics(dec: pd.DataFrame, logret: pd.DataFrame,
                     stress_quantile: float = 0.90) -> dict:
    """Stress-minus-calm gap of every component, labelled as in the paper
    (21-day volatility of the equal-weight market, past-only threshold)."""
    from .regimes import past_only_stress

    signal = logret.mean(axis=1).rolling(21).std().reindex(dec.index)
    stress, valid = past_only_stress(signal, stress_quantile)
    calm = ~stress & valid
    out = {}
    for col in dec.columns:
        out[f'BLK_{col}_gap'] = float(dec.loc[stress, col].mean() - dec.loc[calm, col].mean())
    # Share of the total coupling that lies between blocks, and its gap.
    share = dec['inter'] / dec['total']
    out['BLK_inter_share_gap'] = float(share[stress].mean() - share[calm].mean())
    return out
