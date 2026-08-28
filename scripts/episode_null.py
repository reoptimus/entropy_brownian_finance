"""Calibrated null for the per-episode channel-budget typology.

``episode_channel_budget`` (src/jumps.py) splits each detected entropy-jump
episode into a dependence/odd/even budget. On the S&P 500 panel that split is
roughly balanced (46% dependence-dominant, 54% even-dominant, 0% odd-dominant
episodes) and the single largest episode (COVID) is always even-dominant. The
open question this script answers: is that typology itself just what the
measurement device produces on resampled data with no real episode structure,
the same way jump asymmetry (H3) and channel coupling (H6) turned out to be?

Reuses the exact resampling machinery of ``scripts/placebo_null.py`` (i.i.d.
row resampling by default, or a block bootstrap via ``--block``) so the null
is calibrated the same way as every other statistic in the paper. For each
replication: run the full pipeline, detect jumps, group them into episodes,
budget each episode, and record the same typology statistics computed on the
real data.

Usage::

    python -m scripts.episode_null --config config/sp500.yaml --n-rep 20
    python -m scripts.episode_null --config config/sp500.yaml --n-rep 20 --block 21
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.empirical import DATA_READERS, rolling_measures
from src.jumps import detect_jumps, episode_channel_budget, group_jump_episodes


def typology_of(ent: pd.DataFrame, threshold: float, scale_window: int,
                gap: int, pre: int, post_search: int) -> dict:
    jumps = detect_jumps(ent['J'], threshold, scale_window)
    episodes = group_jump_episodes(jumps, gap=gap, direction='up')
    budget = episode_channel_budget(ent, episodes, pre=pre, post_search=post_search)
    if budget.empty:
        return {'n_episodes': 0}

    dom = budget['dominant_channel']
    top = budget.loc[budget['dJ'].idxmax()]
    # Weighted by episode size, so a handful of tiny episodes can't dominate
    # the corr the way an unweighted Pearson corr over the full table would.
    corr = budget[['dJ', 'share_dependence']].corr().iloc[0, 1] if len(budget) > 2 else np.nan

    return {
        'n_episodes': int(len(budget)),
        'share_dependence_dominant': float((dom == 'dependence').mean()),
        'share_even_dominant': float((dom == 'even').mean()),
        'share_odd_dominant': float((dom == 'odd').mean()),
        'mean_share_dependence': float(budget['share_dependence'].mean()),
        'sd_share_dependence': float(budget['share_dependence'].std()),
        'top_episode_share_dependence': float(top['share_dependence']),
        'top_episode_is_dependence_dominant': bool(top['dominant_channel'] == 'dependence'),
        'corr_dJ_share_dependence': float(corr),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/sp500.yaml')
    p.add_argument('--n-rep', type=int, default=20)
    p.add_argument('--block', type=int, default=1)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--gap', type=int, default=40)
    p.add_argument('--pre', type=int, default=10)
    p.add_argument('--post-search', type=int, default=20)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    logret = DATA_READERS[cfg['data']['type']](cfg)
    logret = logret.loc[str(cfg['sample']['start']):str(cfg['sample']['end'])]

    est = cfg['estimation']
    kw = dict(window=cfg['sample']['window'], method=est['covariance'],
              jump_method=est.get('jump_covariance', 'ewma'),
              halflife=float(est.get('ewma_halflife', 60)),
              shrinkage=float(est.get('ewma_shrinkage', 0.10)),
              shape_estimator=est.get('shape_estimator', 'hyvarinen'))
    jcfg = cfg.get('jumps', {})
    threshold = float(jcfg.get('threshold', 4.0))
    scale_window = int(jcfg.get('scale_window', 250))
    tkw = dict(threshold=threshold, scale_window=scale_window, gap=args.gap,
              pre=args.pre, post_search=args.post_search)

    observed = typology_of(rolling_measures(logret, **kw), **tkw)

    rng = np.random.default_rng(args.seed)
    X = logret.to_numpy()
    T = len(X)
    block = max(1, int(args.block))
    n_blocks = int(np.ceil(T / block))
    rows = []
    for r in range(args.n_rep):
        if block == 1:
            idx = rng.integers(0, T, size=T)
        else:
            starts = rng.integers(0, T, size=n_blocks)
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        boot = pd.DataFrame(X[idx], index=logret.index, columns=logret.columns)
        rows.append(typology_of(rolling_measures(boot, **kw), **tkw))
        print(f'  replication {r + 1}/{args.n_rep} done', flush=True)
    null = pd.DataFrame(rows)

    summary = []
    for k, v in observed.items():
        col = null[k].dropna()
        if len(col) == 0 or not np.isfinite(v if not isinstance(v, bool) else float(v)):
            summary.append({'statistic': k, 'observed': v})
            continue
        vv = float(v) if not isinstance(v, bool) else float(v)
        summary.append({
            'statistic': k,
            'observed': vv,
            'null_mean': float(col.astype(float).mean()),
            'null_sd': float(col.astype(float).std()),
            'null_q05': float(col.astype(float).quantile(0.05)),
            'null_q95': float(col.astype(float).quantile(0.95)),
            'p_greater': float((col.astype(float) >= vv).mean()),
            'p_less': float((col.astype(float) <= vv).mean()),
            'n_rep': int(len(col)),
        })
    out = pd.DataFrame(summary)
    print()
    print(out.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    suffix = '' if block == 1 else f'_block{block}'
    dest = Path(args.out or Path(cfg['outputs']['tables']) / f'episode_typology_null{suffix}.csv')
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    null.to_csv(dest.with_name(dest.stem + '_replications.csv'), index=False)
    print(f'\nWritten to {dest}')


if __name__ == '__main__':
    main()
