"""Sensitivity of the H1 dependence-channel gap to the EWMA shrinkage (B5).

    python -m scripts.shrinkage_sweep --config config/sp500.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.empirical import DATA_READERS, rolling_measures
from src.regimes import past_only_stress


def run(config: str, grid=(0.0, 0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config).read_text())
    logret = DATA_READERS[cfg['data']['type']](cfg)
    logret = logret.loc[str(cfg['sample']['start']):str(cfg['sample']['end'])]
    est = cfg['estimation']
    sig = logret.mean(axis=1).rolling(21).std() * 252 ** 0.5
    stress, valid = past_only_stress(sig, cfg['regimes']['stress_quantile'])
    rows = []
    for sh in grid:
        ent = rolling_measures(
            logret, window=cfg['sample']['window'], method=est['covariance'],
            jump_method=est.get('jump_covariance', 'ewma'),
            halflife=float(est.get('ewma_halflife', 60)), shrinkage=sh,
            shape_estimator=est.get('shape_estimator', 'hyvarinen'))
        s = stress.reindex(ent.index).fillna(False)
        v = valid.reindex(ent.index).fillna(False)
        gap = ent.loc[s & v, 'h_dep_ew'].mean() - ent.loc[~s & v, 'h_dep_ew'].mean()
        rows.append({'shrinkage': sh, 'h_dep_ew_stress_minus_calm': gap})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/sp500.yaml')
    out = run(ap.parse_args().config)
    print(out.to_string(index=False))
