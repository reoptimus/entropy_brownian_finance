"""Plausibility check of the stress ladder against realised history.

Compares the vol multiplier of the k-year scenario with the k-year quantile of
the realised ratio (next-21-day vol / trailing-252-day vol) of the equal-weight
portfolio. Writes outputs/<panel>/tables/ladder_plausibility.csv.

    python -m scripts.ladder_plausibility --config config/sp500.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.empirical import DATA_READERS


def run(config: str) -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config).read_text())
    r = DATA_READERS[cfg['data']['type']](cfg)
    r = r.loc[str(cfg['sample']['start']):str(cfg['sample']['end'])]
    p = r.mean(axis=1)
    fwd = p[::-1].rolling(21).std()[::-1]
    ratio = (fwd / p.rolling(252).std()).dropna().iloc[::21]
    out_t = Path(cfg['outputs']['tables'])
    ladder = pd.read_csv(out_t / 'stress_scenarios.csv')
    ladder = ladder[ladder['return_period_years'] > 0]
    rows = []
    for _, s in ladder.iterrows():
        k = float(s['return_period_years'])
        q = 1.0 - 1.0 / (k * 252 / 21)
        rows.append({'return_period_years': k,
                     'scenario_vol_multiplier': float(s['vol_multiplier']),
                     'realised_vol_ratio_quantile': float(ratio.quantile(q)),
                     'extrapolated': bool(s['extrapolated'])})
    out = pd.DataFrame(rows)
    out.to_csv(out_t / 'ladder_plausibility.csv', index=False)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config/sp500.yaml')
    print(run(ap.parse_args().config).round(2).to_string(index=False))
