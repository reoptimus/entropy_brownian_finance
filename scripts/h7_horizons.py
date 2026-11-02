"""H7 robustness: does entropy add to volatility at other forecast horizons?

The paper tests H7 at 21 trading days (one month). This script re-runs the
same walk-forward test at several horizons h: the forward targets (realised
volatility, drawdown, worst daily loss) are rebuilt over the next h days, the
out-of-sample evaluation dates are spaced h days apart so that targets do not
overlap, and the Diebold--Mariano HAC lag is max(h, 5).

Usage: python scripts/h7_horizons.py config/sp500.yaml [config/default.yaml ...]
Writes paper/tables/<label>_h7_horizons.csv and prints a summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.empirical import main, add_forward_realized_vol, add_forward_tail_risk  # noqa: E402
from src.hypotheses import run_all, h7_incremental_value  # noqa: E402
from src.regimes import past_only_stress  # noqa: E402

HORIZONS = (5, 10, 21, 63)


def run(config: str) -> pd.DataFrame:
    cfg = yaml.safe_load(Path(config).read_text())
    label = cfg['data'].get('type', 'dataset')
    logret, ent = main(cfg)
    ent['stress'], ent['regime_valid'] = past_only_stress(
        ent['stress_signal'], cfg['regimes']['stress_quantile'])
    res = run_all(ent, ent['stress'], cfg, valid=ent['regime_valid'])
    ent = ent.join(res['indicators'][['crisis', 'unhealed']])
    ann = cfg['risk']['realized_vol_annualization']
    rows = []
    for h in HORIZONS:
        e = ent.drop(columns=['future_rv', 'future_worst_loss', 'future_drawdown'])
        e = add_forward_realized_vol(logret, e, h, ann)
        e = add_forward_tail_risk(logret, e, h)
        t = h7_incremental_value(e, thin=h, hac_lags=max(h, 5))
        t.insert(0, 'horizon', h)
        t.insert(0, 'panel', label)
        rows.append(t)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(f'paper/tables/{label}_h7_horizons.csv', index=False)
    return out


if __name__ == '__main__':
    for c in sys.argv[1:]:
        out = run(c)
        x = out[out.extra_regressor != 'none']
        best = x.loc[x.groupby(['horizon', 'target'])['dm_pvalue'].idxmin()]
        with pd.option_context('display.width', 200):
            print(best[['panel', 'horizon', 'target', 'extra_regressor',
                        'oos_mse_vs_vol_only_pct', 'dm_tstat', 'dm_pvalue', 'n_oos']]
                  .round(3).to_string(index=False))
