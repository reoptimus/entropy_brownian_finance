"""Phase B: the entropic CISS on the euro-area multi-market panel.

Steps, all from the files written by ``telechargement_zone_euro/telecharger.py``
(unzipped into ``data/zone_euro/``):

1. build the panel of daily moves in the five CISS segments
   (``src/euro_panel.py``) and the rolling block decomposition of its coupling
   (``src/blocks.py``, rank version, 252-day window);
2. the two calibrated nulls (i.i.d. and 21-day blocks) on the stress-minus-calm
   gap of every component, stress being the top decile of the 21-day volatility
   of the equal-weight equity block, past-only threshold;
3. comparison with the ECB CISS: correlations, encompassing regression of the
   CISS on the coupling components, AUROC against CEPR euro-area recessions;
4. Growth-at-Risk: quantile regressions of 4-quarter-ahead real GDP growth on
   the quarterly average of each indicator.

Usage::

    python -m scripts.euro_ciss --data data/zone_euro --n-rep 200 --jobs 4
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score

from scripts.coupling_null import _resample_index
from src.blocks import rolling_block_coupling
from src.euro_panel import build_euro_panel
from src.regimes import past_only_stress

SEED = 11
NULLS = {'iid': 1, 'block21': 21}
WINDOW = 252
# CEPR Euro Area Business Cycle Dating Committee, peak quarter -> trough quarter.
CEPR_RECESSIONS = [('2008-01-01', '2009-06-30'), ('2011-07-01', '2013-03-31'),
                   ('2019-10-01', '2020-06-30')]
_CACHE: dict = {}


def _panel(folder: str):
    if folder not in _CACHE:
        _CACHE[folder] = build_euro_panel(folder)
    return _CACHE[folder]


def stress_signal(M: pd.DataFrame, blocks: dict) -> pd.Series:
    eq = blocks.get('actions') or blocks.get('intermediaires')
    return M[eq].mean(axis=1).rolling(21).std()


def statistics(dec: pd.DataFrame, M: pd.DataFrame, blocks: dict) -> dict:
    sig = stress_signal(M, blocks).reindex(dec.index)
    stress, valid = past_only_stress(sig, 0.90)
    calm = ~stress & valid
    out = {f'EUR_{c}_gap': float(dec.loc[stress, c].mean() - dec.loc[calm, c].mean())
           for c in dec.columns}
    share = dec['inter'] / dec['total']
    out['EUR_inter_share_gap'] = float(share[stress].mean() - share[calm].mean())
    return out


def run_task(folder: str, null: str, rep: int) -> dict:
    t0 = time.time()
    M, blocks = _panel(folder)
    if null == 'observed':
        data = M
    else:
        ss = np.random.SeedSequence([SEED, 7, NULLS[null], rep])
        idx = _resample_index(len(M), NULLS[null], np.random.default_rng(ss))
        data = pd.DataFrame(M.to_numpy()[idx], index=M.index, columns=M.columns)
    dec = rolling_block_coupling(data, blocks, WINDOW, 'rank')
    return {'null': null, 'rep': rep, 'seconds': round(time.time() - t0, 1),
            'stats': statistics(dec, data, blocks)}


def _auroc(x: pd.Series, y: pd.Series) -> float:
    d = pd.concat([x, y], axis=1).dropna()
    return float(roc_auc_score(d.iloc[:, 1], d.iloc[:, 0])) if d.iloc[:, 1].nunique() == 2 else np.nan


def comparisons(folder: Path, dec: pd.DataFrame, M: pd.DataFrame, blocks: dict) -> dict:
    out: dict = {}
    ind = dec[['total', 'inter']].copy()
    ind['log_vol21'] = np.log(stress_signal(M, blocks)).reindex(ind.index)
    ref = folder / 'reference.csv'
    ciss = None
    if ref.exists():
        r = pd.read_csv(ref, index_col=0, parse_dates=True)
        if 'ciss_zone_euro' in r:
            ciss = r['ciss_zone_euro'].dropna()
    if ciss is not None:
        ind['ciss'] = ciss.reindex(ind.index).ffill(limit=3)
        out['corr_levels'] = ind.corr().round(3).to_dict()
        w = ind.resample('W-FRI').last().dropna()
        z = (w - w.mean()) / w.std()
        enc = []
        for cols in (['inter'], ['total'], ['log_vol21'], ['log_vol21', 'inter'],
                     ['log_vol21', 'total']):
            fit = sm.OLS(z['ciss'], sm.add_constant(z[cols])).fit(
                cov_type='HAC', cov_kwds={'maxlags': 4})
            enc.append({'regressors': ' + '.join(cols), 'n': int(fit.nobs), 'r2': fit.rsquared,
                        **{f't_{c}': fit.tvalues[c] for c in cols},
                        **{f'coef_{c}': fit.params[c] for c in cols}})
        out['ciss_encompassing'] = enc

    y = pd.Series(0.0, index=ind.index)
    for p, t in CEPR_RECESSIONS:
        y[(ind.index >= p) & (ind.index <= t)] = 1.0
    out['auroc_cepr_recession'] = {c: _auroc(ind[c], y) for c in ind.columns}

    if ref.exists() and 'pib_reel_zone_euro' in r:
        gdp = r['pib_reel_zone_euro'].dropna()
        gdp.index = gdp.index.to_period('Q')
        g4 = 100 * (np.log(gdp).shift(-4) - np.log(gdp))
        q = ind.groupby(ind.index.to_period('Q')).mean()
        d = pd.concat([g4.rename('g4'), q], axis=1).dropna()
        d = (d - d.mean()) / d.std().where(d.columns != 'g4', 1.0)
        gar = []
        for c in [c for c in q.columns if c in d]:
            for tau in (0.10, 0.50):
                fit = smf.quantreg(f'g4 ~ {c}', d).fit(q=tau)
                gar.append({'indicator': c, 'quantile': tau, 'n': int(fit.nobs),
                            'coef': fit.params[c], 'pvalue': fit.pvalues[c]})
        out['growth_at_risk'] = gar
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/zone_euro')
    p.add_argument('--n-rep', type=int, default=200)
    p.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--out-dir', default='outputs/euro_ciss')
    args = p.parse_args()

    folder = Path(args.data)
    M, blocks = _panel(str(folder))
    print(f'Panel: {M.shape[0]} days x {M.shape[1]} series, '
          f'{M.index.min().date()} -> {M.index.max().date()}')
    for k, v in blocks.items():
        print(f'  {k:15s} {len(v):2d}  {", ".join(v)}')

    dest = Path('paper/tables')
    dec = rolling_block_coupling(M, blocks, WINDOW, 'rank')
    dec.to_csv(dest / 'euro_block_coupling_series.csv', float_format='%.6g')
    comp = comparisons(folder, dec, M, blocks)
    (dest / 'euro_ciss_comparisons.json').write_text(json.dumps(comp, indent=2, default=float))
    print(json.dumps(comp, indent=1, default=float)[:4000])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / 'checkpoint.jsonl'
    done = set()
    if ckpt.exists():
        done = {(r['null'], r['rep']) for r in map(json.loads, ckpt.read_text().splitlines()) if r}
    tasks = [('observed', -1)] + [(nu, r) for nu in NULLS for r in range(args.n_rep)]
    todo = [t for t in tasks if t not in done]
    t_start = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex, open(ckpt, 'a') as fh:
            futs = [ex.submit(run_task, str(folder), *t) for t in todo]
            for n, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                fh.write(json.dumps(r) + '\n')
                fh.flush()
                el = time.time() - t_start
                print(f'  [{n}/{len(todo)}] {r["null"]} rep {r["rep"]} ({r["seconds"]:.0f} s) '
                      f'~{el / n * (len(todo) - n) / 60:.0f} min left', flush=True)

    recs = [json.loads(l) for l in ckpt.read_text().splitlines() if l.strip()]
    long = pd.DataFrame([{'null': r['null'], 'rep': r['rep'], 'statistic': k, 'value': v}
                         for r in recs for k, v in r['stats'].items()])
    long = long.drop_duplicates(['null', 'rep', 'statistic'], keep='last')
    obs = long[long.null == 'observed'].set_index('statistic')['value']
    rows = []
    for null in NULLS:
        sub = long[long.null == null]
        for stat, v in obs.items():
            col = sub.loc[sub.statistic == stat, 'value'].dropna()
            sd = col.std()
            rows.append({'null': null, 'statistic': stat, 'observed': v,
                         'null_mean': col.mean(), 'null_q05': col.quantile(0.05),
                         'null_q95': col.quantile(0.95),
                         'z_vs_null': (v - col.mean()) / sd if sd > 0 else np.nan,
                         'p_greater': float((col >= v).mean()),
                         'p_less': float((col <= v).mean()), 'n_rep': len(col)})
    summary = pd.DataFrame(rows)
    long.to_csv(dest / 'euro_null_200rep_replications_long.csv', index=False, float_format='%.6g')
    summary.to_csv(dest / 'euro_null_200rep_summary.csv', index=False, float_format='%.4f')
    pd.set_option('display.width', 180)
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
