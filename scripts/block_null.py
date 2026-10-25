"""Calibrated nulls for the block decomposition of the coupling (phase B).

Same resamples as ``scripts/coupling_null.py`` and the published 200-replication
run (``SeedSequence([11, panel, block, rep])``), so the ``total`` gap of the
rank version here equals the ``h_dep_rank`` gap there, replication by
replication. The question this adds: when the cross-section couples under
stress, is it the coupling *between* blocks (the systemic part) that rises
beyond the nulls, or only the coupling *inside* each block?

Usage::

    python -m scripts.block_null --jobs 4
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

from scripts.coupling_null import NULLS, PANELS, SEED, _load, _resample_index
from src.blocks import FF49_BLOCKS, SP500_BLOCKS, block_statistics, rolling_block_coupling

BLOCKS = {'ff49': FF49_BLOCKS, 'sp500': SP500_BLOCKS}


def run_task(panel: str, null: str, rep: int, version: str) -> dict:
    t0 = time.time()
    logret, window, sq = _load(panel)
    if null == 'observed':
        data = logret
    else:
        ss = np.random.SeedSequence([SEED, list(PANELS).index(panel), NULLS[null], rep])
        idx = _resample_index(len(logret), NULLS[null], np.random.default_rng(ss))
        data = pd.DataFrame(logret.to_numpy()[idx], index=logret.index,
                            columns=logret.columns)
    dec = rolling_block_coupling(data, BLOCKS[panel], window, version)
    return {'panel': panel, 'null': null, 'rep': rep, 'version': version,
            'seconds': round(time.time() - t0, 1),
            'stats': block_statistics(dec, data, sq)}


def summarise(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (panel, version), g in long.groupby(['panel', 'version']):
        obs = g[g.null == 'observed'].set_index('statistic')['value']
        for null in NULLS:
            sub = g[g.null == null]
            if sub.empty:
                continue
            for stat, v in obs.items():
                col = sub.loc[sub.statistic == stat, 'value'].dropna()
                sd = col.std()
                rows.append({'panel': panel, 'version': version, 'null': null,
                             'statistic': stat, 'observed': v, 'null_mean': col.mean(),
                             'null_q05': col.quantile(0.05), 'null_q95': col.quantile(0.95),
                             'z_vs_null': (v - col.mean()) / sd if sd > 0 else np.nan,
                             'p_greater': float((col >= v).mean()),
                             'p_less': float((col <= v).mean()), 'n_rep': len(col)})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--n-rep', type=int, default=200)
    p.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--panels', default='ff49,sp500')
    p.add_argument('--version', default='rank', choices=['rank', 'raw', 'devol'])
    p.add_argument('--out-dir', default='outputs/block_null')
    args = p.parse_args()

    panels = [x.strip() for x in args.panels.split(',') if x.strip()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / 'checkpoint.jsonl'
    done = set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r['panel'], r['null'], r['rep'], r['version']))
    tasks = [(pa, 'observed', -1, args.version) for pa in panels]
    tasks += [(pa, nu, r, args.version) for pa in panels for nu in NULLS
              for r in range(args.n_rep)]
    todo = [t for t in tasks if t not in done]
    todo.sort(key=lambda t: (t[0] != 'ff49', t[1] != 'observed'))
    print(f'{len(tasks)} tasks, {len(todo)} to do, {args.jobs} processes', flush=True)

    t_start = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex, open(ckpt, 'a') as fh:
            futs = [ex.submit(run_task, *t) for t in todo]
            for n, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                fh.write(json.dumps(r) + '\n')
                fh.flush()
                el = time.time() - t_start
                print(f'  [{n}/{len(todo)}] {r["panel"]} {r["null"]} rep {r["rep"]} '
                      f'({r["seconds"]:.0f} s) ~{el / n * (len(todo) - n) / 60:.0f} min left',
                      flush=True)

    recs = [json.loads(l) for l in ckpt.read_text().splitlines() if l.strip()]
    long = pd.DataFrame([{'panel': r['panel'], 'null': r['null'], 'rep': r['rep'],
                          'version': r['version'], 'statistic': k, 'value': v}
                         for r in recs for k, v in r['stats'].items()])
    long = long.drop_duplicates(['panel', 'null', 'rep', 'version', 'statistic'], keep='last')
    summary = summarise(long)
    dest = Path('paper/tables')
    long.to_csv(dest / 'block_null_200rep_replications_long.csv', index=False,
                float_format='%.6g')
    summary.to_csv(dest / 'block_null_200rep_summary.csv', index=False, float_format='%.4f')
    pd.set_option('display.width', 180)
    keep = summary.statistic.isin(['BLK_total_gap', 'BLK_inter_gap', 'BLK_inter_share_gap'])
    print(summary[keep].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
