"""Phase A: the two calibrated nulls for the coupling benchmark.

Runs only the coupling measures (``src.coupling.rolling_coupling``), which is
enough to test their stress-minus-calm gaps and several times cheaper than the
full pipeline. Resamples are drawn with exactly the seeds of
``calcul_local/lancer_calcul.py`` (``SeedSequence([11, panel, block, rep])``),
so replication ``r`` here is the same resampled return matrix as replication
``r`` of the published 200-replication run, and the ``h_dep`` gaps must match
``paper/tables/placebo_null_200rep_replications_long.csv`` exactly. The script
checks that.

Resumable: replications already in the checkpoint are not recomputed.

Usage::

    python -m scripts.coupling_null --jobs 4
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
import yaml

from src.coupling import COUPLING_COLUMNS, coupling_statistics, rolling_coupling
from src.empirical import DATA_READERS

PANELS = {'ff49': 'config/default.yaml', 'sp500': 'config/sp500.yaml'}
NULLS = {'iid': 1, 'block21': 21}
SEED = 11
SP500_CSV = Path('calcul_local/data/sp500_prices.csv')
PUBLISHED = Path('paper/tables/placebo_null_200rep_replications_long.csv')
_CACHE: dict = {}


def _load(panel: str):
    if panel not in _CACHE:
        cfg = yaml.safe_load(Path(PANELS[panel]).read_text())
        if panel == 'sp500':
            cfg['data']['raw_file'] = str(SP500_CSV)
        logret = DATA_READERS[cfg['data']['type']](cfg)
        logret = logret.loc[str(cfg['sample']['start']):str(cfg['sample']['end'])]
        sq = float(cfg.get('regimes', {}).get('stress_quantile', 0.90))
        _CACHE[panel] = (logret, int(cfg['sample']['window']), sq)
    return _CACHE[panel]


def _resample_index(T: int, block: int, rng) -> np.ndarray:
    if block == 1:
        return rng.integers(0, T, size=T)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T


def run_task(panel: str, null: str, rep: int) -> dict:
    t0 = time.time()
    logret, window, sq = _load(panel)
    if null == 'observed':
        data = logret
    else:
        ss = np.random.SeedSequence([SEED, list(PANELS).index(panel), NULLS[null], rep])
        idx = _resample_index(len(logret), NULLS[null], np.random.default_rng(ss))
        data = pd.DataFrame(logret.to_numpy()[idx], index=logret.index,
                            columns=logret.columns)
    stats = coupling_statistics(rolling_coupling(data, window), data, sq)
    return {'panel': panel, 'null': null, 'rep': rep,
            'seconds': round(time.time() - t0, 1), 'stats': stats}


def summarise(long: pd.DataFrame) -> pd.DataFrame:
    """Observed value, null distribution and one-sided p-value in the direction
    a coupling market predicts, plus a z-score so measures can be ranked."""
    rows = []
    for panel in long['panel'].unique():
        obs = long[(long.panel == panel) & (long.null == 'observed')].set_index('statistic')['value']
        for null in NULLS:
            sub = long[(long.panel == panel) & (long.null == null)]
            if sub.empty:
                continue
            for stat, v in obs.items():
                col = sub.loc[sub.statistic == stat, 'value'].dropna()
                measure = stat.removeprefix('CPL_').removesuffix('_gap_std').removesuffix('_gap')
                sign = COUPLING_COLUMNS[measure]
                p = float((col <= v).mean()) if sign < 0 else float((col >= v).mean())
                sd = col.std()
                rows.append({'panel': panel, 'null': null, 'statistic': stat,
                             'observed': v, 'null_mean': col.mean(), 'null_sd': sd,
                             'null_q05': col.quantile(0.05), 'null_q95': col.quantile(0.95),
                             'z_vs_null': sign * (v - col.mean()) / sd if sd > 0 else np.nan,
                             'p_predicted_side': p, 'n_rep': len(col)})
    return pd.DataFrame(rows)


def check_against_published(long: pd.DataFrame) -> pd.DataFrame | None:
    if not PUBLISHED.exists():
        return None
    pub = pd.read_csv(PUBLISHED)
    pub = pub[pub.statistic == 'H1_hdep_stress_minus_calm'][['panel', 'null', 'rep', 'value']]
    ours = long[long.statistic == 'CPL_h_dep_gap'][['panel', 'null', 'rep', 'value']]
    m = ours.merge(pub, on=['panel', 'null', 'rep'], suffixes=('_here', '_published'))
    m['abs_diff'] = (m.value_here - m.value_published).abs()
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--n-rep', type=int, default=200)
    p.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--panels', default='ff49,sp500')
    p.add_argument('--out-dir', default='outputs/coupling_null')
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
                done.add((r['panel'], r['null'], r['rep']))
    tasks = [(pa, 'observed', -1) for pa in panels]
    tasks += [(pa, nu, r) for pa in panels for nu in NULLS for r in range(args.n_rep)]
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
                          'statistic': k, 'value': v}
                         for r in recs for k, v in r['stats'].items()])
    long = long.drop_duplicates(['panel', 'null', 'rep', 'statistic'], keep='last')
    summary = summarise(long)
    dest = Path('paper/tables')
    long.to_csv(dest / 'coupling_null_200rep_replications_long.csv', index=False,
                float_format='%.6g')
    summary.to_csv(dest / 'coupling_null_200rep_summary.csv', index=False, float_format='%.4f')

    chk = check_against_published(long)
    if chk is not None and len(chk):
        print(f'\nCheck against the published 200-replication run: {len(chk)} replications '
              f'matched, max |difference| in the h_dep gap = {chk.abs_diff.max():.2e}')
    pd.set_option('display.width', 180)
    # The raw-gap test is the paper's H1 test; its z-score against the null is
    # already unit-free, so it is what ranks the measures. The standardised gap
    # (``_gap_std``) answers a different question and is kept in the CSV only.
    show = summary[~summary.statistic.str.endswith('_gap_std')]
    print(show.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
