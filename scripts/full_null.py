"""All placebo statistics of the JFEC paper, under both nulls, in one run.

For each panel (S&P 500, FF49), each null (i.i.d. row resampling; CCC-FHS
GJR-GARCH filtered historical simulation, see ``scripts/garch_null.py``) and
each replication, one resampled panel is built and every statistic the paper
tests is computed on it:

* the hypothesis statistics of ``scripts/placebo_null.py`` (H1-H6, H8);
* the episode typology of ``scripts/episode_null.py`` (largest episode,
  episodes with falling skewness, dependence-dominated episodes);
* the coupling measures and their within-year gaps of
  ``scripts/garch_null.py``.

Computing them on the same draws keeps the tables consistent, and the run is
checkpointed so it can be stopped and resumed. Each replication has its own
seed (``SeedSequence([SEED, null, panel, rep])``), so results do not depend on
the number of workers or on the order of execution.

p-values are ``(1 + k) / (1 + B)`` in the direction the hypothesis predicts,
which is exact for an exchangeable null; with B = 999 the smallest is 0.001.

Usage::

    python -m scripts.full_null --n-rep 999 --jobs 4
    python -m scripts.full_null --summarise-only
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

from scripts.coupling_null import PANELS, _load
from scripts.episode_null import typology_of
from scripts.garch_null import fit_gjr, simulate_gjr, stats_for
from scripts.placebo_null import statistics_of
from src.empirical import rolling_measures

SEED = 53
NULLS = ('iid', 'ccc_fhs')
_FITS: dict = {}
_CFG: dict = {}


def _settings(panel: str) -> dict:
    if panel not in _CFG:
        cfg = yaml.safe_load(Path(PANELS[panel]).read_text())
        est = cfg['estimation']
        jc = cfg.get('jumps', {})
        _CFG[panel] = dict(
            kw=dict(window=cfg['sample']['window'], method=est['covariance'],
                    jump_method=est.get('jump_covariance', 'ewma'),
                    halflife=float(est.get('ewma_halflife', 60)),
                    shrinkage=float(est.get('ewma_shrinkage', 0.10)),
                    shape_estimator=est.get('shape_estimator', 'hyvarinen')),
            threshold=float(jc.get('threshold', 4.0)),
            scale_window=int(jc.get('scale_window', 250)))
    return _CFG[panel]


def all_stats(data: pd.DataFrame, panel: str) -> dict:
    _, window, sq = _load(panel)
    s = _settings(panel)
    ent = rolling_measures(data, **s['kw'])
    out = {f'PL_{k}': v for k, v in
           statistics_of(ent, data, s['threshold'], s['scale_window'], sq).items()}
    typ = typology_of(ent, threshold=s['threshold'], scale_window=s['scale_window'],
                      gap=40, pre=10, post_search=20)
    out.update({f'EP_{k}': float(v) for k, v in typ.items()})
    out.update(stats_for(data, window, sq))
    return {k: (float(v) if v is not None and np.isfinite(float(v)) else None)
            for k, v in out.items()}


def resample(panel: str, null: str, rep: int) -> pd.DataFrame:
    logret, _, _ = _load(panel)
    T = len(logret)
    ss = np.random.SeedSequence([SEED, NULLS.index(null), list(PANELS).index(panel), rep])
    idx = np.random.default_rng(ss).integers(0, T, size=T)
    if null == 'iid':
        X = logret.to_numpy()[idx]
    else:
        if panel not in _FITS:
            _FITS[panel] = fit_gjr(logret)
        par, Z = _FITS[panel]
        X = simulate_gjr(par, Z, idx)
    return pd.DataFrame(X, index=logret.index, columns=logret.columns)


def run_task(panel: str, null: str, rep: int) -> dict:
    t0 = time.time()
    data = _load(panel)[0] if null == 'observed' else resample(panel, null, rep)
    stats = all_stats(data, panel)
    return {'panel': panel, 'null': null, 'rep': rep,
            'seconds': round(time.time() - t0, 1), 'stats': stats}


def load_checkpoint(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        for k, v in r['stats'].items():
            rows.append({'panel': r['panel'], 'null': r['null'], 'rep': r['rep'],
                         'statistic': k, 'value': v})
    return pd.DataFrame(rows).drop_duplicates(['panel', 'null', 'rep', 'statistic'])


def summarise(long: pd.DataFrame) -> pd.DataFrame:
    """Two-sided tail counts; the build script picks the predicted side."""
    rows = []
    for panel in long['panel'].unique():
        obs = long[(long.panel == panel) & (long.null == 'observed')].set_index('statistic')['value']
        for null in NULLS:
            sub = long[(long.panel == panel) & (long.null == null)]
            if sub.empty:
                continue
            g = sub.groupby('statistic')['value']
            for stat, v in obs.items():
                if v is None or not np.isfinite(v) or stat not in g.groups:
                    continue
                col = g.get_group(stat).dropna().astype(float)
                if col.empty:
                    continue
                B = len(col)
                rows.append({'panel': panel, 'null': null, 'statistic': stat,
                             'observed': v, 'null_mean': col.mean(), 'null_sd': col.std(),
                             'null_q05': col.quantile(0.05), 'null_q95': col.quantile(0.95),
                             'p_greater': (1 + int((col >= v).sum())) / (1 + B),
                             'p_less': (1 + int((col <= v).sum())) / (1 + B),
                             'n_rep': B})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-rep', type=int, default=999)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--panels', default='sp500,ff49')
    ap.add_argument('--nulls', default=','.join(NULLS))
    ap.add_argument('--out-dir', default='outputs/full_null')
    ap.add_argument('--tables', default='paper/tables')
    ap.add_argument('--summarise-only', action='store_true')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck = out / 'checkpoint.jsonl'
    ck.touch()
    panels = args.panels.split(',')
    nulls = args.nulls.split(',')

    if not args.summarise_only:
        done = set()
        for line in ck.read_text().splitlines():
            r = json.loads(line)
            done.add((r['panel'], r['null'], r['rep']))
        tasks = [(pa, 'observed', 0) for pa in panels]
        tasks += [(pa, nu, r) for r in range(args.n_rep) for nu in nulls for pa in panels]
        tasks = [t for t in tasks if t not in done]
        print(f'{len(tasks)} tasks to run ({len(done)} already done)', flush=True)
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.jobs) as ex, ck.open('a') as fh:
            futs = [ex.submit(run_task, *t) for t in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                fh.write(json.dumps(f.result()) + '\n')
                fh.flush()
                if i % 20 == 0 or i == len(tasks):
                    el = time.time() - t0
                    print(f'  [{i}/{len(tasks)}] ~{el / i * (len(tasks) - i) / 60:.0f} min left',
                          flush=True)

    long = load_checkpoint(ck)
    B = int(long[long.null != 'observed'].groupby(['panel', 'null'])['rep'].nunique().min())
    T = Path(args.tables)
    long.to_csv(T / f'full_null_{B}rep_replications_long.csv.gz', index=False)
    s = summarise(long)
    s.to_csv(T / f'full_null_{B}rep_summary.csv', index=False)
    for pa in panels:
        if pa in _FITS or (long.null == 'ccc_fhs').any():
            fit_gjr(_load(pa)[0])[0].to_csv(T / f'{pa}_gjr_params.csv')
    print(s.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
