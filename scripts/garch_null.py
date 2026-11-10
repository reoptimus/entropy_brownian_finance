"""A volatility-preserving null for the coupling result (JFEC revision).

The i.i.d. null of ``scripts/placebo_null.py`` destroys all temporal structure,
including volatility clustering. A referee will ask whether the rise in
coupling under stress is simply what volatility clustering plus the
Forbes-Rigobon (2002) effect produce when the conditional correlation is
constant. This script answers with a constant-conditional-correlation
filtered-historical-simulation null (CCC-FHS):

1. fit a GJR-GARCH(1,1) by Gaussian QMLE to each asset (``arch`` package);
2. collect the T x N matrix of standardised residuals ``z_t``;
3. for each replication, resample *rows* of ``z`` i.i.d. (this keeps fat tails
   and the cross-sectional dependence of a day's shocks, i.e. a constant
   conditional correlation, and destroys any dynamics of that correlation);
4. rebuild returns with the fitted GJR recursions, so every asset keeps its own
   volatility clustering and leverage effect, and all assets share the same
   resampled day (common volatility bursts are preserved);
5. rerun the identical coupling pipeline and regime labelling.

Under this null, volatility moves exactly as in a GJR world, correlations do
not. A coupling gap beyond it is a dependence dynamic, not a volatility one.

Besides the paper's stress-minus-calm gap, we also record a *within-year* gap
(regime dummy with calendar-year fixed effects), which removes slow secular
trends in correlation that could otherwise masquerade as a stress effect.

Usage::

    python -m scripts.garch_null --jobs 4 --n-rep 200
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

from scripts.coupling_null import _load, PANELS
from src.coupling import COUPLING_COLUMNS, coupling_statistics, rolling_coupling
from src.regimes import past_only_stress

SEED = 29
_FITS: dict = {}


def fit_gjr(logret: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-asset GJR-GARCH(1,1) QMLE on percentage returns.

    Returns the parameter table and the T x N matrix of standardised residuals.
    """
    from arch import arch_model
    params, Z = [], []
    for col in logret.columns:
        x = 100.0 * logret[col].to_numpy()
        res = arch_model(x, mean='Constant', vol='GARCH', p=1, o=1, q=1,
                         dist='normal', rescale=False).fit(disp='off')
        p = res.params
        params.append({'asset': col, 'mu': p['mu'], 'omega': p['omega'],
                       'alpha': p['alpha[1]'], 'gamma': p['gamma[1]'],
                       'beta': p['beta[1]'],
                       'persistence': p['alpha[1]'] + 0.5 * p['gamma[1]'] + p['beta[1]']})
        z = np.asarray(res.std_resid)
        Z.append((z - np.nanmean(z)) / np.nanstd(z))
    return pd.DataFrame(params).set_index('asset'), np.column_stack(Z)


def simulate_gjr(par: pd.DataFrame, Z: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Rebuild a return panel from resampled standardised-residual rows."""
    z = Z[idx]
    T, N = z.shape
    mu, om, a, g, b = (par[c].to_numpy() for c in ('mu', 'omega', 'alpha', 'gamma', 'beta'))
    pers = np.minimum(a + 0.5 * g + b, 0.999)
    h = om / (1.0 - pers)                       # start at the unconditional variance
    out = np.empty((T, N))
    for t in range(T):
        e = np.sqrt(h) * z[t]
        out[t] = mu + e
        h = om + (a + g * (e < 0)) * e * e + b * h
    return out / 100.0


def within_year_gap(x: pd.Series, stress: pd.Series, calm: pd.Series) -> float:
    """Stress coefficient in ``x ~ stress + year FE`` on labelled dates."""
    lab = stress | calm
    df = pd.DataFrame({'x': x[lab], 's': stress[lab].astype(float)})
    df['y'] = df.index.year
    dm = df.groupby('y')[['x', 's']].transform(lambda v: v - v.mean())
    den = float((dm['s'] ** 2).sum())
    return float((dm['s'] * dm['x']).sum() / den) if den > 0 else float('nan')


def stats_for(data: pd.DataFrame, window: int, sq: float) -> dict:
    cpl = rolling_coupling(data, window)
    out = coupling_statistics(cpl, data, sq)
    signal = data.mean(axis=1).rolling(21).std().reindex(cpl.index)
    stress, valid = past_only_stress(signal, sq)
    calm = ~stress & valid
    for col in ('h_dep', 'h_dep_devol', 'h_dep_rank'):
        out[f'CPL_{col}_gap_fe'] = within_year_gap(cpl[col], stress, calm)
    return out


def run_task(panel: str, null: str, rep: int) -> dict:
    t0 = time.time()
    logret, window, sq = _load(panel)
    if null == 'observed':
        data = logret
    elif null == 'iid':
        # Same seeds as scripts/coupling_null.py, so replication r is the
        # published i.i.d. replication r (checked on the raw gap).
        ss = np.random.SeedSequence([11, ['ff49', 'sp500'].index(panel), 1, rep])
        idx = np.random.default_rng(ss).integers(0, len(logret), size=len(logret))
        data = pd.DataFrame(logret.to_numpy()[idx], index=logret.index,
                            columns=logret.columns)
    else:
        if panel not in _FITS:
            _FITS[panel] = fit_gjr(logret)
        par, Z = _FITS[panel]
        ss = np.random.SeedSequence([SEED, list(PANELS).index(panel), rep])
        idx = np.random.default_rng(ss).integers(0, len(logret), size=len(logret))
        data = pd.DataFrame(simulate_gjr(par, Z, idx), index=logret.index,
                            columns=logret.columns)
    return {'panel': panel, 'null': null, 'rep': rep,
            'seconds': round(time.time() - t0, 1), 'stats': stats_for(data, window, sq)}


def summarise(long: pd.DataFrame) -> pd.DataFrame:
    """Observed value against each null; p = (1 + k) / (1 + B) in the
    predicted direction (k replications at least as extreme)."""
    rows = []
    for panel in long['panel'].unique():
        obs = long[(long.panel == panel) & (long.null == 'observed')].set_index('statistic')['value']
        for null in ('ccc_fhs', 'iid'):
            sub = long[(long.panel == panel) & (long.null == null)]
            if sub.empty:
                continue
            for stat, v in obs.items():
                if stat.endswith('_gap_std'):
                    continue
                col = sub.loc[sub.statistic == stat, 'value'].dropna()
                measure = stat.removeprefix('CPL_').removesuffix('_gap_fe').removesuffix('_gap')
                sign = COUPLING_COLUMNS[measure]
                k = int((col <= v).sum()) if sign < 0 else int((col >= v).sum())
                sd = col.std()
                rows.append({'panel': panel, 'null': null, 'statistic': stat,
                             'observed': v, 'null_mean': col.mean(), 'null_sd': sd,
                             'null_q05': col.quantile(0.05), 'null_q95': col.quantile(0.95),
                             'z_vs_null': sign * (v - col.mean()) / sd if sd > 0 else np.nan,
                             'p_predicted_side': (1 + k) / (1 + len(col)), 'n_rep': len(col)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-rep', type=int, default=200)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--panels', default='sp500,ff49')
    ap.add_argument('--out-dir', default='outputs/garch_null')
    ap.add_argument('--nulls', default='ccc_fhs', help='comma list: ccc_fhs,iid')
    args = ap.parse_args()
    panels = [x.strip() for x in args.panels.split(',') if x.strip()]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dest = Path('paper/tables')
    for pa in panels:                           # parameter table, for the appendix
        par, _ = fit_gjr(_load(pa)[0])
        par.to_csv(dest / f'{pa}_gjr_params.csv', float_format='%.5f')

    ckpt = out / 'checkpoint.jsonl'
    done = set()
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r['panel'], r['null'], r['rep']))
    tasks = [(pa, 'observed', -1) for pa in panels]
    nulls = [x.strip() for x in args.nulls.split(',') if x.strip()]
    tasks += [(pa, nu, r) for pa in panels for nu in nulls for r in range(args.n_rep)]
    todo = [t for t in tasks if t not in done]
    print(f'{len(tasks)} tasks, {len(todo)} to do, {args.jobs} processes', flush=True)
    t_start = time.time()
    if todo:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex, open(ckpt, 'a') as fh:
            futs = [ex.submit(run_task, *t) for t in todo]
            for n, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                fh.write(json.dumps(r) + '\n')
                fh.flush()
                if n % 10 == 0 or n == len(todo):
                    el = time.time() - t_start
                    print(f'  [{n}/{len(todo)}] ~{el / n * (len(todo) - n) / 60:.0f} min left',
                          flush=True)
    recs = [json.loads(l) for l in ckpt.read_text().splitlines() if l.strip()]
    long = pd.DataFrame([{'panel': r['panel'], 'null': r['null'], 'rep': r['rep'],
                          'statistic': k, 'value': v}
                         for r in recs for k, v in r['stats'].items()])
    long = long.drop_duplicates(['panel', 'null', 'rep', 'statistic'], keep='last')
    summary = summarise(long)
    long.to_csv(dest / 'garch_null_200rep_replications_long.csv', index=False, float_format='%.6g')
    summary.to_csv(dest / 'garch_null_200rep_summary.csv', index=False, float_format='%.4f')
    pd.set_option('display.width', 200)
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
