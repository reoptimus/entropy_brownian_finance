"""Phase A benchmark on observed data: is h_dep better than what already exists?

The calibrated nulls (``scripts/placebo_null.py --coupling`` and
``calcul_local/lancer_calcul.py``) say whether each coupling measure detects
something beyond resampled returns. This script adds the comparisons a
central-bank reader asks for, on the observed series only:

1. how close the measures are to one another (levels and 21-day changes);
2. how well each one separates NBER recession dates from expansion dates, and
   the 12 months before a recession from the rest (AUROC), raw and after the
   part explained by market volatility is removed;
3. if official stress indices are present in ``data/external/``, how much each
   measure adds to the others in explaining them (standardised OLS, HAC).

Official indices are optional because this container cannot download them.
Place either or both files in ``data/external/`` before running:

* ``STLFSI4.csv`` -- St. Louis Fed Financial Stress Index, weekly, from FRED
  (https://fred.stlouisfed.org/series/STLFSI4, "Download", CSV);
* ``OFR_FSI.csv`` -- OFR Financial Stress Index, daily
  (https://www.financialresearch.gov/financial-stress-index/, CSV download).

Usage::

    python -m scripts.coupling_benchmark --config config/sp500.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from sklearn.metrics import roc_auc_score

from src.coupling import COUPLING_COLUMNS
from src.empirical import DATA_READERS, rolling_measures

# NBER business-cycle reference dates (peak month -> trough month) in the
# sample, https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions
NBER_RECESSIONS = [
    ('1990-07-01', '1991-03-31'),
    ('2001-03-01', '2001-11-30'),
    ('2007-12-01', '2009-06-30'),
    ('2020-02-01', '2020-04-30'),
]
EXTERNAL = Path('data/external')


def nber_labels(index: pd.DatetimeIndex, lead_days: int = 0) -> pd.Series:
    """1 inside a recession; with ``lead_days > 0``, 1 on the ``lead_days``
    calendar days before a peak and NaN inside recessions (those dates are
    neither a warning nor a non-warning)."""
    y = pd.Series(0.0, index=index)
    for peak, trough in NBER_RECESSIONS:
        p, t = pd.Timestamp(peak), pd.Timestamp(trough)
        if lead_days:
            y[(index >= p - pd.Timedelta(days=lead_days)) & (index < p)] = 1.0
            y[(index >= p) & (index <= t)] = np.nan
        else:
            y[(index >= p) & (index <= t)] = 1.0
    return y


def _auroc(score: pd.Series, y: pd.Series) -> float:
    d = pd.concat([score, y], axis=1).dropna()
    if d.iloc[:, 1].nunique() < 2:
        return float('nan')
    return float(roc_auc_score(d.iloc[:, 1], d.iloc[:, 0]))


def _orthogonalise(x: pd.Series, logvol: pd.Series) -> pd.Series:
    d = pd.concat([x, logvol], axis=1).dropna()
    fit = sm.OLS(d.iloc[:, 0], sm.add_constant(d.iloc[:, 1])).fit()
    return fit.resid.reindex(x.index)


def _read_external() -> dict:
    out = {}
    f = EXTERNAL / 'STLFSI4.csv'
    if f.exists():
        s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
        out['STLFSI4'] = pd.to_numeric(s, errors='coerce').dropna()
    f = EXTERNAL / 'OFR_FSI.csv'
    if f.exists():
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        col = 'OFR FSI' if 'OFR FSI' in df.columns else df.columns[0]
        out['OFR_FSI'] = pd.to_numeric(df[col], errors='coerce').dropna()
    return out


def _encompassing(target: pd.Series, X: pd.DataFrame, lags: int) -> list[dict]:
    """Standardised OLS of an official index on each measure alone, then on
    h_dep together with each competitor."""
    d = pd.concat([target.rename('y'), X], axis=1).dropna()
    d = (d - d.mean()) / d.std()
    rows = []
    specs = [[c] for c in X.columns] + [['h_dep', c] for c in X.columns if c != 'h_dep']
    for cols in specs:
        fit = sm.OLS(d['y'], sm.add_constant(d[cols])).fit(
            cov_type='HAC', cov_kwds={'maxlags': lags})
        row = {'regressors': ' + '.join(cols), 'n': int(fit.nobs), 'r2': fit.rsquared}
        for c in cols:
            row[f'coef_{c}'] = fit.params[c]
            row[f't_{c}'] = fit.tvalues[c]
        rows.append(row)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/sp500.yaml')
    p.add_argument('--out-prefix', default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    logret = DATA_READERS[cfg['data']['type']](cfg)
    logret = logret.loc[str(cfg['sample']['start']):str(cfg['sample']['end'])]
    est = cfg['estimation']
    ent = rolling_measures(logret, cfg['sample']['window'], method=est['covariance'],
                           jump_method=est.get('jump_covariance', 'ewma'),
                           halflife=float(est.get('ewma_halflife', 60)),
                           shrinkage=float(est.get('ewma_shrinkage', 0.10)),
                           shape_estimator=est.get('shape_estimator', 'hyvarinen'),
                           coupling=True)
    cols = list(COUPLING_COLUMNS)
    # Orient every measure so that larger = more coupled.
    M = pd.DataFrame({c: ent[c] * COUPLING_COLUMNS[c] for c in cols})
    logvol = np.log(logret.mean(axis=1).rolling(21).std().reindex(M.index))

    prefix = args.out_prefix or {'ff49': 'ff49', 'sp500': 'sp500'}.get(cfg['data']['type'], 'eu')
    dest = Path('paper/tables')
    dest.mkdir(parents=True, exist_ok=True)

    series = M.copy()
    series['log_vol21'] = logvol
    series.to_csv(dest / f'{prefix}_coupling_series.csv', float_format='%.6g')

    corr_lvl = M.corr()
    corr_chg = M.diff(21).corr()

    y_rec = nber_labels(M.index)
    y_lead = nber_labels(M.index, lead_days=365)
    auc = []
    for c in cols + ['log_vol21']:
        x = logvol if c == 'log_vol21' else M[c]
        row = {'measure': c,
               'auroc_recession': _auroc(x, y_rec),
               'auroc_12m_before_recession': _auroc(x, y_lead)}
        if c != 'log_vol21':
            xo = _orthogonalise(x, logvol)
            row['auroc_recession_vol_orth'] = _auroc(xo, y_rec)
            row['auroc_12m_before_vol_orth'] = _auroc(xo, y_lead)
            row['corr_with_log_vol21'] = float(x.corr(logvol))
        auc.append(row)
    auc = pd.DataFrame(auc)
    auc.to_csv(dest / f'{prefix}_coupling_auroc.csv', index=False, float_format='%.4f')

    enc_rows = []
    for name, s in _read_external().items():
        if name == 'STLFSI4':
            X = M.resample('W-FRI').last()
            tgt = s.resample('W-FRI').last()
            lags = 4
        else:
            X, tgt, lags = M, s, 21
        for r in _encompassing(tgt, X, lags):
            r['index'] = name
            enc_rows.append(r)
    if enc_rows:
        pd.DataFrame(enc_rows).to_csv(dest / f'{prefix}_coupling_encompassing.csv',
                                      index=False, float_format='%.4f')

    summary = {'n_dates': int(M.dropna().shape[0]),
               'corr_levels': corr_lvl.round(3).to_dict(),
               'corr_21d_changes': corr_chg.round(3).to_dict(),
               'external_indices': sorted(_read_external())}
    (dest / f'{prefix}_coupling_summary.json').write_text(json.dumps(summary, indent=2))

    pd.set_option('display.width', 160)
    print('Correlations of levels (oriented: larger = more coupled)')
    print(corr_lvl.round(3).to_string())
    print('\nCorrelations of 21-day changes')
    print(corr_chg.round(3).to_string())
    print('\nAUROC against NBER recession dates')
    print(auc.round(3).to_string(index=False))
    if enc_rows:
        print('\nEncompassing regressions on official stress indices')
        print(pd.DataFrame(enc_rows).round(3).to_string(index=False))
    else:
        print('\nNo official stress index in data/external/: encompassing step skipped.')


if __name__ == '__main__':
    main()
