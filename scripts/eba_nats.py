"""Cost in nats of the EBA 2025 EU-wide stress test adverse scenario.

The adverse scenario (macro-financial scenario designed by the ESRB, published
by the EBA in January 2025, tables 4.1.6, 4.1.7 and 4.1.10) is read as a shift
of the mean of the one-year distribution of euro-area market variables,
relative to the baseline:

* EU stock prices: -50 % in 2025, baseline flat. Applied to the five national
  indices of the euro panel (DAX, CAC 40, FTSE MIB, IBEX 35, AEX) as a log
  return of ln(0.5).
* Long-term rates, adverse minus baseline, 2025 annual average: Germany
  +110 bp (mapped to the ECB AAA 10-year yield) and EU +168 bp (mapped to the
  ECB all-ratings 10-year yield).
* Exchange rates of the euro against USD, GBP, JPY and CHF: unchanged in both
  scenarios, so a shift of zero (a constraint, not a free variable).

Under a Gaussian reference N(0, S) and a pure mean shift d with the same
covariance, KL = 1/2 d' S^-1 d nats. Variables the scenario does not specify
are left free; the least-improbable completion is the Gaussian conditional
mean, and the cost then depends only on the specified block:
KL_min = 1/2 d_S' (S_SS)^-1 d_S.

S is the covariance of one-year moves. Main estimate: daily moves 2004-09-06
to 2024-12-31 (what a designer had in hand), times 252. Robustness: overlapping
252-day moves; full sample to 2026; stressed window 2008-2012.

Historical comparators: the realised one-year move of the same variables,
priced with the same S (calendar years 2008, 2011, 2022 and the worst rolling
year).

Usage: python scripts/eba_nats.py data/zone_euro  [writes paper/tables/eba_nats.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EQ = ['dax', 'cac40', 'ftse_mib', 'ibex35', 'aex']
RT = ['aaa_10a', 'tous_10a']
FX = ['eur_usd', 'eur_gbp', 'eur_jpy', 'eur_chf']

SHOCK = {**{k: np.log(0.5) for k in EQ},
         'aaa_10a': 1.10, 'tous_10a': 1.68,       # percentage points
         **{k: 0.0 for k in FX}}


def load_levels(folder: Path, max_fill: int = 3) -> pd.DataFrame:
    def rd(name):
        df = pd.read_csv(folder / f'{name}.csv', index_col=0, parse_dates=True)
        return df.apply(pd.to_numeric, errors='coerce').sort_index()
    eq, ob, fx = rd('actions'), rd('obligataire'), rd('change')
    lv = pd.concat([np.log(eq[EQ]), ob[RT], np.log(fx[FX])], axis=1, sort=True)
    bdays = pd.bdate_range(lv.index.min(), lv.index.max())
    lv = lv.reindex(lv.index.union(bdays)).ffill(limit=max_fill).reindex(bdays)
    return lv.dropna()


def cov_daily(lv, start, end):
    d = lv.loc[start:end].diff().dropna()
    return d.cov() * 252


def cov_annual(lv, start, end):
    a = (lv - lv.shift(252)).loc[start:end].dropna()
    return a.cov()


def kl(d: pd.Series, S: pd.DataFrame) -> float:
    k = list(d.index)
    return 0.5 * float(d.values @ np.linalg.solve(S.loc[k, k].values, d.values))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('folder')
    p.add_argument('--out', default='paper/tables/eba_nats.csv')
    a = p.parse_args()
    lv = load_levels(Path(a.folder))
    d_all = pd.Series(SHOCK)[EQ + RT + FX]

    sigmas = {
        'daily_2004_2024': cov_daily(lv, '2004-09-06', '2024-12-31'),
        'annual_2004_2024': cov_annual(lv, '2004-09-06', '2024-12-31'),
        'daily_2004_2026': cov_daily(lv, '2004-09-06', None),
        'daily_stress_2008_2012': cov_daily(lv, '2008-01-01', '2012-12-31'),
    }
    variants = {
        'equities_only': EQ,
        'rates_only': RT,
        'equities_rates': EQ + RT,
        'full_with_fx_fixed': EQ + RT + FX,
    }
    rows = []
    for sname, S in sigmas.items():
        for vname, keys in variants.items():
            rows.append(dict(sigma=sname, item=vname, nats=kl(d_all[keys], S)))
        sd = np.sqrt(np.diag(S.values))
        for k, s in zip(S.index, sd):
            z = d_all[k] / s
            rows.append(dict(sigma=sname, item=f'sd_{k}', nats=s))
            if k in EQ + RT:
                rows.append(dict(sigma=sname, item=f'marginal_{k}', nats=0.5 * z * z))
        rows.append(dict(sigma=sname, item='sum_marginals_eq_rates',
                         nats=sum(0.5 * (d_all[k] / np.sqrt(S.loc[k, k])) ** 2
                                  for k in EQ + RT)))
        C = S.values / np.outer(sd, sd)
        idx = list(S.index)
        rows.append(dict(sigma=sname, item='mean_corr_eq_eq',
                         nats=np.mean([C[idx.index(i), idx.index(j)]
                                       for i in EQ for j in EQ if i < j])))
        rows.append(dict(sigma=sname, item='mean_corr_eq_rates',
                         nats=np.mean([C[idx.index(i), idx.index(j)]
                                       for i in EQ for j in RT])))
    # end of the three-year horizon (2027): stocks -42 %, EU +112 bp, DE +87 bp,
    # priced with the three-year covariance (3 x the one-year one)
    d27 = pd.Series({**{k: np.log(0.58) for k in EQ}, 'aaa_10a': 0.87,
                     'tous_10a': 1.12, **{k: 0.0 for k in FX}})[EQ + RT + FX]
    S3 = 3 * sigmas['daily_2004_2024']
    rows.append(dict(sigma='daily_2004_2024_x3', item='horizon2027_equities_rates',
                     nats=kl(d27[EQ + RT], S3)))
    rows.append(dict(sigma='daily_2004_2024_x3', item='horizon2027_full_with_fx_fixed',
                     nats=kl(d27, S3)))
    # historical one-year moves priced with the main Sigma
    S = sigmas['daily_2004_2024']
    keys = EQ + RT + FX
    yr = lv[keys].resample('YE').last()
    for y in (2008, 2011, 2022):
        x = yr.loc[str(y)].iloc[0] - yr.loc[str(y - 1)].iloc[0]
        rows.append(dict(sigma='daily_2004_2024', item=f'hist_{y}_full', nats=kl(x, S)))
        rows.append(dict(sigma='daily_2004_2024', item=f'hist_{y}_eq_rates',
                         nats=kl(x[EQ + RT], S)))
        for k in ['dax', 'aaa_10a', 'tous_10a']:
            rows.append(dict(sigma='daily_2004_2024', item=f'hist_{y}_move_{k}',
                             nats=float(x[k])))
    roll = (lv[keys] - lv[keys].shift(252)).dropna()
    Sinv = np.linalg.inv(S.loc[keys, keys].values)
    c = 0.5 * np.einsum('ij,jk,ik->i', roll.values, Sinv, roll.values)
    i = int(np.argmax(c))
    rows.append(dict(sigma='daily_2004_2024', item='hist_worst_rolling_full', nats=c[i]))
    rows.append(dict(sigma='daily_2004_2024',
                     item=f'hist_worst_rolling_end_{roll.index[i].date()}', nats=c[i]))
    rows.append(dict(sigma='daily_2004_2024', item='hist_q99_rolling_full',
                     nats=float(np.quantile(c, 0.99))))
    rows.append(dict(sigma='daily_2004_2024', item='hist_median_rolling_full',
                     nats=float(np.median(c))))
    ce = 0.5 * np.einsum('ij,jk,ik->i', roll[EQ + RT].values,
                         np.linalg.inv(S.loc[EQ + RT, EQ + RT].values),
                         roll[EQ + RT].values)
    rows.append(dict(sigma='daily_2004_2024', item='hist_worst_rolling_eq_rates',
                     nats=float(ce.max())))
    rows.append(dict(sigma='daily_2004_2024',
                     item=f'hist_worst_rolling_eq_rates_end_{roll.index[int(np.argmax(ce))].date()}',
                     nats=float(ce.max())))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    with pd.option_context('display.max_rows', 500, 'display.width', 140):
        print(out.round(4).to_string(index=False))
    print('sample', lv.index.min().date(), lv.index.max().date(), len(lv))


if __name__ == '__main__':
    main()
