"""Calibrated null distribution for the jump and relaxation statistics.

The structural index is a non-negative, convex functional of an estimated
covariance and of estimated third and fourth moments. Estimation noise alone
therefore pushes it *up* sharply and lets it decay *smoothly* -- which is
exactly the signature the jump/diffusion postulate predicts. Any claim about
one-sided jumps or about relaxation must be tested against that mechanical null,
not against zero.

This script builds the null by i.i.d. resampling the *rows* of the observed
return matrix. That preserves the cross-sectional dependence structure and the
fat marginal tails of real data, and destroys everything the postulates are
about: volatility clustering, time-varying dependence, and the arrival dynamics
of information. Running the full production pipeline on each replication gives
the distribution of every statistic under "same distribution, no dynamics".

Reported p-values are the share of replications whose statistic is at least as
extreme as the observed one.

Usage::

    python -m scripts.placebo_null --config config/sp500.yaml --n-rep 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.empirical import DATA_READERS, rolling_measures
from src.jumps import detect_jumps, estimate_relaxation, jump_asymmetry


def statistics_of(ent: pd.DataFrame, logret: pd.DataFrame, threshold: float,
                  scale_window: int, stress_quantile: float = 0.90) -> dict:
    jumps = detect_jumps(ent['J'], threshold, scale_window)
    a = jump_asymmetry(jumps)
    rel = estimate_relaxation(ent['J'], jumps)

    signal = logret.mean(axis=1).rolling(21).std().reindex(ent.index)
    stress = signal >= signal.quantile(stress_quantile)
    calm = ~stress

    up = jumps['jump_up'].reindex(ent.index).fillna(False)
    d = ent.dropna(subset=['dh_dep_ew', 'dD', 'dJ', 'dskew_market'])
    dep = -d['dh_dep_ew']
    s_d, c_d = stress.reindex(d.index).fillna(False), calm.reindex(d.index).fillna(True)

    out = {
        # --- H3: jump asymmetry -------------------------------------------
        'n_jump_up': a['n_jump_up'],
        'n_jump_down': a['n_jump_down'],
        'jump_ratio_up_down': a['jump_ratio_up_down'],
        'mean_size_up': a['mean_size_up'],
        'max_size_up': a['max_size_up'],
        'skew_of_changes': a['skew_of_changes'],
        'share_of_variance_up': a['share_of_variance_up'],
        # --- H4: relaxation ------------------------------------------------
        'half_life_days': rel['half_life_days'],
        'kappa': rel['kappa'],
        # --- amplitude of the state variable --------------------------------
        'sd_J': float(ent['J'].std()),
        'range_J': float(ent['J'].max() - ent['J'].min()),
        # --- H1: regime signature -------------------------------------------
        'H1_J_stress_minus_calm': float(ent.loc[stress, 'J'].mean()
                                        - ent.loc[calm, 'J'].mean()),
        'H1_hdep_stress_minus_calm': float(ent.loc[stress, 'h_dep'].mean()
                                           - ent.loc[calm, 'h_dep'].mean()),
        'H1_D_stress_minus_calm': float(ent.loc[stress, 'd_total'].mean()
                                        - ent.loc[calm, 'd_total'].mean()),
        # --- H2: compensation ------------------------------------------------
        'H2_corr_dhvol_dhdep_all': float(ent['dh_vol'].corr(ent['dh_dep'])),
        'H2_corr_dhvol_dhdep_stress': float(ent.loc[stress, 'dh_vol']
                                            .corr(ent.loc[stress, 'dh_dep'])),
        # --- H5: skewness signature -------------------------------------------
        'H5_dskew_on_jump_minus_else': float(ent.loc[up, 'dskew_market'].mean()
                                             - ent.loc[~up, 'dskew_market'].mean()),
        'H5_corr_dJ_dskew': float(ent['dJ'].corr(ent['dskew_market'])),
        # --- H6: channel coupling ----------------------------------------------
        'H6_corr_dep_shape_stress': float(dep[s_d].corr(d.loc[s_d, 'dD'])),
        'H6_corr_dep_shape_calm': float(dep[c_d].corr(d.loc[c_d, 'dD'])),
        'H6_coupling_gap': float(dep[s_d].corr(d.loc[s_d, 'dD'])
                                 - dep[c_d].corr(d.loc[c_d, 'dD'])),
        # --- is J just volatility? ----------------------------------------------
        'corr_J_logvol': float(ent['J'].corr(np.log(signal.replace(0, np.nan)))),
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/sp500.yaml')
    p.add_argument('--n-rep', type=int, default=20)
    p.add_argument('--block', type=int, default=1,
                   help='Resampling block length in days. 1 is the i.i.d. null '
                        '(no temporal structure at all); a value such as 21 '
                        'keeps short-run volatility clustering and destroys only '
                        'the longer-horizon regime dynamics, which is the fairer '
                        'intermediate null for regime-based hypotheses.')
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    logret = DATA_READERS[cfg['data']['type']](cfg)
    logret = logret.loc[str(cfg['sample']['start']):str(cfg['sample']['end'])]

    est = cfg['estimation']
    kw = dict(window=cfg['sample']['window'], method=est['covariance'],
              jump_method=est.get('jump_covariance', 'ewma'),
              halflife=float(est.get('ewma_halflife', 60)),
              shrinkage=float(est.get('ewma_shrinkage', 0.10)),
              shape_estimator=est.get('shape_estimator', 'hyvarinen'))
    jcfg = cfg.get('jumps', {})
    threshold = float(jcfg.get('threshold', 4.0))
    scale_window = int(jcfg.get('scale_window', 250))

    sq = float(cfg.get('regimes', {}).get('stress_quantile', 0.90))
    observed = statistics_of(rolling_measures(logret, **kw), logret,
                             threshold, scale_window, sq)

    rng = np.random.default_rng(args.seed)
    X = logret.to_numpy()
    T = len(X)
    rows = []
    block = max(1, int(args.block))
    n_blocks = int(np.ceil(T / block))
    for r in range(args.n_rep):
        if block == 1:
            idx = rng.integers(0, T, size=T)
        else:
            # Circular block bootstrap: keeps runs of `block` consecutive days,
            # so volatility clustering inside a block survives.
            starts = rng.integers(0, T, size=n_blocks)
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        boot = pd.DataFrame(X[idx], index=logret.index, columns=logret.columns)
        rows.append(statistics_of(rolling_measures(boot, **kw), boot,
                                  threshold, scale_window, sq))
        print(f'  replication {r + 1}/{args.n_rep} done', flush=True)
    null = pd.DataFrame(rows)

    summary = []
    for k, v in observed.items():
        col = null[k].dropna()
        if len(col) == 0 or not np.isfinite(v):
            summary.append({'statistic': k, 'observed': v})
            continue
        summary.append({
            'statistic': k,
            'observed': v,
            'null_mean': float(col.mean()),
            'null_sd': float(col.std()),
            'null_q05': float(col.quantile(0.05)),
            'null_q95': float(col.quantile(0.95)),
            'p_greater': float((col >= v).mean()),
            'p_less': float((col <= v).mean()),
            'n_rep': int(len(col)),
        })
    out = pd.DataFrame(summary)
    print()
    print(out.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    suffix = '' if block == 1 else f'_block{block}'
    dest = Path(args.out or Path(cfg['outputs']['tables']) / f'placebo_null{suffix}.csv')
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    null.to_csv(dest.with_name(dest.stem + '_replications.csv'), index=False)
    print(f'\nWritten to {dest}')


if __name__ == '__main__':
    main()
