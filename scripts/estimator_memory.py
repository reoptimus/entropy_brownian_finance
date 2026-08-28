"""How much of the measured relaxation is the estimator's own memory?

Any rolling or exponentially weighted estimator remembers. If a one-day shock
enters the sample, the resulting entropy series decays as the shock's weight
decays -- with no economics involved at all. A measured half-life is therefore
only evidence of relaxation if it is materially longer than the estimator's
mechanical half-life.

This script measures that null directly. It simulates returns with no
persistence whatsoever -- i.i.d. draws, constant covariance -- injects a single
day of stress, runs the production pipeline on the result, and reports the decay
half-life of the structural index. Whatever comes out is pure estimator memory,
because the data-generating process has none.

A second control runs the same pipeline on clean i.i.d. data with no event, to
confirm the jump detector's false-positive rate at the chosen threshold.

Usage::

    python -m scripts.estimator_memory --n-assets 20 --window 252 --halflife 60
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.empirical import rolling_measures
from src.jumps import detect_jumps, estimate_relaxation


def simulate(n_obs: int, n_assets: int, corr: float, seed: int,
             event_at: int | None, event_vol: float, event_corr: float,
             event_len: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    R = np.full((n_assets, n_assets), corr)
    np.fill_diagonal(R, 1.0)
    sd = 0.015
    cov = R * sd ** 2
    X = rng.multivariate_normal(np.zeros(n_assets), cov, size=n_obs, method='eigh')
    if event_at is not None:
        Re = np.full((n_assets, n_assets), event_corr)
        np.fill_diagonal(Re, 1.0)
        cov_e = Re * (sd * event_vol) ** 2
        X[event_at:event_at + event_len] = rng.multivariate_normal(
            np.zeros(n_assets), cov_e, size=event_len, method='eigh')
    idx = pd.bdate_range('2000-01-03', periods=n_obs)
    return pd.DataFrame(X, index=idx, columns=[f'a{i}' for i in range(n_assets)])


def decay_half_life(J: pd.Series, event_pos: int, baseline: float,
                    horizon: int = 400) -> float:
    """Half-life of the excess of J over its pre-event baseline."""
    seg = J.iloc[event_pos:event_pos + horizon] - baseline
    if len(seg) < 10 or seg.iloc[0] <= 0:
        return np.nan
    peak = seg.max()
    below = np.flatnonzero(seg.to_numpy() <= peak / 2.0)
    peak_pos = int(np.argmax(seg.to_numpy()))
    below = below[below > peak_pos]
    return float(below[0] - peak_pos) if len(below) else np.nan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--n-obs', type=int, default=3000)
    p.add_argument('--n-assets', type=int, default=20)
    p.add_argument('--window', type=int, default=252)
    p.add_argument('--halflife', type=float, default=60.0)
    p.add_argument('--corr', type=float, default=0.35)
    p.add_argument('--event-vol', type=float, default=5.0)
    p.add_argument('--event-corr', type=float, default=0.85)
    p.add_argument('--event-len', type=int, default=1)
    p.add_argument('--threshold', type=float, default=4.0)
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args()

    kw = dict(window=args.window, method='ledoit_wolf', jump_method='ewma',
              halflife=args.halflife, shrinkage=0.10, shape_estimator='hyvarinen')

    event_at = args.window + 600
    shocked = simulate(args.n_obs, args.n_assets, args.corr, args.seed,
                       event_at, args.event_vol, args.event_corr, args.event_len)
    clean = simulate(args.n_obs, args.n_assets, args.corr, args.seed,
                     None, 1.0, args.corr)

    ent_s = rolling_measures(shocked, **kw)
    ent_c = rolling_measures(clean, **kw)

    pos = ent_s.index.get_loc(shocked.index[event_at])
    baseline = float(ent_s['J'].iloc[max(0, pos - 120):pos].mean())
    hl_mech = decay_half_life(ent_s['J'], pos, baseline)

    j_s = detect_jumps(ent_s['J'], args.threshold, 250)
    j_c = detect_jumps(ent_c['J'], args.threshold, 250)
    rel_s = estimate_relaxation(ent_s['J'], j_s)
    rel_c = estimate_relaxation(ent_c['J'], j_c)

    n_c = int(j_c['z'].notna().sum())
    print('=== Estimator memory (i.i.d. data, single injected event) ===')
    print(f'window={args.window}, ewma halflife={args.halflife}, '
          f'N={args.n_assets}, corr={args.corr}')
    print(f'peak jump in J at the event      : '
          f"{float(ent_s['J'].iloc[pos:pos + 5].max() - baseline):.3f} nats")
    print(f'MECHANICAL decay half-life       : {hl_mech:.1f} trading days')
    print(f'OU half-life, shocked series     : {rel_s["half_life_days"]:.1f} days '
          f'(kappa={rel_s["kappa"]:.5f})')
    print()
    print('=== Control: clean i.i.d. data, no event ===')
    print(f'OU half-life                     : {rel_c["half_life_days"]:.1f} days '
          f'(slope p={rel_c["slope_pvalue"]:.3f})')
    print(f'false-positive jumps at {args.threshold} sigma : '
          f"{int(j_c['is_jump'].sum())} of {n_c} dates "
          f"({100 * int(j_c['is_jump'].sum()) / max(n_c, 1):.2f}%), "
          f"up={int(j_c['jump_up'].sum())}, down={int(j_c['jump_down'].sum())}")
    print(f"skew of dJ on clean data         : {float(ent_c['dJ'].skew()):.3f}")


if __name__ == '__main__':
    main()
