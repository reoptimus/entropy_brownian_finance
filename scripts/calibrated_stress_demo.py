"""Demo: a stress scenario composed the way H1 says crises are, not the
simplest way a KL ball can reach the same severity.

:func:`src.stress.gaussian_ball_scenario` only constrains the portfolio's own
projected distribution; left unconstrained, the simplest covariance choice
consistent with that is pure uniform scaling, zero correlation change -- a
default, not a claim about how real crises behave. H1 (the paper's most
robust finding: it clears *both* the i.i.d. and the block-21 null, on every
panel) measures how real crises actually behave -- the dependence channel
moves on its own. This script prices four scenarios of the *same* severity
and shows what the H1-consistent composition costs relative to the default:

1. default        -- dependence_share = 0, the ball's own simplest answer.
2. H1-calibrated   -- dependence_share = the regime-average split H1 actually
                      measures (stress-minus-calm gap in h_dep vs h_vol).
3. H8-refined     -- dependence_share = the same split measured only on the
                      most severe detected episodes (the per-episode channel
                      budget of src.jumps, i.i.d.-null-tested in REPORT.md).
4. classical      -- a textbook committee bundle (vol x2, corr -> 0.90), for
                      scale: how expensive an ungrounded guess turns out to be.

Usage::

    python -m scripts.calibrated_stress_demo --config config/sp500.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.empirical import conditional_state, main
from src.hypotheses import h1_regime_signature
from src.jumps import detect_jumps, episode_channel_budget, group_jump_episodes
from src.stress import (
    classical_scenario,
    composition_calibrated_scenario,
    return_period_of,
    severity_ladder,
)


def h1_dependence_share(ent: pd.DataFrame, stress: pd.Series) -> float:
    """The regime-average dependence share H1 measures: how much of the
    stress-minus-calm entropy gap sits in h_dep versus h_vol, in absolute
    terms. This is H1's own dual-null-cleared statistic, not a new one."""
    h1 = h1_regime_signature(ent, stress).set_index('channel')
    d_dep = abs(float(h1.loc['h_dep', 'diff']))
    d_vol = abs(float(h1.loc['h_vol', 'diff']))
    return d_dep / (d_dep + d_vol) if (d_dep + d_vol) > 0 else 0.5


def h8_severe_episode_share(ent: pd.DataFrame, threshold: float, scale_window: int,
                            gap: int = 40, pre: int = 10, post_search: int = 20,
                            top_frac: float = 0.5) -> tuple:
    """Dependence share measured only on the most severe half of detected
    episodes (ranked by their move in J) -- the severity-conditioned
    refinement of H8, described but not paper-fixed as of REPORT.md."""
    jumps = detect_jumps(ent['J'], threshold, scale_window)
    episodes = group_jump_episodes(jumps, gap=gap, direction='up')
    budget = episode_channel_budget(ent, episodes, pre=pre, post_search=post_search)
    if budget.empty:
        return 0.5, 0
    severe = budget.sort_values('dJ', ascending=False)
    n_top = max(1, int(np.ceil(top_frac * len(severe))))
    top = severe.iloc[:n_top]
    return float(top['share_dependence'].mean()), int(len(top))


def run(config: str) -> None:
    cfg = yaml.safe_load(Path(config).read_text())
    logret, ent = main(cfg)

    q = ent['stress_signal'].quantile(cfg['regimes']['stress_quantile'])
    stress = ent['stress_signal'] >= q

    jcfg = cfg.get('jumps', {})
    threshold = float(jcfg.get('threshold', 4.0))
    scale_window = int(jcfg.get('scale_window', 250))

    share_h1 = h1_dependence_share(ent, stress)
    share_h8, n_severe = h8_severe_episode_share(ent, threshold, scale_window)

    est = cfg['estimation']
    mu0, cov0 = conditional_state(
        logret, cfg['sample']['window'],
        method=est.get('jump_covariance', 'ewma'),
        halflife=float(est.get('ewma_halflife', 60)),
        shrinkage=float(est.get('ewma_shrinkage', 0.10)))
    n_assets = logret.shape[1]
    w_port = np.full(n_assets, 1.0 / n_assets)

    scfg = cfg.get('stress_test', {})
    horizon = int(scfg.get('horizon', 21))
    var_level = float(scfg.get('var_level', 0.99))
    flow = ent['info_flow_h'].dropna()
    rp = 10
    ladder = severity_ladder(flow, return_periods=(rp,), horizon=horizon)
    eta = float(ladder.iloc[0]['eta_nats'])

    print(f'--- {cfg["data"]["type"]}: severity calibration for a {rp}-year scenario '
          f'(eta = {eta:.2f} nats, {n_assets} assets) ---\n')
    print(f'H1 regime-average dependence share : {share_h1:.2f}')
    print(f'H8 severe-episode dependence share  : {share_h8:.2f}  '
          f'(top {n_severe} of the detected episodes)\n')

    rows = []
    for label, share in [('default (KL-ball, pure scaling)', 0.0),
                         ('H1-calibrated', share_h1),
                         ('H8-refined (severe episodes)', share_h8)]:
        sc = composition_calibrated_scenario(mu0, cov0, w_port, eta, share,
                                             risk_level=var_level)
        rows.append({
            'scenario': label,
            'dependence_share': share,
            f'stressed_ES{int(var_level*100)}': sc[f'stressed_ES{int(var_level*100)}'],
            'stressed_vol_ann': sc['stressed_vol_ann'],
            'entropy_price_nats': sc['entropy_price_nats'],
            'excess_cost_over_default_nats': sc['excess_cost_over_default_nats'],
            'implied_return_period_years': return_period_of(
                sc['entropy_price_nats'], flow, horizon),
            'feasible_at_this_share': sc['feasible_at_this_share'],
        })

    classical = classical_scenario(mu0, cov0, w_port,
                                   vol_multiplier=scfg.get('classical_vol_multiplier', 2.0),
                                   target_correlation=scfg.get('classical_correlation', 0.90),
                                   sigma_move=-3.0, risk_level=var_level)
    rows.append({
        'scenario': 'classical committee bundle (vol x2, corr 0.90)',
        'dependence_share': np.nan,
        f'stressed_ES{int(var_level*100)}': classical[f'stressed_ES{int(var_level*100)}'],
        'stressed_vol_ann': classical['stressed_vol_ann'],
        'entropy_price_nats': classical['entropy_price_nats'],
        'excess_cost_over_default_nats': np.nan,
        'implied_return_period_years': return_period_of(
            classical['entropy_price_nats'], flow, horizon),
        'feasible_at_this_share': np.nan,
    })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda x: f'{x:.3f}'))

    dest = Path(cfg['outputs']['tables']) / 'calibrated_stress_demo.csv'
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f'\nWritten to {dest}')


def main_cli() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/sp500.yaml')
    args = p.parse_args()
    run(args.config)


if __name__ == '__main__':
    main_cli()
