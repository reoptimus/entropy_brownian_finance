"""Scale versus structure: what crises and stress scenarios do to each channel,
and what each channel costs in nats.

Outputs (in ``--out``, default ``paper/tables``):

* ``scale_structure_items.csv`` -- for observed crises (stress-minus-calm
  regime gap, largest jump episodes) and for the stress scenarios of the
  paper: the change in the scale channel ``sum log sigma_i`` and in the
  structural index ``J = C + D`` split into ``C`` and ``D``; for scenarios,
  the exact split of the deformation cost (src.scale_split).
* ``scale_structure_exchange.csv`` -- per panel, the small-shock efficiencies
  of the volatility and correlation channels, the correlation ceiling, and
  the exact cheapest cost in each channel for a grid of volatility targets.
* ``paper/jfec/figures/fig_scale_structure{,_fr}.png``.

Usage::

    python -m scripts.scale_structure
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.empirical import conditional_state, main
from src.jumps import detect_jumps, group_jump_episodes
from src.regimes import past_only_stress
from src.scale_split import (
    channel_efficiencies, cheapest_corr_only, cheapest_vol_only, correlation_ceiling,
    gaussian_deformation, lift_cost, minimal_lift_cov, split_deformation,
)
from src.stress import calibrated_covariance

PANELS = {'sp500': 'config/sp500.yaml', 'ff49': 'config/default.yaml'}
V_GRID = (1.1, 1.25, 1.5, 2.0, 3.0)


def channels(row) -> dict:
    return {'scale': float(row['h_vol_ew']), 'C': float(-row['h_dep_ew']),
            'D': float(row['d_total']), 'J': float(row['J'])}


def observed_items(panel: str, cfg: dict, ent: pd.DataFrame, n_top: int) -> list:
    out = []
    stress, valid = past_only_stress(ent['stress_signal'], cfg['regimes']['stress_quantile'])
    e = ent.loc[valid]
    st = stress.loc[valid].astype(bool)
    hi = e.loc[st, ['h_vol_ew', 'h_dep_ew', 'd_total', 'J']].mean()
    lo = e.loc[~st, ['h_vol_ew', 'h_dep_ew', 'd_total', 'J']].mean()
    a, b = channels(hi), channels(lo)
    out.append({'panel': panel, 'kind': 'observed', 'item': 'regime_gap',
                **{f'd_{k}': a[k] - b[k] for k in a}})

    jc = cfg.get('jumps', {})
    jumps = detect_jumps(ent['J'], float(jc.get('threshold', 4.0)),
                         int(jc.get('scale_window', 250)))
    episodes = group_jump_episodes(jumps, gap=40, direction='up')
    pos = {d: i for i, d in enumerate(ent.index)}
    rows = []
    for start, end, _ in episodes:
        i0 = max(0, pos[start] - 10)
        i1 = min(len(ent) - 1, pos[end] + 20)
        win = ent.iloc[pos[start]:i1 + 1]
        if win['J'].isna().all():
            continue
        peak = win['J'].idxmax()
        a, b = channels(ent.loc[peak]), channels(ent.iloc[i0])
        rows.append({'panel': panel, 'kind': 'observed',
                     'item': f'episode_{pd.Timestamp(start):%Y-%m}',
                     'peak': f'{pd.Timestamp(peak):%Y-%m-%d}',
                     **{f'd_{k}': a[k] - b[k] for k in a}})
    rows = sorted(rows, key=lambda r: -r['d_J'])[:n_top]
    return out + rows


def scenario_row(panel, item, cov1, cov0, w) -> dict:
    sp = split_deformation(cov1, cov0)
    v = float(np.sqrt(w @ cov1 @ w / (w @ cov0 @ w)))
    return {'panel': panel, 'kind': 'scenario', 'item': item, 'v': v,
            'd_scale': sp['d_scale'], 'd_C': sp['d_coupling'], 'd_D': 0.0,
            'd_J': sp['d_coupling'], 'cost_total': sp['total'],
            'cost_scale': sp['K_scale'], 'cost_corr': sp['K_corr'],
            'cost_interaction': sp['interaction'], 'cost_min_lift': lift_cost(v)}


def scenario_items(panel, cov0, w, s10) -> list:
    n = len(w)
    sd0 = np.sqrt(np.diag(cov0))
    rows = []
    for m, rho in [(1.5, 0.70), (2.0, 0.90), (3.0, 0.95)]:
        R1 = np.full((n, n), rho)
        np.fill_diagonal(R1, 1.0)
        cov1 = np.outer(m * sd0, m * sd0) * R1
        rows.append(scenario_row(panel, f'classical_x{m:g}_rho{rho:.2f}', cov1, cov0, w))
    v_std = rows[1]['v']
    rows.append(scenario_row(panel, 'uniform_dilation_same_v', v_std ** 2 * cov0, cov0, w))
    rows.append(scenario_row(panel, 'minimal_lift_same_v',
                             minimal_lift_cov(cov0, w, v_std), cov0, w))
    if s10 is not None:
        for share in (0.0, 0.14, 0.49):
            cov1, _ = calibrated_covariance(cov0, w, s10, share)
            rows.append(scenario_row(panel, f'ten_year_share{share:.2f}', cov1, cov0, w))
        v10 = s10 / np.sqrt(w @ cov0 @ w)
        rows.append(scenario_row(panel, 'ten_year_minimal_lift',
                                 minimal_lift_cov(cov0, w, v10), cov0, w))
    return rows


def exchange_rows(panel, cov0, w) -> list:
    eff = channel_efficiencies(cov0, w)
    ceil = correlation_ceiling(cov0, w)
    rows = [{'panel': panel, 'v': np.nan, 'quantity': k, 'value': val}
            for k, val in {**eff, 'corr_ceiling_v': ceil}.items()]
    for v in V_GRID:
        lift = lift_cost(v)
        vol = cheapest_vol_only(cov0, w, v)
        cor = cheapest_corr_only(cov0, w, v)
        sp = split_deformation(minimal_lift_cov(cov0, w, v), cov0)
        vals = {'cost_min_lift': lift,
                'cost_vol_only': vol['cost'],
                'cost_uniform': len(w) * lift,
                'cost_corr_only': cor['cost'],
                'lift_d_scale': sp['d_scale'], 'lift_d_C': sp['d_coupling'],
                'lift_K_scale': sp['K_scale'], 'lift_K_corr': sp['K_corr'],
                'lift_interaction': sp['interaction']}
        rows += [{'panel': panel, 'v': v, 'quantity': k, 'value': val}
                 for k, val in vals.items()]
    return rows


LABELS = {
    'en': {
        'regime_gap': 'stress $-$ calm',
        'episode': 'episode',
        'classical_x1.5_rho0.70': 'classical mild\n(vol $\\times$1.5, $\\rho$=0.70)',
        'classical_x2_rho0.90': 'classical standard\n(vol $\\times$2, $\\rho$=0.90)',
        'classical_x3_rho0.95': 'classical severe\n(vol $\\times$3, $\\rho$=0.95)',
        'uniform_dilation_same_v': 'uniform dilation\n(same loss as standard)',
        'minimal_lift_same_v': 'minimal lift\n(same loss as standard)',
        'scale': r'$\Delta\,\Sigma_i\log\sigma_i$ (scale)',
        'C': r'$\Delta C$ (dependence)', 'D': r'$\Delta D$ (shape)',
        'ta': 'A. Observed crises', 'tb': 'B. Stress scenarios (S&P 500 panel)',
        'xl': 'change (nats)', 'cost': 'deformation cost', 'sp500': 'S&P', 'ff49': 'FF49',
    },
    'fr': {
        'regime_gap': 'stress $-$ calme',
        'episode': 'épisode',
        'classical_x1.5_rho0.70': 'classique modéré\n(vol $\\times$1,5, $\\rho$=0,70)',
        'classical_x2_rho0.90': 'classique standard\n(vol $\\times$2, $\\rho$=0,90)',
        'classical_x3_rho0.95': 'classique sévère\n(vol $\\times$3, $\\rho$=0,95)',
        'uniform_dilation_same_v': 'dilatation uniforme\n(même perte que standard)',
        'minimal_lift_same_v': 'relèvement minimal\n(même perte que standard)',
        'scale': r'$\Delta\,\Sigma_i\log\sigma_i$ (échelle)',
        'C': r'$\Delta C$ (dépendance)', 'D': r'$\Delta D$ (forme)',
        'ta': 'A. Crises observées', 'tb': 'B. Scénarios de stress (panel S&P 500)',
        'xl': 'variation (nats)', 'cost': 'coût de déformation', 'sp500': 'S&P', 'ff49': 'FF49',
    },
}


def _fmt(x, lang):
    s = f'{x:.1f}'
    return s.replace('.', ',') if lang == 'fr' else s


def figure(items: pd.DataFrame, path: Path, lang: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    L = LABELS[lang]
    c_scale, c_C, c_D = '#8c8c8c', '#2b6cb0', '#dd8452'
    obs = items[items.kind == 'observed']
    obs_rows = []
    for panel in ('sp500', 'ff49'):
        sub = obs[obs.panel == panel]
        for _, r in sub.iterrows():
            if r['item'] == 'regime_gap':
                lab = f"{L[panel]}: {L['regime_gap']}"
            else:
                lab = f"{L[panel]}: {L['episode']} {r['peak'][:7]}"
            obs_rows.append((lab, r))
    sc = items[(items.kind == 'scenario') & (items.panel == 'sp500')
               & items['item'].isin([k for k in LABELS['en'] if k.startswith(('classical', 'uniform', 'minimal'))])]
    sc_rows = [(L[r['item']], r) for _, r in sc.iterrows()]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.6),
                             gridspec_kw={'height_ratios': [len(obs_rows), len(sc_rows) + 0.6]})
    for ax, rows, title, show_cost in [(axes[0], obs_rows, L['ta'], False),
                                       (axes[1], sc_rows, L['tb'], True)]:
        y = np.arange(len(rows))[::-1]
        h = 0.36
        for yi, (lab, r) in zip(y, rows):
            ax.barh(yi + h / 2, r['d_scale'], height=h, color=c_scale)
            c, d = r['d_C'], r['d_D']
            ax.barh(yi - h / 2, c, height=h, color=c_C)
            left = c if (c >= 0) == (d >= 0) else 0.0
            ax.barh(yi - h / 2, d, left=left, height=h, color=c_D)
            if show_cost:
                xm = max(r['d_scale'], c + max(d, 0), 0)
                ax.text(xm + 0.4, yi, f"{L['cost']} {_fmt(r['cost_total'], lang)} nats",
                        va='center', fontsize=8, color='#333333')
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows], fontsize=8)
        ax.axvline(0, color='black', lw=0.6)
        ax.set_title(title, loc='left', fontsize=10)
        ax.set_xlabel(L['xl'], fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    xmax = max(axes[1].get_xlim()[1], 34)
    axes[1].set_xlim(right=xmax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (c_scale, c_C, c_D)]
    fig.legend(handles, [L['scale'], L['C'], L['D']], loc='lower center', ncol=3,
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if lang == 'fr':
        import matplotlib.ticker as mt
        for ax in axes:
            ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda x, _: f'{x:g}'.replace('.', ',')))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run(out: Path, figdir: Path, n_top: int) -> None:
    items, exch = [], []
    for panel, cfg_path in PANELS.items():
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        logret, ent = main(cfg)
        items += observed_items(panel, cfg, ent, n_top)
        est = cfg['estimation']
        _, cov0 = conditional_state(
            logret, cfg['sample']['window'], method=est.get('jump_covariance', 'ewma'),
            halflife=float(est.get('ewma_halflife', 60)),
            shrinkage=float(est.get('ewma_shrinkage', 0.10)))
        w = np.full(logret.shape[1], 1.0 / logret.shape[1])
        s10 = None
        f = Path(f'paper/tables/{panel}_stress_scenarios.csv')
        if f.exists():
            t = pd.read_csv(f).set_index('return_period_years')
            s10 = float(t.loc[10, 'stressed_vol_daily'])
        items += scenario_items(panel, cov0, w, s10)
        exch += exchange_rows(panel, cov0, w)
    items = pd.DataFrame(items)
    exch = pd.DataFrame(exch)
    out.mkdir(parents=True, exist_ok=True)
    items.to_csv(out / 'scale_structure_items.csv', index=False)
    exch.to_csv(out / 'scale_structure_exchange.csv', index=False)
    figdir.mkdir(parents=True, exist_ok=True)
    figure(items, figdir / 'fig_scale_structure.png', 'en')
    figure(items, figdir / 'fig_scale_structure_fr.png', 'fr')
    pd.set_option('display.width', 200)
    print(items.round(3).to_string())
    print(exch.pivot_table(index=['panel', 'quantity'], columns='v', values='value',
                           dropna=False).round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='paper/tables')
    ap.add_argument('--figdir', default='paper/jfec/figures')
    ap.add_argument('--n-top', type=int, default=3)
    a = ap.parse_args()
    run(Path(a.out), Path(a.figdir), a.n_top)
