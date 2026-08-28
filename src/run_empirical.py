"""End-to-end run: measures, hypothesis tests, stress ladder, paper figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .empirical import conditional_state, main
from .hypotheses import run_all
from .stress import (
    build_scenario_table,
    classical_scenario,
    return_period_of,
    reverse_stress,
    severity_ladder,
)

plt.rcParams.update({
    'figure.dpi': 160,
    'savefig.dpi': 160,
    'font.size': 9,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_figures(ent: pd.DataFrame, res: dict, scen: pd.DataFrame, out_f: Path,
                 label: str) -> None:
    jumps = res['jumps']

    # 1. The three channels.
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    axes[0].plot(ent.index, ent['h_vol'], lw=0.8, color='#1f77b4')
    axes[0].set_ylabel(r'$H^{vol}$')
    axes[0].set_title(f'{label}: the three entropy channels')
    axes[1].plot(ent.index, ent['h_dep'], lw=0.8, color='#d62728')
    axes[1].set_ylabel(r'$H^{dep}$')
    axes[2].plot(ent.index, ent['d_total'], lw=0.8, color='#2ca02c')
    axes[2].set_ylabel(r'$\mathcal{D}$ (deficit)')
    axes[2].set_xlabel('')
    _save(fig, out_f / 'channels.png')

    # 2. Is any of this just volatility in disguise?
    d = ent.dropna(subset=['stress_signal', 'J', 'd_total', 'h_dep_ew'])
    lv = np.log(d['stress_signal'])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.7))
    for ax, (col, name, sign) in zip(axes, [
            ('h_vol', r'$H^{vol}$ (scale channel)', 1),
            ('h_dep_ew', r'$-H^{dep}$ (dependence channel)', -1),
            ('d_total', r'$\mathcal{D}$ (shape channel)', 1)]):
        y = sign * d[col]
        ax.scatter(lv, y, s=2, alpha=0.15, color='#1f77b4')
        ax.set_xlabel('log trailing volatility')
        ax.set_title(f'{name}\ncorr = {lv.corr(y):+.2f}', fontsize=9)
    fig.suptitle(f'{label}: what each channel shares with volatility', y=1.02)
    _save(fig, out_f / 'channels_vs_volatility.png')

    # 3. Structural index with detected jumps.
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(ent.index, ent['J'], lw=0.8, color='#333333', label=r'$\mathcal{J}_t$')
    up = jumps['jump_up'].reindex(ent.index).fillna(False)
    dn = jumps['jump_down'].reindex(ent.index).fillna(False)
    ax.scatter(ent.index[up], ent['J'][up], s=16, color='#d62728', zorder=3,
               label=f'up-jumps (n={int(up.sum())})')
    ax.scatter(ent.index[dn], ent['J'][dn], s=16, color='#1f77b4', zorder=3,
               label=f'down-jumps (n={int(dn.sum())})')
    ax.set_ylabel('nats')
    ax.legend(frameon=False, ncol=3)
    ax.set_title(f'{label}: structural index and its jumps')
    _save(fig, out_f / 'structural_index_jumps.png')

    # 4. Jump asymmetry: histogram of standardised changes.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    z = jumps['z'].dropna()
    ax.hist(z, bins=140, color='#7f7f7f', alpha=0.85)
    ax.set_yscale('log')
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel(r'standardised $\Delta\mathcal{J}_t$')
    ax.set_ylabel('count (log)')
    ax.set_title(f'{label}: skew={z.skew():.2f}')
    _save(fig, out_f / 'jump_asymmetry.png')

    # 5. Event study around up-jumps.
    ev = res['h4_event_study']
    if not ev.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.fill_between(ev.index, ev['q25'], ev['q75'], alpha=0.2, color='#d62728')
        ax.plot(ev.index, ev['mean'], color='#d62728', lw=1.4)
        ax.axvline(0, color='k', lw=0.6)
        ax.axhline(0, color='k', lw=0.6)
        hl = res['h4_relaxation']['half_life_days'].iloc[0]
        if np.isfinite(hl):
            ax.axvline(hl, color='#1f77b4', ls='--', lw=1.0,
                       label=f'OU half-life = {hl:.0f} d')
            ax.legend(frameon=False)
        ax.set_xlabel('trading days from jump')
        ax.set_ylabel(r'$\mathcal{J}_t-\mathcal{J}_0$ (nats)')
        ax.set_title(f'{label}: jump then relaxation (n={int(ev["n_events"].iloc[0])})')
        _save(fig, out_f / 'relaxation_event_study.png')

    # 6. Compensation scatter (legacy H2 figure, retained).
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.scatter(ent['dh_vol'], ent['dh_dep'], s=4, alpha=0.25, color='#1f77b4')
    ax.axhline(0, color='k', lw=0.6)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel(r'$\Delta H^{vol}$')
    ax.set_ylabel(r'$\Delta H^{dep}$')
    c = ent[['dh_vol', 'dh_dep']].corr().iloc[0, 1]
    ax.set_title(f'{label}: compensation, corr={c:.2f}')
    _save(fig, out_f / 'compensation_scatter.png')

    # 7. Skewness signature.
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(ent.index, ent['skew_market'], lw=0.8, color='#9467bd')
    ax.scatter(ent.index[up], ent['skew_market'][up], s=16, color='#d62728', zorder=3)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_ylabel('market skewness')
    ax.set_title(f'{label}: market skewness, entropy up-jumps marked')
    _save(fig, out_f / 'skewness_signature.png')

    # 8. Stress ladder.
    if not scen.empty:
        lad = scen[scen['return_period_years'] > 0]
        es_col = [c for c in scen.columns if c.startswith('stressed_ES')][0]
        base_es = float(scen.iloc[0][es_col])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
        axes[0].plot(lad['eta_nats'], lad[es_col] * 100, 'o-', color='#d62728')
        axes[0].axhline(base_es * 100, color='#1f77b4', ls=':', lw=1,
                        label='today')
        axes[0].set_xlabel(r'scenario radius $\eta$ (nats)')
        axes[0].set_ylabel(f'stressed {es_col[9:]} (% daily)')
        axes[0].set_title('severity ladder')
        axes[0].legend(frameon=False)
        for _, r in lad.iterrows():
            axes[0].annotate(f"{int(r['return_period_years'])}y",
                             (r['eta_nats'], r[es_col] * 100),
                             textcoords='offset points', xytext=(4, 4), fontsize=7)
        axes[1].plot(lad['eta_nats'], lad['vol_multiplier'], 'o-',
                     label='volatility multiplier')
        axes[1].plot(lad['eta_nats'], -lad['sigma_move'], 'o-',
                     label=r'directional shock ($-\sigma$)')
        axes[1].set_xlabel(r'scenario radius $\eta$ (nats)')
        axes[1].set_title('what one nat buys')
        axes[1].legend(frameon=False)
        _save(fig, out_f / 'stress_ladder.png')


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml')
    parser.add_argument('--label', default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    label = args.label or cfg['data'].get('type', 'dataset')

    logret, ent = main(cfg)
    out_t = Path(cfg['outputs']['tables'])
    out_f = Path(cfg['outputs']['figures'])
    out_t.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

    ent.to_csv(out_t / 'rolling_measures.csv')
    ent.describe().T.to_csv(out_t / 'summary.csv')

    q = ent['stress_signal'].quantile(cfg['regimes']['stress_quantile'])
    ent['stress'] = ent['stress_signal'] >= q

    res = run_all(ent, ent['stress'], cfg)
    for name, table in res.items():
        if isinstance(table, pd.DataFrame) and not table.empty:
            table.to_csv(out_t / f'{name}.csv')

    kappa = float(res['h4_relaxation']['kappa'].iloc[0])
    ent = ent.join(res['indicators'][['crisis', 'unhealed']])

    # --- Stress scenarios -------------------------------------------------
    scfg = cfg.get('stress_test', {})
    horizon = int(scfg.get('horizon', 21))
    var_level = float(scfg.get('var_level', 0.99))
    flow = ent['info_flow_h'].dropna()

    ladder = severity_ladder(
        flow,
        return_periods=tuple(scfg.get('return_periods', (1, 2, 5, 10, 20, 50))),
        horizon=horizon,
    )
    ladder['eta_per_asset'] = ladder['eta_nats'] / logret.shape[1]
    ladder.to_csv(out_t / 'severity_ladder.csv', index=False)

    est = cfg['estimation']
    mu0, cov0 = conditional_state(
        logret, cfg['sample']['window'],
        method=est.get('jump_covariance', 'ewma'),
        halflife=float(est.get('ewma_halflife', 60)),
        shrinkage=float(est.get('ewma_shrinkage', 0.10)),
    )
    n_assets = logret.shape[1]
    w_port = np.full(n_assets, 1.0 / n_assets)

    scen = build_scenario_table(mu0, cov0, w_port, ladder, risk_level=var_level)
    scen['return_period_check'] = [
        return_period_of(e, flow, horizon) for e in scen['eta_nats']]
    scen.to_csv(out_t / 'stress_scenarios.csv', index=False)

    # Price a textbook bundle on the same scale.
    classical_rows = []
    for vm, tc, sm_ in [(1.5, 0.70, -2.0), (2.0, 0.90, -3.0), (3.0, 0.95, -4.0)]:
        c = classical_scenario(mu0, cov0, w_port, vm, tc, sm_, var_level)
        c['implied_return_period_years'] = return_period_of(
            c['entropy_price_nats'], flow, horizon)
        classical_rows.append(c)
    classical = pd.DataFrame(classical_rows)
    classical.to_csv(out_t / 'classical_comparison.csv', index=False)

    # Reverse stress test on the empirical loss distribution.
    R = logret.to_numpy()
    port_loss = -(R @ w_port)
    worst_hist = float(np.max(port_loss))
    rev = pd.DataFrame([
        reverse_stress(port_loss, q) for q in
        [np.quantile(port_loss, 0.99), np.quantile(port_loss, 0.999), worst_hist]
    ])
    rev['label'] = ['loss = empirical VaR99', 'loss = empirical VaR99.9',
                    'loss = worst day in sample']
    rev.to_csv(out_t / 'reverse_stress.csv', index=False)

    make_figures(ent, res, scen, out_f, label)

    summary = {
        'label': label,
        'n_dates': int(len(ent)),
        'n_assets': int(logret.shape[1]),
        'sample': [str(ent.index.min().date()), str(ent.index.max().date())],
        'kappa': kappa,
        'half_life_days': float(res['h4_relaxation']['half_life_days'].iloc[0]),
        'n_jump_up': int(res['h3_jump_asymmetry']['n_jump_up'].iloc[0]),
        'n_jump_down': int(res['h3_jump_asymmetry']['n_jump_down'].iloc[0]),
        'skew_of_dJ': float(res['h3_jump_asymmetry']['skew_of_changes'].iloc[0]),
    }
    (out_t / 'run_summary.json').write_text(json.dumps(summary, indent=2))

    print(f'--- {label}: {summary["n_dates"]} dates, {summary["n_assets"]} assets ---')
    for name in ['h1_regime_signature', 'h2_ceiling_compensation', 'h3_jump_asymmetry',
                 'h4_relaxation', 'h5_skewness_signature', 'h6_tail_dependence_coupling',
                 'h7_incremental_value']:
        print(f'\n== {name} ==')
        print(res[name].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\n== severity ladder ==')
    print(ladder.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\n== stress scenarios ==')
    print(scen.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\n== classical comparison ==')
    print(classical.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\n== reverse stress ==')
    print(rev.to_string(index=False, float_format=lambda v: f'{v:.4f}'))


if __name__ == '__main__':
    run()
