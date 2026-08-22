from pathlib import Path
import argparse
import yaml
import matplotlib.pyplot as plt
from .empirical import main, run_regressions


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    logret, ent = main(cfg)
    out_t = Path(cfg['outputs']['tables']); out_f = Path(cfg['outputs']['figures'])
    out_t.mkdir(parents=True, exist_ok=True); out_f.mkdir(parents=True, exist_ok=True)

    ent.to_csv(out_t / 'rolling_entropy.csv')
    summary = ent[['h_vol','h_dep','h_cov','dh_vol','dh_dep','dh_cov','comp_gap']].describe().T
    summary.to_csv(out_t / 'summary.csv')

    q = ent['stress_signal'].quantile(cfg['regimes']['stress_quantile'])
    ent['stress'] = ent['stress_signal'] >= q
    regime = ent.groupby('stress')[['h_vol','h_dep','h_cov','dh_vol','dh_dep','dh_cov']].mean()
    regime.to_csv(out_t / 'regime_summary.csv')

    reg = run_regressions(ent)
    reg.to_csv(out_t / 'predictive_regressions.csv', index=False)

    ax = ent[['h_vol','h_dep','h_cov']].plot(figsize=(11,6), title='Covariance entropy decomposition')
    ax.set_ylabel('log-scale entropy components')
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(out_f/'entropy_components.png', dpi=160); plt.close(fig)

    ax = ent.plot.scatter(x='dh_vol', y='dh_dep', figsize=(7,6), alpha=.35, title='Volatility–dependence compensation')
    ax.axhline(0); ax.axvline(0); fig=ax.get_figure(); fig.tight_layout(); fig.savefig(out_f/'compensation_scatter.png', dpi=160); plt.close(fig)

    ax = ent['h_cov'].plot(figsize=(11,5), title='0.5 log determinant of covariance')
    ax.set_ylabel('0.5 log det(Sigma)'); fig=ax.get_figure(); fig.tight_layout(); fig.savefig(out_f/'logdet_covariance.png', dpi=160); plt.close(fig)

    print('Empirical pipeline completed.')
    print(reg.to_string(index=False))

if __name__ == '__main__':
    run()
