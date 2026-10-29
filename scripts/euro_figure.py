"""Figure of the euro-area entropic CISS (phase B).

Reads ``paper/tables/euro_block_coupling_series{tag}.csv`` written by
``scripts/euro_ciss.py`` and the ECB CISS from ``data/zone_euro/reference.csv``.

Usage::

    python -m scripts.euro_figure [--overnight-only] [--lang en]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from scripts.euro_ciss import CEPR_RECESSIONS

PAIRS = ['mi_obligataire__intermediaires', 'mi_obligataire__actions',
         'mi_actions__intermediaires', 'mi_intermediaires__change']
LABELS = {
    'fr': {'mi_obligataire__intermediaires': 'Obligations – intermédiaires',
           'mi_obligataire__actions': 'Obligations – actions',
           'mi_actions__intermediaires': 'Actions – intermédiaires',
           'mi_intermediaires__change': 'Intermédiaires – change',
           'total': 'Couplage total (rangs)', 'ciss': 'CISS BCE', 'sd': 'écart-type',
           't0': 'CISS entropique zone euro : couplage total et CISS (centrés réduits)',
           'inter': 'Entre segments (nats)', 'share': 'Part entre segments', 'part': 'part',
           't1': 'Couplage entre les cinq segments du CISS',
           't2': 'Carte de contagion : information mutuelle entre paires de segments',
           'note': 'Zones grises : récessions CEPR. Fenêtre glissante de 252 jours, '
                   'scores normaux, Ledoit-Wolf.'},
    'en': {'mi_obligataire__intermediaires': 'Bonds – intermediaries',
           'mi_obligataire__actions': 'Bonds – equities',
           'mi_actions__intermediaires': 'Equities – intermediaries',
           'mi_intermediaires__change': 'Intermediaries – FX',
           'total': 'Total coupling (ranks)', 'ciss': 'ECB CISS', 'sd': 'std. dev.',
           't0': 'Euro-area entropic CISS: total coupling and CISS (standardised)',
           'inter': 'Between segments (nats)', 'share': 'Between-segment share',
           'part': 'share', 't1': 'Coupling between the five CISS segments',
           't2': 'Contagion map: mutual information between pairs of segments',
           'note': 'Grey: CEPR recessions. Rolling 252-day window, normal scores, '
                   'Ledoit-Wolf.'},
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/zone_euro')
    p.add_argument('--overnight-only', action='store_true')
    p.add_argument('--lang', choices=('fr', 'en'), default='fr')
    args = p.parse_args()
    tag = '_overnight_only' if args.overnight_only else ''
    L = LABELS[args.lang]
    dec = pd.read_csv(f'paper/tables/euro_block_coupling_series{tag}.csv',
                      index_col=0, parse_dates=True)
    ref = pd.read_csv(Path(args.data) / 'reference.csv', index_col=0, parse_dates=True)
    ciss = ref['ciss_zone_euro'].dropna().loc[dec.index.min():]

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    z = lambda s: (s - s.mean()) / s.std()
    ax[0].plot(z(dec['total']), lw=0.9, label=L['total'])
    ax[0].plot(z(ciss.resample('W-FRI').last()), lw=0.9, alpha=0.8, label=L['ciss'])
    ax[0].set_ylabel(L['sd'])
    ax[0].set_title(L['t0'])
    ax[1].plot(dec['inter'], lw=0.9, color='C2', label=L['inter'])
    ax1b = ax[1].twinx()
    ax1b.plot(dec['inter'] / dec['total'], lw=0.7, color='C3', alpha=0.7,
              label=L['share'])
    ax[1].set_ylabel('nats')
    ax1b.set_ylabel(L['part'])
    ax[1].set_title(L['t1'])
    for k in PAIRS:
        ax[2].plot(dec[k], lw=0.9, label=L[k])
    ax[2].set_ylabel('nats')
    ax[2].set_title(L['t2'])
    for a in ax:
        for s, e in CEPR_RECESSIONS:
            a.axvspan(pd.Timestamp(s), pd.Timestamp(e), color='grey', alpha=0.15, lw=0)
        a.legend(loc='upper left', fontsize=8, frameon=False)
    ax1b.legend(loc='upper right', fontsize=8, frameon=False)
    fig.text(0.01, 0.005, L['note'], fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    suffix = '_en' if args.lang == 'en' else ''
    out = Path(f'paper/figures/euro_ciss_entropique{tag}{suffix}.png')
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == '__main__':
    main()
