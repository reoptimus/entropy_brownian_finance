"""Figure of the euro-area entropic CISS (phase B).

Reads ``paper/tables/euro_block_coupling_series{tag}.csv`` written by
``scripts/euro_ciss.py`` and the ECB CISS from ``data/zone_euro/reference.csv``.

Usage::

    python -m scripts.euro_figure [--overnight-only]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from scripts.euro_ciss import CEPR_RECESSIONS

PAIRS = {'mi_obligataire__intermediaires': 'Obligations – intermédiaires',
         'mi_obligataire__actions': 'Obligations – actions',
         'mi_actions__intermediaires': 'Actions – intermédiaires',
         'mi_intermediaires__change': 'Intermédiaires – change'}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/zone_euro')
    p.add_argument('--overnight-only', action='store_true')
    args = p.parse_args()
    tag = '_overnight_only' if args.overnight_only else ''
    dec = pd.read_csv(f'paper/tables/euro_block_coupling_series{tag}.csv',
                      index_col=0, parse_dates=True)
    ref = pd.read_csv(Path(args.data) / 'reference.csv', index_col=0, parse_dates=True)
    ciss = ref['ciss_zone_euro'].dropna().loc[dec.index.min():]

    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    z = lambda s: (s - s.mean()) / s.std()
    ax[0].plot(z(dec['total']), lw=0.9, label='Couplage total (rangs)')
    ax[0].plot(z(ciss.resample('W-FRI').last()), lw=0.9, alpha=0.8, label='CISS BCE')
    ax[0].set_ylabel('écart-type')
    ax[0].set_title('CISS entropique zone euro : couplage total et CISS (centrés réduits)')
    ax[1].plot(dec['inter'], lw=0.9, color='C2', label='Entre segments (nats)')
    ax1b = ax[1].twinx()
    ax1b.plot(dec['inter'] / dec['total'], lw=0.7, color='C3', alpha=0.7,
              label='Part entre segments')
    ax[1].set_ylabel('nats')
    ax1b.set_ylabel('part')
    ax[1].set_title('Couplage entre les cinq segments du CISS')
    for k, lab in PAIRS.items():
        ax[2].plot(dec[k], lw=0.9, label=lab)
    ax[2].set_ylabel('nats')
    ax[2].set_title('Carte de contagion : information mutuelle entre paires de segments')
    for a in ax:
        for s, e in CEPR_RECESSIONS:
            a.axvspan(pd.Timestamp(s), pd.Timestamp(e), color='grey', alpha=0.15, lw=0)
        a.legend(loc='upper left', fontsize=8, frameon=False)
    ax1b.legend(loc='upper right', fontsize=8, frameon=False)
    fig.text(0.01, 0.005, 'Zones grises : récessions CEPR. Fenêtre glissante de 252 jours, '
             'scores normaux, Ledoit-Wolf.', fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = Path(f'paper/figures/euro_ciss_entropique{tag}.png')
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == '__main__':
    main()
