"""Copy the figures the paper includes from the run outputs into paper/figures.

The pipeline writes one figure set per dataset under ``outputs/<dataset>/figures``.
The paper references a flat, prefixed set. This script keeps the two in sync so
that regenerating a result and rebuilding the paper is two commands, not a
manual copy.

Usage::

    python -m src.run_empirical --config config/sp500.yaml
    python -m src.run_empirical --config config/pilot_eu_stock_markets.yaml
    python -m scripts.build_paper_figures
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SOURCES = {
    'sp500': Path('outputs/sp500/figures'),
    'eu': Path('outputs/pilot_eu/figures'),
}

WANTED = [
    'channels.png',
    'channels_vs_volatility.png',
    'structural_index_jumps.png',
    'jump_asymmetry.png',
    'relaxation_event_study.png',
    'compensation_scatter.png',
    'skewness_signature.png',
    'stress_ladder.png',
]


TABLES = [
    'h1_regime_signature.csv',
    'h2_ceiling_compensation.csv',
    'h3_jump_asymmetry.csv',
    'h4_relaxation.csv',
    'h5_skewness_signature.csv',
    'h6_tail_dependence_coupling.csv',
    'h7_incremental_value.csv',
    'severity_ladder.csv',
    'stress_scenarios.csv',
    'classical_comparison.csv',
    'reverse_stress.csv',
    'placebo_null.csv',
    'placebo_null_block21.csv',
    'negentropy_benchmark.csv',
    'run_summary.json',
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--dest', default='paper/figures')
    p.add_argument('--tables-dest', default='paper/tables')
    args = p.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    copied, missing = [], []
    for prefix, src in SOURCES.items():
        for name in WANTED:
            source = src / name
            target = dest / f'{prefix}_{name}'
            if source.exists():
                shutil.copyfile(source, target)
                copied.append(target.name)
            else:
                missing.append(str(source))

    # The result tables the paper quotes are small and belong under version
    # control, so every number in the text can be traced to a file.
    tdest = Path(args.tables_dest)
    tdest.mkdir(parents=True, exist_ok=True)
    for prefix, src in SOURCES.items():
        for name in TABLES:
            source = src.parent / 'tables' / name
            if source.exists():
                shutil.copyfile(source, tdest / f'{prefix}_{name}')
                copied.append(f'{prefix}_{name}')

    for name in copied:
        print(f'  copied {name}')
    if missing:
        print('\nNot found (run the pipeline for that config first):')
        for m in missing:
            print(f'  {m}')


if __name__ == '__main__':
    main()
