"""Euro-area multi-market panel for the entropic CISS (phase B).

Reads the CSV files written by ``telechargement_zone_euro/telecharger.py`` (one
per CISS segment, dates in the first column, one series per column) and turns
them into one matrix of daily moves:

* rates and yields (money market, bonds): first difference of the level, plus
  two spreads that carry stress on their own (Euribor 3M minus the overnight
  rate; all-ratings minus AAA 10-year yield);
* prices (equity indices, banks and insurers, exchange rates): log returns.

Everything enters through correlations, which are invariant to the unit of each
series, so rate changes and returns can sit in the same matrix.

Calendars differ across exchanges; a level is carried forward for at most
``max_fill`` business days before a move is computed, and the panel keeps only
dates on which every retained series has a value.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = ('monetaire', 'obligataire', 'actions', 'intermediaires', 'change')
LEVEL_SEGMENTS = ('monetaire', 'obligataire')


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.apply(pd.to_numeric, errors='coerce').sort_index()


def _overnight(mm: pd.DataFrame) -> pd.Series | None:
    """EONIA until the €STR starts, then the €STR. Only changes are used, so
    the constant 8.5 bp gap between the two does not matter, except on the
    splice date, whose change is dropped."""
    eonia, estr = mm.get('eonia'), mm.get('estr')
    if eonia is None and estr is None:
        return None
    if estr is None:
        return eonia
    if eonia is None:
        return estr
    start = estr.first_valid_index()
    return pd.concat([eonia[eonia.index < start], estr[estr.index >= start]])


def build_euro_panel(folder: str | Path, start: str = '2005-01-01',
                     end: str | None = None, max_fill: int = 3,
                     min_coverage: float = 0.95) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Return ``(moves, blocks)``.

    Series covering less than ``min_coverage`` of the business days in
    ``[start, end]`` are dropped (and reported in the returned block map only if
    kept), so one late-starting stock does not shorten the whole panel.
    """
    folder = Path(folder)
    levels: dict[str, pd.Series] = {}
    blocks: dict[str, list[str]] = {}
    splice_dates = []
    for seg in SEGMENTS:
        f = folder / f'{seg}.csv'
        if not f.exists():
            continue
        df = _read(f)
        names = []
        if seg == 'monetaire':
            on = _overnight(df)
            if on is not None:
                levels['mm_overnight'] = on
                names.append('mm_overnight')
                if 'estr' in df and 'eonia' in df:
                    splice_dates.append(df['estr'].first_valid_index())
            for c in [c for c in df.columns if c.startswith('euribor')]:
                levels[f'mm_{c}'] = df[c]
                names.append(f'mm_{c}')
            if on is not None and 'euribor_3m' in df:
                levels['mm_spread_3m_on'] = df['euribor_3m'] - on.reindex(df.index).ffill()
                names.append('mm_spread_3m_on')
        elif seg == 'obligataire':
            for c in df.columns:
                levels[f'bd_{c}'] = df[c]
                names.append(f'bd_{c}')
            if {'tous_10a', 'aaa_10a'} <= set(df.columns):
                levels['bd_spread_10a'] = df['tous_10a'] - df['aaa_10a']
                names.append('bd_spread_10a')
        else:
            prefix = {'actions': 'eq', 'intermediaires': 'fi', 'change': 'fx'}[seg]
            for c in df.columns:
                levels[f'{prefix}_{c}'] = df[c]
                names.append(f'{prefix}_{c}')
        blocks[seg] = names

    L = pd.DataFrame(levels)
    days = pd.bdate_range(start, end or L.index.max())
    L = L.reindex(L.index.union(days)).ffill(limit=max_fill).reindex(days)

    moves = {}
    for seg, names in blocks.items():
        for n in names:
            x = L[n]
            moves[n] = x.diff() if seg in LEVEL_SEGMENTS else np.log(x).diff()
    M = pd.DataFrame(moves)
    for d in splice_dates:
        if d in M.index:
            M.loc[d, 'mm_overnight'] = np.nan

    coverage = M.notna().mean()
    keep = coverage[coverage >= min_coverage].index
    M = M[keep]
    # A series that never moves inside a window (a rate pinned at a floor)
    # would make the correlation matrix singular; flat stretches are left to
    # the Ledoit-Wolf shrinkage, but an all-zero series is dropped.
    M = M.loc[:, M.abs().sum() > 0]
    blocks = {seg: [n for n in names if n in M.columns] for seg, names in blocks.items()}
    blocks = {k: v for k, v in blocks.items() if v}
    M = M.dropna(how='any')
    return M, blocks
