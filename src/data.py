from __future__ import annotations
from pathlib import Path
import io
import zipfile
import pandas as pd
import numpy as np


def read_ff49_daily(zip_path: str | Path, exclude_other: bool = True) -> pd.DataFrame:
    """Read the daily value-weighted FF49 portfolio returns.

    French returns are percentages. We convert to decimal simple returns and
    then to log returns via log(1+r). Rows containing the -99.99 missing-value
    marker are removed.
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        csv_name = next(n for n in names if n.lower().endswith('.csv'))
        raw = z.read(csv_name).decode('latin1')

    lines = raw.splitlines()
    # Locate the daily value-weighted section. The exact header text can vary.
    start = next(i for i, line in enumerate(lines)
                 if 'Average Value Weighted Returns -- Daily' in line)
    header = start + 1
    end = next((i for i in range(header, len(lines)) if not lines[i].strip()), len(lines))

    block = '\n'.join(lines[header:end])
    df = pd.read_csv(io.StringIO(block), skipinitialspace=True)
    df.columns = df.columns.str.strip()
    df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    df['date'] = pd.to_datetime(df['date'].astype(str).str.strip(), format='%Y%m%d', errors='coerce')
    df = df.dropna(subset=['date']).set_index('date')
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.replace([-99.99, -999.0, -99.0], np.nan)

    if exclude_other and 'Other' in df.columns:
        df = df.drop(columns=['Other'])

    df = df.dropna(how='all')
    simple = df / 100.0
    if (simple <= -1).any().any():
        raise ValueError('Return <= -100% encountered; cannot take log1p safely.')
    logret = np.log1p(simple)
    logret = logret.dropna(how='any')
    if logret.empty:
        raise ValueError(
            'Parsed zero rows of daily returns. The section header search '
            '("Average Value Weighted Returns -- Daily") or the CSV delimiter '
            'assumption (comma-separated) likely no longer matches the '
            'downloaded file format; inspect the raw CSV before proceeding.'
        )
    return logret


def read_eu_stock_markets_daily(csv_path: str | Path) -> pd.DataFrame:
    """Read the real daily closing-price panel for DAX, SMI, CAC and FTSE.

    Source: the classic ``EuStockMarkets`` dataset (Bollerslev & Ghysels),
    bundled with base R and mirrored as CSV by the Rdatasets project at
    https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/EuStockMarkets.csv
    1860 contemporaneous trading days, 1991-01-02 through 1998-12-31,
    covering the 1997 Asian and 1998 Russian/LTCM stress episodes.

    The upstream file only carries a row index, not calendar dates, so dates
    here are reconstructed as a plain business-day sequence starting
    1991-01-02. Country-specific holidays are not removed, so individual
    dates can drift by a few days from the true calendar by the end of the
    sample; only the chronological order and spacing matter for the rolling
    covariance pipeline, not the exact calendar label.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    cols = [c for c in ['DAX', 'SMI', 'CAC', 'FTSE'] if c in df.columns]
    if len(cols) != 4:
        raise ValueError(f'Expected columns DAX, SMI, CAC, FTSE; found {list(df.columns)}')
    prices = df[cols].apply(pd.to_numeric, errors='coerce')
    if prices.isna().any().any():
        raise ValueError('Non-numeric or missing price observed in EuStockMarkets CSV.')
    dates = pd.bdate_range('1991-01-02', periods=len(prices))
    prices.index = dates
    logret = np.log(prices).diff().dropna(how='any')
    return logret
