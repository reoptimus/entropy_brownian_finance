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
    end = next(i for i in range(header, len(lines)) if not lines[i].strip())

    block = '\n'.join(lines[header:end])
    df = pd.read_csv(io.StringIO(block), skipinitialspace=True)
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
    return logret
