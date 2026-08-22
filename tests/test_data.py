import textwrap
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import read_ff49_daily, read_eu_stock_markets_daily


def _write_fake_ff49_zip(path: Path) -> None:
    body = textwrap.dedent("""\
        This file was created by CMPT_ME_ALL_dai_49.....

          Average Value Weighted Returns -- Daily
        ,Agric,Food  ,Beer,Other
        19260701,  0.15,  0.20, -0.10,  0.05
        19260702, -0.30,  0.05,  0.40,  0.11
        19260703,-99.99,-99.99,-99.99,-99.99
        19260706,  0.10,  0.10,  0.10,  0.02

          Average Equal Weighted Returns -- Daily
        ,Agric,Food  ,Beer,Other
        19260701,  0.10,  0.10, -0.05,  0.01
        """)
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('49_Industry_Portfolios_daily.CSV', body)


def test_read_ff49_daily_parses_value_weighted_block(tmp_path):
    zpath = tmp_path / 'ff49.zip'
    _write_fake_ff49_zip(zpath)
    df = read_ff49_daily(zpath)

    assert list(df.columns) == ['Agric', 'Food', 'Beer']
    assert len(df) == 3  # the -99.99 missing-value row is dropped
    assert pd.Timestamp('1926-07-03') not in df.index
    assert np.isclose(df.loc['1926-07-01', 'Agric'], np.log1p(0.0015))


def test_read_ff49_daily_keeps_other_when_not_excluded(tmp_path):
    zpath = tmp_path / 'ff49.zip'
    _write_fake_ff49_zip(zpath)
    df = read_ff49_daily(zpath, exclude_other=False)
    assert 'Other' in df.columns


def test_read_ff49_daily_raises_on_unparseable_format(tmp_path):
    # Space-delimited instead of the real comma-delimited French format: the
    # header collapses into a single unnamed column and no numeric data
    # survives, which must fail loudly rather than return an empty frame.
    body = (
        '  Average Value Weighted Returns -- Daily\n'
        '        Agric      Food     Beer\n'
        '19260701   0.15      0.20    -0.10\n'
    )
    zpath = tmp_path / 'ff49_bad.zip'
    with zipfile.ZipFile(zpath, 'w') as z:
        z.writestr('bad.CSV', body)
    with pytest.raises(ValueError):
        read_ff49_daily(zpath)


def test_read_eu_stock_markets_daily(tmp_path):
    csv_path = tmp_path / 'eu.csv'
    csv_path.write_text(
        'rownames,DAX,SMI,CAC,FTSE\n'
        '1,1000,1000,1000,1000\n'
        '2,1010,995,1005,998\n'
        '3,1005,1002,1010,1001\n'
    )
    logret = read_eu_stock_markets_daily(csv_path)
    assert list(logret.columns) == ['DAX', 'SMI', 'CAC', 'FTSE']
    assert len(logret) == 2
    assert np.isclose(logret.iloc[0]['DAX'], np.log(1010 / 1000))
    assert isinstance(logret.index, pd.DatetimeIndex)
