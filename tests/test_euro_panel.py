import numpy as np
import pandas as pd

from src.euro_panel import build_euro_panel


def _write(folder, name, df):
    df.to_csv(folder / f'{name}.csv')


def test_panel_builds_moves_blocks_and_splice(tmp_path):
    rng = np.random.default_rng(0)
    idx = pd.bdate_range('2004-01-01', '2021-12-31')
    walk = lambda s: pd.Series(np.cumsum(rng.standard_normal(len(idx))) * s, index=idx)
    mm = pd.DataFrame({'euribor_3m': 2 + walk(0.01), 'eonia': 1.5 + walk(0.01)})
    estr = (1.4 + walk(0.01)).where(idx >= '2019-10-01')
    mm['estr'] = estr
    _write(tmp_path, 'monetaire', mm)
    _write(tmp_path, 'obligataire', pd.DataFrame({'aaa_10a': 3 + walk(0.02),
                                                  'tous_10a': 3.5 + walk(0.02)}))
    eq = pd.DataFrame({'dax': np.exp(8 + walk(0.01))})
    eq.loc[eq.index[::7], 'dax'] = np.nan          # holidays on another calendar
    _write(tmp_path, 'actions', eq)
    late = pd.Series(np.exp(3 + walk(0.02)), index=idx).where(idx >= '2015-01-01')
    _write(tmp_path, 'intermediaires', pd.DataFrame({'bnp': np.exp(4 + walk(0.02)),
                                                     'late_ipo': late}))
    _write(tmp_path, 'change', pd.DataFrame({'eur_usd': np.exp(walk(0.005))}))

    M, blocks = build_euro_panel(tmp_path, start='2005-01-01', end='2021-12-31')
    assert set(blocks) == {'monetaire', 'obligataire', 'actions', 'intermediaires', 'change'}
    assert 'mm_spread_3m_on' in blocks['monetaire']
    assert 'bd_spread_10a' in blocks['obligataire']
    assert 'fi_late_ipo' not in M.columns            # under the coverage threshold
    assert not M.isna().any().any()
    # The EONIA -> €STR splice day carries no overnight move.
    assert pd.Timestamp('2019-10-01') not in M.index
    assert M['eq_dax'].abs().max() < 0.2
