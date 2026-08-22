import numpy as np
import pandas as pd
from src.empirical import rolling_entropy, main as empirical_main


def test_synthetic_pipeline():
    rng = np.random.default_rng(123)
    A = np.array([[1.0, .5, .2], [.0, .9, .3], [.0, .0, .7]])
    cov = A @ A.T / 10000
    x = rng.multivariate_normal(np.zeros(3), cov, size=600)
    dates = pd.bdate_range('2020-01-01', periods=len(x))
    df = pd.DataFrame(x, index=dates, columns=['a','b','c'])
    out = rolling_entropy(df, 120)
    assert len(out) == 481
    assert np.isfinite(out[['h_vol','h_dep','h_cov']].to_numpy()).all()


def test_empirical_main_dispatches_on_data_type(tmp_path):
    csv_path = tmp_path / 'eu.csv'
    rng = np.random.default_rng(7)
    n = 300
    prices = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n, 4)), axis=0))
    rows = ['rownames,DAX,SMI,CAC,FTSE']
    for i, p in enumerate(prices, start=1):
        rows.append(f'{i},{p[0]},{p[1]},{p[2]},{p[3]}')
    csv_path.write_text('\n'.join(rows))

    cfg = {
        'data': {'type': 'eu_stock_markets', 'raw_file': str(csv_path)},
        'sample': {'start': '1991-01-01', 'end': '2099-01-01', 'window': 60},
        'estimation': {'covariance': 'ledoit_wolf', 'exclude_other': False},
        'risk': {'forecast_horizon': 10, 'realized_vol_annualization': 260},
    }
    logret, ent = empirical_main(cfg)
    assert list(logret.columns) == ['DAX', 'SMI', 'CAC', 'FTSE']
    assert {'h_vol', 'h_dep', 'h_cov', 'comp_gap', 'stress_signal'}.issubset(ent.columns)
    assert np.isfinite(ent[['h_vol', 'h_dep', 'h_cov']].to_numpy()).all()
