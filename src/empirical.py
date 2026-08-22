from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .data import read_ff49_daily, read_eu_stock_markets_daily
from .entropy import estimate_covariance, covariance_decomposition

DATA_READERS = {
    'ff49': lambda cfg: read_ff49_daily(cfg['data']['raw_file'], cfg['estimation']['exclude_other']),
    'eu_stock_markets': lambda cfg: read_eu_stock_markets_daily(cfg['data']['raw_file']),
}


def rolling_entropy(logret: pd.DataFrame, window: int, method='ledoit_wolf') -> pd.DataFrame:
    rows = []
    for end in range(window, len(logret) + 1):
        X = logret.iloc[end-window:end].to_numpy()
        cov = estimate_covariance(X, method)
        d = covariance_decomposition(cov)
        d['date'] = logret.index[end-1]
        rows.append(d)
    out = pd.DataFrame(rows).set_index('date')
    return out


def add_forward_realized_vol(logret: pd.DataFrame, entropy: pd.DataFrame, horizon: int, ann=252):
    # Equal-weight cross-sectional market return; used only as a simple
    # predictive-risk target. Robustness should repeat this by portfolio.
    mkt = logret.mean(axis=1)
    future = []
    for dt in entropy.index:
        pos = logret.index.get_loc(dt)
        x = mkt.iloc[pos+1:pos+1+horizon]
        future.append(np.sqrt(ann * np.sum(x.to_numpy()**2)) if len(x) == horizon else np.nan)
    entropy = entropy.copy()
    entropy['future_rv'] = future
    return entropy


def run_regressions(df: pd.DataFrame):
    d = df.dropna(subset=['future_rv', 'h_vol', 'h_dep']).copy()
    split = int(len(d) * 0.70)
    train, test = d.iloc[:split], d.iloc[split:]
    results = []
    for name, cols in [('vol_only', ['h_vol']), ('vol_plus_dependence', ['h_vol', 'h_dep'])]:
        Xtr = sm.add_constant(train[cols])
        Xte = sm.add_constant(test[cols], has_constant='add')
        model = sm.OLS(train['future_rv'], Xtr).fit(cov_type='HAC', cov_kwds={'maxlags': 21})
        pred = model.predict(Xte)
        mse = float(np.mean((test['future_rv'].to_numpy() - pred.to_numpy())**2))
        mae = float(np.mean(np.abs(test['future_rv'].to_numpy() - pred.to_numpy())))
        results.append({
            'model': name,
            'train_r2': model.rsquared,
            'train_adj_r2': model.rsquared_adj,
            'test_mse': mse,
            'test_mae': mae,
            'dep_coef': model.params.get('h_dep', np.nan),
            'dep_pvalue': model.pvalues.get('h_dep', np.nan),
        })
    return pd.DataFrame(results)


def main(cfg):
    data_type = cfg['data'].get('type', 'ff49')
    if data_type not in DATA_READERS:
        raise ValueError(f"Unknown data.type '{data_type}'; expected one of {list(DATA_READERS)}")
    logret = DATA_READERS[data_type](cfg)
    start, end = cfg['sample']['start'], cfg['sample']['end']
    logret = logret.loc[start:end]
    ent = rolling_entropy(logret, cfg['sample']['window'], cfg['estimation']['covariance'])
    ent['dh_vol'] = ent['h_vol'].diff()
    ent['dh_dep'] = ent['h_dep'].diff()
    ent['dh_cov'] = ent['h_cov'].diff()
    ent['comp_gap'] = ent['dh_vol'] + ent['dh_dep']
    market = logret.mean(axis=1)
    ent['stress_signal'] = market.rolling(21).std() * np.sqrt(252)
    ent = add_forward_realized_vol(logret, ent, cfg['risk']['forecast_horizon'], cfg['risk']['realized_vol_annualization'])
    return logret, ent
