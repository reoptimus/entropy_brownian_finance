"""Rolling estimation of the three entropy channels and the information flow.

For every date the pipeline produces:

Scale channel
    ``h_vol = sum_i log sigma_i`` -- the marginal fluctuation scale.

Dependence channel
    ``h_dep = 1/2 log det R <= 0`` -- the covariance-volume contraction caused
    by correlation. ``-h_dep`` is the Gaussian multi-information.

Shape channel
    ``d_total`` -- the marginal entropy deficit (negentropy), the uncertainty
    the covariance matrix cannot see. Split into an asymmetry channel
    (``d_odd``) and a tail channel (``d_even``).

Their combinations are the objects the paper tests:

``h_cov = h_vol + h_dep``
    the Gaussian entropy ceiling, i.e. what the covariance matrix alone implies.

``h_tot = h_cov - d_total``
    the entropy actually available given the shape of the distribution.

``J = d_total - h_dep >= 0``
    the *structural index*: the divergence of the conditional joint return
    distribution from a product of Gaussian marginals with the same scales. It
    is scale-free, so it is not a repackaged volatility index, and it is the
    state variable whose jumps and relaxation the postulates concern.

Two flow measures are computed alongside:

``surprisal``
    ``(M_t^2 - N)/2`` with ``M_t`` the Mahalanobis length of today's return
    under yesterday's conditional model: the excess self-information of the
    day's observation, in nats. Its expectation is the KL divergence of the
    model from reality.

``info_flow``
    ``KL(p_t || p_{t-1})`` between consecutive Gaussian conditional estimates:
    how much the market revised its own conditional distribution today. This is
    the quantity that calibrates the stress-scenario radius, so that a scenario
    severity and an observed information event are measured in the same unit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .data import read_ff49_daily, read_eu_stock_markets_daily, read_sp500_daily
from .entropy import (
    covariance_decomposition,
    estimate_covariance,
    ewma_weights,
    gaussian_kl,
)
from .shape import entropy_deficit, structural_index, tail_dependence

DATA_READERS = {
    'ff49': lambda cfg: read_ff49_daily(cfg['data']['raw_file'], cfg['estimation']['exclude_other']),
    'eu_stock_markets': lambda cfg: read_eu_stock_markets_daily(cfg['data']['raw_file']),
    'sp500': lambda cfg: read_sp500_daily(cfg['data']['raw_file']),
}


def rolling_measures(
    logret: pd.DataFrame,
    window: int,
    method: str = 'ledoit_wolf',
    jump_method: str = 'ewma',
    halflife: float = 60.0,
    shrinkage: float = 0.10,
    shape_estimator: str = 'hyvarinen',
    tail_quantile: float = 0.05,
    flow_horizon: int = 21,
) -> pd.DataFrame:
    """Compute every entropy channel on a rolling window.

    The baseline covariance (``method``) drives the level decomposition and
    reproduces the earlier draft's H1/H2 statistics. The responsive covariance
    (``jump_method``, exponentially weighted) drives the shape channel, the
    structural index and the flow measures: a box-car window makes an extreme
    observation *leaving* the sample look like an event, which would contaminate
    any jump test run on the resulting series.
    """
    X = logret.to_numpy()
    T, N = X.shape
    w_ewma = ewma_weights(window, halflife)
    uniform = np.full(window, 1.0 / window)

    from collections import deque

    rows = []
    prev_mu = prev_cov = None
    history: deque = deque(maxlen=flow_horizon)
    for end in range(window, T + 1):
        Xw = X[end - window:end]

        cov_base = estimate_covariance(Xw, method, halflife, shrinkage)
        base = covariance_decomposition(cov_base)

        cov_jump = estimate_covariance(Xw, jump_method, halflife, shrinkage)
        resp = covariance_decomposition(cov_jump)

        weights = w_ewma if jump_method == 'ewma' else uniform
        shape = entropy_deficit(Xw, weights=weights, method=shape_estimator)

        rec = {'date': logret.index[end - 1]}
        rec.update({k: v for k, v in base.items()})
        rec.update({f'{k}_ew': v for k, v in resp.items()})
        rec.update(shape)
        rec['h_tot'] = base['h_cov'] - shape['d_total']
        rec['J'] = structural_index(resp['h_dep'], shape['d_total'])
        rec['J_lw'] = structural_index(base['h_dep'], shape['d_total'])
        rec['tail_dep'] = tail_dependence(Xw, tail_quantile)

        mu = weights @ Xw
        if prev_cov is not None:
            rec['info_flow'] = gaussian_kl(mu, cov_jump, prev_mu, prev_cov)
            x = X[end - 1] - prev_mu
            m2 = float(x @ np.linalg.solve(prev_cov, x))
            rec['mahalanobis2'] = m2
            rec['surprisal'] = 0.5 * (m2 - N)
        else:
            rec['info_flow'] = np.nan
            rec['mahalanobis2'] = np.nan
            rec['surprisal'] = np.nan
        if len(history) == flow_horizon:
            mu_h, cov_h = history[0]
            rec['info_flow_h'] = gaussian_kl(mu, cov_jump, mu_h, cov_h)
        else:
            rec['info_flow_h'] = np.nan
        history.append((mu, cov_jump))
        prev_mu, prev_cov = mu, cov_jump
        rows.append(rec)

    out = pd.DataFrame(rows).set_index('date')
    diffs = {
        'dh_vol': 'h_vol', 'dh_dep': 'h_dep', 'dh_cov': 'h_cov', 'dh_tot': 'h_tot',
        'dh_vol_ew': 'h_vol_ew', 'dh_dep_ew': 'h_dep_ew', 'dh_cov_ew': 'h_cov_ew',
        'dD': 'd_total', 'dD_odd': 'd_odd', 'dD_even': 'd_even', 'dJ': 'J',
        'dskew_market': 'skew_market', 'dexkurt_market': 'exkurt_market',
        'dtail_dep': 'tail_dep',
    }
    for new, src in diffs.items():
        out[new] = out[src].diff()
    out['comp_gap'] = out['dh_vol'] + out['dh_dep']
    return out


# Backwards-compatible alias: the earlier draft's entry point.
def rolling_entropy(logret: pd.DataFrame, window: int, method: str = 'ledoit_wolf') -> pd.DataFrame:
    return rolling_measures(logret, window, method=method, jump_method=method)


def add_forward_realized_vol(logret: pd.DataFrame, entropy: pd.DataFrame, horizon: int, ann=252):
    """Forward realised volatility of the equal-weighted cross-sectional return."""
    mkt = logret.mean(axis=1)
    future = []
    for dt in entropy.index:
        pos = logret.index.get_loc(dt)
        x = mkt.iloc[pos + 1:pos + 1 + horizon]
        future.append(np.sqrt(ann * np.sum(x.to_numpy() ** 2)) if len(x) == horizon else np.nan)
    entropy = entropy.copy()
    entropy['future_rv'] = future
    return entropy


def add_forward_tail_risk(logret: pd.DataFrame, entropy: pd.DataFrame, horizon: int):
    """Forward worst daily loss and drawdown of the equal-weighted market."""
    mkt = logret.mean(axis=1)
    worst, dd = [], []
    for dt in entropy.index:
        pos = logret.index.get_loc(dt)
        x = mkt.iloc[pos + 1:pos + 1 + horizon].to_numpy()
        if len(x) < horizon:
            worst.append(np.nan)
            dd.append(np.nan)
            continue
        worst.append(float(-x.min()))
        cum = np.cumsum(x)
        dd.append(float(-(cum - np.maximum.accumulate(np.r_[0.0, cum])[:-1]).min()))
    entropy = entropy.copy()
    entropy['future_worst_loss'] = worst
    entropy['future_drawdown'] = dd
    return entropy


def run_regressions(df: pd.DataFrame, target: str = 'future_rv', extra: str = 'h_dep'):
    """Volatility-only benchmark against a model adding one entropy channel."""
    cols_needed = [target, 'h_vol', extra]
    d = df.dropna(subset=cols_needed).copy()
    split = int(len(d) * 0.70)
    train, test = d.iloc[:split], d.iloc[split:]
    results = []
    for name, cols in [('vol_only', ['h_vol']), (f'vol_plus_{extra}', ['h_vol', extra])]:
        Xtr = sm.add_constant(train[cols])
        Xte = sm.add_constant(test[cols], has_constant='add')
        model = sm.OLS(train[target], Xtr).fit(cov_type='HAC', cov_kwds={'maxlags': 21})
        pred = model.predict(Xte)
        resid = test[target].to_numpy() - pred.to_numpy()
        results.append({
            'target': target,
            'model': name,
            'train_r2': model.rsquared,
            'train_adj_r2': model.rsquared_adj,
            'test_mse': float(np.mean(resid ** 2)),
            'test_mae': float(np.mean(np.abs(resid))),
            'extra_coef': model.params.get(extra, np.nan),
            'extra_pvalue': model.pvalues.get(extra, np.nan),
        })
    return pd.DataFrame(results)


def main(cfg):
    data_type = cfg['data'].get('type', 'ff49')
    if data_type not in DATA_READERS:
        raise ValueError(f"Unknown data.type '{data_type}'; expected one of {list(DATA_READERS)}")
    logret = DATA_READERS[data_type](cfg)
    start, end = cfg['sample']['start'], cfg['sample']['end']
    logret = logret.loc[str(start):str(end)]

    est = cfg['estimation']
    ent = rolling_measures(
        logret,
        cfg['sample']['window'],
        method=est['covariance'],
        jump_method=est.get('jump_covariance', 'ewma'),
        halflife=float(est.get('ewma_halflife', 60)),
        shrinkage=float(est.get('ewma_shrinkage', 0.10)),
        shape_estimator=est.get('shape_estimator', 'hyvarinen'),
        flow_horizon=int(cfg.get('stress_test', {}).get('horizon', 21)),
    )

    market = logret.mean(axis=1)
    ent['stress_signal'] = market.rolling(21).std() * np.sqrt(252)
    ent = add_forward_realized_vol(
        logret, ent, cfg['risk']['forecast_horizon'], cfg['risk']['realized_vol_annualization'])
    ent = add_forward_tail_risk(logret, ent, cfg['risk']['forecast_horizon'])
    return logret, ent


def conditional_state(logret: pd.DataFrame, window: int, method: str = 'ewma',
                      halflife: float = 60.0, shrinkage: float = 0.10):
    """Conditional mean and covariance on the final ``window`` observations.

    This is the base distribution a stress test is run *from*: what the market
    looks like today, not on average over the sample.
    """
    X = logret.to_numpy()[-window:]
    w = ewma_weights(window, halflife) if method == 'ewma' else np.full(window, 1.0 / window)
    mu = w @ X
    cov = estimate_covariance(X, method, halflife, shrinkage)
    return mu, cov
