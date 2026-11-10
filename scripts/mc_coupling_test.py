"""Monte Carlo study of the coupling test (JFEC revision).

Two questions a referee asks of any resampling test:

1. *Finite-sample bias.* Under independence, the sample coupling
   ``C_hat = -1/2 log det R_hat`` is positive. Bartlett's (1950) sphericity
   statistic gives ``E[C_hat] ~= N(N-1) / (4 (T - 1 - (2N+5)/6))``. We check the
   approximation for Gaussian and Student-t(5) data, raw and with Ledoit-Wolf
   shrinkage (the paper's estimator).

2. *Size and power of the stress-minus-calm coupling test.* We simulate panels
   from four data-generating processes and apply the paper's test with each of
   the two nulls (i.i.d. row bootstrap; CCC-GJR filtered historical simulation):

   - ``iid_t``     i.i.d. Student-t(5), constant correlation        -> size
   - ``ccc_gjr``   GJR-GARCH(1,1) with t(5) shocks, constant corr.  -> size
   - ``ms_vol``    Markov-switching volatility (x2.5), constant corr.-> size
   - ``ms_corr``   same, correlation 0.30 calm -> 0.50 in stress    -> power

   Panels: N = 10 assets, T = 4000 days, window 252, stress = past-only 90 %
   quantile of the 21-day volatility of the equal-weight portfolio (as in the
   paper). The estimator in the Monte Carlo is the plain sample correlation, so
   that thousands of pipelines run in minutes; the paper's own estimator is
   used for the empirical work.

Outputs: paper/tables/mc_bias.csv, paper/tables/mc_size_power.csv.

Usage::

    python -m scripts.mc_coupling_test --n-mc 200 --n-boot 99 --jobs 4
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

WINDOW, N, T, MIN_HIST, Q = 252, 10, 4000, 504, 0.90
RHO = 0.30
DGPS = ['iid_t', 'ccc_gjr', 'ms_vol', 'ms_corr']


# ----------------------------------------------------------------- estimators
def rolling_half_logdet_corr(X: np.ndarray, w: int = WINDOW) -> np.ndarray:
    """``1/2 log det R`` of the sample correlation on every rolling window."""
    Tn, n = X.shape
    c1 = np.vstack([np.zeros((1, n)), np.cumsum(X, axis=0)])
    outer = np.einsum('ti,tj->tij', X, X)
    c2 = np.concatenate([np.zeros((1, n, n)), np.cumsum(outer, axis=0)])
    s1 = c1[w:] - c1[:-w]
    s2 = c2[w:] - c2[:-w]
    m = s1 / w
    cov = s2 / w - np.einsum('ti,tj->tij', m, m)
    d = np.sqrt(np.einsum('tii->ti', cov))
    R = cov / (d[:, :, None] * d[:, None, :])
    return 0.5 * np.linalg.slogdet(R)[1]           # index k <-> window ending at k+w-1


def stress_labels(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sig = pd.Series(X.mean(axis=1)).rolling(21).std()
    thr = sig.expanding(min_periods=MIN_HIST).quantile(Q).shift(1)
    valid = (thr.notna() & sig.notna()).to_numpy()
    stress = (sig >= thr).to_numpy() & valid
    return stress, valid


def coupling_gap(X: np.ndarray) -> float:
    """Stress-minus-calm gap of ``1/2 log det R`` (negative = coupling)."""
    h = rolling_half_logdet_corr(X)
    stress, valid = stress_labels(X)
    s, v = stress[WINDOW - 1:], valid[WINDOW - 1:]
    calm = v & ~s
    return float(h[s].mean() - h[calm].mean())


# ----------------------------------------------------------------------- DGPs
def _corr_chol(rho: float) -> np.ndarray:
    R = np.full((N, N), rho) + (1 - rho) * np.eye(N)
    return np.linalg.cholesky(R)


def _t_shocks(rng, size, nu=5.0):
    """Unit-variance multivariate-t innovations (common chi-square mixing)."""
    g = rng.standard_normal(size)
    w = rng.chisquare(nu, size=size[0])[:, None] / nu
    return g / np.sqrt(w) * np.sqrt((nu - 2) / nu)


def simulate(dgp: str, rng) -> np.ndarray:
    if dgp == 'iid_t':
        return 0.01 * _t_shocks(rng, (T, N)) @ _corr_chol(RHO).T
    if dgp == 'ccc_gjr':
        z = _t_shocks(rng, (T, N)) @ _corr_chol(RHO).T
        om, a, g, b = 0.02, 0.03, 0.08, 0.91
        h = np.full(N, om / (1 - a - g / 2 - b))
        out = np.empty((T, N))
        for t in range(T):
            e = np.sqrt(h) * z[t]
            out[t] = e
            h = om + (a + g * (e < 0)) * e * e + b * h
        return out / 100.0
    if dgp in ('ms_vol', 'ms_corr'):
        p_cs, p_sc = 0.004, 0.03              # calm->stress, stress->calm
        state = np.zeros(T, dtype=bool)
        for t in range(1, T):
            u = rng.random()
            state[t] = (u < p_cs) if not state[t - 1] else (u >= p_sc)
        Lc = _corr_chol(RHO)
        Ls = _corr_chol(0.50 if dgp == 'ms_corr' else RHO)
        z = _t_shocks(rng, (T, N))
        out = np.where(state[:, None], 2.5 * (z @ Ls.T), z @ Lc.T)
        return 0.01 * out
    raise ValueError(dgp)


# ---------------------------------------------------------------------- nulls
def fit_gjr(X: np.ndarray):
    from arch import arch_model
    P, Z = [], []
    for i in range(X.shape[1]):
        r = arch_model(100 * X[:, i], mean='Constant', vol='GARCH', p=1, o=1, q=1,
                       rescale=False).fit(disp='off')
        p = r.params
        P.append([p['mu'], p['omega'], p['alpha[1]'], p['gamma[1]'], p['beta[1]']])
        z = np.asarray(r.std_resid)
        Z.append((z - z.mean()) / z.std())
    return np.array(P), np.column_stack(Z)


def sim_fhs(P: np.ndarray, Z: np.ndarray, idx: np.ndarray) -> np.ndarray:
    z = Z[idx]
    mu, om, a, g, b = P.T
    pers = np.minimum(a + g / 2 + b, 0.999)
    h = om / (1 - pers)
    out = np.empty_like(z)
    for t in range(z.shape[0]):
        e = np.sqrt(h) * z[t]
        out[t] = mu + e
        h = om + (a + g * (e < 0)) * e * e + b * h
    return out / 100.0


def one_mc(args) -> dict:
    dgp, m, n_boot = args
    rng = np.random.default_rng(np.random.SeedSequence([41, DGPS.index(dgp), m]))
    X = simulate(dgp, rng)
    obs = coupling_gap(X)
    iid = np.array([coupling_gap(X[rng.integers(0, T, T)]) for _ in range(n_boot)])
    P, Z = fit_gjr(X)
    fhs = np.array([coupling_gap(sim_fhs(P, Z, rng.integers(0, T, T))) for _ in range(n_boot)])
    return {'dgp': dgp, 'mc': m, 'observed_gap': obs,
            'p_iid': (1 + (iid <= obs).sum()) / (n_boot + 1),
            'p_fhs': (1 + (fhs <= obs).sum()) / (n_boot + 1)}


# ----------------------------------------------------------------------- bias
def bias_table(n_mc: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for n, w in [(10, 252), (20, 252), (48, 252), (20, 504)]:
        approx = n * (n - 1) / (4 * (w - 1 - (2 * n + 5) / 6))
        for dist in ('gaussian', 't5'):
            raw, lw = [], []
            for _ in range(n_mc if n < 48 else n_mc // 4):
                X = rng.standard_normal((w, n)) if dist == 'gaussian' else rng.standard_t(5, (w, n))
                Rm = np.corrcoef(X, rowvar=False)
                raw.append(-0.5 * np.linalg.slogdet(Rm)[1])
                c = ledoit_wolf(X)[0]
                d = np.sqrt(np.diag(c))
                lw.append(-0.5 * np.linalg.slogdet(c / np.outer(d, d))[1])
            rows.append({'N': n, 'window': w, 'law': dist, 'bartlett_approx': approx,
                         'mc_mean_raw': np.mean(raw), 'mc_sd_raw': np.std(raw),
                         'mc_mean_lw': np.mean(lw), 'mc_sd_lw': np.std(lw)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-mc', type=int, default=200)
    ap.add_argument('--n-boot', type=int, default=99)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--dgps', default='iid_t,ccc_gjr,ms_vol,ms_corr')
    args = ap.parse_args()
    dest = Path('paper/tables')
    t0 = time.time()
    b = bias_table()
    b.to_csv(dest / 'mc_bias.csv', index=False, float_format='%.4f')
    print(b.round(3).to_string(index=False), f'\n[{time.time() - t0:.0f} s]', flush=True)

    tasks = [(d, m, args.n_boot) for d in args.dgps.split(',') for m in range(args.n_mc)]
    with ProcessPoolExecutor(args.jobs) as ex:
        res = list(ex.map(one_mc, tasks, chunksize=4))
    long = pd.DataFrame(res)
    long.to_csv(dest / 'mc_size_power_replications.csv', index=False, float_format='%.5g')
    summ = long.groupby('dgp').agg(
        n_mc=('mc', 'size'), mean_gap=('observed_gap', 'mean'),
        reject_iid_5=('p_iid', lambda p: (p <= 0.05).mean()),
        reject_fhs_5=('p_fhs', lambda p: (p <= 0.05).mean()),
        reject_iid_1=('p_iid', lambda p: (p <= 0.01).mean()),
        reject_fhs_1=('p_fhs', lambda p: (p <= 0.01).mean())).reset_index()
    summ.to_csv(dest / 'mc_size_power.csv', index=False, float_format='%.4f')
    print(summ.round(3).to_string(index=False), f'\n[{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
