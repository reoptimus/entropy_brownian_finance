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

   - ``ms_corr_small`` same, correlation 0.30 calm -> 0.40 in stress -> power

   Panels: N = 10, 20 or 48 assets (the sizes of the paper's panels), T = 4000
   days, window 250 days evaluated every 5th day, stress = past-only 90 % quantile of the 21-day volatility
   of the equal-weight portfolio (as in the paper). The estimator in the Monte Carlo is the plain sample correlation, so
   that thousands of pipelines run in minutes; the paper's own estimator is
   used for the empirical work.

Outputs: paper/tables/mc_bias.csv, paper/tables/mc_size_power.csv.

Usage::

    python -m scripts.mc_coupling_test --sizes 10:1000,20:500,48:300 --n-boot 199 --jobs 4
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf

WINDOW, STEP, T, MIN_HIST, Q = 250, 5, 4000, 504, 0.90
RHO = 0.30
DGPS = ['iid_t', 'ccc_gjr', 'ms_vol', 'ms_corr', 'ms_corr_small']
STRESS_RHO = {'ms_vol': RHO, 'ms_corr': 0.50, 'ms_corr_small': 0.40}


# ----------------------------------------------------------------- estimators
def rolling_half_logdet_corr(X: np.ndarray, w: int = WINDOW, step: int = STEP):
    """``1/2 log det R`` of the sample correlation on rolling windows of ``w``
    days, evaluated every ``step`` days. Cross-products are summed by blocks of
    ``step`` days, so the cost falls by ``step`` against a daily evaluation.

    Returns the values and the (0-based) index of each window's last day.
    """
    Tn, n = X.shape
    nb = Tn // step
    Xb = X[:nb * step].reshape(nb, step, n)
    b1 = Xb.sum(axis=1)
    b2 = np.einsum('bti,btj->bij', Xb, Xb)
    c1 = np.vstack([np.zeros((1, n)), np.cumsum(b1, axis=0)])
    c2 = np.concatenate([np.zeros((1, n, n)), np.cumsum(b2, axis=0)])
    k = w // step
    s1, s2 = c1[k:] - c1[:-k], c2[k:] - c2[:-k]
    m = s1 / w
    cov = s2 / w - np.einsum('ti,tj->tij', m, m)
    d = np.sqrt(np.einsum('tii->ti', cov))
    R = cov / (d[:, :, None] * d[:, None, :])
    ends = np.arange(k, nb + 1) * step - 1
    return 0.5 * np.linalg.slogdet(R)[1], ends


def rolling_half_logdet_lw(X: np.ndarray, w: int = WINDOW, step: int = STEP):
    """Same as ``rolling_half_logdet_corr`` with the paper's estimator: the
    correlation implied by Ledoit-Wolf shrinkage on each window."""
    from src.lw import ledoit_wolf_cov
    Tn = X.shape[0]
    ends = np.arange(w - 1, (Tn // step) * step, step)
    ends = ends[(ends + 1) % step == 0]
    out = np.empty(len(ends))
    for i, e in enumerate(ends):
        c = ledoit_wolf_cov(X[e - w + 1:e + 1])
        d = np.sqrt(np.diag(c))
        out[i] = 0.5 * np.linalg.slogdet(c / np.outer(d, d))[1]
    return out, ends


def stress_labels(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sig = pd.Series(X.mean(axis=1)).rolling(21).std()
    thr = sig.expanding(min_periods=MIN_HIST).quantile(Q).shift(1)
    valid = (thr.notna() & sig.notna()).to_numpy()
    stress = (sig >= thr).to_numpy() & valid
    return stress, valid


def coupling_gap(X: np.ndarray, est: str = 'sample') -> float:
    """Stress-minus-calm gap of ``1/2 log det R`` (negative = coupling)."""
    h, ends = (rolling_half_logdet_lw if est == 'lw' else rolling_half_logdet_corr)(X)
    stress, valid = stress_labels(X)
    s, v = stress[ends], valid[ends]
    calm = v & ~s
    return float(h[s].mean() - h[calm].mean())


# ----------------------------------------------------------------------- DGPs
def _corr_chol(rho: float, N: int) -> np.ndarray:
    R = np.full((N, N), rho) + (1 - rho) * np.eye(N)
    return np.linalg.cholesky(R)


def _t_shocks(rng, size, nu=5.0):
    """Unit-variance multivariate-t innovations (common chi-square mixing)."""
    g = rng.standard_normal(size)
    w = rng.chisquare(nu, size=size[0])[:, None] / nu
    return g / np.sqrt(w) * np.sqrt((nu - 2) / nu)


def simulate(dgp: str, rng, N: int = 10) -> np.ndarray:
    if dgp == 'iid_t':
        return 0.01 * _t_shocks(rng, (T, N)) @ _corr_chol(RHO, N).T
    if dgp == 'ccc_gjr':
        z = _t_shocks(rng, (T, N)) @ _corr_chol(RHO, N).T
        om, a, g, b = 0.02, 0.03, 0.08, 0.91
        h = np.full(N, om / (1 - a - g / 2 - b))
        out = np.empty((T, N))
        for t in range(T):
            e = np.sqrt(h) * z[t]
            out[t] = e
            h = om + (a + g * (e < 0)) * e * e + b * h
        return out / 100.0
    if dgp in STRESS_RHO:
        p_cs, p_sc = 0.004, 0.03              # calm->stress, stress->calm
        state = np.zeros(T, dtype=bool)
        for t in range(1, T):
            u = rng.random()
            state[t] = (u < p_cs) if not state[t - 1] else (u >= p_sc)
        Lc = _corr_chol(RHO, N)
        Ls = _corr_chol(STRESS_RHO[dgp], N)
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
    """Rebuild returns from resampled residual rows. ``idx`` is (T,) or (B, T);
    the GJR recursion runs on all B draws at once."""
    one = idx.ndim == 1
    idx = np.atleast_2d(idx)
    z = Z[idx]                                   # B x T x N
    mu, om, a, g, b = P.T
    pers = np.minimum(a + g / 2 + b, 0.999)
    h = np.broadcast_to(om / (1 - pers), z[:, 0].shape).copy()
    out = np.empty_like(z)
    for t in range(z.shape[1]):
        e = np.sqrt(h) * z[:, t]
        out[:, t] = mu + e
        h = om + (a + g * (e < 0)) * e * e + b * h
    out /= 100.0
    return out[0] if one else out


def one_mc(args) -> dict:
    dgp, m, n_boot, N, est = args
    rng = np.random.default_rng(np.random.SeedSequence([41, DGPS.index(dgp), m, N]))
    X = simulate(dgp, rng, N)
    obs = coupling_gap(X, est)
    iid = np.array([coupling_gap(X[rng.integers(0, T, T)], est) for _ in range(n_boot)])
    P, Z = fit_gjr(X)
    sims = sim_fhs(P, Z, rng.integers(0, T, (n_boot, T)))
    fhs = np.array([coupling_gap(x, est) for x in sims])
    return {'dgp': dgp, 'N': N, 'mc': m, 'est': est, 'observed_gap': obs,
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
    ap.add_argument('--sizes', default='10:1000,20:500,48:300',
                    help='comma list N:number_of_Monte_Carlo_runs')
    ap.add_argument('--n-boot', type=int, default=199)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--dgps', default=','.join(DGPS))
    ap.add_argument('--checkpoint', default='outputs/mc/checkpoint.jsonl')
    ap.add_argument('--skip-bias', action='store_true')
    ap.add_argument('--summarise-only', action='store_true')
    ap.add_argument('--est', default='sample', help='sample or lw (Ledoit-Wolf, the paper estimator)')
    args = ap.parse_args()
    dest = Path('paper/tables')
    t0 = time.time()
    if not (args.skip_bias or args.summarise_only):
        b = bias_table()
        b.to_csv(dest / 'mc_bias.csv', index=False, float_format='%.4f')
        print(b.round(3).to_string(index=False), f'\n[{time.time() - t0:.0f} s]', flush=True)

    ck = Path(args.checkpoint)
    ck.parent.mkdir(parents=True, exist_ok=True)
    ck.touch()
    done = {(r['dgp'], r['N'], r['mc'], r.get('est', 'sample'))
            for r in map(json.loads, ck.read_text().splitlines())}
    sizes = [tuple(int(v) for v in x.split(':')) for x in args.sizes.split(',')]
    tasks = [(d, m, args.n_boot, n, args.est) for n, n_mc in sizes for m in range(n_mc)
             for d in args.dgps.split(',') if (d, n, m, args.est) not in done]
    if args.summarise_only:
        tasks = []
    print(f'{len(tasks)} Monte Carlo runs to do ({len(done)} done)', flush=True)
    with ProcessPoolExecutor(args.jobs) as ex, ck.open('a') as fh:
        futs = [ex.submit(one_mc, t) for t in tasks]
        for i, f in enumerate(as_completed(futs), 1):
            r = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in f.result().items()}
            fh.write(json.dumps(r) + '\n')
            fh.flush()
            if i % 50 == 0:
                el = time.time() - t0
                print(f'  [{i}/{len(tasks)}] ~{el / i * (len(tasks) - i) / 60:.0f} min left', flush=True)

    long = pd.DataFrame([json.loads(x) for x in ck.read_text().splitlines()])
    long['est'] = long['est'].fillna('sample') if 'est' in long else 'sample'
    long = long.drop_duplicates(['est', 'dgp', 'N', 'mc']).sort_values(['est', 'N', 'dgp', 'mc'])
    long.to_csv(dest / 'mc_size_power_replications.csv', index=False, float_format='%.5g')
    both = lambda a: (lambda d: ((d.p_iid <= a) & (d.p_fhs <= a)).mean())
    rows = []
    for (est, n, dgp), d in long.groupby(['est', 'N', 'dgp']):
        rows.append({'est': est, 'N': n, 'dgp': dgp, 'n_mc': len(d), 'mean_gap': d.observed_gap.mean(),
                     'reject_iid_5': (d.p_iid <= .05).mean(), 'reject_fhs_5': (d.p_fhs <= .05).mean(),
                     'reject_both_5': both(.05)(d),
                     'reject_iid_1': (d.p_iid <= .01).mean(), 'reject_fhs_1': (d.p_fhs <= .01).mean(),
                     'reject_both_1': both(.01)(d)})
    summ = pd.DataFrame(rows)
    summ.to_csv(dest / 'mc_size_power.csv', index=False, float_format='%.4f')
    print(summ.round(3).to_string(index=False), f'\n[{time.time() - t0:.0f} s]')


if __name__ == '__main__':
    main()
