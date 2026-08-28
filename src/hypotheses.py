"""The paper's hypotheses, each as a function returning a tidy result frame.

H1  Regime signature       stress raises h_vol, lowers h_dep, raises D, raises J
H2  Partial compensation   d h_dep = -beta * d h_vol + u with beta > 0: the two
                           channels move in opposite directions and in
                           proportion. beta is the compensation rate (0 = none,
                           1 = det Sigma conserved), reported rather than
                           asserted; see compensation_slope()
H3  Jump asymmetry         changes in J are one-sided: jumps up, diffusion down
H4  Relaxation             J mean-reverts at an estimable rate after a jump
H5  Skewness signature     entropy jumps carry negative market skewness (P2)
H6  Tail-dependence coupling  correlation and tail weight surge together
H7  Predictive power       knowing today's entropy says something about the
                           next 21 days that today's volatility does not --
                           judged only on dates unseen at estimation time

Every test that compares regimes or averages an overlapping series uses a
stationary block bootstrap, because a 252-day rolling window makes consecutive
observations share 251 of 252 underlying return days; ordinary standard errors
on such a series are meaningless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .jumps import (
    crisis_diffusion_indicators,
    detect_jumps,
    estimate_relaxation,
    event_study,
    jump_asymmetry,
)

__all__ = [
    'block_bootstrap_diff',
    'h1_regime_signature',
    'h2_ceiling_compensation',
    'compensation_slope',
    'h3_jump_asymmetry',
    'h4_relaxation',
    'h5_skewness_signature',
    'h6_tail_dependence_coupling',
    'h7_incremental_value',
    'run_all',
]


def block_bootstrap_diff(x: pd.Series, mask: pd.Series, block: int = 252,
                         n_boot: int = 2000, seed: int = 0) -> dict:
    """Difference in means between two regimes with a block-bootstrap p-value.

    Resamples circular blocks of length ``block`` from the joint (value, regime)
    series, which preserves both the persistence of the entropy series and the
    clustering of the regime indicator.
    """
    d = pd.DataFrame({'x': x, 'm': mask}).dropna()
    v, m = d['x'].to_numpy(), d['m'].to_numpy().astype(bool)
    n = len(v)
    if n == 0 or m.sum() == 0 or (~m).sum() == 0:
        return {'diff': np.nan, 'p_value': np.nan, 'n': n}
    obs = v[m].mean() - v[~m].mean()

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        # Null: regime labels carry no information about the level. Pairing a
        # block-resampled value path with the *original* label sequence breaks
        # the link while preserving both the persistence of the series and the
        # clustering of the regime indicator.
        vb = v[idx]
        stats[b] = vb[m].mean() - vb[~m].mean()
    stats = stats[~np.isnan(stats)]
    p = float(np.mean(np.abs(stats - np.nanmean(stats)) >= abs(obs)))
    return {
        'stress_mean': float(v[m].mean()),
        'calm_mean': float(v[~m].mean()),
        'diff': float(obs),
        'p_value': p,
        'n_stress': int(m.sum()),
        'n_calm': int((~m).sum()),
    }


def h1_regime_signature(ent: pd.DataFrame, stress: pd.Series,
                        block: int = 252, n_boot: int = 2000) -> pd.DataFrame:
    """Regime means of every channel, with block-bootstrap significance."""
    cols = ['h_vol', 'h_dep', 'h_cov', 'd_total', 'd_odd', 'd_even', 'h_tot', 'J',
            'skew_market', 'exkurt_market', 'mean_corr', 'n_eff_modes', 'tail_dep']
    rows = []
    for c in cols:
        if c not in ent.columns:
            continue
        r = block_bootstrap_diff(ent[c], stress, block, n_boot)
        r['channel'] = c
        rows.append(r)
    out = pd.DataFrame(rows)
    order = ['channel', 'stress_mean', 'calm_mean', 'diff', 'p_value', 'n_stress', 'n_calm']
    return out[[c for c in order if c in out.columns]]


def _variance_reduction(a: pd.Series, b: pd.Series) -> dict:
    d = pd.DataFrame({'a': a, 'b': b}).dropna()
    va, vb = d['a'].var(), d['b'].var()
    vs = (d['a'] + d['b']).var()
    return {
        'corr': float(d['a'].corr(d['b'])),
        'sd_sum': float(np.sqrt(vs)),
        'sd_benchmark': float(np.sqrt(va + vb)),
        'var_reduction_pct': float(100.0 * (1.0 - vs / (va + vb))),
        'n': int(len(d)),
    }


def h2_ceiling_compensation(ent: pd.DataFrame, stress: pd.Series) -> pd.DataFrame:
    """Compensation between the scale and dependence channels, by regime.

    Also reports the same statistic for the *total* entropy ``h_tot``, whose
    changes are ``d h_cov - d D``. If the Gaussian ceiling is conserved while
    total entropy is not, the informational content of a crisis lies in the
    shape channel -- which is the claim that motivates the rest of the paper.
    """
    rows = []
    regimes = {'all': pd.Series(True, index=ent.index),
               'calm': ~stress.astype(bool),
               'stress': stress.astype(bool)}
    for name, m in regimes.items():
        sub = ent.loc[m.reindex(ent.index).fillna(False)]
        r = _variance_reduction(sub['dh_vol'], sub['dh_dep'])
        r['regime'] = name
        r['pair'] = 'dh_vol vs dh_dep'
        rows.append(r)
        r2 = _variance_reduction(sub['dh_cov'], -sub['dD'])
        r2['regime'] = name
        r2['pair'] = 'dh_cov vs -dD'
        rows.append(r2)
        rows.append({'regime': name, 'pair': 'levels',
                     'corr': np.nan, 'sd_sum': float(sub['dh_cov'].std()),
                     'sd_benchmark': float(sub['dh_tot'].std()),
                     'var_reduction_pct': np.nan, 'n': int(len(sub))})
    out = pd.DataFrame(rows)
    return out[['regime', 'pair', 'corr', 'sd_sum', 'sd_benchmark', 'var_reduction_pct', 'n']]


def compensation_slope(ent: pd.DataFrame, stress: pd.Series, hac_lags: int = 21) -> pd.DataFrame:
    """Estimate the compensation rate beta of H2: d h_dep = -beta * d h_vol + u.

    The earlier formulation of H2 bundled a sign claim and a magnitude claim into
    one binary verdict ("the Gaussian ceiling is more stable than its parts").
    Separating them is what makes the hypothesis reportable: beta is the fraction
    of a scale-entropy move absorbed by dependence, so beta = 0 is no
    compensation, beta = 1 is conservation of det Sigma, and beta < 0 is
    amplification. The hypothesis is then simply beta > 0, with the magnitude
    reported rather than asserted.
    """
    rows = []
    regimes = {'all': pd.Series(True, index=ent.index),
               'calm': ~stress.astype(bool),
               'stress': stress.astype(bool)}
    for name, m in regimes.items():
        sub = ent.loc[m.reindex(ent.index).fillna(False)].dropna(
            subset=['dh_vol', 'dh_dep'])
        if len(sub) < 30:
            continue
        model = sm.OLS(sub['dh_dep'], sm.add_constant(sub['dh_vol'])).fit(
            cov_type='HAC', cov_kwds={'maxlags': hac_lags})
        beta = -float(model.params['dh_vol'])
        se = float(model.bse['dh_vol'])
        rows.append({
            'regime': name,
            'beta': beta,
            'beta_se': se,
            'beta_lo95': beta - 1.96 * se,
            'beta_hi95': beta + 1.96 * se,
            'r2': float(model.rsquared),
            'significantly_positive': bool(beta - 1.96 * se > 0),
            'n': int(len(sub)),
        })
    return pd.DataFrame(rows)


def h3_jump_asymmetry(ent: pd.DataFrame, threshold: float = 4.0,
                      scale_window: int = 250, column: str = 'J') -> tuple:
    """One-sidedness of the jump component of the structural index."""
    jumps = detect_jumps(ent[column], threshold, scale_window)
    stats = jump_asymmetry(jumps)
    stats['column'] = column
    stats['threshold'] = threshold
    return pd.DataFrame([stats]), jumps


def h4_relaxation(ent: pd.DataFrame, jumps: pd.DataFrame, column: str = 'J',
                  pre: int = 30, post: int = 120) -> tuple:
    """Mean reversion of the structural index, estimated off jump dates."""
    rel = estimate_relaxation(ent[column], jumps)
    rel['column'] = column
    ev = event_study(ent[column], jumps, pre, post, 'up')
    return pd.DataFrame([rel]), ev


def h5_skewness_signature(ent: pd.DataFrame, jumps: pd.DataFrame) -> pd.DataFrame:
    """Do entropy jumps carry the negative sign that postulate P2 requires?

    A squared negentropy is blind to direction, so the sign has to come from
    somewhere else. P2 says the constraint information imposes is one-sided:
    liquidity is withdrawn on the sell side, so an entropy jump should arrive
    with negative market skewness. Tested three ways: the contemporaneous
    correlation between the jump and the change in market skewness, the mean
    skewness change on jump dates against all other dates, and the split of the
    asymmetry channel ``d_odd`` between the two groups.
    """
    d = ent.copy()
    d['is_jump_up'] = jumps['jump_up'].reindex(d.index).fillna(False)
    rows = []
    for col in ['dskew_market', 'skew_market', 'dD_odd', 'd_odd', 'dtail_dep']:
        if col not in d.columns:
            continue
        g = d.dropna(subset=[col])
        up = g.loc[g['is_jump_up'], col]
        rest = g.loc[~g['is_jump_up'], col]
        rows.append({
            'variable': col,
            'mean_on_jump_up': float(up.mean()) if len(up) else np.nan,
            'mean_elsewhere': float(rest.mean()),
            'diff': float(up.mean() - rest.mean()) if len(up) else np.nan,
            'n_jump_up': int(len(up)),
            'corr_with_dJ': float(g[col].corr(g['dJ'])) if 'dJ' in g else np.nan,
        })
    return pd.DataFrame(rows)


def h6_tail_dependence_coupling(ent: pd.DataFrame, stress: pd.Series) -> pd.DataFrame:
    """Do the dependence and tail channels move together, and by how much?

    Reports, by regime, the correlation between the two entropy-destroying
    channels and the share of the variance of ``dJ`` each contributes
    (``cov(channel, dJ) / var(dJ)``, which sums to one by bilinearity).
    """
    rows = []
    regimes = {'all': pd.Series(True, index=ent.index),
               'calm': ~stress.astype(bool),
               'stress': stress.astype(bool)}
    for name, m in regimes.items():
        sub = ent.loc[m.reindex(ent.index).fillna(False)].dropna(
            subset=['dh_dep_ew', 'dD', 'dJ', 'dD_even', 'dD_odd'])
        dep = -sub['dh_dep_ew']
        var_j = sub['dJ'].var()
        rows.append({
            'regime': name,
            'n': int(len(sub)),
            'corr_dep_shape': float(dep.corr(sub['dD'])),
            'corr_dep_tail': float(dep.corr(sub['dD_even'])),
            'corr_dep_skewchan': float(dep.corr(sub['dD_odd'])),
            'share_var_dJ_dependence': float(dep.cov(sub['dJ']) / var_j) if var_j else np.nan,
            'share_var_dJ_shape': float(sub['dD'].cov(sub['dJ']) / var_j) if var_j else np.nan,
            'corr_taildep_J': float(sub['dtail_dep'].corr(sub['dJ']))
            if 'dtail_dep' in sub else np.nan,
        })
    return pd.DataFrame(rows)


def _walk_forward_predict(d: pd.DataFrame, target: str, cols: list,
                          min_train: int, refit: int) -> pd.Series:
    """Expanding-window walk-forward predictions, refitting every ``refit`` dates."""
    y = d[target]
    preds = pd.Series(np.nan, index=d.index)
    start = min_train
    while start < len(d):
        stop = min(start + refit, len(d))
        train = d.iloc[:start]
        test = d.iloc[start:stop]
        try:
            model = sm.OLS(train[target], sm.add_constant(train[cols])).fit()
            preds.iloc[start:stop] = model.predict(
                sm.add_constant(test[cols], has_constant='add')).to_numpy()
        except Exception:  # pragma: no cover - degenerate design matrix
            pass
        start = stop
    return preds


def _diebold_mariano(e1: np.ndarray, e2: np.ndarray, lags: int) -> tuple:
    """Diebold--Mariano statistic on squared-error differentials, HAC-corrected."""
    d = e1 ** 2 - e2 ** 2
    d = d[np.isfinite(d)]
    if len(d) < 30:
        return np.nan, np.nan
    model = sm.OLS(d, np.ones(len(d))).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return float(model.tvalues[0]), float(model.pvalues[0])


def h7_incremental_value(ent: pd.DataFrame,
                         targets=('future_rv', 'future_drawdown', 'future_worst_loss'),
                         extras=('J', 'd_total', 'h_dep', 'crisis', 'unhealed'),
                         hac_lags: int = 21, min_train: int | None = None,
                         refit: int = 252, thin: int = 21) -> pd.DataFrame:
    """Does an entropy channel add to volatility in forecasting forward risk?

    The earlier draft's predictive hypothesis was evaluated on a single 70/30
    chronological split and its own Limitations section asked for a genuine
    walk-forward at a non-overlapping frequency. That is what happens here: an
    expanding window refitted annually, evaluated only on dates spaced
    ``thin`` apart so that consecutive forecast targets do not overlap, and a
    Diebold--Mariano test of the squared-error differential against the
    volatility-only benchmark rather than a bare MSE comparison.
    """
    # One common evaluation grid, so every model is scored on the same dates
    # and the Diebold--Mariano differentials are actually paired.
    eval_dates = ent.index[::thin]
    if min_train is None:
        # Enough history to fit, but never more than 40% of a short sample.
        min_train = max(252, min(1500, int(0.40 * len(ent))))

    rows = []
    for target in targets:
        if target not in ent.columns:
            continue
        base_errors = None
        for extra in ('none',) + tuple(extras):
            if extra != 'none' and extra not in ent.columns:
                continue
            cols = ['h_vol'] + ([] if extra == 'none' else [extra])
            d = ent.dropna(subset=[target] + cols)
            if len(d) < min_train + 2 * min(refit, 126):
                continue
            preds = _walk_forward_predict(d, target, cols, min_train,
                                          min(refit, max(63, len(d) // 8)))
            err = (d[target] - preds).dropna()
            err = err[err.index.isin(eval_dates)]
            full = sm.OLS(d[target], sm.add_constant(d[cols])).fit(
                cov_type='HAC', cov_kwds={'maxlags': hac_lags})
            row = {
                'target': target,
                'extra_regressor': extra,
                'full_sample_r2': float(full.rsquared),
                'extra_coef': float(full.params.get(extra, np.nan)),
                'extra_pvalue': float(full.pvalues.get(extra, np.nan)),
                'oos_mse': float(np.mean(err.to_numpy() ** 2)),
                'oos_mae': float(np.mean(np.abs(err.to_numpy()))),
                'n_oos': int(len(err)),
            }
            if extra == 'none':
                base_errors = err
                row['dm_tstat'] = np.nan
                row['dm_pvalue'] = np.nan
                row['oos_mse_vs_vol_only_pct'] = 0.0
            elif base_errors is not None:
                common = err.index.intersection(base_errors.index)
                t, p = _diebold_mariano(base_errors.loc[common].to_numpy(),
                                        err.loc[common].to_numpy(), hac_lags)
                row['dm_tstat'] = t
                row['dm_pvalue'] = p
                row['oos_mse_vs_vol_only_pct'] = float(
                    100.0 * (row['oos_mse'] / float(np.mean(base_errors.to_numpy() ** 2)) - 1.0))
            rows.append(row)
    return pd.DataFrame(rows)


def run_all(ent: pd.DataFrame, stress: pd.Series, cfg: dict) -> dict:
    """Run every hypothesis test and return the tables as a dict."""
    jcfg = cfg.get('jumps', {})
    threshold = float(jcfg.get('threshold', 4.0))
    scale_window = int(jcfg.get('scale_window', 250))
    pre, post = int(jcfg.get('event_pre', 30)), int(jcfg.get('event_post', 120))

    h3, jumps = h3_jump_asymmetry(ent, threshold, scale_window, 'J')
    h4, ev = h4_relaxation(ent, jumps, 'J', pre, post)

    # The operational indicators are themselves candidate predictors, so they
    # are built here and handed to H7 alongside the raw channels.
    ind = crisis_diffusion_indicators(ent['J'], jumps, float(h4['kappa'].iloc[0]))
    ent = ent.join(ind[['crisis', 'unhealed']])

    return {
        'h1_regime_signature': h1_regime_signature(ent, stress),
        'h2_ceiling_compensation': h2_ceiling_compensation(ent, stress),
        'h2_compensation_slope': compensation_slope(ent, stress),
        'h3_jump_asymmetry': h3,
        'h4_relaxation': h4,
        'h4_event_study': ev,
        'h5_skewness_signature': h5_skewness_signature(ent, jumps),
        'h6_tail_dependence_coupling': h6_tail_dependence_coupling(ent, stress),
        'h7_incremental_value': h7_incremental_value(ent),
        'indicators': ind,
        'jumps': jumps,
    }
