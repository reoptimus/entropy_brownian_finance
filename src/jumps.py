"""Jump detection and relaxation estimation for the entropy state variable.

The paper's central postulate is an asymmetry: information arrival removes
entropy *discontinuously* (a constraint is added to what the market knows, and
by the constraint-monotonicity proposition the entropy ceiling can only fall),
while the return toward the unconstrained maximum is *diffusive* and gradual.
Written on the structural index :math:`\\mathcal J_t` (which rises when entropy
falls), the postulated dynamics are a non-Gaussian Ornstein--Uhlenbeck process
with one-sided jumps,

.. math:: d\\mathcal J_t = -\\kappa\\,\\mathcal J_t\\,dt + \\eta\\,dW_t + \\int z\\,N(dt,dz),\\quad z>0,

so the two things to measure are (i) the asymmetry of the jump component and
(ii) the relaxation rate :math:`\\kappa`, equivalently the half-life
:math:`\\ln 2/\\kappa`.

This module provides:

* :func:`local_scale` -- a jump-robust local scale for a difference series,
  built from bipower variation (Barndorff-Nielsen & Shephard, 2004).
* :func:`detect_jumps` -- threshold detection with that scale, returning signed
  jump sizes and the asymmetry statistics that test the postulate.
* :func:`estimate_relaxation` -- OU mean-reversion estimated on non-jump dates
  only, so the diffusive channel is identified without contamination.
* :func:`event_study` -- average path of the index around detected jumps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

__all__ = [
    'local_scale',
    'detect_jumps',
    'estimate_relaxation',
    'event_study',
    'crisis_diffusion_indicators',
]

_BP_SCALE = np.sqrt(2.0 / np.pi)  # E|Z| for a standard normal


def local_scale(dx: pd.Series, window: int = 250, min_periods: int = 60) -> pd.Series:
    """Jump-robust trailing scale of a difference series.

    Bipower variation, ``mean(|dx_t| |dx_{t-1}|) / (E|Z|)^2``, is consistent for
    the diffusive variance even in the presence of jumps, because a jump
    contaminates only the two products it enters. The trailing window is shifted
    by one so the scale used to judge date *t* does not include date *t*.
    """
    a = dx.abs()
    bp = (a * a.shift(1)).rolling(window, min_periods=min_periods).mean() / _BP_SCALE ** 2
    return np.sqrt(bp.clip(lower=0.0)).shift(1)


def detect_jumps(
    series: pd.Series,
    threshold: float = 4.0,
    window: int = 250,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Detect jumps in ``series`` from its first differences.

    Returns a frame indexed like ``series`` with the difference, the robust
    local scale, the standardised difference, a boolean jump flag and the signed
    jump size (zero on non-jump dates). A *positive* jump in the structural
    index is an entropy-destroying event.
    """
    dx = series.diff()
    scale = local_scale(dx, window, min_periods)
    z = dx / scale
    is_jump = z.abs() >= threshold
    out = pd.DataFrame({
        'dx': dx,
        'scale': scale,
        'z': z,
        'is_jump': is_jump.fillna(False),
    })
    out['jump_size'] = np.where(out['is_jump'], out['dx'], 0.0)
    out['jump_up'] = out['is_jump'] & (out['dx'] > 0)
    out['jump_down'] = out['is_jump'] & (out['dx'] < 0)
    return out


def jump_asymmetry(jumps: pd.DataFrame, periods_per_year: float = 252.0) -> dict:
    """Summary statistics testing the one-sided-jump postulate."""
    up = jumps.loc[jumps['jump_up'], 'dx']
    down = jumps.loc[jumps['jump_down'], 'dx']
    n_obs = int(jumps['z'].notna().sum())
    years = n_obs / periods_per_year if n_obs else np.nan

    n_up, n_down = len(up), len(down)
    # Binomial test of equal up/down jump counts under the symmetric null.
    if n_up + n_down > 0:
        from scipy import stats as _st
        p_binom = float(_st.binomtest(n_up, n_up + n_down, 0.5).pvalue)
    else:
        p_binom = np.nan

    return {
        'n_obs': n_obs,
        'n_jump_up': n_up,
        'n_jump_down': n_down,
        'jump_ratio_up_down': float(n_up / n_down) if n_down else np.inf,
        'p_binomial_symmetry': p_binom,
        'intensity_up_per_year': float(n_up / years) if years else np.nan,
        'intensity_down_per_year': float(n_down / years) if years else np.nan,
        'mean_size_up': float(up.mean()) if n_up else np.nan,
        'mean_size_down': float(down.mean()) if n_down else np.nan,
        'max_size_up': float(up.max()) if n_up else np.nan,
        'max_size_down': float(down.min()) if n_down else np.nan,
        'skew_of_changes': float(jumps['dx'].skew()),
        'share_of_variance_up': float((up ** 2).sum() / (jumps['dx'] ** 2).sum())
        if n_obs else np.nan,
        'share_of_variance_down': float((down ** 2).sum() / (jumps['dx'] ** 2).sum())
        if n_obs else np.nan,
    }


def estimate_relaxation(
    series: pd.Series,
    jumps: pd.DataFrame | None = None,
    hac_lags: int = 21,
) -> dict:
    """Estimate the OU relaxation rate of ``series`` on non-jump dates.

    Regresses ``dJ_t`` on ``J_{t-1}`` with an intercept, excluding dates flagged
    as jumps (and the date immediately after a jump, whose difference still
    contains the jump). The slope is ``-kappa`` per period; the half-life is
    ``ln 2 / kappa``. A negative slope with a positive long-run level is exactly
    the relaxation postulate; an insignificant or positive slope would refute
    it.
    """
    df = pd.DataFrame({'level': series.shift(1), 'd': series.diff()}).dropna()
    if jumps is not None:
        mask = jumps['is_jump'].reindex(df.index).fillna(False)
        mask = mask | jumps['is_jump'].shift(1).reindex(df.index).fillna(False)
        df = df.loc[~mask.astype(bool)]
    X = sm.add_constant(df[['level']])
    model = sm.OLS(df['d'], X).fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
    slope = float(model.params['level'])
    kappa = -slope
    return {
        'n_used': int(len(df)),
        'slope': slope,
        'slope_se': float(model.bse['level']),
        'slope_tstat': float(model.tvalues['level']),
        'slope_pvalue': float(model.pvalues['level']),
        'kappa': kappa,
        'half_life_days': float(np.log(2.0) / kappa) if kappa > 0 else np.nan,
        'long_run_level': float(-model.params['const'] / slope) if slope != 0 else np.nan,
    }


def event_study(
    series: pd.Series,
    jumps: pd.DataFrame,
    pre: int = 30,
    post: int = 120,
    direction: str = 'up',
) -> pd.DataFrame:
    """Average path of ``series`` around detected jumps, demeaned at t=0.

    Returns a frame indexed by event time with the mean and the 25th/75th
    percentile of the level relative to its value on the event date.
    """
    col = 'jump_up' if direction == 'up' else 'jump_down'
    idx = np.flatnonzero(jumps[col].to_numpy())
    values = series.to_numpy()
    paths = []
    for i in idx:
        lo, hi = i - pre, i + post + 1
        if lo < 0 or hi > len(values):
            continue
        paths.append(values[lo:hi] - values[i])
    if not paths:
        return pd.DataFrame()
    P = np.vstack(paths)
    return pd.DataFrame({
        'mean': P.mean(axis=0),
        'q25': np.percentile(P, 25, axis=0),
        'q75': np.percentile(P, 75, axis=0),
        'n_events': P.shape[0],
    }, index=pd.Index(range(-pre, post + 1), name='event_day'))


def crisis_diffusion_indicators(
    series: pd.Series,
    jumps: pd.DataFrame,
    kappa: float,
    lookback: int = 250,
) -> pd.DataFrame:
    """Two operational indicators built from the jump/diffusion split.

    ``crisis`` is the standardised positive-jump component: the size of today's
    entropy-destroying jump in units of the local diffusive scale, zero when no
    jump is detected. It answers "is information arriving now, and how much".

    ``unhealed`` is the share of the trailing maximum of the index that has not
    yet relaxed away, ``J_t / max(J_{t-lookback..t})``. It answers "how far is
    the market from having absorbed what already arrived". Under pure
    relaxation at rate ``kappa`` it decays with half-life ``ln2/kappa``; a value
    that stays near one for longer than that is a market that is not healing.
    """
    crisis = np.where(jumps['jump_up'], jumps['z'], 0.0)
    trailing_max = series.rolling(lookback, min_periods=20).max()
    out = pd.DataFrame({
        'crisis': crisis,
        'unhealed': (series / trailing_max).clip(upper=1.0),
    }, index=series.index)
    out['expected_half_life'] = np.log(2.0) / kappa if kappa > 0 else np.nan
    return out
