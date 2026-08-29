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
    'group_jump_episodes',
    'episode_channel_budget',
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


def group_jump_episodes(jumps: pd.DataFrame, gap: int = 40,
                        direction: str = 'up') -> list[tuple]:
    """Merge individually detected jumps into episodes.

    A single flagged jump is rarely an isolated arrival: the event study
    (:func:`event_study`) shows the index typically keeps rising for another
    dozen-odd days after the date a jump is *signalled*, i.e. detected jumps
    cluster into runs belonging to the same information episode (a crisis
    unfolds over weeks, not one trading day). Two jump dates less than ``gap``
    trading days apart (by position in ``jumps.index``, not calendar days) are
    merged into the same episode.

    Returns a list of ``(start, end, member_dates)`` tuples in chronological
    order, ``start``/``end`` being the first/last *detected* jump date in the
    episode -- callers that need the episode's full extent (e.g. to find where
    the index actually peaks) should look some days past ``end``, since the
    detector's threshold crossing is not the same as the index's local
    maximum.
    """
    col = 'jump_up' if direction == 'up' else 'jump_down'
    flagged = jumps.index[jumps[col].to_numpy()]
    if len(flagged) == 0:
        return []
    pos = {d: i for i, d in enumerate(jumps.index)}

    episodes = []
    start = prev = flagged[0]
    members = [flagged[0]]
    for d in flagged[1:]:
        if pos[d] - pos[prev] > gap:
            episodes.append((start, prev, members))
            start, members = d, []
        members.append(d)
        prev = d
    episodes.append((start, prev, members))
    return episodes


def episode_channel_budget(ent: pd.DataFrame, episodes: list[tuple],
                           pre: int = 10, post_search: int = 20) -> pd.DataFrame:
    """Decompose each episode's move in J into its three additive channels.

    ``J = (-h_dep_ew) + d_odd + d_even`` is an exact identity (structural
    index = dependence contribution + asymmetry channel + tail-weight
    channel), so for any two dates the change in each term sums exactly to the
    change in J. This computes that split from just before an episode starts
    to wherever J actually peaks within (and slightly past) the episode, which
    gives a per-crisis budget instead of the single number pooled over an
    entire stress regime that :func:`~src.hypotheses.h6_tail_dependence_coupling`
    reports.

    Reading the three shares against the paper's volume-time bridge
    (``paper/main_fr.tex``, Section 2 and its contrapositive, Section 2.4):
    a random, information-free volume clock produces a *symmetric* scale
    mixture -- fatter tails without directional skew, i.e. the **even**
    (tail-weight) channel -- while a genuinely constraining, directional piece
    of news produces **asymmetry**, i.e. the **odd** channel (this is also
    postulate P2's mechanism). The **dependence** channel is a third,
    logically separate source: news common to the whole cross-section, which
    couples assets together regardless of whether it is itself symmetric or
    not. So a dependence-dominated episode reads as systemic/common-factor, an
    odd-dominated one as directional/informational, and an even-dominated one
    as consistent with elevated, undirected trading intensity -- a volume-
    clock signature -- though none of the three panels used here carries
    actual traded volume, so this reading is a model-consistent interpretation
    of the channel split, not a direct measurement of volume.

    Note this uses the *level* channels (``h_dep_ew``/``d_odd``/``d_even``),
    not their date-to-date differences, so a single noisy day cannot flip the
    attribution; only the well-defined start/peak comparison matters.
    """
    n = len(ent)
    pos = {d: i for i, d in enumerate(ent.index)}
    rows = []
    for start, end, members in episodes:
        i_start = max(0, pos[start] - pre)
        i_end = min(n - 1, pos[end] + post_search)
        base = ent.iloc[i_start]
        window = ent.iloc[pos[start]:i_end + 1]
        if window['J'].isna().all():
            continue
        peak_date = window['J'].idxmax()
        peak = ent.loc[peak_date]

        d_dep = float(-peak['h_dep_ew'] - (-base['h_dep_ew']))
        d_odd = float(peak['d_odd'] - base['d_odd'])
        d_even = float(peak['d_even'] - base['d_even'])
        total = d_dep + d_odd + d_even
        dj = float(peak['J'] - base['J'])

        dominant = max(
            [('dependence', d_dep), ('odd', d_odd), ('even', d_even)],
            key=lambda kv: kv[1],
        )[0]
        rows.append({
            'start': start, 'end': end, 'peak': peak_date,
            'n_jumps': len(members), 'days_to_peak': pos[peak_date] - pos[start],
            'base_date': base.name, 'dJ': dj,
            'd_dependence': d_dep, 'd_odd': d_odd, 'd_even': d_even,
            'share_dependence': d_dep / total if total else np.nan,
            'share_odd': d_odd / total if total else np.nan,
            'share_even': d_even / total if total else np.nan,
            'dominant_channel': dominant,
        })
    return pd.DataFrame(rows)


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
