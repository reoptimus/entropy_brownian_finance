"""Entropy-calibrated stress scenarios.

A classical stress test picks shocks by hand -- volatility doubles, correlations
go to 0.9, the index falls three sigma -- and has no way of saying how
implausible the bundle is, nor whether its pieces are mutually consistent. The
entropy view replaces the hand-picked bundle by a single scalar:

    a scenario of severity ``eta`` is the worst conditional return distribution
    lying within a Kullback--Leibler ball of radius ``eta`` nats around the
    market's current conditional distribution.

Three things follow.

1. The scenario cannot be internally inconsistent: the mean shift and the
   variance inflation are chosen jointly, on one budget, rather than shocked
   piecemeal.
2. ``eta`` is measured in the same unit as the information the market actually
   revealed, ``I_t^{(h)} = KL(p_t \\Vert p_{t-h})``, so a severity carries a
   return period instead of an adjective.
3. Any *existing* scenario -- a regulatory template, a historical replay, a
   committee's judgement -- can be priced on the same scale
   (:func:`price_scenario`), which turns "is this severe enough?" into an
   arithmetic question.

The worst-case-within-a-KL-ball construction is Glasserman & Xu (2014); relative
entropy as a plausibility constraint on stress scenarios is Breuer & Csiszar
(2013). What is added here is the calibration of the radius from the market's
own realised information flow rather than from the modeller's judgement.

Geometry of the Gaussian ball
-----------------------------
For a linear portfolio only the one-dimensional projection matters. Writing
``m0``, ``s0`` for today's portfolio mean and standard deviation and
``u=(m-m0)/s0``, ``v=s/s0``, the projected KL constraint is

.. math:: g(u,v) = \\tfrac12\\left(u^2 + v^2 - 1 - 2\\log v\\right) \\le \\eta,

and the worst case for any risk measure of the form ``-m + c s`` (Gaussian VaR
and expected shortfall are both of this form) solves

.. math:: u = -t,\\qquad v - v^{-1} = c\\,t,

with ``t \\ge 0`` fixed by ``g(u,v)=\\eta``. Note the immediate sanity check: for
a pure mean shock (``c=0``) the worst case is exactly a
``\\sqrt{2\\eta}``-sigma move, so one nat buys 1.41 sigma.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .entropy import gaussian_kl

__all__ = [
    'es_coefficient',
    'gaussian_ball_scenario',
    'price_scenario',
    'classical_scenario',
    'calibrated_covariance',
    'composition_calibrated_scenario',
    'severity_ladder',
    'return_period_of',
    'build_scenario_table',
    'tilt_weights',
    'kl_of_tilt',
    'worst_case_tilt',
    'reverse_stress',
]


# --------------------------------------------------------------------------
# Gaussian KL ball: the main scenario generator
# --------------------------------------------------------------------------

def es_coefficient(level: float = 0.99) -> float:
    """Coefficient ``c`` such that Gaussian ES at ``level`` is ``-m + c*s``."""
    z = stats.norm.ppf(level)
    return float(stats.norm.pdf(z) / (1.0 - level))


def _ball_constraint(t: float, c: float) -> tuple:
    u = -t
    v = 0.5 * (c * t + np.sqrt(c * c * t * t + 4.0))
    g = 0.5 * (u * u + v * v - 1.0 - 2.0 * np.log(v))
    return u, v, g


def gaussian_ball_scenario(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    eta: float,
    risk_level: float = 0.99,
    objective: str = 'es',
) -> dict:
    """Worst-case conditional distribution within a KL ball of radius ``eta``.

    ``objective`` selects what the scenario is worst *for*: ``'es'`` (expected
    shortfall at ``risk_level``), ``'var'``, or ``'mean'`` (pure expected loss,
    for which the answer is the ``sqrt(2*eta)``-sigma move).

    Returns the stressed portfolio moments, the implied volatility multiplier
    and sigma move, and the two extreme readings of where the variance
    inflation came from: entirely from marginal volatilities, or entirely from
    correlation. The truth is between them, and the split is an empirical
    question the paper answers from the observed composition of entropy jumps.
    """
    mu = np.asarray(mu, dtype=float)
    w = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    m0 = float(w @ mu)
    s0 = float(np.sqrt(w @ cov @ w))

    c = {'es': es_coefficient(risk_level), 'var': float(stats.norm.ppf(risk_level)),
         'mean': 0.0}[objective]

    if eta <= 0:
        u, v = 0.0, 1.0
    else:
        hi = 1.0
        while _ball_constraint(hi, c)[2] < eta:
            hi *= 1.6
            if hi > 1e6:
                raise RuntimeError('Radius too large to bracket.')
        t = optimize.brentq(lambda x: _ball_constraint(x, c)[2] - eta, 0.0, hi,
                            xtol=1e-14, rtol=1e-12)
        u, v, _ = _ball_constraint(t, c)

    m, s = m0 + u * s0, v * s0
    z = float(stats.norm.ppf(risk_level))

    # Where could the variance inflation have come from?
    sd = np.sqrt(np.diag(cov))
    own = float(np.sum((w * sd) ** 2))                    # sum_i w_i^2 sigma_i^2
    cross = float((w @ sd) ** 2 - own)                    # sum_{i != j} w_i w_j s_i s_j
    implied_corr = (s ** 2 - own) / cross if cross > 0 else np.nan

    return {
        'eta': float(eta),
        'objective': objective,
        'sigma_move': float(u),
        'vol_multiplier': float(v),
        'base_mean': m0,
        'base_vol_daily': s0,
        'stressed_mean': float(m),
        'stressed_vol_daily': float(s),
        'base_vol_ann': float(s0 * np.sqrt(252)),
        'stressed_vol_ann': float(s * np.sqrt(252)),
        f'base_VaR{int(risk_level * 100)}': float(-m0 + z * s0),
        f'stressed_VaR{int(risk_level * 100)}': float(-m + z * s),
        f'base_ES{int(risk_level * 100)}': float(-m0 + es_coefficient(risk_level) * s0),
        f'stressed_ES{int(risk_level * 100)}': float(-m + es_coefficient(risk_level) * s),
        'implied_mean_corr_if_all_dependence': float(implied_corr),
        'feasible_as_correlation_only': bool(np.isfinite(implied_corr) and implied_corr <= 1.0),
        'base_mean_corr': float((np.sum(cov / np.outer(sd, sd)) - len(sd))
                                / (len(sd) * (len(sd) - 1))),
    }


def price_scenario(mu1, cov1, mu0, cov0) -> float:
    """Informational price, in nats, of an arbitrary scenario.

    ``KL(N(mu1, cov1) || N(mu0, cov0))``: how far the scenario's distribution
    sits from the market's current conditional distribution. Paired with a
    severity ladder this converts any hand-built scenario into a return period.
    """
    return gaussian_kl(mu1, cov1, mu0, cov0)


def classical_scenario(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    vol_multiplier: float = 2.0,
    target_correlation: float = 0.90,
    sigma_move: float = -3.0,
    risk_level: float = 0.99,
) -> dict:
    """A textbook stress bundle, and what it costs in nats.

    Marginal volatilities multiplied, every pairwise correlation forced to a
    single number, and a directional shock of ``sigma_move`` base standard
    deviations. Each leg is defensible on its own; the bundle has no
    plausibility measure attached to it -- until it is priced against the
    market's own conditional distribution, which is what this function does.
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(mu)

    sd0 = np.sqrt(np.diag(cov))
    R0 = cov / np.outer(sd0, sd0)
    sd1 = sd0 * vol_multiplier
    R1 = np.full((n, n), target_correlation)
    np.fill_diagonal(R1, 1.0)
    cov1 = np.outer(sd1, sd1) * R1

    s0 = float(np.sqrt(w @ cov @ w))
    # Cheapest mean shift delivering a portfolio move of ``sigma_move`` base
    # standard deviations. Its informational price is exactly sigma_move^2 / 2.
    mu1 = mu + sigma_move * (cov @ w) / s0

    m1 = float(w @ mu1)
    s1 = float(np.sqrt(w @ cov1 @ w))
    z = float(stats.norm.ppf(risk_level))

    # Price each leg separately, holding the others at their current values.
    leg_mean = price_scenario(mu1, cov, mu, cov)
    leg_vol = price_scenario(mu, np.outer(sd1, sd1) * R0, mu, cov)
    leg_corr = price_scenario(mu, np.outer(sd0, sd0) * R1, mu, cov)

    return {
        'vol_multiplier': vol_multiplier,
        'target_correlation': target_correlation,
        'sigma_move': sigma_move,
        'stressed_mean': m1,
        'stressed_vol_daily': s1,
        'stressed_vol_ann': float(s1 * np.sqrt(252)),
        f'stressed_VaR{int(risk_level * 100)}': float(-m1 + z * s1),
        f'stressed_ES{int(risk_level * 100)}': float(-m1 + es_coefficient(risk_level) * s1),
        'price_mean_leg_nats': leg_mean,
        'price_vol_leg_nats': leg_vol,
        'price_correlation_leg_nats': leg_corr,
        'entropy_price_nats': price_scenario(mu1, cov1, mu, cov),
    }


# --------------------------------------------------------------------------
# Composition-calibrated scenario: which channel pays for the severity
# --------------------------------------------------------------------------

def _vol_only_covariance(cov: np.ndarray, weights: np.ndarray, s_target: float) -> np.ndarray:
    """Uniform-scaling covariance hitting ``s_target``, correlations unchanged.

    This is the composition implicit in :func:`gaussian_ball_scenario`: its
    geometry only ever constrains the *projected* (portfolio-level) KL ball,
    is silent on how a multi-asset covariance should move, and uniform
    scaling is the simplest choice consistent with that geometry -- not a
    proven ambient-KL minimum. (The actual ambient-KL-minimising covariance
    for a fixed portfolio variance is a rank-one perturbation of the
    *precision* matrix, a different and less interpretable object; this
    module does not use it, for the same reason the docstring below gives:
    composition is an empirical question, not one to solve by fiat.)
    """
    s0 = float(np.sqrt(weights @ cov @ weights))
    k = s_target / s0
    return (k * k) * cov


def calibrated_covariance(
    cov: np.ndarray, weights: np.ndarray, s_target: float, dependence_share: float,
) -> tuple:
    """Split a target portfolio volatility between the scale and dependence channels.

    A diversified portfolio's volatility cannot be pushed arbitrarily high by
    correlation alone: at fixed marginal volatilities, ``rho=1`` (everything
    moving together) is the ceiling, and it is a modest one for a large book
    -- for the equal-weight, 20-name S&P sector panel this construction is
    tested on, it is only about 1.5x today's volatility. So "the dependence
    channel explains ``share`` of a severe scenario" cannot mean literal
    uniform correlation solving ``w'*Sigma*w = s_target**2`` on its own once
    ``s_target`` clears that ceiling -- there is no such correlation matrix.
    What it can mean, and what this function builds: correlation is pushed as
    far toward its ceiling as ``dependence_share`` calls for (capped at 1, the
    physical limit), and whatever variance is still missing is closed by a
    uniform scaling on top. That scaling is applied last and uniformly, so
    the portfolio severity ``s_target`` is matched *exactly* for every
    ``dependence_share`` in ``[0, 1]`` -- what changes with the share is only
    how the same severity is spread across assets, never the severity itself.

    ``share=0`` reproduces :func:`gaussian_ball_scenario`'s own default
    (pure uniform scaling, see :func:`_vol_only_covariance`); ``share=1``
    pushes correlation to its ceiling first and scales only what is left. H1
    measures how much of a real regime's entropy destruction the dependence
    channel actually accounts for; that is the number this is meant to be
    called with, not a free parameter to guess.

    Returns ``(cov1, feasible)``; ``feasible`` is ``True`` when a uniform
    correlation at or below 1 could reach ``s_target`` on its own (so
    ``share=1`` needs no extra scaling), ``False`` when even ``rho=1`` falls
    short and scaling is unavoidable regardless of the requested share.
    """
    cov = np.asarray(cov, dtype=float)
    weights = np.asarray(weights, dtype=float)
    sd0 = np.sqrt(np.diag(cov))
    n = len(sd0)
    own = float(np.sum((weights * sd0) ** 2))
    cross_full = float((weights @ sd0) ** 2 - own)  # cross term at rho=1

    if cross_full <= 0:
        rho_cap, feasible = 0.0, False
    else:
        rho_needed = (s_target * s_target - own) / cross_full
        rho_cap = float(np.clip(rho_needed, -1.0, 1.0))
        feasible = rho_needed <= 1.0

    R_cap = np.full((n, n), rho_cap)
    np.fill_diagonal(R_cap, 1.0)
    cov_cap = np.outer(sd0, sd0) * R_cap

    share = float(np.clip(dependence_share, 0.0, 1.0))
    cov_mix = (1.0 - share) * cov + share * cov_cap
    v_mix = float(np.sqrt(weights @ cov_mix @ weights))
    k = s_target / v_mix
    return (k * k) * cov_mix, feasible


def composition_calibrated_scenario(
    mu: np.ndarray, cov: np.ndarray, weights: np.ndarray, eta: float,
    dependence_share: float, risk_level: float = 0.99, objective: str = 'es',
) -> dict:
    """A KL-ball scenario of severity ``eta``, composed the way H1 says crises are.

    :func:`gaussian_ball_scenario` fixes the severity by constraining only the
    portfolio's own projected distribution; left unconstrained, the simplest
    consistent choice is pure uniform scaling of the covariance, correlations
    untouched (:func:`_vol_only_covariance`) -- a default, not a claim about
    how real crises behave. H1 measures how real crises actually behave: the
    dependence channel moves on its own, past both the i.i.d. and the
    block-21 null, in every panel. This function keeps the severity fixed at
    ``eta`` (same portfolio mean and volatility as
    :func:`gaussian_ball_scenario`) but builds the *asset-level* covariance
    with the measured ``dependence_share`` instead of the default, and reports
    what that composition costs in ambient nats relative to the default's own
    ambient cost -- the price of choosing a historically-grounded composition
    over the simplest one.
    """
    base = gaussian_ball_scenario(mu, cov, weights, eta, risk_level, objective)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    w = np.asarray(weights, dtype=float)
    s0 = base['base_vol_daily']

    mu1 = mu + base['sigma_move'] * (cov @ w) / s0
    cov_default = _vol_only_covariance(cov, w, base['stressed_vol_daily'])
    cov1, feasible = calibrated_covariance(
        cov, w, base['stressed_vol_daily'], dependence_share)

    # ``eta`` is the *projected*, portfolio-level KL radius (see the module
    # docstring); it is not on the same scale as the full ambient KL that
    # price_scenario reports, so the composition's cost is measured against
    # the ambient cost of its own default (share=0) twin, not against eta.
    base['dependence_share'] = float(dependence_share)
    base['feasible_at_this_share'] = feasible
    base['default_composition_cost_nats'] = price_scenario(mu1, cov_default, mu, cov)
    base['entropy_price_nats'] = price_scenario(mu1, cov1, mu, cov)
    base['excess_cost_over_default_nats'] = (
        base['entropy_price_nats'] - base['default_composition_cost_nats'])
    base['stressed_cov'] = cov1
    base['stressed_mu'] = mu1
    return base


# --------------------------------------------------------------------------
# Calibration of the radius
# --------------------------------------------------------------------------

def severity_ladder(
    information_flow: pd.Series,
    return_periods=(1, 2, 5, 10, 20, 50),
    horizon: int = 21,
    periods_per_year: float = 252.0,
    non_overlapping: bool = True,
) -> pd.DataFrame:
    """Map return periods to KL radii from realised information flow.

    ``information_flow`` is the ``h``-step conditional revision
    ``I_t^{(h)} = KL(p_t || p_{t-h})`` in nats -- a proper divergence between two
    conditional distributions, hence directly comparable with a scenario
    radius. Overlapping observations are thinned to one per ``horizon`` before
    the quantiles are taken, so the tail quantiles are not inflated by counting
    the same episode ``h`` times.
    """
    s = information_flow.dropna()
    if non_overlapping:
        s = s.iloc[::horizon]
    blocks_per_year = periods_per_year / horizon
    rows = []
    # A KL divergence grows roughly with the dimension of the return vector, so
    # a radius is only comparable across universes once divided by it. The
    # ladder is therefore calibrated separately for each portfolio; the
    # per-asset column exists only to make magnitudes readable across datasets.
    for rp in return_periods:
        p_exceed = 1.0 / (rp * blocks_per_year)
        if not 0.0 < p_exceed < 1.0:
            continue
        rows.append({
            'return_period_years': rp,
            'exceedance_prob_per_block': p_exceed,
            'eta_nats': float(s.quantile(1.0 - p_exceed)),
            'n_blocks': int(len(s)),
            # A quantile beyond 1/n cannot be read off the sample; it is an
            # interpolation of the empirical tail and is flagged as such.
            'extrapolated': bool(p_exceed < 1.0 / max(len(s), 1)),
        })
    return pd.DataFrame(rows)


def return_period_of(eta: float, information_flow: pd.Series, horizon: int = 21,
                     periods_per_year: float = 252.0,
                     non_overlapping: bool = True) -> float:
    """Return period, in years, implied by a scenario radius of ``eta`` nats."""
    s = information_flow.dropna()
    if non_overlapping:
        s = s.iloc[::horizon]
    if len(s) == 0:
        return np.nan
    p = float((s >= eta).mean())
    blocks_per_year = periods_per_year / horizon
    if p <= 0:
        return float(len(s) / blocks_per_year)  # censored: beyond the sample
    return float(1.0 / (p * blocks_per_year))


def build_scenario_table(
    mu: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    ladder: pd.DataFrame,
    risk_level: float = 0.99,
    objective: str = 'es',
) -> pd.DataFrame:
    """Scenario ladder in the language a risk committee already reads."""
    rows = []
    base = gaussian_ball_scenario(mu, cov, weights, 0.0, risk_level, objective)
    base['return_period_years'] = 0
    rows.append(base)
    for _, r in ladder.iterrows():
        sc = gaussian_ball_scenario(mu, cov, weights, float(r['eta_nats']),
                                    risk_level, objective)
        sc['return_period_years'] = r['return_period_years']
        rows.append(sc)
    out = pd.DataFrame(rows).rename(columns={'eta': 'eta_nats'})
    front = ['return_period_years', 'eta_nats', 'sigma_move', 'vol_multiplier']
    return out[front + [c for c in out.columns if c not in front]]


# --------------------------------------------------------------------------
# Non-parametric tilt: reverse stress testing on the empirical loss sample
# --------------------------------------------------------------------------

def _normalise(p0, n: int) -> np.ndarray:
    if p0 is None:
        return np.full(n, 1.0 / n)
    p0 = np.asarray(p0, dtype=float)
    if p0.shape != (n,):
        raise ValueError('Base weights must have the same length as the sample.')
    if np.any(p0 < 0):
        raise ValueError('Base weights must be non-negative.')
    return p0 / p0.sum()


def tilt_weights(loss: np.ndarray, theta: float, p0=None) -> np.ndarray:
    """Exponentially tilted probabilities ``p_i ∝ p0_i exp(theta * loss_i)``."""
    loss = np.asarray(loss, dtype=float)
    p0 = _normalise(p0, loss.size)
    a = theta * loss
    a -= a.max()
    w = p0 * np.exp(a)
    return w / w.sum()


def kl_of_tilt(loss: np.ndarray, theta: float, p0=None) -> float:
    loss = np.asarray(loss, dtype=float)
    p0 = _normalise(p0, loss.size)
    w = tilt_weights(loss, theta, p0)
    a = theta * loss
    m = a.max()
    log_mgf = m + np.log(np.sum(p0 * np.exp(a - m)))
    return float(theta * np.sum(w * loss) - log_mgf)


def _solve_theta(objective, target: float) -> float:
    if target <= 0:
        return 0.0
    hi = 1.0
    for _ in range(200):
        if objective(hi) >= target:
            break
        hi *= 1.6
    else:
        raise RuntimeError('Could not bracket the tilt parameter; target too extreme.')
    return float(optimize.brentq(lambda t: objective(t) - target, 0.0, hi,
                                 xtol=1e-12, rtol=1e-10))


def worst_case_tilt(loss: np.ndarray, eta: float, p0=None) -> dict:
    """Worst-case *empirical* measure within a KL ball, with a degeneracy check.

    On a finite sample of ``T`` atoms the largest attainable divergence is
    ``log T``, reached when all mass sits on the single worst observation. The
    reported ``effective_sample_size`` (Kish) is the diagnostic: when it falls
    to a handful of observations the tilted measure is a historical replay
    dressed up as a distribution, and the Gaussian-ball construction above
    should be used instead.
    """
    loss = np.asarray(loss, dtype=float)
    p0 = _normalise(p0, loss.size)
    theta = _solve_theta(lambda t: kl_of_tilt(loss, t, p0), eta)
    w = tilt_weights(loss, theta, p0)
    return {
        'eta': float(eta),
        'theta': theta,
        'weights': w,
        'expected_loss': float(np.sum(w * loss)),
        'base_expected_loss': float(np.sum(p0 * loss)),
        'effective_sample_size': float(1.0 / np.sum(w ** 2)),
        'max_attainable_eta': float(np.log(loss.size)),
    }


def reverse_stress(loss: np.ndarray, target_loss: float, p0=None) -> dict:
    """Cheapest scenario, in nats, that produces a given expected loss.

    Instead of asking what a scenario of a given severity does, this asks how
    implausible a scenario has to be to produce a given outcome. The answer is a
    distance in nats, which :func:`return_period_of` converts to a frequency.
    """
    loss = np.asarray(loss, dtype=float)
    p0 = _normalise(p0, loss.size)
    base = float(np.sum(p0 * loss))
    if target_loss <= base:
        return {'target_loss': float(target_loss), 'theta': 0.0, 'eta': 0.0,
                'expected_loss': base, 'effective_sample_size': float(loss.size)}
    theta = _solve_theta(lambda t: float(np.sum(tilt_weights(loss, t, p0) * loss)),
                         target_loss)
    w = tilt_weights(loss, theta, p0)
    return {
        'target_loss': float(target_loss),
        'theta': theta,
        'eta': kl_of_tilt(loss, theta, p0),
        'expected_loss': float(np.sum(w * loss)),
        'effective_sample_size': float(1.0 / np.sum(w ** 2)),
    }
