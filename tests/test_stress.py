import numpy as np
import pandas as pd
import pytest

from src.stress import (
    build_scenario_table,
    calibrated_covariance,
    classical_scenario,
    composition_calibrated_scenario,
    es_coefficient,
    gaussian_ball_scenario,
    kl_of_tilt,
    price_scenario,
    return_period_of,
    reverse_stress,
    severity_ladder,
    tilt_weights,
    worst_case_tilt,
)


@pytest.fixture
def market():
    rng = np.random.default_rng(0)
    n = 6
    A = rng.standard_normal((n, n))
    cov = (A @ A.T / n + np.eye(n)) * 1e-4
    mu = np.full(n, 2e-4)
    w = np.full(n, 1.0 / n)
    return mu, cov, w


def test_pure_mean_objective_is_the_sqrt_two_eta_sigma_move(market):
    """The construction's sanity check: one nat buys sqrt(2) sigma."""
    mu, cov, w = market
    for eta in [0.5, 1.0, 2.0, 5.0]:
        sc = gaussian_ball_scenario(mu, cov, w, eta, objective='mean')
        assert sc['sigma_move'] == pytest.approx(-np.sqrt(2 * eta), rel=1e-8)
        assert sc['vol_multiplier'] == pytest.approx(1.0, rel=1e-8)


def test_zero_radius_reproduces_the_base_distribution(market):
    mu, cov, w = market
    sc = gaussian_ball_scenario(mu, cov, w, 0.0)
    assert sc['sigma_move'] == 0.0
    assert sc['vol_multiplier'] == 1.0
    assert sc['stressed_mean'] == pytest.approx(sc['base_mean'])
    assert sc['stressed_ES99'] == pytest.approx(sc['base_ES99'])


def test_scenario_is_monotone_in_the_radius(market):
    mu, cov, w = market
    etas = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    es = [gaussian_ball_scenario(mu, cov, w, e)['stressed_ES99'] for e in etas]
    vol = [gaussian_ball_scenario(mu, cov, w, e)['vol_multiplier'] for e in etas]
    assert all(np.diff(es) > 0)
    assert all(np.diff(vol) > 0)


def test_scenario_solution_saturates_the_kl_budget(market):
    """The optimum sits on the boundary of the ball, not inside it."""
    mu, cov, w = market
    eta = 2.0
    sc = gaussian_ball_scenario(mu, cov, w, eta)
    u, v = sc['sigma_move'], sc['vol_multiplier']
    g = 0.5 * (u ** 2 + v ** 2 - 1.0 - 2.0 * np.log(v))
    assert g == pytest.approx(eta, rel=1e-8)


def test_es_dominates_var_which_dominates_mean(market):
    mu, cov, w = market
    eta = 2.0
    es = gaussian_ball_scenario(mu, cov, w, eta, objective='es')
    var = gaussian_ball_scenario(mu, cov, w, eta, objective='var')
    mean = gaussian_ball_scenario(mu, cov, w, eta, objective='mean')
    # Each objective maximises its own functional over the same ball.
    assert es['stressed_ES99'] >= var['stressed_ES99']
    assert var['stressed_VaR99'] >= mean['stressed_VaR99']
    assert es_coefficient(0.99) > 1.0


def test_price_scenario_is_zero_at_the_base_and_positive_elsewhere(market):
    mu, cov, w = market
    assert price_scenario(mu, cov, mu, cov) == pytest.approx(0.0, abs=1e-12)
    assert price_scenario(mu * 2, cov, mu, cov) > 0
    assert price_scenario(mu, cov * 1.5, mu, cov) > 0


def test_classical_mean_leg_costs_exactly_half_sigma_squared(market):
    mu, cov, w = market
    for s in [-1.0, -2.0, -3.0]:
        c = classical_scenario(mu, cov, w, vol_multiplier=1.0,
                               target_correlation=0.0, sigma_move=s)
        assert c['price_mean_leg_nats'] == pytest.approx(0.5 * s ** 2, rel=1e-8)


def test_classical_legs_are_non_negative_and_bundle_is_dearer(market):
    mu, cov, w = market
    c = classical_scenario(mu, cov, w, 2.0, 0.90, -3.0)
    for leg in ['price_mean_leg_nats', 'price_vol_leg_nats', 'price_correlation_leg_nats']:
        assert c[leg] >= 0
    assert c['entropy_price_nats'] > c['price_mean_leg_nats']


def test_calibrated_covariance_preserves_portfolio_variance_at_any_share(market):
    mu, cov, w = market
    s_target = 1.6 * float(np.sqrt(w @ cov @ w))
    for share in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cov1, _ = calibrated_covariance(cov, w, s_target, share)
        assert float(np.sqrt(w @ cov1 @ w)) == pytest.approx(s_target, rel=1e-8)


def test_calibrated_covariance_endpoints_match_pure_scale_and_feasible_corr(market):
    mu, cov, w = market
    s0 = float(np.sqrt(w @ cov @ w))
    s_target = 1.6 * s0  # below the rho=1 ceiling for this fixture (~2.54x s0)
    cov_share0, feasible = calibrated_covariance(cov, w, s_target, 0.0)
    k = s_target / s0
    assert np.allclose(cov_share0, (k ** 2) * cov)
    assert feasible

    # Feasible severities: pure correlation reaches the target with marginal
    # volatilities left exactly at their base values.
    cov_share1, _ = calibrated_covariance(cov, w, s_target, 1.0)
    sd0 = np.sqrt(np.diag(cov))
    assert np.allclose(np.sqrt(np.diag(cov_share1)), sd0)


def test_calibrated_covariance_caps_correlation_at_its_ceiling(market):
    mu, cov, w = market
    s0 = float(np.sqrt(w @ cov @ w))
    # Beyond the rho=1 ceiling (~2.54x s0): no correlation matrix reaches this
    # on its own, so even share=1 must lean on some uniform scaling too --
    # but the portfolio severity is still matched exactly.
    s_target = 3.5 * s0
    cov1, feasible = calibrated_covariance(cov, w, s_target, dependence_share=1.0)
    assert not feasible
    assert float(np.sqrt(w @ cov1 @ w)) == pytest.approx(s_target, rel=1e-8)
    sd1 = np.sqrt(np.diag(cov1))
    sd0 = np.sqrt(np.diag(cov))
    assert np.all(sd1 > sd0)  # the residual gap still had to be closed by scaling


def test_calibrated_covariance_severity_invariant_holds_beyond_the_ceiling(market):
    mu, cov, w = market
    s0 = float(np.sqrt(w @ cov @ w))
    s_target = 3.5 * s0  # above the rho=1 ceiling
    for share in [0.0, 0.25, 0.5, 0.75, 1.0]:
        cov1, _ = calibrated_covariance(cov, w, s_target, share)
        assert float(np.sqrt(w @ cov1 @ w)) == pytest.approx(s_target, rel=1e-8)


def test_composition_calibrated_scenario_matches_ball_at_zero_excess_cost(market):
    mu, cov, w = market
    sc = composition_calibrated_scenario(mu, cov, w, eta=2.0, dependence_share=0.0)
    assert sc['excess_cost_over_default_nats'] == pytest.approx(0.0, abs=1e-8)
    assert sc['entropy_price_nats'] == pytest.approx(sc['default_composition_cost_nats'])


def test_composition_calibrated_scenario_severity_is_share_invariant(market):
    """The portfolio-level severity (mean, vol, ES/VaR) never depends on the
    composition -- only the ambient price and the asset-level covariance do."""
    mu, cov, w = market
    # eta chosen so the target stays within the rho=1 ceiling (~2.5x s0 for
    # this fixture): at share=1.0 exactly beyond that ceiling, cov_cap is a
    # rank-one (perfectly-correlated) matrix and its KL price is undefined --
    # a real degeneracy, not something to paper over in this invariance test.
    scs = [composition_calibrated_scenario(mu, cov, w, eta=0.5, dependence_share=s)
           for s in [0.0, 0.3, 0.6, 0.9, 1.0]]
    for key in ['stressed_mean', 'stressed_vol_daily', 'stressed_ES99']:
        vals = [sc[key] for sc in scs]
        assert all(v == pytest.approx(vals[0]) for v in vals)
    # Composition is not a free lunch: every non-default share changes the
    # ambient price (up or down -- the default is not an ambient-KL minimum,
    # only the ball's own simplest choice; see _vol_only_covariance).
    prices = [sc['entropy_price_nats'] for sc in scs]
    assert len(set(np.round(prices, 6))) > 1


def test_severity_ladder_is_monotone_and_flags_extrapolation():
    rng = np.random.default_rng(1)
    flow = pd.Series(np.abs(rng.standard_normal(2000)) * 0.5,
                     index=pd.bdate_range('2000-01-03', periods=2000))
    lad = severity_ladder(flow, return_periods=(1, 5, 20, 100), horizon=21)
    assert lad['eta_nats'].is_monotonic_increasing
    assert bool(lad.iloc[-1]['extrapolated'])
    assert not bool(lad.iloc[0]['extrapolated'])


def test_return_period_inverts_the_ladder():
    rng = np.random.default_rng(2)
    flow = pd.Series(np.abs(rng.standard_normal(4000)),
                     index=pd.bdate_range('2000-01-03', periods=4000))
    lad = severity_ladder(flow, return_periods=(2, 5), horizon=21)
    for _, r in lad.iterrows():
        rp = return_period_of(r['eta_nats'], flow, horizon=21)
        assert rp == pytest.approx(r['return_period_years'], rel=0.5)


def test_build_scenario_table_columns(market):
    mu, cov, w = market
    lad = pd.DataFrame({'return_period_years': [1, 5], 'eta_nats': [1.0, 3.0]})
    tab = build_scenario_table(mu, cov, w, lad)
    assert len(tab) == 3                      # base row plus two rungs
    assert tab.iloc[0]['eta_nats'] == 0.0
    assert list(tab.columns[:4]) == ['return_period_years', 'eta_nats',
                                     'sigma_move', 'vol_multiplier']


def test_tilt_matches_its_kl_budget():
    rng = np.random.default_rng(3)
    loss = rng.standard_normal(5000) * 0.01
    for eta in [0.05, 0.2, 1.0]:
        wc = worst_case_tilt(loss, eta)
        assert kl_of_tilt(loss, wc['theta']) == pytest.approx(eta, rel=1e-6)
        assert wc['expected_loss'] > wc['base_expected_loss']


def test_tilt_degeneracy_is_reported():
    rng = np.random.default_rng(4)
    loss = rng.standard_normal(500) * 0.01
    small = worst_case_tilt(loss, 0.05)
    huge = worst_case_tilt(loss, 0.95 * np.log(len(loss)))
    assert small['effective_sample_size'] > 100
    assert huge['effective_sample_size'] < 10
    assert huge['max_attainable_eta'] == pytest.approx(np.log(len(loss)))


def test_reverse_stress_round_trip():
    rng = np.random.default_rng(5)
    loss = rng.standard_normal(4000) * 0.01
    target = float(np.quantile(loss, 0.99))
    rev = reverse_stress(loss, target)
    assert rev['expected_loss'] == pytest.approx(target, rel=1e-6)
    assert rev['eta'] > 0
    w = tilt_weights(loss, rev['theta'])
    assert float(np.sum(w * loss)) == pytest.approx(target, rel=1e-6)


def test_reverse_stress_is_free_below_the_base_loss():
    loss = np.linspace(-0.02, 0.02, 500)
    rev = reverse_stress(loss, float(loss.mean()) - 0.001)
    assert rev['eta'] == 0.0
