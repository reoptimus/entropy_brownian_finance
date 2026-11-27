import numpy as np

from src.scale_split import (
    channel_efficiencies, cheapest_corr_only, cheapest_vol_only, correlation_ceiling,
    gaussian_deformation, lift_cost, minimal_lift_cov, split_deformation,
)


def _cov(n=6, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, 2))
    return F @ F.T + np.diag(rng.uniform(0.5, 1.5, n))


def test_split_is_exact():
    cov0 = _cov()
    rng = np.random.default_rng(1)
    G = rng.standard_normal((6, 6))
    cov1 = cov0 + 0.3 * G @ G.T
    sp = split_deformation(cov1, cov0)
    assert abs(sp['check']) < 1e-12
    assert abs(sp['total'] - gaussian_deformation(cov1, cov0)) < 1e-12


def test_no_interaction_from_independence():
    cov0 = np.diag([1.0, 2.0, 0.5, 3.0])
    cov1 = _cov(4, 3)
    sp = split_deformation(cov1, cov0)
    assert abs(sp['interaction']) < 1e-12
    assert abs(sp['K_corr'] - sp['d_coupling']) < 1e-12


def test_minimal_lift_entropy_rise_is_log_v():
    cov0 = _cov()
    w = np.full(6, 1 / 6)
    for v in (1.2, 2.0, 3.0):
        sp = split_deformation(minimal_lift_cov(cov0, w, v), cov0)
        assert abs(sp['d_scale'] - sp['d_coupling'] - np.log(v)) < 1e-10
        assert abs(sp['total'] - lift_cost(v)) < 1e-10


def test_efficiencies_match_exact_small_shocks():
    cov0 = _cov()
    w = np.full(6, 1 / 6)
    e = channel_efficiencies(cov0, w)
    v = 1.002
    lift = lift_cost(v)
    assert abs(lift / cheapest_vol_only(cov0, w, v)['cost'] - e['e_vol']) < 2e-3
    assert abs(lift / cheapest_corr_only(cov0, w, v)['cost'] - e['e_corr']) < 2e-3


def test_orthogonal_at_independence():
    e = channel_efficiencies(np.diag([1.0, 2.0, 3.0]), np.array([0.2, 0.5, 0.3]))
    assert abs(e['e_vol'] + e['e_corr'] - 1) < 1e-12


def test_correlation_ceiling():
    cov0 = _cov()
    w = np.full(6, 1 / 6)
    vmax = correlation_ceiling(cov0, w)
    assert np.isinf(cheapest_corr_only(cov0, w, vmax + 0.01)['cost'])
    r = cheapest_corr_only(cov0, w, 1 + 0.5 * (vmax - 1))
    assert r['success']
    c1 = r['cov1']
    assert abs(np.sqrt(w @ c1 @ w / (w @ cov0 @ w)) - (1 + 0.5 * (vmax - 1))) < 1e-8
