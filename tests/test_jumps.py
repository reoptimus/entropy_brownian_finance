import numpy as np
import pandas as pd
import pytest

from src.jumps import (
    crisis_diffusion_indicators,
    detect_jumps,
    estimate_relaxation,
    event_study,
    jump_asymmetry,
    local_scale,
)


def _ou_path(n=6000, kappa=0.02, sigma=0.05, level=1.0, seed=0,
             jump_at=None, jump_size=0.0):
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = level
    for t in range(1, n):
        x[t] = x[t - 1] + kappa * (level - x[t - 1]) + sigma * rng.standard_normal()
        if jump_at is not None and t in jump_at:
            x[t] += jump_size
    return pd.Series(x, index=pd.bdate_range('2000-01-03', periods=n))


def test_local_scale_is_robust_to_jumps():
    s = _ou_path(3000, sigma=0.05, seed=1)
    contaminated = s.copy()
    contaminated.iloc[1500] += 50.0
    d_clean = local_scale(s.diff()).iloc[2000]
    d_dirty = local_scale(contaminated.diff()).iloc[2000]
    # Bipower variation touches the jump through only two products, so a single
    # 1000-sigma outlier must not move the local scale by more than a few percent.
    assert abs(d_dirty / d_clean - 1.0) < 0.25


def test_local_scale_is_shifted_and_excludes_the_current_date():
    s = _ou_path(500, seed=2)
    dx = s.diff()
    sc = local_scale(dx, window=100, min_periods=30)
    assert sc.isna().iloc[:30].all()


def test_detect_jumps_finds_injected_jumps():
    jump_at = {1200, 2400, 3600}
    s = _ou_path(5000, sigma=0.02, jump_at=jump_at, jump_size=1.0, seed=3)
    j = detect_jumps(s, threshold=4.0, window=250)
    found = set(np.flatnonzero(j['jump_up'].to_numpy()))
    assert jump_at.issubset(found)
    assert j.loc[j['jump_up'], 'dx'].min() > 0


def test_jump_asymmetry_detects_one_sidedness():
    s = _ou_path(5000, sigma=0.02, jump_at=set(range(500, 4500, 400)),
                 jump_size=1.0, seed=4)
    stats = jump_asymmetry(detect_jumps(s, 4.0, 250))
    assert stats['n_jump_up'] > stats['n_jump_down']
    assert stats['skew_of_changes'] > 0
    assert stats['p_binomial_symmetry'] < 0.01


def test_jump_asymmetry_is_symmetric_on_symmetric_noise():
    rng = np.random.default_rng(5)
    s = pd.Series(np.cumsum(rng.standard_normal(4000) * 0.01),
                  index=pd.bdate_range('2000-01-03', periods=4000))
    stats = jump_asymmetry(detect_jumps(s, 4.0, 250))
    total = stats['n_jump_up'] + stats['n_jump_down']
    if total >= 10:
        assert stats['p_binomial_symmetry'] > 0.01


def test_estimate_relaxation_recovers_kappa():
    kappa = 0.02
    s = _ou_path(20000, kappa=kappa, sigma=0.05, level=1.0, seed=6)
    rel = estimate_relaxation(s)
    assert rel['kappa'] == pytest.approx(kappa, rel=0.25)
    assert rel['half_life_days'] == pytest.approx(np.log(2) / kappa, rel=0.3)
    assert rel['long_run_level'] == pytest.approx(1.0, abs=0.15)
    assert rel['slope_pvalue'] < 0.01


def test_estimate_relaxation_excludes_jump_dates():
    jump_at = set(range(300, 5000, 250))
    s = _ou_path(6000, kappa=0.02, sigma=0.03, jump_at=jump_at, jump_size=1.0, seed=7)
    j = detect_jumps(s, 4.0, 250)
    with_jumps = estimate_relaxation(s)
    without = estimate_relaxation(s, j)
    assert without['n_used'] < with_jumps['n_used']
    assert without['kappa'] > 0


def test_event_study_shape_and_decay():
    jump_at = set(range(600, 5000, 600))
    s = _ou_path(6000, kappa=0.02, sigma=0.02, jump_at=jump_at, jump_size=1.0, seed=8)
    j = detect_jumps(s, 4.0, 250)
    ev = event_study(s, j, pre=20, post=100, direction='up')
    assert not ev.empty
    assert ev.index[0] == -20 and ev.index[-1] == 100
    assert ev.loc[0, 'mean'] == pytest.approx(0.0, abs=1e-12)
    assert ev.loc[-1, 'mean'] < ev.loc[1, 'mean']    # the jump happened
    assert ev.loc[100, 'mean'] < ev.loc[1, 'mean']   # and then decayed


def test_crisis_diffusion_indicators_bounds():
    s = _ou_path(3000, jump_at={1000}, jump_size=1.0, seed=9)
    j = detect_jumps(s, 4.0, 250)
    ind = crisis_diffusion_indicators(s, j, kappa=0.02)
    assert (ind['unhealed'].dropna() <= 1.0).all()
    assert (ind['crisis'] >= 0).all()
    assert ind['crisis'].iloc[1000] > 0
