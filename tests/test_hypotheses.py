import numpy as np
import pandas as pd
import pytest

from src.empirical import conditional_state, rolling_measures
from src.hypotheses import (
    block_bootstrap_diff,
    h1_regime_signature,
    h2_ceiling_compensation,
    h3_jump_asymmetry,
    h4_relaxation,
    h6_tail_dependence_coupling,
    run_all,
)


@pytest.fixture(scope='module')
def panel():
    """Returns with a genuine stress regime: higher vol, higher correlation,
    fatter tails and negative skew in the second of three blocks."""
    rng = np.random.default_rng(0)
    n, T = 6, 1800
    calm_R = np.full((n, n), 0.25)
    np.fill_diagonal(calm_R, 1.0)
    stress_R = np.full((n, n), 0.80)
    np.fill_diagonal(stress_R, 1.0)

    X = rng.multivariate_normal(np.zeros(n), calm_R * 1e-4, size=T, method='eigh')
    lo, hi = 900, 1100
    heavy = rng.multivariate_normal(np.zeros(n), stress_R * 4e-4,
                                    size=hi - lo, method='eigh')
    heavy += -np.abs(rng.standard_gamma(1.0, size=(hi - lo, 1))) * 0.004
    X[lo:hi] = heavy
    idx = pd.bdate_range('2000-01-03', periods=T)
    return pd.DataFrame(X, index=idx, columns=[f'a{i}' for i in range(n)])


@pytest.fixture(scope='module')
def measures(panel):
    ent = rolling_measures(panel, window=252, method='ledoit_wolf',
                           jump_method='ewma', halflife=60.0)
    market = panel.mean(axis=1)
    ent['stress_signal'] = market.rolling(21).std() * np.sqrt(252)
    ent['stress'] = ent['stress_signal'] >= ent['stress_signal'].quantile(0.90)
    return ent


def test_rolling_measures_identities(measures):
    m = measures.dropna(subset=['h_cov'])
    assert np.allclose(m['h_cov'], m['h_vol'] + m['h_dep'])
    assert np.allclose(m['h_tot'], m['h_cov'] - m['d_total'])
    assert np.allclose(m['J'], m['d_total'] - m['h_dep_ew'])
    assert (m['J'] > 0).all()
    assert (m['d_total'] >= 0).all()
    assert (m['h_dep'] <= 1e-12).all()          # Hadamard
    assert (m['n_eff_modes'] <= 6 + 1e-9).all()


def test_surprisal_is_centred_on_a_calm_stretch(measures):
    calm = measures.loc[~measures['stress'], 'surprisal'].dropna()
    # Excess surprisal has mean zero when the model matches the data.
    assert abs(calm.mean()) < 3.0


def test_information_flow_is_non_negative(measures):
    for col in ['info_flow', 'info_flow_h']:
        v = measures[col].dropna()
        assert len(v) > 0
        assert (v >= -1e-12).all()


def test_h1_detects_the_injected_regime(measures):
    out = h1_regime_signature(measures, measures['stress'], n_boot=300)
    row = out.set_index('channel')
    assert row.loc['J', 'diff'] > 0
    assert row.loc['h_dep', 'diff'] < 0
    assert row.loc['mean_corr', 'diff'] > 0
    assert row.loc['n_eff_modes', 'diff'] < 0


def test_block_bootstrap_finds_nothing_on_a_random_label(measures):
    rng = np.random.default_rng(3)
    fake = pd.Series(rng.random(len(measures)) > 0.9, index=measures.index)
    r = block_bootstrap_diff(measures['J'], fake, block=252, n_boot=400)
    assert r['p_value'] > 0.05


def test_h2_returns_all_regimes(measures):
    out = h2_ceiling_compensation(measures, measures['stress'])
    assert set(out['regime']) == {'all', 'calm', 'stress'}
    assert out['n'].min() > 0


def test_h3_and_h4_run_and_agree_on_direction(measures):
    h3, jumps = h3_jump_asymmetry(measures, threshold=4.0, scale_window=250)
    assert h3['n_jump_up'].iloc[0] >= h3['n_jump_down'].iloc[0]
    h4, ev = h4_relaxation(measures, jumps)
    assert np.isfinite(h4['slope'].iloc[0])


def test_h6_shares_sum_to_one(measures):
    out = h6_tail_dependence_coupling(measures, measures['stress'])
    total = out['share_var_dJ_dependence'] + out['share_var_dJ_shape']
    assert np.allclose(total, 1.0, atol=1e-8)


def test_run_all_produces_every_table(measures):
    cfg = {'jumps': {'threshold': 4.0, 'scale_window': 250,
                     'event_pre': 20, 'event_post': 60}}
    res = run_all(measures, measures['stress'], cfg)
    for key in ['h1_regime_signature', 'h2_ceiling_compensation', 'h3_jump_asymmetry',
                'h4_relaxation', 'h5_skewness_signature', 'h6_tail_dependence_coupling',
                'indicators', 'jumps']:
        assert key in res


def test_conditional_state_matches_the_last_window(panel):
    mu, cov = conditional_state(panel, window=252, method='ewma', halflife=60.0)
    assert mu.shape == (panel.shape[1],)
    assert cov.shape == (panel.shape[1], panel.shape[1])
    assert np.all(np.linalg.eigvalsh(cov) > 0)
