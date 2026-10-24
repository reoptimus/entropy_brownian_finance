import numpy as np
import pandas as pd

from src.coupling import (
    absorption_ratio,
    coupling_statistics,
    ewma_devolatilize,
    normal_scores,
    rolling_coupling,
)
from src.empirical import rolling_measures


def _panel(T=700, N=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range('2000-01-03', periods=T)
    return pd.DataFrame(rng.standard_normal((T, N)) * 0.01, index=idx,
                        columns=[f'a{i}' for i in range(N)])


def test_devolatilize_uses_only_the_past():
    X = _panel().to_numpy()
    Z = ewma_devolatilize(X)
    X2 = X.copy()
    X2[400] *= 50.0
    Z2 = ewma_devolatilize(X2)
    # Day 400 itself changes, everything before it is untouched.
    np.testing.assert_allclose(Z[:400], Z2[:400], rtol=1e-12)
    assert np.isnan(Z[:21]).all()


def test_devolatilize_clips_stale_series():
    X = _panel().to_numpy().copy()
    X[:100, 0] = 0.0
    X[100, 0] = -1.0
    assert np.nanmax(np.abs(ewma_devolatilize(X, clip=8.0))) <= 8.0


def test_absorption_ratio_bounds():
    assert absorption_ratio(np.eye(10)) == np.float64(0.2)
    v = np.ones(10)
    one_factor = np.outer(v, v) + 1e-6 * np.eye(10)
    assert absorption_ratio(one_factor) > 0.999


def test_normal_scores_invariant_to_monotone_maps():
    X = _panel(T=252).to_numpy()
    np.testing.assert_allclose(normal_scores(X), normal_scores(np.exp(50 * X)))


def test_rolling_coupling_matches_pipeline_h_dep():
    lr = _panel(T=320)
    a = rolling_coupling(lr, window=252)
    b = rolling_measures(lr, 252, coupling=True)
    np.testing.assert_allclose(a['h_dep'], b['h_dep'], rtol=1e-10, atol=1e-12)
    for col in ['h_dep_rank', 'absorption_ratio', 'eig1_share']:
        np.testing.assert_allclose(a[col], b[col], rtol=1e-10, atol=1e-12)


def test_coupling_gap_has_predicted_sign_when_stress_couples():
    rng = np.random.default_rng(3)
    T, N = 1800, 6
    idx = pd.bdate_range('2000-01-03', periods=T)
    calm = rng.standard_normal((T, N)) * 0.01
    # Turbulent spells where one common factor dominates.
    stress = np.zeros(T, bool)
    for s in range(700, T, 300):
        stress[s:s + 80] = True
    f = rng.standard_normal(T) * 0.03
    X = calm + np.where(stress[:, None], f[:, None], 0.0)
    lr = pd.DataFrame(X, index=idx)
    out = coupling_statistics(rolling_coupling(lr, window=60), lr)
    assert out['CPL_h_dep_gap'] < 0
    assert out['CPL_absorption_ratio_gap'] > 0
    assert out['CPL_eig1_share_gap'] > 0
