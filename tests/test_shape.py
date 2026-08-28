import numpy as np
import pytest

from src.entropy import covariance_decomposition, effective_dimension, gaussian_kl
from src.shape import (
    entropy_deficit,
    marginal_shape,
    negentropy_hyvarinen,
    negentropy_hyvarinen_channels,
    negentropy_moments,
    structural_index,
    tail_dependence,
)


def _w(n):
    return np.full(n, 1.0 / n)


def test_negentropy_vanishes_for_gaussian():
    rng = np.random.default_rng(0)
    z = rng.standard_normal(200_000)
    z = (z - z.mean()) / z.std()
    assert negentropy_hyvarinen(z[:, None], _w(z.size))[0] < 5e-4


def test_hyvarinen_tracks_the_true_laplace_negentropy():
    # h(N(0,1)) - h(Laplace with unit variance) = 1/2 log(2 pi e) - (1 + log(2/sqrt2))
    true = 0.5 * np.log(2 * np.pi * np.e) - (1.0 + np.log(np.sqrt(2.0)))
    rng = np.random.default_rng(1)
    z = rng.laplace(size=200_000)
    z = (z - z.mean()) / z.std()
    est = negentropy_hyvarinen(z[:, None], _w(z.size))[0]
    assert abs(est - true) < 0.02


def test_edgeworth_diverges_on_fat_tails_and_hyvarinen_does_not():
    """The reason the pipeline never uses the moment estimator."""
    rng = np.random.default_rng(2)
    z = rng.standard_t(4, size=200_000)
    z = (z - z.mean()) / z.std()
    skew = float(np.mean(z ** 3))
    exkurt = float(np.mean(z ** 4)) - 3.0
    j_s, j_k = negentropy_moments(np.array([skew]), np.array([exkurt]))
    edgeworth = float(j_s[0] + j_k[0])
    hyv = negentropy_hyvarinen(z[:, None], _w(z.size))[0]
    assert edgeworth > 1.0        # true value is about 0.10 nats
    assert hyv < 0.25


def test_hyvarinen_channels_are_non_negative_and_sum_to_total():
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((5000, 4))
    Z = (Z - Z.mean(0)) / Z.std(0)
    odd, even = negentropy_hyvarinen_channels(Z, _w(Z.shape[0]))
    assert (odd >= 0).all() and (even >= 0).all()
    assert np.allclose(odd + even, negentropy_hyvarinen(Z, _w(Z.shape[0])))


def test_odd_channel_responds_to_asymmetry_and_even_channel_to_tails():
    rng = np.random.default_rng(4)
    n = 100_000
    skewed = rng.exponential(size=n)
    heavy = rng.standard_t(5, size=n)
    skewed = (skewed - skewed.mean()) / skewed.std()
    heavy = (heavy - heavy.mean()) / heavy.std()
    o_s, e_s = negentropy_hyvarinen_channels(skewed[:, None], _w(n))
    o_h, e_h = negentropy_hyvarinen_channels(heavy[:, None], _w(n))
    assert o_s[0] > o_h[0]
    assert e_h[0] > o_h[0]


def test_structural_index_is_the_kl_to_independent_gaussians():
    """For a Gaussian sample the deficit vanishes and J reduces to -h_dep."""
    rng = np.random.default_rng(5)
    n = 4
    A = rng.standard_normal((n, n))
    cov = A @ A.T / n + np.eye(n) * 0.3
    X = rng.multivariate_normal(np.zeros(n), cov, size=200_000)
    d = entropy_deficit(X)
    dec = covariance_decomposition(np.cov(X, rowvar=False))
    J = structural_index(dec['h_dep'], d['d_total'])
    assert d['d_total'] < 0.01
    assert J > 0
    assert abs(J - (-dec['h_dep'])) < 0.02


def test_structural_index_rises_with_both_channels():
    rng = np.random.default_rng(6)
    n, T = 5, 40_000
    gauss = rng.multivariate_normal(np.zeros(n), np.eye(n), size=T)
    heavy = rng.standard_t(4, size=(T, n))
    base = structural_index(covariance_decomposition(np.cov(gauss, rowvar=False))['h_dep'],
                            entropy_deficit(gauss)['d_total'])
    shaped = structural_index(covariance_decomposition(np.cov(heavy, rowvar=False))['h_dep'],
                              entropy_deficit(heavy)['d_total'])
    assert shaped > base

    R = np.full((n, n), 0.8)
    np.fill_diagonal(R, 1.0)
    corr = rng.multivariate_normal(np.zeros(n), R, size=T)
    dependent = structural_index(covariance_decomposition(np.cov(corr, rowvar=False))['h_dep'],
                                 entropy_deficit(corr)['d_total'])
    assert dependent > base


def test_marginal_shape_recovers_known_moments():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((100_000, 2))
    m = marginal_shape(x)
    assert np.allclose(m['skew'], 0, atol=0.03)
    assert np.allclose(m['exkurt'], 0, atol=0.06)


def test_entropy_deficit_market_signature_is_signed():
    """Negentropy is even, so the sign has to come from the market factor."""
    rng = np.random.default_rng(8)
    n, T = 4, 60_000
    common = -rng.standard_gamma(1.0, size=T)  # left-skewed common factor
    X = 0.6 * common[:, None] + rng.standard_normal((T, n))
    d = entropy_deficit(X)
    assert d['skew_market'] < -0.3
    assert d['d_odd'] > 0


def test_tail_dependence_bounds():
    rng = np.random.default_rng(9)
    T, n, q = 40_000, 4, 0.05
    indep = rng.standard_normal((T, n))
    assert tail_dependence(indep, q) == pytest.approx(q, abs=0.02)

    common = rng.standard_normal(T)
    identical = np.tile(common[:, None], (1, n))
    assert tail_dependence(identical, q) == pytest.approx(1.0, abs=1e-6)


def test_effective_dimension_and_gaussian_kl():
    assert effective_dimension(np.eye(6)) == pytest.approx(6.0)
    R = np.full((6, 6), 0.999)
    np.fill_diagonal(R, 1.0)
    assert effective_dimension(R) < 1.5

    cov = np.eye(3) * 2.0
    assert gaussian_kl(np.zeros(3), cov, np.zeros(3), cov) == pytest.approx(0.0, abs=1e-12)
    assert gaussian_kl(np.ones(3), cov, np.zeros(3), cov) > 0
