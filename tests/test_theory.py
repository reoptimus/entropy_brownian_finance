import numpy as np
from src.entropy import covariance_decomposition, maxent_entropy_from_cov


def test_entropy_decomposition_identity():
    A = np.array([[1.0, .2, .1], [.2, 1.4, .3], [.1, .3, .9]])
    cov = A @ A.T
    d = covariance_decomposition(cov)
    assert np.isclose(d['h_cov'], d['h_vol'] + d['h_dep'])


def test_bivariate_determinant():
    s1, s2, rho = .2, .3, .7
    cov = np.array([[s1**2, rho*s1*s2], [rho*s1*s2, s2**2]])
    d = covariance_decomposition(cov)
    expected = np.log(s1*s2) + .5*np.log(1-rho**2)
    assert np.isclose(d['h_cov'], expected)


def test_maxent_entropy_positive_definite():
    cov = np.eye(2)
    h = maxent_entropy_from_cov(cov)
    assert np.isfinite(h)
