import numpy as np
from sklearn.covariance import ledoit_wolf

from src.lw import ledoit_wolf_cov


def test_matches_sklearn():
    rng = np.random.default_rng(0)
    for n, p in [(252, 20), (252, 48), (60, 5), (30, 40), (100, 2)]:
        X = rng.standard_t(4, (n, p)) @ rng.normal(size=(p, p))
        np.testing.assert_allclose(ledoit_wolf_cov(X), ledoit_wolf(X)[0], rtol=1e-11, atol=1e-14)
