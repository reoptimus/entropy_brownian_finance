import numpy as np
import pandas as pd
from src.empirical import rolling_entropy


def test_synthetic_pipeline():
    rng = np.random.default_rng(123)
    A = np.array([[1.0, .5, .2], [.0, .9, .3], [.0, .0, .7]])
    cov = A @ A.T / 10000
    x = rng.multivariate_normal(np.zeros(3), cov, size=600)
    dates = pd.bdate_range('2020-01-01', periods=len(x))
    df = pd.DataFrame(x, index=dates, columns=['a','b','c'])
    out = rolling_entropy(df, 120)
    assert len(out) == 481
    assert np.isfinite(out[['h_vol','h_dep','h_cov']].to_numpy()).all()
