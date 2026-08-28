"""Entropy deficit: the non-Gaussian (shape) channel of the decomposition.

For a conditional return density :math:`p_t` and the Gaussian :math:`q_t`
matching its first two moments, Proposition 1 gives

.. math:: \\mathcal{D}_t \\;=\\; h(q_t) - h(p_t) \\;=\\; D_{KL}(p_t \\Vert q_t) \\;\\ge\\; 0,

the *entropy deficit* (negentropy): the part of the market's uncertainty that no
covariance matrix can see, and where skewness and fat tails live.

Rather than attempt a 20- or 48-dimensional negentropy on a rolling window, we
use the exact, rotation-free chain rule

.. math:: \\mathcal{J} \\;\\equiv\\; D_{KL}\\!\\left(p \\,\\Vert\\, \\textstyle\\prod_i q_i\\right)
          \\;=\\; \\underbrace{\\sum_i J(x_i)}_{\\text{marginal shape}}
          \\;+\\; \\underbrace{I(x)}_{\\text{dependence}},

where :math:`J(x_i)=D_{KL}(p_i\\Vert q_i)` is the marginal negentropy of asset
*i* and :math:`I(x)=D_{KL}(p\\Vert\\prod_i p_i)` the multi-information. Replacing
:math:`I(x)` by its linear (Gaussian-copula) part
:math:`-\\tfrac12\\log\\det R` gives the estimator used throughout:

.. math:: \\hat{\\mathcal J}_t \\;=\\; \\sum_i \\hat J(x_{i,t}) \\;-\\; \\tfrac12\\log\\det \\hat R_t .

Both channels are scale-free, so :math:`\\hat{\\mathcal J}` is invariant to the
level of volatility -- it is not a repackaged volatility index. What it omits is
non-linear/tail dependence, reported separately by a tail-dependence
diagnostic.

Estimating :math:`J`
--------------------
``hyvarinen`` (default) uses Hyvarinen's bounded-contrast approximation. Its two
terms split naturally into an odd (asymmetry) and an even (tail-weight)
channel, which is exactly the skewness/fat-tail decomposition the paper needs.

``moments`` uses the Edgeworth form :math:`J \\approx S^2/12 + K^2/48`. It makes
the mechanism transparent -- an entropy deficit *is* skewness plus excess
kurtosis -- but it is unusable as an estimator on fat-tailed daily returns: on a
Student-t(4) sample it returns 2.19 nats against a true value of 0.095. It is
kept for the interpretive decomposition and for the benchmark table produced by
``scripts/negentropy_benchmark.py``, never as the reported deficit.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    'negentropy_moments',
    'negentropy_hyvarinen',
    'negentropy_hyvarinen_channels',
    'marginal_shape',
    'entropy_deficit',
    'structural_index',
    'tail_dependence',
]

# Hyvarinen & Oja (2000), eq. (25).
_K1 = 36.0 / (8.0 * np.sqrt(3.0) - 9.0)
_K2 = 24.0 / (16.0 * np.sqrt(3.0) - 27.0)
_SQRT_HALF = np.sqrt(0.5)


def _standardise(X: np.ndarray, weights: np.ndarray):
    mu = weights @ X
    Xc = X - mu
    var = weights @ (Xc ** 2)
    sd = np.sqrt(np.clip(var, 1e-300, None))
    return Xc / sd, sd


def negentropy_moments(skew, exkurt):
    """Edgeworth negentropy split into its skewness and kurtosis channels."""
    j_skew = np.asarray(skew, dtype=float) ** 2 / 12.0
    j_kurt = np.asarray(exkurt, dtype=float) ** 2 / 48.0
    return j_skew, j_kurt


def negentropy_hyvarinen_channels(Z: np.ndarray, weights: np.ndarray):
    """Hyvarinen negentropy per column, split into odd and even channels.

    The odd term ``k1 (E[z exp(-z^2/2)])^2`` responds to asymmetry; the even
    term ``k2 (E[exp(-z^2/2)] - sqrt(1/2))^2`` responds to tail weight (and to
    bimodality). Both are non-negative and sum to the negentropy estimate.
    """
    e = np.exp(-0.5 * Z ** 2)
    odd = weights @ (Z * e)
    even = weights @ e - _SQRT_HALF
    return _K1 * odd ** 2, _K2 * even ** 2


def negentropy_hyvarinen(Z: np.ndarray, weights: np.ndarray) -> np.ndarray:
    odd, even = negentropy_hyvarinen_channels(Z, weights)
    return odd + even


def marginal_shape(X: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """Per-asset shape statistics and negentropies for a (T x N) window."""
    if weights is None:
        weights = np.full(X.shape[0], 1.0 / X.shape[0])
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    Z, _ = _standardise(X, weights)
    skew = weights @ (Z ** 3)
    exkurt = weights @ (Z ** 4) - 3.0
    j_odd, j_even = negentropy_hyvarinen_channels(Z, weights)
    j_skew_mom, j_kurt_mom = negentropy_moments(skew, exkurt)
    return {
        'skew': skew,
        'exkurt': exkurt,
        'j_odd': j_odd,
        'j_even': j_even,
        'j_hyv': j_odd + j_even,
        'j_moments': j_skew_mom + j_kurt_mom,
        'j_skew_moments': j_skew_mom,
        'j_kurt_moments': j_kurt_mom,
    }


def entropy_deficit(
    X: np.ndarray,
    weights: np.ndarray | None = None,
    market_weights: np.ndarray | None = None,
    method: str = 'hyvarinen',
) -> dict:
    """Marginal entropy deficit of a return window, plus its signed signature.

    ``X`` is (T x N). ``weights`` are observation weights (EWMA or uniform);
    ``market_weights`` are cross-sectional weights defining the market factor
    whose *signed* skewness carries the direction an even functional cannot.
    """
    if weights is None:
        weights = np.full(X.shape[0], 1.0 / X.shape[0])
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    if market_weights is None:
        market_weights = np.full(X.shape[1], 1.0 / X.shape[1])

    m = marginal_shape(X, weights)

    mkt = X @ np.asarray(market_weights, dtype=float)
    zm, _ = _standardise(mkt[:, None], weights)
    zm = zm[:, 0]
    skew_mkt = float(weights @ zm ** 3)
    exkurt_mkt = float(weights @ zm ** 4) - 3.0
    j_odd_mkt, j_even_mkt = negentropy_hyvarinen_channels(zm[:, None], weights)

    d_hyv = float(m['j_hyv'].sum())
    d_mom = float(m['j_moments'].sum())
    out = {
        'd_hyvarinen': d_hyv,
        'd_moments': d_mom,
        'd_odd': float(m['j_odd'].sum()),
        'd_even': float(m['j_even'].sum()),
        'd_skew_moments': float(m['j_skew_moments'].sum()),
        'd_kurt_moments': float(m['j_kurt_moments'].sum()),
        'skew_mean': float(m['skew'].mean()),
        'exkurt_mean': float(m['exkurt'].mean()),
        'skew_market': skew_mkt,
        'exkurt_market': exkurt_mkt,
        'j_market': float(j_odd_mkt[0] + j_even_mkt[0]),
    }
    if method == 'hyvarinen':
        out['d_total'] = d_hyv
    elif method == 'moments':
        out['d_total'] = d_mom
    else:
        raise ValueError(f'Unknown negentropy method: {method}')
    return out


def structural_index(h_dep: float, d_total: float) -> float:
    """Structural information index J = D - h_dep (nats, non-negative).

    The divergence of the joint conditional return distribution from the product
    of Gaussian marginals with the same scales, under the Gaussian-copula
    approximation of dependence. It aggregates the two entropy-destroying
    channels -- dependence (``-h_dep``) and marginal non-Gaussianity (``D``) --
    and is invariant to the level of volatility.
    """
    return float(d_total - h_dep)


def tail_dependence(X: np.ndarray, q: float = 0.05) -> float:
    """Empirical lower-tail co-exceedance rate, averaged over asset pairs.

    For each pair, the share of dates on which both assets fall below their own
    ``q`` quantile, normalised by ``q`` so that independence gives ``q`` and
    perfect co-movement gives 1. Measures the non-linear dependence that the
    Gaussian-copula term of the structural index omits.
    """
    T, N = X.shape
    thr = np.quantile(X, q, axis=0)
    below = (X <= thr).astype(float)
    co = below.T @ below / T
    off = co[~np.eye(N, dtype=bool)]
    return float(off.mean() / q)
