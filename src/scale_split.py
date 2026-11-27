"""Scale versus structure in Gaussian stress scenarios.

The three-channel identity of the paper writes the Gaussian entropy as
``const + sum_i log sigma_i - C(R)``: the scale channel raises entropy in a
crisis, the coupling channel ``C`` lowers it. A Gaussian scenario moves only
these two (its shape deficit is zero by construction), so its deformation cost
in nats can be read channel by channel. This module gives

* :func:`split_deformation` -- the exact three-term split of
  ``KL(N(0, Sigma1) || N(0, Sigma0))`` into a pure-volatility cost, a
  pure-correlation cost and an interaction term, with ``Sigma = D R D``;
* :func:`channel_efficiencies` -- the small-shock exchange rate: the share of
  the minimal-lift efficiency (Proposition "Minimal lift") that a shock
  restricted to volatilities, or to correlations, can achieve, in closed form;
* :func:`cheapest_vol_only`, :func:`cheapest_corr_only` -- the exact
  finite-shock optimum inside each channel (numerical), and
  :func:`correlation_ceiling`, the largest portfolio volatility a
  correlation-only shock can reach.

All costs are deformation costs (the location cost ``u^2/2`` is separable
exactly and is left out).
"""
from __future__ import annotations

import numpy as np
from scipy import optimize


def _vol_corr(cov: np.ndarray) -> tuple:
    sd = np.sqrt(np.diag(cov))
    return sd, cov / np.outer(sd, sd)


def coupling(R: np.ndarray) -> float:
    """``C(R) = -1/2 log det R`` (nats)."""
    return float(-0.5 * np.linalg.slogdet(R)[1])


def gaussian_deformation(cov1: np.ndarray, cov0: np.ndarray) -> float:
    """``KL(N(0, cov1) || N(0, cov0))``, nats."""
    n = cov0.shape[0]
    A = np.linalg.solve(cov0, cov1)
    return float(0.5 * (np.trace(A) - n - np.linalg.slogdet(A)[1]))


def split_deformation(cov1: np.ndarray, cov0: np.ndarray) -> dict:
    """Exact split of the deformation cost into scale, correlation, interaction.

    With ``Sigma_k = D_k R_k D_k`` and ``S = D_1 D_0^{-1}``:

    * ``K_scale = KL(D1 R0 D1 || D0 R0 D0)``: volatilities move, correlations
      stay at ``R0``;
    * ``K_corr = KL(R1 || R0)``: correlations move, volatilities stay (it does
      not depend on ``D0``);
    * ``I = 1/2 tr(R0^{-1} (S E S - E))`` with ``E = R1 - R0``.

    ``total = K_scale + K_corr + I`` holds exactly. ``I = 0`` when ``S = I``,
    when ``R1 = R0`` or when ``R0 = I`` (independent reference); with
    ``R0 = I``, ``K_corr = C(R1)``: coupling is the nats cost of correlation
    measured from independence. Also returned: ``d_scale = sum log s_i`` and
    ``d_coupling = C(R1) - C(R0)``, the two entropy channels the scenario moves.
    """
    sd0, R0 = _vol_corr(np.asarray(cov0, float))
    sd1, R1 = _vol_corr(np.asarray(cov1, float))
    s = sd1 / sd0
    S = np.diag(s)
    R0inv = np.linalg.inv(R0)
    E = R1 - R0
    k_scale = gaussian_deformation(S @ R0 @ S, R0)
    k_corr = gaussian_deformation(R1, R0)
    inter = float(0.5 * np.trace(R0inv @ (S @ E @ S - E)))
    total = gaussian_deformation(cov1, cov0)
    return {
        'total': total,
        'K_scale': k_scale,
        'K_corr': k_corr,
        'interaction': inter,
        'check': total - (k_scale + k_corr + inter),
        'd_scale': float(np.log(s).sum()),
        'd_coupling': coupling(R1) - coupling(R0),
    }


def minimal_lift_cov(cov0: np.ndarray, w: np.ndarray, v: float) -> np.ndarray:
    """``Sigma0 + (v^2 - 1) s0^2 beta beta'`` (Proposition "Minimal lift")."""
    s02 = float(w @ cov0 @ w)
    beta = cov0 @ w / s02
    return cov0 + (v * v - 1.0) * s02 * np.outer(beta, beta)


def lift_cost(v: float) -> float:
    """Deformation cost of the minimal lift, ``1/2 (v^2 - 1 - 2 log v)``."""
    return 0.5 * (v * v - 1.0 - 2.0 * np.log(v))


def correlation_ceiling(cov0: np.ndarray, w: np.ndarray) -> float:
    """Largest ``v`` a correlation-only shock can give: ``sum |w_i| sigma_i / s0``."""
    sd = np.sqrt(np.diag(cov0))
    return float(np.abs(w) @ sd / np.sqrt(w @ cov0 @ w))


def channel_efficiencies(cov0: np.ndarray, w: np.ndarray) -> dict:
    """Small-shock efficiencies of the volatility and correlation channels.

    Raise the portfolio variance by a factor ``1 + eps`` (``eps`` small). The
    minimal lift costs ``eps^2/4``; the cheapest shock restricted to a channel
    costs ``eps^2 / (4 e)`` with

    * volatilities only: ``e_vol = 2 c' (I + R0 o R0^{-1})^{-1} c``, ``c`` the
      Euler risk contributions (``sum c = 1``), ``o`` the Hadamard product;
    * correlations only: ``e_corr = 1 - (rho^2)' (R0 o R0)^{-1} (rho^2)``,
      ``rho_i`` the correlation of asset ``i`` with the portfolio;
    * uniform dilation: ``e_unif = 1/N`` (Corollary "Uniform dilation").

    ``e_corr / e_vol`` is the exchange rate: nats of volatility shock per nat
    of correlation shock for the same small loss. At ``R0 = I`` the two
    channels are Fisher-orthogonal and ``e_vol + e_corr = 1``.
    """
    sd, R0 = _vol_corr(np.asarray(cov0, float))
    n = len(sd)
    a = w * sd
    s02 = float(a @ R0 @ a)
    b = R0 @ a
    c = a * b / s02
    rho = b / np.sqrt(s02)
    M = np.eye(n) + R0 * np.linalg.inv(R0)
    e_vol = float(2.0 * c @ np.linalg.solve(M, c))
    r2 = rho ** 2
    e_corr = float(1.0 - r2 @ np.linalg.solve(R0 * R0, r2))
    return {'e_vol': e_vol, 'e_corr': e_corr, 'e_uniform': 1.0 / n,
            'exchange_rate_corr_per_vol': e_corr / e_vol}


def cheapest_vol_only(cov0: np.ndarray, w: np.ndarray, v: float) -> dict:
    """Cheapest scenario with correlations fixed and portfolio sd ``v s0``.

    Minimises ``1/2 (s' (R0 o R0^{-1}) s - N) - sum log s`` subject to
    ``(s o a)' R0 (s o a) = v^2 s0^2``, ``a = D0 w``; the objective is convex
    in ``s``.
    """
    sd, R0 = _vol_corr(np.asarray(cov0, float))
    n = len(sd)
    a = w * sd
    s02 = float(a @ R0 @ a)
    H = R0 * np.linalg.inv(R0)

    def f(y):
        s = np.exp(y)
        return 0.5 * (s @ H @ s - n) - y.sum(), (s * (H @ s)) - 1.0

    def g(y):
        s = np.exp(y)
        sa = s * a
        return (sa @ R0 @ sa) / s02 - v * v

    def gj(y):
        s = np.exp(y)
        sa = s * a
        return 2.0 * sa * (R0 @ sa) / s02

    y0 = np.full(n, np.log(v))
    res = optimize.minimize(f, y0, jac=True, method='SLSQP',
                            constraints=[{'type': 'eq', 'fun': g, 'jac': gj}],
                            options={'maxiter': 1000, 'ftol': 1e-14})
    s = np.exp(res.x)
    cov1 = np.outer(sd * s, sd * s) * R0
    return {'cost': float(res.fun), 'success': bool(res.success), 'cov1': cov1,
            'scales': s}


def cheapest_corr_only(cov0: np.ndarray, w: np.ndarray, v: float,
                       n_steps: int = 40) -> dict:
    """Cheapest scenario with volatilities fixed and portfolio sd ``v s0``.

    Minimises ``KL(R1 || R0)`` over correlation matrices subject to
    ``a' R1 a = v^2 s0^2`` (``a = D0 w``). The first-order conditions give
    the optimum in closed form up to ``N + 1`` multipliers: the precision
    matrix moves by a rank-one term along ``a`` plus a diagonal,
    ``R1^{-1} = R0^{-1} - 2 lam a a'/s0^2 - diag(m)``, with ``m`` and
    ``lam`` fixed by ``diag R1 = 1`` and the variance target. They are found
    by Newton steps with continuation in ``v``. Returns ``cost = inf`` above
    the correlation ceiling.
    """
    sd, R0 = _vol_corr(np.asarray(cov0, float))
    n = len(sd)
    a = w * sd
    ah = a / np.sqrt(float(a @ R0 @ a))
    if v >= correlation_ceiling(cov0, w) - 1e-9:
        return {'cost': np.inf, 'success': False, 'cov1': None}
    P0 = np.linalg.inv(R0)
    theta = np.zeros(n + 1)  # (lam, m)

    def solve_R(th):
        P = P0 - 2.0 * th[0] * np.outer(ah, ah) - np.diag(th[1:])
        return np.linalg.inv(P), P

    ok = True
    for vk in np.linspace(1.0, v, n_steps + 1)[1:]:
        for _ in range(100):
            R, P = solve_R(theta)
            Ra = R @ ah
            F = np.concatenate([[ah @ Ra - vk * vk], np.diag(R) - 1.0])
            if np.max(np.abs(F)) < 1e-12:
                break
            Jm = np.zeros((n + 1, n + 1))
            Jm[0, 0] = 2.0 * (ah @ Ra) ** 2
            Jm[0, 1:] = Ra ** 2
            Jm[1:, 0] = 2.0 * Ra ** 2
            Jm[1:, 1:] = R ** 2
            step = np.linalg.solve(Jm, -F)
            t = 1.0
            while t > 1e-6:
                cand = theta + t * step
                Pc = P0 - 2.0 * cand[0] * np.outer(ah, ah) - np.diag(cand[1:])
                if np.all(np.linalg.eigvalsh(Pc) > 0):
                    break
                t *= 0.5
            theta = theta + t * step
        else:
            ok = False
    R1, _ = solve_R(theta)
    cost = gaussian_deformation(R1, R0)
    return {'cost': cost, 'success': ok and bool(np.max(np.abs(np.diag(R1) - 1)) < 1e-9),
            'cov1': np.outer(sd, sd) * R1, 'lam': float(theta[0])}
