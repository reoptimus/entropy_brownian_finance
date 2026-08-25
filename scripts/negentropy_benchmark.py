"""Accuracy benchmark for the marginal-negentropy estimators.

Reproduces the table quoted in the paper's methodology appendix: the Edgeworth
form S^2/12 + K^2/48 is transparent but diverges on fat-tailed samples, whereas
Hyvarinen's bounded-contrast approximation tracks the true negentropy closely
over the range of shapes daily equity returns actually take.

Reference negentropy is computed with the Vasicek spacing estimator of
differential entropy, which is consistent and free of any parametric
assumption.

Usage::

    python -m scripts.negentropy_benchmark [--n 200000] [--out outputs/tables/negentropy_benchmark.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.shape import negentropy_hyvarinen, negentropy_moments

GAUSS_ENTROPY = 0.5 * np.log(2 * np.pi * np.e)


def vasicek_negentropy(x: np.ndarray) -> float:
    """Negentropy of a standardised sample via the Vasicek spacing estimator."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    m = max(1, int(np.sqrt(n)))
    d = x[m:] - x[:-m]
    h = float(np.mean(np.log(n / m * d)))
    return GAUSS_ENTROPY - h


def build_cases(n: int, seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        'Gaussian': rng.standard_normal(n),
        'Uniform': rng.uniform(-1.0, 1.0, n),
        'Skew-normal (a=-4)': stats.skewnorm.rvs(-4, size=n, random_state=seed + 1),
        'Laplace': rng.laplace(size=n),
        'Student t(8)': rng.standard_t(8, size=n),
        'Student t(5)': rng.standard_t(5, size=n),
        'Student t(4)': rng.standard_t(4, size=n),
        'Lognormal (s=0.5)': rng.lognormal(0.0, 0.5, n),
        'Exponential': rng.exponential(size=n),
    }


def run(n: int = 200_000, seed: int = 1) -> pd.DataFrame:
    w = np.full(n, 1.0 / n)
    rows = []
    for name, sample in build_cases(n, seed).items():
        z = (sample - sample.mean()) / sample.std()
        skew = float(stats.skew(z))
        exkurt = float(stats.kurtosis(z))
        j_s, j_k = negentropy_moments(np.array([skew]), np.array([exkurt]))
        rows.append({
            'distribution': name,
            'skewness': skew,
            'excess_kurtosis': exkurt,
            'J_true_vasicek': vasicek_negentropy(z),
            'J_edgeworth': float(j_s[0] + j_k[0]),
            'J_hyvarinen': float(negentropy_hyvarinen(z[:, None], w)[0]),
        })
    df = pd.DataFrame(rows)
    df['edgeworth_ratio'] = df['J_edgeworth'] / df['J_true_vasicek'].clip(lower=1e-6)
    df['hyvarinen_ratio'] = df['J_hyvarinen'] / df['J_true_vasicek'].clip(lower=1e-6)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=200_000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out', default='outputs/tables/negentropy_benchmark.csv')
    args = parser.parse_args()

    df = run(args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
