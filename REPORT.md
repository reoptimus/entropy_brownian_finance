# Research execution report — 22 August 2026

## What was completed

### A — Mathematical development
- Formalized the maximum-entropy result under fixed mean and positive-definite covariance.
- Separated the exact MaxEnt theorem from the additional temporal assumptions required for Brownian motion.
- Corrected the Itô drift relation between log-prices and prices.
- Derived the covariance decomposition
  `det(Sigma) = (prod sigma_i^2) det(R)`
  and the associated maximum-entropy decomposition.
- Reframed the proposed conservation law as an empirical hypothesis about changes in `log det(Sigma)`, not as a theorem.
- Added the bivariate interpretation and the important sign-symmetry caveat for correlation.
- Added a route for studying non-Gaussianity via KL divergence to the Gaussian MaxEnt reference.

### B — Empirical methodology
- Selected the Kenneth French 49 Industry Portfolios daily value-weighted returns as the primary public dataset.
- The official current source reports daily data from 1 July 1926 through 30 June 2026.
- Designed rolling covariance estimation with Ledoit–Wolf shrinkage.
- Defined the three main empirical quantities: `H_vol`, `H_dep`, `H_cov`.
- Defined stress independently from the entropy decomposition using a trailing 21-day equal-weight market volatility signal.
- Added predictive evaluation using a 70/30 chronological train/test split.
- Added unit tests for the mathematical identities and the synthetic rolling pipeline.

### C — Paper and reproducible code
- Created a LaTeX paper draft in `paper/main.tex`.
- Created Python data ingestion, entropy decomposition, rolling estimation, regime analysis and predictive regression code.
- Created configuration in `config/default.yaml`.
- Created tests and continuous-integration configuration.
- Initialized a local Git repository and committed the project.

## Data status

The public dataset has been identified and its current official availability verified through Kenneth French's Data Library. However, the execution environment used for this work could not resolve external network requests to download the ZIP file. Therefore **no real-data empirical coefficient, p-value, figure, or confirmation is reported as a finding**.

This is deliberate. It would be scientifically wrong to manufacture or infer an empirical confirmation from the data description alone.

## Synthetic validation

The code was run on a synthetic two-regime Gaussian system. The tests confirm that:

1. the covariance entropy decomposition is numerically exact;
2. the bivariate determinant identity is recovered;
3. the rolling covariance pipeline produces finite estimates;
4. all automated tests pass (`4 passed`).

The synthetic experiment is only a software/theory validation. It is **not evidence for the financial hypothesis**.

## What deserves attention

### 1. The central empirical hypothesis is still unconfirmed
The key quantity is
`Delta log det(Sigma)`. The paper should not claim entropy conservation until this quantity is tested on real data.

### 2. Covariance estimation is probably the biggest empirical risk
With 49 industries and a 252-day rolling window, the covariance matrix is estimable, but determinant estimates can still be noisy and biased. Results should be repeated with different windows, sample covariance, Ledoit–Wolf shrinkage, and preferably a nonlinear-shrinkage or factor-based estimator.

### 3. Industry portfolios are not individual assets
They are useful for a first test, but common factors and portfolio construction may mechanically create the dependence structure. A CRSP/WRDS individual-stock robustness analysis would be important for a serious journal submission.

### 4. `log(det R)` is not the same thing as average correlation
In two dimensions it depends on `rho^2`; in higher dimensions it is a spectral volume measure. The paper should use the term `dependence-volume component` rather than treating it as a scalar correlation index.

### 5. Stress-regime definition must be robust
The baseline uses trailing market volatility, but the final paper should repeat the analysis using drawdowns, VIX where available, realized volatility, and dated crisis windows. This prevents the stress result from being an artifact of one regime definition.

### 6. Predictive results need genuine out-of-sample evaluation
The repository now uses a chronological 70/30 split, but a final paper should add rolling/expanding forecasts, benchmark models, Diebold–Mariano tests where appropriate, and formal VaR/ES backtests.

### 7. Non-Gaussianity is not a minor detail
The MaxEnt Gaussian is a reference distribution. Actual financial returns have skewness, excess kurtosis and dependence structures that the Gaussian does not capture. The proposed KL extension is worth developing.

### 8. Differential entropy is unit-dependent
Absolute entropy levels should not be given a physical interpretation. Changes, differences, normalized values, and `log det Sigma` are safer empirical objects.

## Recommended next empirical run

1. Download the FF49 daily file using `python -m src.download_data`.
2. Run `python -m src.run_empirical --config config/default.yaml`.
3. Inspect the sign and magnitude of `corr(dH_vol, dH_dep)`.
4. Compare `std(dH_cov)` with the component standard deviations in calm and stress regimes.
5. Repeat with windows 126, 252 and 504 days.
6. Repeat with sample covariance and alternative shrinkage.
7. Test 25/49 industry universes and, if available, individual-stock CRSP data.
8. Only then decide whether the compensation hypothesis deserves to be presented as a stylized fact.
