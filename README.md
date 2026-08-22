# Maximum Entropy, Brownian Motion and Volatility–Dependence Compensation

Research repository for the paper:

**From Maximum Entropy to Geometric Brownian Motion: Covariance Entropy and Volatility–Dependence Compensation in Financial Markets**

## Research question

The project separates three claims:

1. **Exact mathematical result:** among absolutely continuous distributions with fixed mean and covariance, the Gaussian maximizes differential entropy.
2. **Process construction:** Gaussian increments with independent increments, variance proportional to elapsed time, and continuity lead to Brownian motion; exponentiating log-prices gives geometric Brownian motion after Itô's formula.
3. **Empirical hypothesis:** financial stress may involve a compensation between marginal volatility entropy and dependence-induced covariance-volume contraction. The testable object is
   \[
   \Delta H^{cov}_t = \frac12\Delta\log\det\Sigma_t,
   \]
   with
   \[
   H^{vol}_t=\sum_i\log\sigma_{i,t},\qquad H^{dep}_t=\frac12\log\det R_t.
   \]

The repository **does not assume that the compensation hypothesis is true**.

## Data

The primary planned dataset is the **Kenneth R. French 49 Industry Portfolios, daily returns**, covering July 1, 1926 through June 30, 2026 according to the current Data Library page. The portfolios are built from NYSE, AMEX and NASDAQ firms using SIC-based industry assignments.

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_49_ind_port.html

Download endpoint used by the code:
`https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip`

A second robustness source should ideally be run later using individual-stock data (CRSP/Compustat through WRDS, if available) because industry portfolios mechanically reduce idiosyncratic noise and are not independent firms.

## Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.download_data
python -m src.run_empirical --config config/default.yaml
pytest -q
```

The current environment used to prepare this repository could inspect the official data source but could not execute the remote download. Therefore **no real-data empirical result is claimed by this repository yet**. The code includes a synthetic-data validation that checks the identities and estimation pipeline.

## Main outputs

- `outputs/tables/summary.csv`
- `outputs/tables/regime_summary.csv`
- `outputs/tables/predictive_regressions.csv`
- `outputs/figures/entropy_components.png`
- `outputs/figures/compensation_scatter.png`
- `outputs/figures/logdet_covariance.png`

## Main hypothesis tests

### H1: stress contraction
Stress is associated with higher volatility and lower `det(R)`.

### H2: compensation
Changes in `H_vol` and `H_dep` are negatively related and their sum is more stable than the individual components in selected regimes.

### H3: predictive value
`H_dep` adds information for forecasting future realized volatility and tail risk beyond marginal volatility.

## Important caveats

- `log(det R)` is a measure of dependence-induced volume contraction, not a generic measure of correlation and not sensitive to the sign of pairwise correlation in the bivariate case.
- Differential entropy depends on units. The empirical paper therefore emphasizes changes and normalized covariance entropy rather than an absolute physical interpretation.
- A sample covariance determinant can be badly biased/unstable in high dimensions. The pipeline uses Ledoit–Wolf shrinkage by default.
- The MaxEnt result is exact for fixed first and second moments; it does not imply that actual financial returns are Gaussian.
