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

The primary dataset is the **Kenneth R. French 49 Industry Portfolios, daily value-weighted returns**. The daily file is available from July 1, 1969 through June 30, 2026 (the monthly series goes back further, to 1926, but the daily series does not). The portfolios are built from NYSE, AMEX and NASDAQ firms using SIC-based industry assignments.

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_49_ind_port.html

Download endpoint used by the code:
`https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip`

A second robustness source should ideally be run later using individual-stock data (CRSP/Compustat through WRDS, if available) because industry portfolios mechanically reduce idiosyncratic noise and are not independent firms.

## Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.download_data --config config/default.yaml
python -m src.run_empirical --config config/default.yaml
pytest -q
```

Note on the development environment: the sandbox this repository was developed in sits behind an outbound network allowlist that does not include `mba.tuck.dartmouth.edu`, so `download_data.py` cannot reach it from there — a network-policy restriction of that sandbox, not a code defect, and it does not affect a normal machine. This is why the results below were produced from a manually supplied copy of the official zip rather than a live download in that session.

### Real FF49 result (included, reproducible)

The paper's intended empirical test has been run on the official FF49 daily file: **48 industries** (the residual *Other* portfolio excluded), **1990-01-01 to 2026-06-30**, 8,939 rolling 252-day windows. Findings — see `paper/main.tex`, Section "Empirical results: the FF49 industry cross-section", for the full statement and caveats:

- **H1 (stress contraction): confirmed.** Stress dates show higher `H_vol` and lower `H_dep` than calm dates.
- **H2 (compensation): confirmed.** `corr(ΔH_vol, ΔH_dep) = −0.63` over the full sample; `sd(ΔH_cov)` is well below the value implied by treating the two components as uncorrelated, in both calm and stress regimes.
- **H3 (predictive value): significant in sample, not confirmed out of sample.** Adding `H_dep` to `H_vol` is highly significant under HAC standard errors (p ≈ 9.5×10⁻⁵) and raises train R² from 0.28 to 0.31, but does not reduce test MSE on a chronological 70/30 split — most likely because the 252-day rolling window induces heavy autocorrelation that a simple HAC correction does not fully absorb (see Limitations in the paper for a proposed walk-forward fix).

The stress-regime dates independently line up with known market history (1997–98, 2000–02, 2008–11, 2015–16, 2018, 2020, 2022), which is itself a sanity check on the pipeline unrelated to the paper's hypotheses.

### Second real dataset (robustness / pilot)

The identical pipeline was also run, unmodified, on a second, independent real dataset: the classic `EuStockMarkets` panel (DAX, SMI, CAC, FTSE daily closes, 1991–1998, Bollerslev & Ghysels), mirrored as CSV by the Rdatasets project:

`https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/EuStockMarkets.csv`

```bash
python -m src.download_data --config config/pilot_eu_stock_markets.yaml
python -m src.run_empirical --config config/pilot_eu_stock_markets.yaml
```

This is a smaller (4-asset, single 8-year window) but genuinely independent real-data check, kept as a robustness diagnostic — see `paper/main.tex`, Section "Robustness check on a second, independent real dataset". It replicates the sign and shape of H1–H2 on an unrelated market and period.

## Main outputs

Written under `outputs/tables` and `outputs/figures` (FF49 run) or `outputs/pilot_eu/{tables,figures}` (pilot run):

- `summary.csv`
- `regime_summary.csv`
- `predictive_regressions.csv`
- `entropy_components.png`
- `compensation_scatter.png`
- `logdet_covariance.png`

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
- The pilot dataset's dates are reconstructed as a plain business-day sequence (the upstream file has no calendar column), so individual dates can drift by a few days from the true exchange calendar; only chronological order and spacing are used by the pipeline.

## Development

`pip install -e .` (via `pyproject.toml`) installs the package for local development. `pytest -q` runs the full test suite, including a fixture-based test of the real FF49 CSV parser (`tests/test_data.py`) so a change to the parsing logic — or a change in the upstream file format — is caught without needing network access. CI (`.github/workflows/tests.yml`) runs the same suite on push/PR.
