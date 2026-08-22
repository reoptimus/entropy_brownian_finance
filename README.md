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
python -m src.download_data --config config/default.yaml
python -m src.run_empirical --config config/default.yaml
pytest -q
```

The environment used to develop this repository is on an outbound network allowlist that does not include `mba.tuck.dartmouth.edu`, so the FF49 download above could not be executed from here — this is a network-policy restriction of that environment, not a code defect. **No FF49 real-data result is claimed by this repository yet.**

### Pilot real-data run (included, reproducible)

To avoid shipping only a synthetic validation, the same pipeline was also run, unmodified, on a second real dataset reachable from that environment: the classic `EuStockMarkets` panel (DAX, SMI, CAC, FTSE daily closes, 1991–1998, Bollerslev & Ghysels), mirrored as CSV by the Rdatasets project:

`https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/EuStockMarkets.csv`

```bash
python -m src.download_data --config config/pilot_eu_stock_markets.yaml
python -m src.run_empirical --config config/pilot_eu_stock_markets.yaml
```

This is a genuine real-data run (not synthetic), but it is a 4-asset pilot over one 8-year window, not the paper's intended 49-industry, century-scale test — see `paper/main.tex`, Section "Pilot empirical illustration on real data", for the numbers and their interpretation. Switch back to `config/default.yaml` for the FF49 analysis as soon as the download is reachable (e.g. run locally, outside this sandbox).

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
