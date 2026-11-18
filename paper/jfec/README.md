# Replication package — "What Survives a Placebo? Entropy, Market Coupling and the Plausibility of Stress Scenarios"

Sébastien Gallet, November 2026. Prepared for submission to the *Journal of Financial Econometrics* (JFEC).

## Contents

| Folder | What it holds |
| --- | --- |
| `paper/` | `main.tex` / `main.pdf` (JFEC submission copy, double-spaced, tables and figures at the end), `online_appendix.tex` / `.pdf`, `references.bib`, `numbers.tex` (generated), `tables/`, `figures/` |
| `note_maths/` | `math_note_EN.tex` / `.pdf`: the mathematics of the paper, step by step, at master's level |
| `francais/` | French version of the paper (`main_fr.tex` / `.pdf`) |
| `code/` | the Python code: `src/` (library), `scripts/` (one script per table), `config/`, `requirements.txt` |
| `data/` | raw inputs (`raw/`) and every result table the paper reads (`results/`) |

## Data

| File | Source | Licence / access |
| --- | --- | --- |
| `data/raw/sp500_prices.csv` | daily closing prices of 20 S&P 500 constituents, 1990-01-02 to 2022-12-28, from the example dataset of `skfolio` | BSD-3 |
| `data/raw/49_Industry_Portfolios_daily_CSV.zip` | Kenneth R. French Data Library, 49 industry portfolios, daily, value-weighted | public |
| euro-area panel (not included) | ECB Statistical Data Warehouse, 28 daily series, built by `code/scripts/euro_ciss.py` from the zip produced by the download script of the central-bank project | public, but the bundle was built on the author's machine |
| EBA 2025 scenario | EBA, *2025 EU-wide stress test: macro-financial scenario* (ESRB), tables 4.1.6, 4.1.7, 4.1.10 | public |

The S&P 500 sample contains firms that survived the whole period (survivorship bias); the FF49 panel is the survivorship-free check.

## How to reproduce

Python 3.11 or later. From `code/`:

```bash
pip install -r requirements.txt arch
mkdir -p calcul_local/data data && cp ../data/raw/sp500_prices.csv calcul_local/data/ && cp ../data/raw/49_Industry_Portfolios_daily_CSV.zip data/
mkdir -p paper/tables && cp ../data/results/*.csv paper/tables/
```

| Paper item | Command | Output (in `paper/tables/`) | Time (4 cores) |
| --- | --- | --- | --- |
| Tables 4 and 5 (coupling, verdicts), Figures 1–2, online tables OA.1–OA.2 | `python -m scripts.full_null --n-rep 999 --jobs 4` (resumable; every placebo statistic under the i.i.d. and CCC-FHS nulls on the same draws) | `full_null_999rep_*`, `*_gjr_params.csv` | ~10 h |
| Tables 2 and 3 (bias, Monte Carlo) | `python -m scripts.mc_coupling_test --sizes 10:1000,20:500,48:300 --n-boot 199 --jobs 4`, then `python -m scripts.mc_coupling_test --sizes 48:200 --n-boot 99 --est lw --dgps iid_t,ccc_gjr,ms_vol,ms_corr --skip-bias --jobs 4` | `mc_bias.csv`, `mc_size_power*.csv` | ~10 h + 2 h |
| Table 6 (forecasts, DM and Clark–West) | `python scripts/h7_horizons.py config/sp500.yaml config/default.yaml` | `*_h7_horizons.csv` | ~10 min |
| Table 7 (classical scenarios) | `python -m src.run_empirical --config config/sp500.yaml` | `outputs/sp500/tables/` (copied to `paper/tables/sp500_classical_comparison.csv`) | ~5 min |
| Table 8 (EBA) | `python scripts/eba_nats.py <unzipped euro panel>` | `eba_nats.csv` | 1 min |
| Online appendix, entropic CISS and its two nulls | `python -m scripts.euro_ciss --data <unzipped euro panel> --overnight-only --n-rep 999 --nulls iid,ccc_fhs` | `euro_null_999rep_*`, `euro_*` | ~4 h |
| All LaTeX tables, number macros and figures | `python -m scripts.jfec_build` | `paper/jfec/` | 1 min |

Then compile `paper/main.tex` with `pdflatex`, `bibtex`, `pdflatex` ×2.

Random seeds are fixed in every script (one `SeedSequence` per replication), so reruns reproduce the published numbers exactly, whatever the number of workers. `src/lw.py` is a lean copy of scikit-learn's Ledoit–Wolf estimator (identical to 1e-15, checked by `tests/test_lw.py`) that makes the runs about three times faster.

## What was and was not recomputed for this version

Recomputed in November 2026 for the submission version: every placebo statistic of both US panels under both nulls with 999 replications, the within-year gaps, the euro-area block coupling under both nulls with 999 replications, the Monte Carlo study (10, 20 and 48 assets, plus the Ledoit–Wolf check), the bias table, the H7 forecasts with the Clark–West test, the Holm-adjusted verdicts and the figures.

Carried over unchanged from the stored result tables of the project (same code, same data): the classical-scenario audit and the EBA 2025 pricing. The euro-area panel (ECB, FRED and Yahoo Finance series) is not redistributed; `telecharger.py` in the project rebuilds it.
