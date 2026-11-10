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
| Table 4, i.i.d. columns; online table OA.2 | `python -m scripts.coupling_null --jobs 4` | `coupling_null_200rep_*.csv` | ~1 h |
| Table 4, CCC-FHS columns and within-year rows; Figure 1 | `python -m scripts.garch_null --jobs 4 --nulls ccc_fhs,iid` | `garch_null_200rep_*.csv`, `*_gjr_params.csv` | ~1.5 h |
| Tables 2 and 3 (bias, Monte Carlo) | `python -m scripts.mc_coupling_test --jobs 4` | `mc_bias.csv`, `mc_size_power*.csv` | ~20 min |
| Table 5 (verdicts) | `python -m scripts.placebo_null` (200-rep placebo nulls; `calcul_local/lancer_calcul.py` is the stand-alone version), then `python -m scripts.episode_null` | `*_placebo_null.csv`, `sp500_episode_null_200rep.csv` | ~7 CPU-hours |
| Table 6 (forecasts, DM and Clark–West) | `python scripts/h7_horizons.py config/sp500.yaml config/default.yaml` | `*_h7_horizons.csv` | ~10 min |
| Table 7 (classical scenarios) | `python -m src.run_empirical --config config/sp500.yaml` | `outputs/sp500/tables/` (copied to `paper/tables/sp500_classical_comparison.csv`) | ~5 min |
| Table 8 (EBA) | `python scripts/eba_nats.py <unzipped euro panel>` | `eba_nats.csv` | 1 min |
| Online appendix, entropic CISS | `python -m scripts.euro_ciss` (needs the euro panel) | `euro_*.csv` | ~2 h |
| All LaTeX tables, number macros and figures | `python -m scripts.jfec_build` | `paper/jfec/` | 1 min |

Then compile `paper/main.tex` with `pdflatex`, `bibtex`, `pdflatex` ×2.

Random seeds are fixed in every script (`SeedSequence`), so reruns reproduce the published numbers exactly; the i.i.d. replications of `garch_null.py` use the same seeds as `coupling_null.py` and match its raw gaps.

## What was and was not recomputed for this version

Recomputed in November 2026 for the JFEC version: the CCC-FHS null and the within-year gaps (both US panels), the Monte Carlo study, the bias table, the H7 forecasts with the Clark–West test, the Holm-adjusted verdicts, the two main-text figures.

Carried over unchanged from the stored result tables of the project (same code, same data): the 200-replication i.i.d. placebo nulls of all hypotheses, the episode null, the classical-scenario audit, the EBA 2025 pricing and the euro-area entropic CISS. The euro-area panel is not redistributed here, so the CCC-FHS null was not run on it.
