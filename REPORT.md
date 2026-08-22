# Research execution report — 22 August 2026 (updated)

This update follows a full review of the repository (math, code, reproducibility) and closes the three action items raised by that review: code fixes, a mathematical consistency correction in the paper, and a real-data run to replace the earlier synthetic-only validation.

## A — Code fixes

- **Bug found and fixed:** `read_ff49_daily` silently returned an *empty* DataFrame — no error — when the CSV delimiter assumption did not hold (verified by deliberately feeding it a space-delimited variant of the real format). It now raises `ValueError` when parsing yields zero rows, and no longer crashes with an opaque `StopIteration` when the target section runs to end-of-file with no trailing blank line. Column names are now stripped of incidental whitespace.
- Harmonized error handling: `maxent_entropy_from_cov` now raises `ValueError` on a non-positive-definite covariance, consistent with `covariance_decomposition` (it previously returned `NaN` silently, which could let a degenerate estimate propagate unnoticed).
- Added `tests/test_data.py`: fixture-based tests for `read_ff49_daily` (value-weighted block parsing, `Other`-column handling, the empty-result failure mode) and for the new `read_eu_stock_markets_daily` reader — the real-data parsing path had zero test coverage before this pass.
- Added a dispatch test for `empirical.main` covering the new `data.type` config switch.
- Generalized `src/data.py`, `src/download_data.py`, `src/empirical.py` to support multiple data sources via `config['data']['type']` (`ff49` or `eu_stock_markets`) instead of being hard-wired to the FF49 zip.
- Added `.gitignore` and removed previously (accidentally) committed `__pycache__` bytecode.
- Added `pyproject.toml` so the package installs with `pip install -e .`.
- Added `.github/workflows/tests.yml` — CI was claimed in the previous version of this report but did not exist; it now does.
- Docstrings added to `src/entropy.py` and `src/data.py` explaining the formulas and the calendar-reconstruction caveat.

All 9 tests pass (`pytest -q`), up from 4.

## B — Mathematical development

- **Itô-consistency fix (Section 6 of the paper):** the paper carefully derives the drift correction between log-price and price via Itô's formula (Section 4), but the differential form of the dependence-entropy compensation, `dH_dep = 1/2 tr(R^{-1} dR)`, implicitly treated `R_t` as finite-variation. If `R_t` is itself a diffusion, `log det R_t` needs the analogous second-order (quadratic-variation) correction. The paper now states this explicitly, gives the corrected differential, and notes that the empirical analysis sidesteps the issue by using discrete finite differences rather than continuous-time differentials.
- **Tightened the proof of Proposition 1** (Gaussian maximizes entropy under fixed mean/covariance): replaced the informal Lagrangian-only sketch with the direct, rigorous Kullback–Leibler argument (`h(p) ≤ h(q)` from `D_KL(p‖q) ≥ 0`), keeping the Lagrangian as a heuristic for *why* the maximizer has Gaussian form rather than as the proof of optimality/uniqueness.
- **Fixed a LaTeX source bug:** the References and Conclusion sections were corrupted — a stray extra backslash and literal `\n` text sequences (not real line breaks) had been embedded in the source, which would have rendered as garbled raw text in the compiled PDF. Fixed and recompiled cleanly (`pdflatex`, 7 pages, no warnings).
- Cross-referenced the non-Gaussianity/KL section to the corrected Proposition 1 proof, and to the new pilot-data section.

## C — Real-data empirical run

The primary target (Kenneth French 49 Industry Portfolios) remains unreachable from this environment: `python -m src.download_data` fails with a proxy-level 403 on `mba.tuck.dartmouth.edu` — an outbound network-policy restriction of the sandbox, confirmed via the proxy status endpoint, not a bug in the download script. Per policy this was not circumvented.

Rather than leave the empirical claim at "synthetic only," the identical pipeline was run against a second dataset that the sandbox's network policy does permit: `raw.githubusercontent.com`, which hosts the Rdatasets mirror of the classic `EuStockMarkets` panel (DAX, SMI, CAC, FTSE daily closes, 1991–1998; Bollerslev & Ghysels). This is genuine, independently verifiable real market data — not synthetic, not fabricated — obtained via `python -m src.download_data --config config/pilot_eu_stock_markets.yaml` and analyzed with `python -m src.run_empirical --config config/pilot_eu_stock_markets.yaml`.

Results (see `paper/main.tex`, "Pilot empirical illustration on real data", for the full statement and caveats):

- **H1 (stress contraction):** confirmed directionally. Stress dates (top decile of trailing 21-day market vol; cluster around Sep 1992 ERM crisis, Sep 1993 ERM turbulence, Oct–Nov 1997 Asian crisis) show higher `H_vol` (−18.25 vs −18.93) and lower `H_dep` (−1.147 vs −0.911) than calm dates.
- **H2 (compensation):** `corr(ΔH_vol, ΔH_dep) = −0.23` (negative, as required). In the stress subsample, `sd(ΔH_cov) = 0.021` is below the "independent-components" benchmark `sqrt(sd(ΔH_vol)² + sd(ΔH_dep)²) = 0.0315`, consistent with partial compensation.
- **H3 (predictive value):** inconclusive at this scale. Test MSE improves only marginally (0.0849 vs 0.0869) and the `H_dep` coefficient is not significant (p = 0.64) on the 70/30 split.

This is a 4-asset, single-window pilot — explicitly not a substitute for the intended 49-industry, century-scale test, and reported as such in the paper. It demonstrates the pipeline on genuine market data and gives directionally consistent, non-confirmatory evidence for H1–H2.

## What still deserves attention

1. Run the real FF49 analysis as soon as the download is reachable (`config/default.yaml` is unchanged and ready).
2. The pilot's business-day date reconstruction (no exchange-holiday calendar) should not be relied on for exact-date event studies, only for the rolling-window ordering used here.
3. Points 2–8 of the previous version of this report (covariance estimation robustness, industry portfolios vs. individual assets, `log det R` terminology, stress-regime robustness, genuine walk-forward evaluation, non-Gaussianity via KL, unit-dependence of differential entropy) still stand and are unaffected by this update.
