# Research execution report

## 25 August 2026 — structural rework: entropy jumps, relaxation, stress testing

The paper was rebuilt around a different logical spine, on request. What changed,
in order of importance.

### 1. The volume bridge is now argued, not asserted

The previous draft moved from "maximum entropy under fixed first two moments
gives a Gaussian" to "log-prices are Brownian" by adding temporal assumptions,
but never said where the variance came from. Section 2 now defines the price as
accumulated signed order flow (Kyle-style linear impact), makes the variance
budget proportional to traded volume, and applies maximum entropy *in volume
time*. The result is the mixture-of-distributions hypothesis (Clark 1973; Ané &
Geman 2000) derived rather than posited, and a subordinated Brownian motion
rather than a calendar-time one. Monroe's theorem is cited for what it implies:
subordination alone has no content — maximum entropy is what selects Brownian
motion from the class it allows. Section 2.5 lists where the bridge is weak
(square-root impact, autocorrelated order flow) and notes that no empirical
result depends on it.

### 2. A third entropy channel, and a scale-free state variable

The old decomposition had two channels, `H_vol` and `H_dep`. It is now three:
scale, dependence, and **shape** — the entropy deficit `D = KL(p‖q) ≥ 0`, which
by Proposition 1 is exactly the non-Gaussianity of the conditional distribution.
The Edgeworth identity `J ≈ S²/12 + K²/48` makes explicit that the shape channel
*is* skewness and fat tails, which is what the rework asked to be shown.

Combining the two scale-free channels gives the **structural index**

    J = D − ½ log det R = KL( p ‖ ∏ᵢ qᵢ ) ≥ 0

via an exact chain rule. This replaces `H_cov` as the paper's object of study,
for a reason the data supplied: `H_cov` *rises* in a crisis, because the scale
channel dominates it. Any indicator built on the level of Gaussian entropy has
the sign backwards.

### 3. Jump/diffusion: one theorem and two postulates, clearly separated

The requested postulate — information suddenly lowers entropy, which then
relaxes back — is now split into the part that is a theorem and the part that is
not. Conditioning cannot raise expected entropy (`E[h(X|Y)] = h(X) − I(X;Y)`), so
the downward jump follows, and *its size in nats is the mutual information of the
news*. What is postulated is P1 (diffusive relaxation) and P2 (the constraint is
one-sided, hence negative skewness). The answer to "does the link between fat
tails, correlation and entropy jumps need an extra postulate?" is: no for the
coupling, which follows from the news being common to the cross-section; yes for
the sign, because negentropy is even in skewness and cannot supply a direction.

### 4. The placebo null, and what it did to the results

This is the substantive finding of the rework and it is not the one that was
expected.

`J` is a non-negative convex functional of estimated second, third and fourth
moments. A fat-tailed observation entering the estimator pushes it up sharply;
the estimator's memory lets it decay smoothly. **Sawtooth dynamics are what the
measurement device produces on leptokurtic data, whether or not information
arrives in parcels.** Two nulls were built to measure this:

- `scripts/estimator_memory.py` — i.i.d. Gaussian data, constant covariance, one
  injected shock. Mechanical decay half-life **35 days**; on clean data with no
  event, the 4σ detector still finds **37 up-jumps and 0 down-jumps** and
  skew(ΔJ) = +2.9.
- `scripts/placebo_null.py` — twenty resamplings of the *observed* returns,
  preserving the fat marginal tails and the cross-sectional dependence. Two
  versions: **i.i.d.** rows (destroys all temporal structure) and **21-day
  blocks** (keeps short-run volatility clustering, destroys longer-horizon
  regime dynamics). Full pipeline on each, including the same volatility filter
  that defines the stress regime.

Verdicts against both nulls:

| | data | i.i.d. | block-21 | verdict |
|---|---|---|---|---|
| H1 `h_dep` regime gap | −1.72 | −0.29 | −0.64 | **clears both** |
| H1 `J` regime gap | +2.89 | +2.32 | +2.21 | i.i.d. only |
| H1 `D` regime gap | +0.48 | +1.35 | +0.74 | fails both |
| H2 corr(ΔH_vol, ΔH_dep) | +0.10 | +0.14 | +0.16 | fails both |
| H3 up-jumps | 158 | 147.7 | 129.5 | block only |
| H3 skew(ΔJ) | +12.3 | +14.6 | +13.8 | fails both |
| H4 half-life | 120 d | 92 d | 162 d | **not identified** |
| H4 range(J) | 18.95 | 10.49 | 14.23 | **clears both** |
| H5 corr(ΔJ, Δskew) | −0.24 | −0.07 | −0.01 | edge, *p* ≈ 0.05–0.10 |
| H6 coupling gap | 0.62 | 0.75 | 0.67 | fails both |
| H7 out-of-sample | — | — | — | rejected outright |

**Exactly two statistics clear both nulls.** The dependence channel's regime
signal — what distinguishes a crisis from a run of large returns is that the
cross-section couples, not that the marginals fatten — and the range of `J`,
which travels further in real markets than in any resampling of the same
returns.

The half-life is the instructive failure: at 120 days it sits *above* the i.i.d.
null (92) and *below* the block null (162). Block resampling occasionally
concatenates high-volatility blocks and manufactures long excursions; i.i.d.
resampling breaks up the runs real markets contain and manufactures short ones.
The level is an artefact of whichever null one chooses to believe, so P1 is
supported by the amplitude result rather than by a relaxation rate.

Neither null is nested in the other in a way that makes one uniformly
conservative — several null statistics come out *larger* than the observed ones,
because scrambling makes extreme days arrive out of calm baselines.

### 5. H2 does not replicate on individual firms

The FF49 result from the previous draft (corr(ΔH_vol, ΔH_dep) = −0.63, 57%
variance reduction) does not reproduce on twenty individual stocks: +0.10 over
the full sample, +0.28 in calm markets, −0.40 only in stress. Either industry
aggregation manufactures the compensation, or cross-sectional size drives it; the
data here cannot separate the two. Reported as a negative result, since the
earlier draft's own Limitations section asked for exactly this test.

### 6. Stress testing calibrated in nats

A scenario of severity η is the worst conditional distribution within a KL ball
of radius η around today's (Glasserman & Xu 2014; Breuer & Csiszár 2013). The
contribution is the calibration: η is read off the market's own realised
information flow `KL(pₜ‖pₜ₋ₕ)`, measured in the same unit, so a severity carries
a return period. Closed-form for a linear portfolio; **one nat buys √2 sigma**.

Pricing textbook bundles on the same scale gave the applied result:

| bundle | price | ES₉₉ | entropy scenario at the same ES₉₉ | overpayment |
|---|---|---|---|---|
| vol×1.5, ρ→0.70, −2σ | 6.80 nats | 9.1% | 1.85 nats (2.3 y) | 73% |
| vol×2.0, ρ→0.90, −3σ | 9.97 nats | 13.7% | 5.47 nats (10.6 y) | 45% |
| vol×3.0, ρ→0.95, −4σ | 20.19 nats | 20.3% | 13.74 nats (~32 y) | 32% |

A committee running the standard bundle believes it has specified a ~30-year
event; measured by the loss it actually produces, it has specified a 10-year
event and paid 30-year implausibility for it. The expensive leg is the uniform
volatility multiplier (16.1 nats alone), not the correlation shock (8.1) or the
directional shock (4.5 = σ²/2 exactly). The legs are strongly non-additive:
forcing correlations up collapses the covariance onto a near-rank-one subspace
the reference already grants variance to, which makes the volatility scaling
cheap.

### 7. Data

The Kenneth French site is unreachable from this sandbox (only
`raw.githubusercontent.com` and PyPI are allowlisted), so the FF49 reader and
config are kept intact but the primary panel is now **twenty S&P 500
constituents, daily, 1990–2022** — individual firms, which is the robustness test
the previous draft listed as its most-wanted future work. `EuStockMarkets`
remains as replication. Every FF49 number quoted is attributed to the earlier
draft rather than recomputed.

### 8. Code

New modules: `src/shape.py` (negentropy, channels, structural index),
`src/jumps.py` (bipower scale, detection, OU relaxation, event study,
indicators), `src/stress.py` (KL-ball scenarios, ladder, scenario pricing,
reverse stress), `src/hypotheses.py` (H1–H7 as tidy tables). `src/entropy.py`
gained EWMA covariance and spectral diagnostics; `src/empirical.py` now computes
surprisal and information flow alongside the channels; `src/run_empirical.py`
runs everything and draws eight figures. Four scripts under `scripts/`.

Two estimator decisions worth recording. The shape channel uses **Hyvärinen's
bounded-contrast negentropy**, not the Edgeworth form: benchmarked against a
Vasicek reference, Edgeworth overstates a Student-t(4) by a factor of 69 while
Hyvärinen is within 3%. And the jump analysis uses an **EWMA** covariance rather
than the 252-day box-car window, because with a box-car an extreme observation
*leaving* the window creates a discontinuity on a date when nothing happened —
which is exactly the artefact the previous draft flagged in its own Limitations.

Tests: 54, all passing, no network required.

---


# Research execution report — 22 August 2026 (updated twice)

## Second update: the real FF49 result

The user supplied the official `49_Industry_Portfolios_daily_CSV.zip` directly (the sandbox's network policy still blocks a live download of it from this session; this file was obtained by the user on their own machine and uploaded). It was placed at `data/raw/49_Industry_Portfolios_daily_CSV.zip` and run through the unmodified, already-tested pipeline via `python -m src.run_empirical --config config/default.yaml`.

The parser (`read_ff49_daily`, fixed and tested in the first update below) worked correctly on the first try: 48 industries (`Other` excluded), daily data from 1969-07-01 to 2026-06-30 (the daily file starts later than the monthly one, which goes back to 1926 — the earlier version of this README/report incorrectly stated 1926 for the daily series; now corrected), restricted to the configured sample 1990-01-01–2026-06-30, giving 8,939 rolling 252-day windows in ~38 seconds.

This is now the paper's intended empirical test, not a synthetic or small pilot substitute. Results (full statement in `paper/main.tex`, Section "Empirical results: the FF49 industry cross-section"):

- **H1 confirmed:** stress dates (top decile of trailing 21-day market vol) show `H_vol = -190.6` vs `-210.1` calm, and `H_dep = -31.6` vs `-23.1` calm.
- **H2 confirmed, and stronger than the earlier 4-asset pilot:** `corr(ΔH_vol, ΔH_dep) = -0.63` overall (-0.74 calm, -0.40 stress). `sd(ΔH_cov)` is well below the "independent components" benchmark in both regimes.
- **H3 significant in sample, not confirmed out of sample:** `H_dep` coefficient p ≈ 9.5e-5, train R² 0.282 → 0.311, but test MSE is not improved (0.2066 → 0.2079) on the 70/30 chronological split. Diagnosed as most likely an artifact of the 252-day window overlap inflating apparent significance beyond what the HAC correction absorbs — flagged explicitly in the paper's Limitations with a proposed fix (non-overlapping frequency, walk-forward evaluation).
- Independent sanity check: the stress-date clustering lines up with known crises without having been tuned to do so — 1997–98 (Asian crisis/LTCM), 2000–02 (dot-com), 2008–11 (GFC + Euro debt crisis), 2015–16, 2018, 2020 (COVID), 2022 (rate hikes).

The paper now reports the FF49 result as the primary empirical section and keeps the earlier EuStockMarkets pilot as a secondary, independent robustness check (both datasets agree on the sign of H1–H2). Abstract, Limitations and Conclusion were updated to match. README updated with the real numbers and corrected data-availability date (1969, not 1926, for the daily series).

## First update

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
