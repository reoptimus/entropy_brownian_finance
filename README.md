# Entropy jumps and information relaxation in financial markets

Research code for the paper in `paper/main.tex`: maximum entropy in volume time,
a three-channel decomposition of return entropy, a jump/diffusion account of how
information moves it, and a stress-testing construction calibrated in nats.

## The argument in one page

**Maximum entropy gives a Gaussian.** Fix a mean and a covariance and the
least-committal distribution is Gaussian (`src/entropy.py`). Applied in *volume
time* — with the price defined as accumulated order flow and traded volume
supplying the variance budget — this delivers Gaussian returns conditional on
volume, i.e. the mixture-of-distributions hypothesis derived rather than
assumed. Temporal consistency turns that into a subordinated Brownian motion and
Itô turns it into a geometric Brownian price.

**So non-Gaussian prices measure information.** The entropy deficit
`D = KL(p‖q) ≥ 0` between the true conditional distribution and the Gaussian
with the same first two moments is exactly the departure from maximal ignorance
(`src/shape.py`).

**Three channels.** Entropy splits into scale (`Σ log σᵢ`), dependence
(`½ log det R ≤ 0`) and shape (`−D`). The last two are scale-free; their sum is
the **structural index**

```
J = D − ½ log det R = KL( p ‖ ∏ᵢ qᵢ ) ≥ 0
```

— the divergence from a product of Gaussian marginals with the same scales.
Zero iff returns are independent and Gaussian; invariant to the level of
volatility.

**Dynamics: one theorem, two postulates.** Conditioning cannot raise expected
entropy, so an information event lowers the entropy ceiling by exactly the
mutual information of the news — *downward jumps are a theorem*. What is
postulated is that entropy relaxes back diffusively (P1) and that the constraint
is one-sided in the loss direction (P2). Together: a non-Gaussian
Ornstein–Uhlenbeck process for `J`, up in jumps and down by diffusion
(`src/jumps.py`).

**Stress testing in nats.** A scenario of severity `η` is the worst conditional
distribution within a KL ball of radius `η` around today's, with the radius
calibrated on the market's own realised information flow `KL(pₜ‖pₜ₋ₕ)`
(`src/stress.py`). One nat buys √2 sigma. The same metric prices scenarios you
already run.

## What the data say

Primary panel: twenty S&P 500 constituents, daily, 1990–2022 (individual firms,
not industry portfolios). Replication: `EuStockMarkets`, four indices,
1991–1998.

Every result below is reported against **two** calibrated nulls
(`scripts/placebo_null.py`), both resampling the observed returns and running
the identical pipeline twenty times:

- **i.i.d. null** — rows resampled independently. Keeps the fat marginal tails
  and the cross-sectional dependence, destroys *all* temporal structure.
- **block-21 null** — circular blocks of 21 days. Additionally keeps short-run
  volatility clustering; destroys only the longer-horizon regime dynamics.

A statistic that clears both is evidence. One that clears one and fails the
other is not identified.

| | in the data | vs i.i.d. null | vs block null | verdict |
|---|---|---|---|---|
| **H1** `h_dep` regime gap | −1.72 | −0.29 | −0.64 | ✅ **clears both** |
| **H1** `J` regime gap | +2.89 | +2.32 | +2.21 | i.i.d. only |
| **H1** `D` regime gap | +0.48 | +1.35 | +0.74 | ❌ null is larger |
| **H2** corr(ΔH_vol, ΔH_dep) | +0.10 | +0.14 | +0.16 | ❌ |
| **H3** up-jumps | 158 | 147.7 | 129.5 | block only |
| **H3** skew(ΔJ) | +12.3 | +14.6 | +13.8 | ❌ null is larger |
| **H4** half-life | 120 d | 92 d | 162 d | ⚠️ **not identified** (data between the nulls) |
| **H4** range(`J`) | 18.95 | 10.49 | 14.23 | ✅ **clears both** |
| **H5** corr(ΔJ, Δskew) | −0.24 | −0.07 | −0.01 | ⚠️ edge, *p* ≈ 0.05–0.10 |
| **H6** coupling gap | 0.62 | 0.75 | 0.67 | ❌ nulls couple *more* |
| **H7** out-of-sample forecast | in-sample *p* < 1e-4 | — | — | ❌ no gain over volatility |

**Exactly two statistics clear both nulls.** The dependence channel's regime
signal: what separates a crisis from a run of large returns is that the
cross-section couples — the effective number of independent risk modes falls
from 13.2 to 10.5 out of 20 — not that the marginals fatten. And the range of
`J`: the index travels further in real markets than in any resampling of the
same returns.

The reason the nulls matter: `J` is a non-negative convex functional of estimated
second, third and fourth moments. A single fat-tailed observation entering the
estimator pushes it up sharply and the estimator's memory lets it decay
smoothly. Sawtooth dynamics, one-sided jumps, a skewness signature and
regime-dependent channel coupling are what that measurement device produces on
leptokurtic data, whether or not information arrives in parcels. **Anyone
building a crisis indicator from a convex functional of estimated moments should
run this null before believing their own figure.** A separate controlled
simulation (`scripts/estimator_memory.py`) puts the purely mechanical floor at a
35-day decay half-life and 37 up-jumps / 0 down-jumps on i.i.d. Gaussian data
with no events at all.

The stress-testing construction in `src/stress.py` does not depend on any of
H1–H7: it is a measurement convention plus a convex optimisation.

## Layout

```
src/entropy.py      Gaussian entropy, covariance estimators (Ledoit–Wolf, EWMA), spectral diagnostics
src/shape.py        entropy deficit: Hyvärinen negentropy, asymmetry/tail channels, structural index
src/jumps.py        bipower local scale, jump detection, OU relaxation, event study, indicators
src/stress.py       KL-ball scenarios, severity ladder, scenario pricing, reverse stress
src/empirical.py    rolling estimation of every channel plus surprisal and information flow
src/hypotheses.py   H1–H7 as functions returning tidy tables
src/run_empirical.py  end-to-end run: tables, hypothesis tests, stress ladder, figures
src/data.py         readers for FF49, EuStockMarkets and the S&P 500 panel

scripts/negentropy_benchmark.py  accuracy of the negentropy estimators against a Vasicek reference
scripts/estimator_memory.py      how much relaxation a memoryless world already produces
scripts/placebo_null.py          calibrated nulls (i.i.d. and block bootstrap) for every hypothesis
scripts/build_paper_figures.py   sync run outputs into paper/figures
```

## Running it

```bash
pip install -r requirements.txt
pip install skfolio            # only needed for the S&P 500 panel

python -m src.download_data  --config config/sp500.yaml
python -m src.run_empirical  --config config/sp500.yaml --label "S&P 500 (20 stocks)"

python -m src.download_data  --config config/pilot_eu_stock_markets.yaml
python -m src.run_empirical  --config config/pilot_eu_stock_markets.yaml --label "EuStockMarkets"

python -m scripts.negentropy_benchmark
python -m scripts.estimator_memory
python -m scripts.placebo_null --config config/sp500.yaml --n-rep 20             # i.i.d. null, ~1 h
python -m scripts.placebo_null --config config/sp500.yaml --n-rep 20 --block 21 # block null, ~1 h

python -m scripts.build_paper_figures
cd paper && latexmk -pdf main.tex        # English
cd paper && latexmk -pdf main_fr.tex     # French (needs texlive-lang-french)
```

The paper exists in two languages: `paper/main.tex` (English, authoritative) and
`paper/main_fr.tex` (French translation). Both report the same numbers from the
same run and share `paper/figures/`. Edits to results must be made in both.

`config/default.yaml` targets the Kenneth French 49-industry daily file. The
reader and config are kept intact so the whole analysis can be reproduced on it,
but the file is fetched from Dartmouth and was not reachable from the
environment these results were produced in; download it manually to
`data/raw/` and run the same commands.

Tests: `python -m pytest tests/ -q` (54 tests, no network access needed).

## Reading the outputs

Each run writes to `outputs/<dataset>/`:

- `tables/rolling_measures.csv` — every channel, every date
- `tables/h1_…csv` … `h7_…csv` — one file per hypothesis
- `tables/severity_ladder.csv`, `stress_scenarios.csv` — the stress ladder
- `tables/classical_comparison.csv` — textbook bundles priced in nats
- `tables/reverse_stress.csv` — what a given loss costs
- `tables/placebo_null.csv`, `placebo_null_block21.csv` — observed statistics against each null
- `figures/*.png` — the paper's figures

## Caveats worth knowing before using any of this

- The shape channel is estimated **marginally**; non-linear tail dependence is
  measured separately (co-exceedance) rather than folded into `J`, so `J`
  understates the true divergence, most under stress.
- A KL radius grows with the dimension of the return vector. Severity ladders
  are **not comparable across universes** and must be recalibrated per portfolio.
- Rolling windows make consecutive observations share 251 of 252 return days.
  Every regime test here uses a block bootstrap; in-sample *t*-statistics on
  these series are not meaningful and are not reported as evidence.
- Jump arrivals cluster (see the event study). The Poisson arrivals in the
  stated model are an idealisation; a Hawkes process is the natural next step.
