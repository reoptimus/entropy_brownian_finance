"""Build the tables, figures and number macros of the JFEC version.

Reads only CSV files in ``paper/tables`` (written by the analysis scripts) and
writes:

- ``paper/jfec/numbers.tex``         LaTeX macros for every number quoted in
                                    the text that comes from a new analysis;
- ``paper/jfec/tables/tables.tex``  all tables, one per page (endfloat);
- ``paper/jfec/figures/*.png``      the two main-text figures (300 dpi).

Usage::

    python -m scripts.jfec_build
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

T = Path('paper/tables')
OUT = Path('paper/jfec')
(OUT / 'tables').mkdir(parents=True, exist_ok=True)
(OUT / 'figures').mkdir(parents=True, exist_ok=True)


def pct(x: float) -> str:
    return f'{100 * x:.0f}\\%'


def fp(p: float) -> str:
    """Format a resampling p-value."""
    return '<0.005' if p <= 0.005 + 1e-9 else f'{p:.3f}'


def conv(p_share: float, n: int = 200) -> float:
    """Published p-values are shares k/B; convert to (1+k)/(1+B)."""
    return (round(p_share * n) + 1) / (n + 1)


# ------------------------------------------------------------------ numbers
macros: dict[str, str] = {}

g = pd.read_csv(T / 'garch_null_200rep_summary.csv')


def gn(panel, null, stat, field):
    r = g[(g.panel == panel) & (g.null == null) & (g.statistic == stat)]
    return float(r[field].iloc[0]) if len(r) else float('nan')


for pa, tag in (('sp500', 'Sp'), ('ff49', 'Ff')):
    # coupling C = -h_dep: report gaps as positive numbers in nats
    macros[f'gn{tag}RawNull'] = f'{-gn(pa, "ccc_fhs", "CPL_h_dep_gap", "null_mean"):.2f}'
    macros[f'gn{tag}FeObs'] = f'{-gn(pa, "ccc_fhs", "CPL_h_dep_gap_fe", "observed"):.2f}'
    macros[f'gn{tag}FeNull'] = f'{-gn(pa, "ccc_fhs", "CPL_h_dep_gap_fe", "null_mean"):.2f}'
    raw = -gn(pa, 'ccc_fhs', 'CPL_h_dep_gap', 'observed')
    fe = -gn(pa, 'ccc_fhs', 'CPL_h_dep_gap_fe', 'observed')
    macros[f'gn{tag}FeShrink'] = pct(1 - fe / raw)
    macros[f'gn{tag}FeDevolP'] = fp(gn(pa, 'ccc_fhs', 'CPL_h_dep_devol_gap_fe', 'p_predicted_side'))

mc = pd.read_csv(T / 'mc_size_power.csv').set_index('dgp')
for d, tag in (('iid_t', 'SizeIid'), ('ccc_gjr', 'Garch'), ('ms_vol', 'Msvol'), ('ms_corr', 'Pow')):
    macros[f'mc{tag}Iid'] = pct(mc.loc[d, 'reject_iid_5'])
    macros[f'mc{tag}Fhs'] = pct(mc.loc[d, 'reject_fhs_5'])

size = mc.loc[['iid_t', 'ccc_gjr', 'ms_vol'], ['reject_iid_1', 'reject_fhs_1']].to_numpy().max()
macros['mcWorstOne'] = pct(size)
# joint rule: reject only if both nulls reject
mcr = pd.read_csv(T / 'mc_size_power_replications.csv')
mcr['both5'] = (mcr.p_iid <= 0.05) & (mcr.p_fhs <= 0.05)
mcr['both1'] = (mcr.p_iid <= 0.01) & (mcr.p_fhs <= 0.01)
both = mcr.groupby('dgp')[['both5', 'both1']].mean()
mc = mc.join(both)
for d, tag in (('iid_t', 'SizeIid'), ('ccc_gjr', 'Garch'), ('ms_vol', 'Msvol'), ('ms_corr', 'Pow')):
    macros[f'mcBoth{tag}'] = pct(mc.loc[d, 'both5'])
macros['mcBothWorst'] = f"{100 * mc.loc[['iid_t', 'ccc_gjr', 'ms_vol'], 'both5'].max():.1f}\\%"
macros['mcBothWorstOne'] = f"{100 * mc.loc[['iid_t', 'ccc_gjr', 'ms_vol'], 'both1'].max():.1f}\\%"

b = pd.read_csv(T / 'mc_bias.csv')


def bias(n, w, law, col):
    return float(b[(b.N == n) & (b.window == w) & (b.law == law)][col].iloc[0])


macros['biasTwentyApprox'] = f'{bias(20, 252, "gaussian", "bartlett_approx"):.2f}'
macros['biasTwentyRaw'] = f'{bias(20, 252, "gaussian", "mc_mean_raw"):.2f}'
macros['biasTwentyT'] = f'{bias(20, 252, "t5", "mc_mean_raw"):.2f}'
macros['biasTwentyLW'] = f'{bias(20, 252, "gaussian", "mc_mean_lw"):.3f}'
macros['biasFortyEightApprox'] = f'{bias(48, 252, "gaussian", "bartlett_approx"):.2f}'

for pa, tag in (('sp500', 'Sp'), ('ff49', 'Ff')):
    par = pd.read_csv(T / f'{pa}_gjr_params.csv')
    macros[f'gjr{tag}Pers'] = f'{par.persistence.mean():.3f}'
    for c, n in (('alpha', 'Alpha'), ('gamma', 'Gamma'), ('beta', 'Beta')):
        macros[f'gjr{tag}{n}'] = f'{par[c].median():.3f}'

h7 = {pa: pd.read_csv(T / f'{pa}_h7_horizons.csv') for pa in ('sp500', 'ff49')}


def h7row(pa, h, col, target='future_rv', extra='unhealed'):
    d = h7[pa]
    r = d[(d.horizon == h) & (d.target == target) & (d.extra_regressor == extra)]
    return float(r[col].iloc[0])


for pa, tag in (('sp500', 'Sp'), ('ff49', 'Ff')):
    macros[f'cw{tag}Five'] = f'{h7row(pa, 5, "cw_pvalue"):.3f}'
    macros[f'cw{tag}Ten'] = f'{h7row(pa, 10, "cw_pvalue"):.3f}'

# ------------------------------------------------------- verdicts with Holm
pl = {pa: pd.read_csv(T / f'{pa}_placebo_null.csv').set_index('statistic') for pa in ('sp500', 'ff49')}
ep = pd.read_csv(T / 'sp500_episode_null_200rep.csv').set_index('statistic')
TESTS = [  # (hypothesis, description, source, statistic, direction, fmt)
    ('H1', 'coupling gap (nats)', 'pl', 'H1_hdep_stress_minus_calm', 'less', '{:.2f}'),
    ('H1', 'index $\\JJ$ gap (nats)', 'pl', 'H1_J_stress_minus_calm', 'greater', '{:.2f}'),
    ('H1', 'shape $\\DD$ gap (nats)', 'pl', 'H1_D_stress_minus_calm', 'greater', '{:.2f}'),
    ('H2', 'compensation slope $\\beta$ (stress)', 'pl', 'H2_beta_stress', 'greater', '{:.2f}'),
    ('H3', 'upward jumps (count)', 'pl', 'n_jump_up', 'greater', '{:.0f}'),
    ('H3', 'largest episode (nats)', 'ep', 'max_episode_dJ', 'greater', '{:.1f}'),
    ('H4', 'half-life of $\\JJ$ (days)', 'pl', 'half_life_days', 'greater', '{:.0f}'),
    ('H4', 'range of $\\JJ$ (nats)', 'pl', 'range_J', 'greater', '{:.1f}'),
    ('H4', 'median half-life from peak (days)', 'ep', 'median_episode_half_life', 'greater', '{:.0f}'),
    ('H5', 'corr$(\\Delta\\JJ,\\Delta S)$, daily', 'pl', 'H5_corr_dJ_dskew', 'less', '{:.2f}'),
    ('H5', 'episodes with falling skewness', 'ep', 'share_episodes_dskew_negative', 'greater', '{:.2f}'),
    ('H6', 'channel co-movement gap', 'pl', 'H6_coupling_gap', 'greater', '{:.2f}'),
    ('H8', 'episodes dominated by dependence', 'pl', 'H8_share_dependence_dominant', 'greater', '{:.2f}'),
]
rows = []
for hyp, desc, src, stat, direc, f in TESTS:
    for pa in ('sp500', 'ff49'):
        tab = pl[pa] if src == 'pl' else (ep if pa == 'sp500' else None)
        if tab is None or stat not in tab.index:
            continue
        r = tab.loc[stat]
        obs, nm = float(r.observed), float(r.null_mean)
        if stat == 'H1_hdep_stress_minus_calm':
            obs, nm = -obs, -nm
        p = conv(float(r['p_less' if direc == 'less' else 'p_greater']))
        from scipy.stats import norm as _n
        sd = float(r.null_sd)
        z = (float(r.observed) - float(r.null_mean)) / sd if sd > 0 else 0.0
        z = -z if direc == 'less' else z
        rows.append({'hyp': hyp, 'desc': desc, 'panel': pa, 'obs': f.format(obs),
                     'null': f.format(nm), 'p': p, 'z': z, 'p_norm': float(_n.sf(z))})
V = pd.DataFrame(rows)
m = len(V)
# With B = 200 the smallest rank p-value is 1/201, so a Holm adjustment over m
# tests cannot go below m/201. The adjustment is therefore applied to the
# Gaussian-tail approximation of each p-value (from the z-score against the
# null distribution); a test passes if its exact rank p-value is <= 0.05 AND
# its Holm-adjusted tail p-value is <= 0.05.
order = V.p_norm.sort_values().index
adj, running = {}, 0.0
for i, ix in enumerate(order):
    running = max(running, min(1.0, (m - i) * V.p_norm[ix]))
    adj[ix] = running
V['p_holm'] = pd.Series(adj)
V['passes'] = (V.p <= 0.05) & (V.p_holm <= 0.05)
macros['holmNtests'] = str(m)
macros['holmNsurv'] = str(int(V.passes.sum()))
V.to_csv(T / 'jfec_verdicts_holm.csv', index=False)

# ------------------------------------------------------------- write macros
(OUT / 'numbers.tex').write_text(
    '% Generated by scripts/jfec_build.py -- do not edit by hand.\n'
    + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in macros.items()))

def to_fr(v: str) -> str:
    import re
    v = re.sub(r'(?<=\d)\.(?=\d)', '{,}', v)
    return v.replace('\\%', '\\,\\%').replace('<', '<')


(OUT / 'francais').mkdir(exist_ok=True)
(OUT / 'francais' / 'numbers_fr.tex').write_text(
    '% Generated by scripts/jfec_build.py -- do not edit by hand.\n'
    + ''.join(f'\\newcommand{{\\{k}}}{{{to_fr(v)}}}\n' for k, v in macros.items()))

# ------------------------------------------------------------------- tables
tex: list[str] = []
add = tex.append

add(r'''\begin{table}[p]\centering\small
\caption{Hypotheses tested}\label{tab:hyp}
\begin{tabular}{@{}lp{0.8\linewidth}@{}}\toprule
& Statement \\\midrule
H1 & \emph{Regime signature.} Under stress, coupling rises, the shape deficit rises, hence $\JJ$ rises. \\
H2 & \emph{Compensation.} When volatilities rise, dependence absorbs part of the entropy increase. \\
H3 & \emph{One-sided jumps.} Large moves of $\JJ$ are rises; rising episodes are more numerous and larger. \\
H4 & \emph{Relaxation.} After an episode, $\JJ$ decays with a half-life longer than the estimator's memory. \\
H5 & \emph{Sign.} Jumps and rising episodes coincide with negative market skewness. \\
H6 & \emph{Channel co-movement.} Dependence and tails move independently in calm markets and together under stress. \\
H7 & \emph{Forecasting.} Entropy improves out-of-sample short-term risk forecasts beyond volatility. \\
H8 & \emph{Crisis composition.} In crisis episodes, dependence is the dominant channel more often than without dynamics. \\
\bottomrule\end{tabular}
\par\medskip\begin{minipage}{0.95\linewidth}\footnotesize
H3, H4, H5 and H8 concern jumps of $\JJ$; they are tested day by day and by episode (jumps less than 40 days apart).
\end{minipage}
\end{table}
''')

# bias
lines = []
for _, r in b.iterrows():
    law = 'Gaussian' if r.law == 'gaussian' else 'Student-$t_5$'
    lines.append(f"{int(r.N)} & {int(r.window)} & {law} & {r.bartlett_approx:.3f} & "
                 f"{r.mc_mean_raw:.3f} ({r.mc_sd_raw:.3f}) & {r.mc_mean_lw:.3f} ({r.mc_sd_lw:.3f}) \\\\")
add(r'''\begin{table}[p]\centering\small
\caption{Finite-sample bias of sample coupling under independence}\label{tab:bias}
\begin{tabular}{@{}rrlrrr@{}}\toprule
$N$ & Window $T$ & Law & $b(N,T)$, eq.~\eqref{eq:bias} & Sample correlation & Ledoit--Wolf \\\midrule
''' + '\n'.join(lines) + r'''
\bottomrule\end{tabular}
\par\medskip\begin{minipage}{0.95\linewidth}\footnotesize
Mean (standard deviation) of $C(\hat R)=-\frac12\log\det\hat R$ in nats over 2,000 simulated windows (500 for $N=48$) of independent assets.
Student-$t_5$ draws are independent across assets. The Ledoit--Wolf column applies the shrinkage used in the empirical work.
\end{minipage}
\end{table}
''')

# Monte Carlo
names = {'iid_t': 'i.i.d.\\ Student-$t_5$, constant correlation $0.30$',
         'ccc_gjr': 'GJR-GARCH(1,1), $t_5$ shocks, constant correlation $0.30$',
         'ms_vol': 'Markov-switching volatility ($\\times2.5$), constant correlation',
         'ms_corr': 'Markov-switching volatility, correlation $0.30\\to0.50$ in stress'}
lines = []
for d in ('iid_t', 'ccc_gjr', 'ms_vol', 'ms_corr'):
    r = mc.loc[d]
    kind = 'power' if d == 'ms_corr' else 'size'
    lines.append(f"{names[d]} & {kind} & {-r.mean_gap:.2f} & {r.reject_iid_5:.3f} & {r.reject_fhs_5:.3f} & "
                 f"{r.both5:.3f} & {r.reject_iid_1:.3f} & {r.reject_fhs_1:.3f} & {r.both1:.3f} \\\\")
nmc = int(mc.n_mc.iloc[0])
add(r'''\begin{table}[p]\centering\small
\caption{Size and power of the coupling test: Monte Carlo}\label{tab:mc}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}llrrrrrrr@{}}\toprule
& & Mean gap & \multicolumn{3}{c}{Rejection at 5\%} & \multicolumn{3}{c}{Rejection at 1\%} \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}
Data-generating process & & (nats) & i.i.d. & CCC-FHS & both & i.i.d. & CCC-FHS & both \\\midrule
''' + '\n'.join(lines) + r'''
\bottomrule\end{tabular}}
\par\medskip\begin{minipage}{0.95\linewidth}\footnotesize
''' + f'{nmc}' + r''' simulated panels per process, $N=10$ assets, $T=4{,}000$ days, 252-day windows, stress = past-only 90\% quantile
of the 21-day volatility of the equal-weight portfolio. Each test uses 99 null replications; the CCC-FHS null re-estimates the
GJR-GARCH models on each simulated panel. ``Both'': rejection by both nulls, the rule used in the paper. Mean gap: stress-minus-calm coupling, positive when assets couple under stress.
\end{minipage}
\end{table}
''')

# central coupling table
cs = pd.read_csv(T / 'coupling_null_200rep_summary.csv')


def iid_pub(pa, stat, field):
    r = cs[(cs.panel == pa) & (cs.null == 'iid') & (cs.statistic == stat)]
    return float(r[field].iloc[0])


MEAS = [('Coupling, raw', 'h_dep', -1), ('Coupling, devolatilised', 'h_dep_devol', -1),
        ('Coupling, rank', 'h_dep_rank', -1), ('Absorption ratio', 'absorption_ratio', 1),
        ('First-eigenvalue share', 'eig1_share', 1)]
lines = []
for lab, key, sgn in MEAS:
    cells = [lab]
    for pa in ('sp500', 'ff49'):
        st = f'CPL_{key}_gap'
        obs = sgn * gn(pa, 'ccc_fhs', st, 'observed')
        ni = sgn * iid_pub(pa, st, 'null_mean')
        pi = conv(iid_pub(pa, st, 'p_predicted_side'))
        nf = sgn * gn(pa, 'ccc_fhs', st, 'null_mean')
        pf = gn(pa, 'ccc_fhs', st, 'p_predicted_side')
        d = 3 if key in ('absorption_ratio', 'eig1_share') else 2
        cells += [f'{obs:.{d}f}', f'{ni:.{d}f}', fp(pi), f'{nf:.{d}f}', fp(pf)]
    lines.append(' & '.join(cells) + r' \\')
lines.append(r'\midrule\multicolumn{11}{@{}l}{\emph{Within-year gap (calendar-year fixed effects)}} \\')
for lab, key, sgn in MEAS[:3]:
    cells = [lab]
    for pa in ('sp500', 'ff49'):
        st = f'CPL_{key}_gap_fe'
        obs = -gn(pa, 'ccc_fhs', st, 'observed')
        ni = -gn(pa, 'iid', st, 'null_mean')
        pi = gn(pa, 'iid', st, 'p_predicted_side')
        nf = -gn(pa, 'ccc_fhs', st, 'null_mean')
        pf = gn(pa, 'ccc_fhs', st, 'p_predicted_side')
        cells += [f'{obs:.2f}', f'{ni:.2f}' if np.isfinite(ni) else '--', fp(pi) if np.isfinite(pi) else '--',
                  f'{nf:.2f}', fp(pf)]
    lines.append(' & '.join(cells) + r' \\')
add(r'''\begin{table}[p]\centering\small
\caption{Coupling under stress against two placebo nulls}\label{tab:coupling}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lrrrrrrrrrr@{}}\toprule
& \multicolumn{5}{c}{S\&P~500, 20 stocks} & \multicolumn{5}{c}{48 industries} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}
& & \multicolumn{2}{c}{i.i.d.\ null} & \multicolumn{2}{c}{CCC-FHS null} & & \multicolumn{2}{c}{i.i.d.\ null} & \multicolumn{2}{c}{CCC-FHS null} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
Measure & Obs. & mean & $p$ & mean & $p$ & Obs. & mean & $p$ & mean & $p$ \\\midrule
''' + '\n'.join(lines) + r'''
\bottomrule\end{tabular}}
\par\medskip\begin{minipage}{0.95\linewidth}\footnotesize
Stress-minus-calm gap of each measure (coupling in nats, signed so that a positive value means more coupling under stress;
absorption ratio and first-eigenvalue share in variance shares). Stress: past-only 90\% quantile of 21-day equal-weight volatility.
$p$: $(1+k)/(1+B)$ with $k$ the number of the $B=200$ replications at least as extreme in the predicted direction;
$<0.005$ means no replication reached the observed value. Devolatilised: returns divided by their RiskMetrics volatility.
Rank: normal scores. CCC-FHS: constant-correlation GJR-GARCH filtered historical simulation.
\end{minipage}
\end{table}
''')

# verdicts
lines = []
for _, r in V.iterrows():
    pan = 'stocks' if r.panel == 'sp500' else 'industries'
    ver = '\\textbf{passes}' if r.passes else ('null stronger' if r.p > 0.95 else 'fails')
    ph = '<0.001' if r.p_holm < 0.001 else f'{r.p_holm:.3f}'
    lines.append(f"{r.hyp} & {r.desc} & {pan} & {r.obs} & {r['null']} & {fp(r.p)} & {r.z:.1f} & {ph} & {ver} \\\\")
add(r'''\begin{table}[p]\centering\small
\caption{Verdicts against the i.i.d.\ null with multiple-testing control}\label{tab:verdicts}
\resizebox{\linewidth}{!}{%
\begin{tabular}{@{}lllrrrrrl@{}}\toprule
& Statistic & Panel & Observed & Null mean & $p$ & $z$ & Holm $p$ & Verdict \\\midrule
''' + '\n'.join(lines) + r'''
\bottomrule\end{tabular}}
\par\medskip\begin{minipage}{0.95\linewidth}\footnotesize
200 replications of the i.i.d.\ null. $p$: exact rank $p$-value $(1+k)/201$, one-sided in the direction the hypothesis predicts (minimum $0.005$). $z$: distance to the null mean in null standard deviations, in the predicted direction. Holm $p$: the
\citet{holm1979} step-down adjustment over the ''' + str(m) + r''' tests of the table, applied to the Gaussian-tail $p$-value implied by $z$, because with 200 replications no rank $p$-value can survive it. ``Passes'': rank $p\le0.05$ and Holm $p\le0.05$. ``Null stronger'': the null produces a larger effect
than the data ($p>0.95$). Episode statistics are computed on the stock panel only. The coupling gap is in nats, positive for coupling.
\end{minipage}
\end{table}
''')

# H7
lines = []
for pa, lab in (('sp500', 'S\\&P~500, 20 stocks'), ('ff49', '48 industries')):
    cells = [lab]
    for h in (5, 10, 21, 63):
        cells.append(f'{h7row(pa, h, "oos_mse_vs_vol_only_pct"):+.1f}\\%')
    lines.append(' & '.join(cells) + r' \\')
    for col, nm in (('dm_pvalue', '\\quad DM $p$ (two-sided)'), ('cw_pvalue', '\\quad CW $p$ (one-sided)')):
        cells = [nm] + [f'{h7row(pa, h, col):.3f}' for h in (5, 10, 21, 63)]
        lines.append(' & '.join(cells) + r' \\')
add(r'''\begin{table}[p]\centering\small
\caption{Out-of-sample forecasts of realised volatility}\label{tab:h7}
\begin{tabular}{@{}lrrrr@{}}\toprule
Horizon (trading days) & 5 & 10 & 21 & 63 \\\midrule
''' + '\n'.join(lines) + r'''
\bottomrule\end{tabular}
\par\medskip\begin{minipage}{0.95\linewidth}\footnotesize
Change in the mean squared error of the realised-volatility forecast when the unhealed-shock indicator
$\JJ_t/\max_{252}\JJ$ is added to log volatility (negative = better). Expanding-window OLS refitted yearly; evaluation dates
spaced by the horizon. DM: \citet{dm1995}; CW: \citet{clarkwest2007}, the test for nested models.
\end{minipage}
\end{table}
''')

# classical scenarios (from paper/tables/sp500_classical_comparison.csv and the published audit)
add(r'''\begin{table}[p]\centering\small
\caption{Cost in nats of classical stress scenarios}\label{tab:classical}
\begin{tabular}{@{}lrrr@{}}\toprule
Classical scenario & moderate & standard & severe \\\midrule
volatility multiplier & $1.5$ & $2.0$ & $3.0$ \\
all correlations forced to & $0.70$ & $0.90$ & $0.95$ \\
directional shock & $-2\sigma$ & $-3\sigma$ & $-4\sigma$ \\\midrule
one-day ES$_{99}$ & $9.1\%$ & $13.7\%$ & $20.3\%$ \\
full cost (20 assets) & $6.80$ & $9.97$ & $20.19$ \\
cost seen by the portfolio & $2.75$ & $7.24$ & $16.22$ \\
minimal cost for the same ES$_{99}$ (Proposition~\ref{prop:lift}) & $\mathbf{1.85}$ & $\mathbf{5.47}$ & $\mathbf{13.74}$ \\
ratio full / minimal & $3.7$ & $1.8$ & $1.5$ \\\midrule
\multicolumn{4}{@{}l}{\emph{Cost of each component alone}} \\
\quad directional shock & $2.00$ & $4.50$ & $8.00$ \\
\quad volatility dilation & $4.39$ & $16.14$ & $58.03$ \\
\quad forced correlations & $2.43$ & $8.10$ & $13.50$ \\
\bottomrule\end{tabular}
\par\medskip\begin{minipage}{0.95\linewidth}\footnotesize
Equal-weight portfolio of the twenty stocks; current state: annual volatility 20\%, one-day ES$_{99}$ 3.4\%. The full cost is the
Gaussian divergence \eqref{eq:klgauss} on the 20 assets; the cost seen by the portfolio is the divergence of the portfolio's law.
The minimal cost is the portfolio divergence $g(u,v)$ of the entropy-ball scenario with the same ES$_{99}$, attained in full
dimension by the rank-one lift \eqref{eq:lift}. The largest 21-day information flow observed in 33 years is about 2 nats.
\end{minipage}
\end{table}
''')

add(r'''\begin{table}[p]\centering\small
\caption{Cost in nats of the EBA 2025 adverse scenario and of observed years}\label{tab:eba}
\begin{tabular}{@{}lrr@{}}\toprule
& Equities and rates & With FX fixed \\\midrule
\textbf{EBA 2025 adverse scenario, year 1} & $\mathbf{10.5}$ & $\mathbf{11.1}$ \\
\quad equities only / rates only & $6.8$ / $3.1$ & \\
\quad sum of variable-by-variable costs & $31.9$ & \\
\quad with a 2008--2012 crisis covariance & $8.4$ & $9.0$ \\\midrule
Observed year 2011 / 2008 / 2022 & $3.1$ / $11.2$ / $11.9$ & $3.1$ / $18.2$ / $13.2$ \\
Worst rolling year (March 2008--March 2009) & $14.4$ & $19.5$ \\
\bottomrule\end{tabular}
\par\medskip\begin{minipage}{0.95\linewidth}\footnotesize
Five equity indices (DAX, CAC~40, FTSE~MIB, IBEX~35, AEX), two ten-year euro-area yields (AAA and all ratings) and four euro
exchange rates; annual covariance estimated on daily changes from September 2004 to December 2024. Cost
$\frac12\Delta^\top\Sigma^{-1}\Delta$ of the year-1 deviations from baseline (equities $-50\%$, rates $+110$ and $+168$~bp,
FX unchanged). Computed with \texttt{scripts/eba\_nats.py} on the euro-area panel; carried over from the
central-bank version of this work.
\end{minipage}
\end{table}
''')

(OUT / 'tables' / 'tables.tex').write_text('\n'.join(tex))

FR = [
    ('Hypotheses tested', 'Hypothèses testées'), ('Statement', 'Énoncé'),
    ('\\emph{Regime signature.} Under stress, coupling rises, the shape deficit rises, hence $\\JJ$ rises.',
     '\\emph{Signature de régime.} Sous stress, le couplage monte, le déficit de forme monte, donc $\\JJ$ monte.'),
    ('\\emph{Compensation.} When volatilities rise, dependence absorbs part of the entropy increase.',
     '\\emph{Compensation.} Quand les volatilités montent, la dépendance absorbe une partie de la hausse d\'entropie.'),
    ('\\emph{One-sided jumps.} Large moves of $\\JJ$ are rises; rising episodes are more numerous and larger.',
     '\\emph{Sauts unilatéraux.} Les grands mouvements de $\\JJ$ sont des hausses ; les épisodes de hausse sont plus nombreux et plus grands.'),
    ('\\emph{Relaxation.} After an episode, $\\JJ$ decays with a half-life longer than the estimator\'s memory.',
     '\\emph{Relaxation.} Après un épisode, $\\JJ$ décroît avec une demi-vie plus longue que la mémoire de l\'estimateur.'),
    ('\\emph{Sign.} Jumps and rising episodes coincide with negative market skewness.',
     '\\emph{Signe.} Les sauts et les épisodes de hausse coïncident avec une asymétrie négative du marché.'),
    ('\\emph{Channel co-movement.} Dependence and tails move independently in calm markets and together under stress.',
     '\\emph{Co-mouvement des canaux.} Dépendance et queues bougent indépendamment en marché calme et ensemble sous stress.'),
    ('\\emph{Forecasting.} Entropy improves out-of-sample short-term risk forecasts beyond volatility.',
     '\\emph{Prévision.} L\'entropie améliore hors échantillon la prévision du risque à court terme au-delà de la volatilité.'),
    ('\\emph{Crisis composition.} In crisis episodes, dependence is the dominant channel more often than without dynamics.',
     '\\emph{Composition des crises.} Dans les épisodes de crise, la dépendance est plus souvent le canal dominant que sans dynamique.'),
    ('H3, H4, H5 and H8 concern jumps of $\\JJ$; they are tested day by day and by episode (jumps less than 40 days apart).',
     'H3, H4, H5 et H8 portent sur les sauts de $\\JJ$ ; elles sont testées jour par jour et par épisode (sauts distants de moins de 40 jours).'),
    ('Finite-sample bias of sample coupling under independence', 'Biais à distance finie du couplage empirique sous indépendance'),
    ('Window $T$', 'Fenêtre $T$'), ('Law', 'Loi'), ('Gaussian', 'Gaussienne'),
    ('Sample correlation', 'Corrélation empirique'),
    ('Mean (standard deviation) of', 'Moyenne (écart-type) de'),
    ('in nats over 2,000 simulated windows (500 for $N=48$) of independent assets.',
     'en nats sur 2\\,000 fenêtres simulées (500 pour $N=48$) d\'actifs indépendants.'),
    ('Student-$t_5$ draws are independent across assets. The Ledoit--Wolf column applies the shrinkage used in the empirical work.',
     'Les tirages Student-$t_5$ sont indépendants entre actifs. La colonne Ledoit--Wolf applique le rétrécissement utilisé dans la partie empirique.'),
    ('Size and power of the coupling test: Monte Carlo', 'Taille et puissance du test de couplage : Monte-Carlo'),
    ('i.i.d.\\ Student-$t_5$, constant correlation $0.30$', 'Student-$t_5$ i.i.d., corrélation constante $0.30$'),
    ('GJR-GARCH(1,1), $t_5$ shocks, constant correlation $0.30$', 'GJR-GARCH(1,1), chocs $t_5$, corrélation constante $0.30$'),
    ('Markov-switching volatility ($\\times2.5$), constant correlation', 'volatilité à régimes markoviens ($\\times2.5$), corrélation constante'),
    ('Markov-switching volatility, correlation $0.30\\to0.50$ in stress', 'volatilité à régimes, corrélation $0.30\\to0.50$ sous stress'),
    (' & size & ', ' & taille & '), (' & power & ', ' & puissance & '),
    ('Mean gap', 'Écart moyen'), ('Rejection at', 'Rejet à'), ('Data-generating process', 'Processus générateur'),
    ('CCC-FHS & both', 'CCC-FHS & les deux'),
    ('simulated panels per process, $N=10$ assets, $T=4{,}000$ days, 252-day windows, stress = past-only 90\\% quantile',
     'panels simulés par processus, $N=10$ actifs, $T=4\\,000$ jours, fenêtres de 252 jours, stress = quantile à 90\\,\\% passé seulement'),
    ('of the 21-day volatility of the equal-weight portfolio. Each test uses 99 null replications; the CCC-FHS null re-estimates the',
     'de la volatilité à 21 jours du portefeuille équipondéré. Chaque test utilise 99 répétitions du témoin ; le témoin CCC-FHS réestime les'),
    ("GJR-GARCH models on each simulated panel. ``Both'': rejection by both nulls, the rule used in the paper. Mean gap: stress-minus-calm coupling, positive when assets couple under stress.",
     "modèles GJR-GARCH sur chaque panel simulé. « Les deux » : rejet par les deux témoins, règle utilisée dans l'article. Écart moyen : couplage stress moins calme, positif quand les actifs se couplent sous stress."),
    ('Coupling under stress against two placebo nulls', 'Couplage sous stress face à deux témoins placebo'),
    ('S\\&P~500, 20 stocks', 'S\\&P~500, 20 actions'), ('48 industries', '48 secteurs'),
    ('i.i.d.\\ null', 'témoin i.i.d.'), ('CCC-FHS null', 'témoin CCC-FHS'),
    ('Measure & Obs.', 'Mesure & Obs.'), ('& mean & $p$', '& moy. & $p$'),
    ('Coupling, raw', 'Couplage, brut'), ('Coupling, devolatilised', 'Couplage, dévolatilisé'),
    ('Coupling, rank', 'Couplage, rangs'), ('Absorption ratio', "Ratio d'absorption"),
    ('First-eigenvalue share', 'Part de la 1re valeur propre'),
    ('\\emph{Within-year gap (calendar-year fixed effects)}', '\\emph{Écart au sein de l\'année (effets fixes par année civile)}'),
    ('Stress-minus-calm gap of each measure (coupling in nats, signed so that a positive value means more coupling under stress;',
     'Écart stress moins calme de chaque mesure (couplage en nats, signé pour qu\'une valeur positive signifie plus de couplage sous stress ;'),
    ('absorption ratio and first-eigenvalue share in variance shares). Stress: past-only 90\\% quantile of 21-day equal-weight volatility.',
     "ratio d'absorption et part de la première valeur propre en parts de variance). Stress : quantile à 90\\,\\% passé seulement de la volatilité équipondérée à 21 jours."),
    ('$p$: $(1+k)/(1+B)$ with $k$ the number of the $B=200$ replications at least as extreme in the predicted direction;',
     '$p$ : $(1+k)/(1+B)$, $k$ étant le nombre des $B=200$ répétitions au moins aussi extrêmes dans le sens prédit ;'),
    ('$<0.005$ means no replication reached the observed value. Devolatilised: returns divided by their RiskMetrics volatility.',
     "$<0.005$ signifie qu'aucune répétition n'a atteint la valeur observée. Dévolatilisé : rendements divisés par leur volatilité RiskMetrics."),
    ('Rank: normal scores. CCC-FHS: constant-correlation GJR-GARCH filtered historical simulation.',
     'Rangs : scores normaux. CCC-FHS : simulation historique filtrée GJR-GARCH à corrélation constante.'),
    ('Verdicts against the i.i.d.\\ null with multiple-testing control', 'Verdicts face au témoin i.i.d.\\ avec contrôle des tests multiples'),
    ('& Statistic & Panel & Observed & Null mean & $p$ & $z$ & Holm $p$ & Verdict', '& Statistique & Panel & Observé & Moy. témoin & $p$ & $z$ & $p$ de Holm & Verdict'),
    ('coupling gap (nats)', 'écart de couplage (nats)'), ('index $\\JJ$ gap (nats)', 'écart de l\'indice $\\JJ$ (nats)'),
    ('shape $\\DD$ gap (nats)', 'écart de forme $\\DD$ (nats)'),
    ('compensation slope $\\beta$ (stress)', 'pente de compensation $\\beta$ (stress)'),
    ('upward jumps (count)', 'sauts vers le haut (nombre)'), ('largest episode (nats)', 'plus grand épisode (nats)'),
    ('half-life of $\\JJ$ (days)', 'demi-vie de $\\JJ$ (jours)'), ('range of $\\JJ$ (nats)', 'étendue de $\\JJ$ (nats)'),
    ('median half-life from peak (days)', 'demi-vie médiane depuis le pic (jours)'),
    ('corr$(\\Delta\\JJ,\\Delta S)$, daily', 'corr$(\\Delta\\JJ,\\Delta S)$, quotidienne'),
    ('episodes with falling skewness', 'épisodes à asymétrie décroissante'),
    ('channel co-movement gap', 'écart de co-mouvement des canaux'),
    ('episodes dominated by dependence', 'épisodes dominés par la dépendance'),
    (' & stocks & ', ' & actions & '), (' & industries & ', ' & secteurs & '),
    ('\\textbf{passes}', '\\textbf{passe}'), ('null stronger', 'témoin plus fort'), (' & fails \\\\', ' & échoue \\\\'),
    ('200 replications of the i.i.d.\\ null. $p$: exact rank $p$-value $(1+k)/201$, one-sided in the direction the hypothesis predicts (minimum $0.005$). $z$: distance to the null mean in null standard deviations, in the predicted direction. Holm $p$: the',
     "200 répétitions du témoin i.i.d. $p$ : $p$-valeur exacte de rang $(1+k)/201$, unilatérale dans le sens prédit (minimum $0.005$). $z$ : distance à la moyenne du témoin en écarts-types du témoin, dans le sens prédit. $p$ de Holm : l'ajustement séquentiel de"),
    ('step-down adjustment over the', 'sur les'),
    ("tests of the table, applied to the Gaussian-tail $p$-value implied by $z$, because with 200 replications no rank $p$-value can survive it. ``Passes'': rank $p\\le0.05$ and Holm $p\\le0.05$. ``Null stronger'': the null produces a larger effect",
     "tests du tableau, appliqué à la $p$-valeur de queue gaussienne implicite dans $z$, car avec 200 répétitions aucune $p$-valeur de rang ne peut y survivre. « Passe » : $p$ de rang $\\le0.05$ et $p$ de Holm $\\le0.05$. « Témoin plus fort » : le témoin produit un effet plus grand"),
    ('than the data ($p>0.95$). Episode statistics are computed on the stock panel only. The coupling gap is in nats, positive for coupling.',
     'que les données ($p>0.95$). Les statistiques par épisode ne sont calculées que sur le panel d\'actions. L\'écart de couplage est en nats, positif pour un couplage.'),
    ('Out-of-sample forecasts of realised volatility', 'Prévisions hors échantillon de la volatilité réalisée'),
    ('Horizon (trading days)', 'Horizon (jours de bourse)'),
    ('\\quad DM $p$ (two-sided)', '\\quad $p$ DM (bilatérale)'), ('\\quad CW $p$ (one-sided)', '\\quad $p$ CW (unilatérale)'),
    ('Change in the mean squared error of the realised-volatility forecast when the unhealed-shock indicator',
     "Variation de l'erreur quadratique moyenne de la prévision de volatilité réalisée quand l'indicateur de choc non cicatrisé"),
    ('is added to log volatility (negative = better). Expanding-window OLS refitted yearly; evaluation dates',
     "est ajouté au log de la volatilité (négatif = meilleur). MCO sur fenêtre croissante réestimés chaque année ; dates d'évaluation"),
    ('spaced by the horizon. DM: \\citet{dm1995}; CW: \\citet{clarkwest2007}, the test for nested models.',
     "espacées de l'horizon. DM : \\citet{dm1995} ; CW : \\citet{clarkwest2007}, le test pour modèles emboîtés."),
    ('Cost in nats of classical stress scenarios', 'Coût en nats de scénarios de stress classiques'),
    ('Classical scenario & moderate & standard & severe', 'Scénario classique & modéré & standard & sévère'),
    ('volatility multiplier', 'multiplicateur de volatilité'), ('all correlations forced to', 'toutes les corrélations forcées à'),
    ('directional shock &', 'choc directionnel &'), ('one-day ES$_{99}$', 'ES$_{99}$ à un jour'),
    ('full cost (20 assets)', 'coût total (20 actifs)'), ('cost seen by the portfolio', 'coût vu par le portefeuille'),
    ('minimal cost for the same ES$_{99}$', 'coût minimal pour la même ES$_{99}$'),
    ('ratio full / minimal', 'rapport total / minimal'), ('\\emph{Cost of each component alone}', '\\emph{Coût de chaque composante seule}'),
    ('\\quad directional shock', '\\quad choc directionnel'), ('\\quad volatility dilation', '\\quad dilatation des volatilités'),
    ('\\quad forced correlations', '\\quad corrélations forcées'),
    ('Equal-weight portfolio of the twenty stocks; current state: annual volatility 20\\%, one-day ES$_{99}$ 3.4\\%. The full cost is the',
     'Portefeuille équipondéré des vingt actions ; état actuel : volatilité annuelle 20\\,\\%, ES$_{99}$ à un jour 3.4\\,\\%. Le coût total est la'),
    ("Gaussian divergence \\eqref{eq:klgauss} on the 20 assets; the cost seen by the portfolio is the divergence of the portfolio's law.",
     'divergence gaussienne \\eqref{eq:klgauss} sur les 20 actifs ; le coût vu par le portefeuille est la divergence de la loi du portefeuille.'),
    ('The minimal cost is the portfolio divergence $g(u,v)$ of the entropy-ball scenario with the same ES$_{99}$, attained in full',
     "Le coût minimal est la divergence $g(u,v)$ du scénario de boule d'entropie de même ES$_{99}$, atteinte en pleine"),
    ('dimension by the rank-one lift \\eqref{eq:lift}. The largest 21-day information flow observed in 33 years is about 2 nats.',
     "dimension par le relèvement de rang un \\eqref{eq:lift}. Le plus grand flux d'information à 21 jours observé en 33 ans est d'environ 2 nats."),
    ('Cost in nats of the EBA 2025 adverse scenario and of observed years', "Coût en nats du scénario adverse 2025 de l'ABE et des années observées"),
    ('& Equities and rates & With FX fixed', '& Actions et taux & Changes fixés'),
    ('\\textbf{EBA 2025 adverse scenario, year 1}', '\\textbf{Scénario adverse ABE 2025, année 1}'),
    ('\\quad equities only / rates only', '\\quad actions seules / taux seuls'),
    ('\\quad sum of variable-by-variable costs', '\\quad somme des coûts variable par variable'),
    ('\\quad with a 2008--2012 crisis covariance', '\\quad avec une covariance de crise 2008--2012'),
    ('Observed year 2011 / 2008 / 2022', 'Année observée 2011 / 2008 / 2022'),
    ('Worst rolling year (March 2008--March 2009)', 'Pire année glissante (mars 2008--mars 2009)'),
    ('Five equity indices (DAX, CAC~40, FTSE~MIB, IBEX~35, AEX), two ten-year euro-area yields (AAA and all ratings) and four euro',
     'Cinq indices actions (DAX, CAC~40, FTSE~MIB, IBEX~35, AEX), deux taux à dix ans de la zone euro (AAA et toutes notations) et quatre taux de change'),
    ('exchange rates; annual covariance estimated on daily changes from September 2004 to December 2024. Cost',
     "de l'euro ; covariance annuelle estimée sur les variations quotidiennes de septembre 2004 à décembre 2024. Coût"),
    ('of the year-1 deviations from baseline (equities $-50\\%$, rates $+110$ and $+168$~bp,',
     'des écarts de la première année au scénario central (actions $-50\\,\\%$, taux $+110$ et $+168$~pb,'),
    ('FX unchanged). Computed with \\texttt{scripts/eba\\_nats.py} on the euro-area panel; carried over from the',
     'changes inchangés). Calculé avec \\texttt{scripts/eba\\_nats.py} sur le panel de la zone euro ; repris de la'),
    ('central-bank version of this work.', 'version banque centrale de ce travail.'),
]
import re as _re
fr = '\n'.join(tex)
for a_, b_ in FR:
    fr = fr.replace(a_, b_)
fr = _re.sub(r'(?<=\d)\.(?=\d)', '{,}', fr)
fr = _re.sub(r'(\d)\{,\}(\d+\\linewidth)', r'\1.\2', fr)
(OUT / 'tables' / 'tables_fr.tex').write_text(fr)


# ------------------------------------------------------------------ figures
LAB = {
    'en': dict(iid='i.i.d. null', fhs='CCC-FHS (GARCH) null', obs='observed',
               t1='S&P 500, 20 stocks', t2='48 industry portfolios',
               x='stress-minus-calm coupling gap (nats)', y='replications',
               st='stress dates (past-only rule)', c='coupling $C$', yl='coupling (nats)',
               nm='effective number of modes', suf=''),
    'fr': dict(iid='témoin i.i.d.', fhs='témoin CCC-FHS (GARCH)', obs='observé',
               t1='S&P 500, 20 actions', t2='48 portefeuilles sectoriels',
               x='écart de couplage stress moins calme (nats)', y='répétitions',
               st='dates de stress (règle passée seulement)', c='couplage $C$',
               yl='couplage (nats)', nm='nombre effectif de modes', suf='_fr'),
}


def figures() -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gl = pd.read_csv(T / 'garch_null_200rep_replications_long.csv')
    il = pd.read_csv(T / 'coupling_null_200rep_replications_long.csv')
    from scripts.coupling_null import _load
    from src.coupling import rolling_coupling
    from src.regimes import past_only_stress
    logret, window, sq = _load('sp500')
    cpl = rolling_coupling(logret, window)
    for lang, L in LAB.items():
        _figs(gl, il, logret, window, sq, cpl, past_only_stress, L)


def _figs(gl, il, logret, window, sq, cpl, past_only_stress, L) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (pa, title) in zip(axes, (('sp500', L['t1']), ('ff49', L['t2']))):
        iid = -il[(il.panel == pa) & (il.null == 'iid') & (il.statistic == 'CPL_h_dep_gap')].value
        fhs = -gl[(gl.panel == pa) & (gl.null == 'ccc_fhs') & (gl.statistic == 'CPL_h_dep_gap')].value
        obs = -float(gl[(gl.panel == pa) & (gl.null == 'observed') & (gl.statistic == 'CPL_h_dep_gap')].value.iloc[0])
        bins = np.linspace(min(iid.min(), fhs.min()) - 0.05, max(obs, fhs.max()) + 0.1, 45)
        ax.hist(iid, bins=bins, color='#8c8c8c', alpha=0.75, label=L['iid'])
        ax.hist(fhs, bins=bins, color='#3b6ea5', alpha=0.75, label=L['fhs'])
        ax.axvline(obs, color='#b2182b', lw=2, label=L['obs'])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(L['x'])
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel(L['y'])
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / f"fig_null_distributions{L['suf']}.png", dpi=300)
    plt.close(fig)

    C = -cpl['h_dep']
    sig = logret.mean(axis=1).rolling(21).std().reindex(cpl.index)
    stress, _ = past_only_stress(sig, sq)
    n = logret.shape[1]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ymax = float(C.max()) * 1.05
    ax.fill_between(C.index, 0, ymax, where=stress.to_numpy(), color='#f4a582', alpha=0.5, lw=0,
                    label=L['st'])
    ax.plot(C.index, C, color='#2166ac', lw=1, label=L['c'])
    ax.set_ylim(0, ymax)
    ax.set_ylabel(L['yl'])
    ax.spines[['top']].set_visible(False)
    ax2 = ax.twinx()
    lo, hi = ax.get_ylim()
    ax2.set_ylim(lo, hi)
    ticks = [n * np.exp(-2 * c / n) for c in ax.get_yticks() if lo <= c <= hi]
    ax2.set_yticks([c for c in ax.get_yticks() if lo <= c <= hi])
    ax2.set_yticklabels([f'{t:.1f}' for t in ticks])
    ax2.set_ylabel(L['nm'])
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / f"fig_coupling_series{L['suf']}.png", dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    figures()
    print(open(OUT / 'numbers.tex').read())
    print(V.to_string())
