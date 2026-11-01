"""Write the body rows of the two placebo tables of the paper (FR and EN).

Reads paper/tables/{panel}_placebo_null.csv (i.i.d. null, 200 replications)
and writes paper/tables/tab_placebo_{panel}.tex (French) and
paper/tables/tab_placebo_{panel}_en.tex (English), included by
paper/main_fr.tex and paper/main.tex.

Only the i.i.d. null is used. The 21-day block null (still computed, in
paper/tables/{panel}_placebo_null_block21.csv) preserves volatility clustering
and the within-month dynamics of crises, which are part of what the measure is
meant to capture; it is therefore not used as a criterion.

A statistic "clears" the null when at most 5% of replications are at least as
extreme as the observed value in the direction predicted by its hypothesis; it
runs "against" when that holds in the opposite direction.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# (group, FR label, EN label, statistic, predicted direction, decimals)
ROWS = [
    ('H1', r'écart $H^{dep}$', r'gap $H^{dep}$', 'H1_hdep_stress_minus_calm', 'less', 2),
    ('H1', r'écart $\JJ$', r'gap $\JJ$', 'H1_J_stress_minus_calm', 'greater', 2),
    ('H1', r'écart $\DD$', r'gap $\DD$', 'H1_D_stress_minus_calm', 'greater', 2),
    None,
    ('H2', r'$\mathrm{corr}(\Delta H^{vol}\!,\Delta H^{dep})$', r'$\mathrm{corr}(\Delta H^{vol}\!,\Delta H^{dep})$', 'H2_corr_dhvol_dhdep_all', 'less', 2),
    ('H2', r'$\beta$ (stress)', r'$\beta$ (stress)', 'H2_beta_stress', 'greater', 2),
    None,
    ('H3', 'sauts positifs', 'positive jumps', 'n_jump_up', 'greater', 1),
    ('H3', 'sauts négatifs', 'negative jumps', 'n_jump_down', 'less', 1),
    ('H3', 'saut positif moyen (nats)', 'mean positive jump (nats)', 'mean_size_up', 'greater', 2),
    ('H3', r'asymétrie de $\Delta\JJ$', r'skewness of $\Delta\JJ$', 'skew_of_changes', 'greater', 2),
    ('H3', r'part de var.\ sauts pos.', r'var.\ share of pos.\ jumps', 'share_of_variance_up', 'greater', 3),
    None,
    ('H4', 'demi-vie (jours)', 'half-life (days)', 'half_life_days', 'greater', 1),
    ('H4', r'$\mathrm{sd}(\JJ)$', r'$\mathrm{sd}(\JJ)$', 'sd_J', 'greater', 2),
    ('H4', r'étendue$(\JJ)$', r'range$(\JJ)$', 'range_J', 'greater', 2),
    None,
    ('H5', r'écart $\Delta S_{mkt}$', r'gap $\Delta S_{mkt}$', 'H5_dskew_on_jump_minus_else', 'less', 3),
    ('H5', r'$\mathrm{corr}(\Delta\JJ,\Delta S_{mkt})$', r'$\mathrm{corr}(\Delta\JJ,\Delta S_{mkt})$', 'H5_corr_dJ_dskew', 'less', 3),
    None,
    ('H6', 'couplage, stress', 'coupling, stress', 'H6_corr_dep_shape_stress', 'greater', 2),
    ('H6', 'écart de couplage', 'coupling gap', 'H6_coupling_gap', 'greater', 3),
    None,
    ('--', r'$\mathrm{corr}(\JJ,\log\mathrm{vol})$', r'$\mathrm{corr}(\JJ,\log\mathrm{vol})$', 'corr_J_logvol', 'none', 3),
]

WORDS = {
    'fr': {'clear': r'\textbf{franchit}', 'against': 'à contre-sens', 'fail': 'échoue'},
    'en': {'clear': r'\textbf{passes}', 'against': 'wrong side', 'fail': 'fails'},
}


def fmt(x: float, d: int, lang: str) -> str:
    s = f'{x:.{d}f}'
    return s.replace('.', '{,}') if lang == 'fr' else s


def num(x: float, d: int, lang: str) -> str:
    return f'${fmt(x, d, lang)}$'


def rng(a: float, b: float, d: int, lang: str) -> str:
    return f'${fmt(a, d, lang)}$--${fmt(b, d, lang)}$'


def pval(r, direction: str, lang: str) -> str:
    if direction == 'none':
        return '---'
    p = r['p_greater'] if direction == 'greater' else r['p_less']
    if p < 0.005:
        return '$<' + fmt(0.005, 3, lang) + '$'
    return num(p, 3, lang)


def verdict(r, direction: str, lang: str) -> str:
    if direction == 'none':
        return '---'
    key, opp = ('p_greater', 'p_less') if direction == 'greater' else ('p_less', 'p_greater')
    w = WORDS[lang]
    if r[key] <= 0.05:
        return w['clear']
    if r[opp] <= 0.05:
        return w['against']
    return w['fail']


def build(panel: str, lang: str) -> str:
    iid = pd.read_csv(f'paper/tables/{panel}_placebo_null.csv').set_index('statistic')
    out = []
    for row in ROWS:
        if row is None:
            out.append(r'\midrule')
            continue
        grp, lab_fr, lab_en, st, direction, d = row
        a = iid.loc[st]
        out.append(' & '.join([
            grp, lab_fr if lang == 'fr' else lab_en, num(a['observed'], d, lang),
            num(a['null_mean'], d, lang), rng(a['null_q05'], a['null_q95'], d, lang),
            pval(a, direction, lang), verdict(a, direction, lang)]) + r' \\')
    return '\n'.join(out) + '\n\\bottomrule\n'


if __name__ == '__main__':
    for panel in ('sp500', 'ff49'):
        for lang, suffix in (('fr', ''), ('en', '_en')):
            path = Path(f'paper/tables/tab_placebo_{panel}{suffix}.tex')
            path.write_text(build(panel, lang), encoding='utf-8')
            print(path)
