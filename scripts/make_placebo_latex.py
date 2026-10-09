"""Write the body rows of the two placebo tables of the French paper.

Reads paper/tables/{panel}_placebo_null{,_block21}.csv and writes
paper/tables/tab_placebo_{panel}.tex, included by paper/main_fr.tex.

A statistic "clears" a null when at most 5% of replications are at least as
extreme as the observed value in the direction predicted by its hypothesis; it
runs "against" when that holds in the opposite direction.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# (group, label, statistic, predicted direction, decimals)
ROWS = [
    ('H1', r'écart $H^{dep}$', 'H1_hdep_stress_minus_calm', 'less', 2),
    ('H1', r'écart $\JJ$', 'H1_J_stress_minus_calm', 'greater', 2),
    ('H1', r'écart $\DD$', 'H1_D_stress_minus_calm', 'greater', 2),
    None,
    ('H2', r'$\mathrm{corr}(\Delta H^{vol}\!,\Delta H^{dep})$', 'H2_corr_dhvol_dhdep_all', 'less', 2),
    ('H2', r'$\beta$ (stress)', 'H2_beta_stress', 'greater', 2),
    None,
    ('H3', 'sauts positifs', 'n_jump_up', 'greater', 1),
    ('H3', 'sauts négatifs', 'n_jump_down', 'less', 1),
    ('H3', 'saut positif moyen (nats)', 'mean_size_up', 'greater', 2),
    ('H3', r'asymétrie de $\Delta\JJ$', 'skew_of_changes', 'greater', 2),
    ('H3', r'part de var.\ sauts pos.', 'share_of_variance_up', 'greater', 3),
    None,
    ('H4', 'demi-vie (jours)', 'half_life_days', 'half', 1),
    ('H4', r'$\mathrm{sd}(\JJ)$', 'sd_J', 'greater', 2),
    ('H4', r'étendue$(\JJ)$', 'range_J', 'greater', 2),
    None,
    ('H5', r'écart $\Delta S_{mkt}$', 'H5_dskew_on_jump_minus_else', 'less', 3),
    ('H5', r'$\mathrm{corr}(\Delta\JJ,\Delta S_{mkt})$', 'H5_corr_dJ_dskew', 'less', 3),
    None,
    ('H6', 'couplage, stress', 'H6_corr_dep_shape_stress', 'greater', 2),
    ('H6', 'écart de couplage', 'H6_coupling_gap', 'greater', 3),
    None,
    ('--', r'$\mathrm{corr}(\JJ,\log\mathrm{vol})$', 'corr_J_logvol', 'none', 3),
]


def fr(x: float, d: int) -> str:
    s = f'{x:.{d}f}'.replace('.', '{,}')
    return s.replace('-', '-') if x >= 0 else s


def num(x: float, d: int) -> str:
    return f'${fr(x, d)}$'


def rng(a: float, b: float, d: int) -> str:
    return f'${fr(a, d)}$--${fr(b, d)}$'


def verdict(r_iid, r_blk, direction: str) -> str:
    if direction == 'none':
        return '---'
    obs = r_iid['observed']
    if direction == 'half':
        lo, hi = sorted([r_iid['null_mean'], r_blk['null_mean']])
        return 'entre les deux témoins' if lo <= obs <= hi else 'hors de l\'intervalle des témoins'
    key, opp = ('p_greater', 'p_less') if direction == 'greater' else ('p_less', 'p_greater')
    clear = [r[key] <= 0.05 for r in (r_iid, r_blk)]
    against = [r[opp] <= 0.05 for r in (r_iid, r_blk)]
    if all(clear):
        return r'\textbf{franchit les deux}'
    if any(clear):
        i = clear.index(True)
        other = 1 - i
        name = ['i.i.d.', 'blocs'][i]
        if against[other]:
            return f'contradictoire ({name} franchit, l\'autre à contre-sens)'
        return f'{name}\\ seulement' if name == 'i.i.d.' else 'blocs seulement'
    if all(against):
        return 'à contre-sens (les deux)'
    if any(against):
        return f'à contre-sens ({["i.i.d.", "blocs"][against.index(True)]})'
    return 'échoue aux deux'


def build(panel: str) -> str:
    iid = pd.read_csv(f'paper/tables/{panel}_placebo_null.csv').set_index('statistic')
    blk = pd.read_csv(f'paper/tables/{panel}_placebo_null_block21.csv').set_index('statistic')
    out = []
    for row in ROWS:
        if row is None:
            out.append(r'\midrule')
            continue
        grp, label, st, direction, d = row
        a, b = iid.loc[st], blk.loc[st]
        out.append(' & '.join([
            grp, label, num(a['observed'], d),
            num(a['null_mean'], d), rng(a['null_q05'], a['null_q95'], d),
            num(b['null_mean'], d), rng(b['null_q05'], b['null_q95'], d),
            verdict(a, b, direction)]) + r' \\')
    return '\n'.join(out) + '\n\\bottomrule\n'


if __name__ == '__main__':
    for panel in ('sp500', 'ff49'):
        Path(f'paper/tables/tab_placebo_{panel}.tex').write_text(build(panel), encoding='utf-8')
        print(f'paper/tables/tab_placebo_{panel}.tex')
