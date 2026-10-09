"""Calcul lourd des deux témoins calibrés (i.i.d. et blocs de 21 jours),
200 réplications, sur FF49 et sur le S&P 500, en parallèle sur vos cœurs.

Une seule commande :

    python calcul_local/lancer_calcul.py      (depuis la racine du dépôt)

À la fin, un seul fichier à renvoyer : ``calcul_local/resultats/resultats_calcul_local.zip``.

Le calcul est reprenable : si vous l'interrompez (Ctrl+C, veille, coupure),
relancez la même commande, les réplications déjà faites sont conservées.

Options utiles :
    --test          contrôle rapide de l'installation (2 réplications, S&P 500)
    --jobs N        nombre de processus (défaut : nombre de cœurs - 1)
    --n-rep N       réplications par témoin (défaut : 200)
"""
from __future__ import annotations

import os

# Un seul fil BLAS par processus : c'est le parallélisme entre réplications qui
# accélère, pas celui des produits matriciels (sinon les processus se gênent).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import hashlib
import json
import platform
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ICI = Path(__file__).resolve().parent      # calcul_local/
RACINE = ICI.parent                          # racine du dépôt
os.chdir(RACINE)
sys.path.insert(0, str(RACINE))

import numpy as np
import pandas as pd
import yaml

from src.empirical import DATA_READERS, rolling_measures
from scripts.placebo_null import statistics_of
from scripts.episode_null import typology_of

PANELS = {'ff49': 'config/default.yaml', 'sp500': 'config/sp500.yaml'}
NULLS = {'iid': 1, 'block21': 21}
SEED = 11
# Version du code de calcul : un point de reprise d'une autre version est ignoré,
# pour ne jamais mélanger d'anciens résultats avec ceux du code courant.
CODE_VERSION = 'phase0-v1'
# Le panel S&P 500 est livré dans calcul_local/data : aucun téléchargement.
SP500_CSV = ICI / 'data' / 'sp500_prices.csv'
FF49_ZIP = RACINE / 'data' / '49_Industry_Portfolios_daily_CSV.zip'


def _code_commit() -> str:
    try:
        import subprocess
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=RACINE, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return 'inconnu (pas de git)'

_CACHE: dict = {}


def _load(panel: str):
    """Données et paramètres d'un panel, mis en cache dans chaque processus."""
    if panel not in _CACHE:
        cfg = yaml.safe_load(Path(PANELS[panel]).read_text())
        if panel == 'sp500':
            cfg['data']['raw_file'] = str(SP500_CSV)
        logret = DATA_READERS[cfg['data']['type']](cfg)
        logret = logret.loc[str(cfg['sample']['start']):str(cfg['sample']['end'])]
        est = cfg['estimation']
        kw = dict(window=cfg['sample']['window'], method=est['covariance'],
                  jump_method=est.get('jump_covariance', 'ewma'),
                  halflife=float(est.get('ewma_halflife', 60)),
                  shrinkage=float(est.get('ewma_shrinkage', 0.10)),
                  shape_estimator=est.get('shape_estimator', 'hyvarinen'))
        jcfg = cfg.get('jumps', {})
        thr = float(jcfg.get('threshold', 4.0))
        sw = int(jcfg.get('scale_window', 250))
        sq = float(cfg.get('regimes', {}).get('stress_quantile', 0.90))
        _CACHE[panel] = (logret, kw, thr, sw, sq)
    return _CACHE[panel]


def _resample_index(T: int, block: int, rng) -> np.ndarray:
    # Même construction que scripts/placebo_null.py.
    if block == 1:
        return rng.integers(0, T, size=T)
    n_blocks = int(np.ceil(T / block))
    starts = rng.integers(0, T, size=n_blocks)
    return (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T


def run_task(panel: str, null: str, rep: int) -> dict:
    """Une réplication (rep >= 0) ou la statistique observée (null == 'observed')."""
    t0 = time.time()
    logret, kw, thr, sw, sq = _load(panel)
    if null == 'observed':
        data = logret
    else:
        # Graine propre à (panel, témoin, réplication) : résultat identique quel
        # que soit l'ordre d'exécution ou le nombre de processus.
        ss = np.random.SeedSequence([SEED, list(PANELS).index(panel), NULLS[null], rep])
        idx = _resample_index(len(logret), NULLS[null], np.random.default_rng(ss))
        data = pd.DataFrame(logret.to_numpy()[idx], index=logret.index,
                            columns=logret.columns)
    ent = rolling_measures(data, **kw)
    stats = statistics_of(ent, data, thr, sw, sq)
    typo = typology_of(ent, threshold=thr, scale_window=sw, gap=40, pre=10, post_search=20)
    stats.update({f'H8_{k}': v for k, v in typo.items()})
    stats = {k: (float(v) if v is not None else None) for k, v in stats.items()}
    return {'code_version': CODE_VERSION, 'panel': panel, 'null': null, 'rep': rep, 'seconds': round(time.time() - t0, 1),
            'stats': stats}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _summaries(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for panel in df['panel'].unique():
        obs = df[(df.panel == panel) & (df.null == 'observed')]
        if obs.empty:
            continue
        obs = obs.set_index('statistic')['value']
        for null in [n for n in NULLS if n in set(df.null)]:
            sub = df[(df.panel == panel) & (df.null == null)]
            for stat, v in obs.items():
                col = sub.loc[sub.statistic == stat, 'value'].dropna()
                row = {'panel': panel, 'null': null, 'statistic': stat, 'observed': v}
                if len(col) and np.isfinite(v):
                    row.update(null_mean=col.mean(), null_sd=col.std(),
                               null_q05=col.quantile(0.05), null_q95=col.quantile(0.95),
                               p_greater=(col >= v).mean(), p_less=(col <= v).mean(),
                               n_rep=len(col))
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n-rep', type=int, default=200)
    p.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--panels', default='ff49,sp500')
    p.add_argument('--test', action='store_true')
    args = p.parse_args()

    panels = [x.strip() for x in args.panels.split(',') if x.strip()]
    n_rep = args.n_rep
    out_dir = ICI / 'resultats'
    if args.test:
        panels, n_rep, out_dir = ['sp500'], 2, ICI / 'resultats_test'
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / 'checkpoint.jsonl'

    done = set()
    stale = 0
    if ckpt.exists():
        for line in ckpt.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get('code_version') != CODE_VERSION:
                    stale += 1
                    continue
                done.add((r['panel'], r['null'], r['rep']))
    if stale:
        print(f'ATTENTION : {stale} résultats d\'une ancienne version du code ignorés '
              f'(point de reprise périmé). Ils seront recalculés.', flush=True)

    tasks = [(pa, 'observed', -1) for pa in panels]
    tasks += [(pa, nu, r) for pa in panels for nu in NULLS for r in range(n_rep)]
    todo = [t for t in tasks if t not in done]
    # Les tâches FF49 (les plus longues) d'abord, pour finir sans traîne.
    todo.sort(key=lambda t: (t[0] != 'ff49', t[1] != 'observed'))

    print(f'Panels : {panels} | réplications par témoin : {n_rep} | processus : {args.jobs}')
    print(f'Tâches : {len(tasks)} au total, {len(tasks) - len(todo)} déjà faites, '
          f'{len(todo)} à faire.', flush=True)

    t_start = time.time()
    n_done = 0
    if todo:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex, \
                open(ckpt, 'a', encoding='utf-8') as fh:
            futs = {ex.submit(run_task, *t): t for t in todo}
            for fut in as_completed(futs):
                r = fut.result()
                fh.write(json.dumps(r) + '\n')
                fh.flush()
                n_done += 1
                el = time.time() - t_start
                eta = el / n_done * (len(todo) - n_done)
                print(f'  [{n_done}/{len(todo)}] {r["panel"]:5s} {r["null"]:8s} '
                      f'rep {r["rep"]:3d}  ({r["seconds"]:.0f} s)  '
                      f'reste ~{eta / 3600:.1f} h', flush=True)

    # Assemblage : un CSV long (toutes les réplications), un résumé, les métadonnées.
    recs = [r for r in (json.loads(l) for l in ckpt.read_text(encoding='utf-8').splitlines()
                        if l.strip()) if r.get('code_version') == CODE_VERSION]
    long = pd.DataFrame([{'panel': r['panel'], 'null': r['null'], 'rep': r['rep'],
                          'statistic': k, 'value': v}
                         for r in recs for k, v in r['stats'].items()])
    long = long.drop_duplicates(['panel', 'null', 'rep', 'statistic'], keep='last')
    summary = _summaries(long)
    long.to_csv(out_dir / 'replications_long.csv', index=False)
    summary.to_csv(out_dir / 'resume_temoins.csv', index=False)

    import scipy, sklearn, statsmodels
    info = {
        'code_commit': _code_commit(), 'code_version': CODE_VERSION,
        'seed': SEED, 'n_rep': n_rep, 'panels': panels, 'nulls': NULLS,
        'jobs': args.jobs, 'test_mode': args.test,
        'wall_seconds_this_session': round(time.time() - t_start, 1),
        'cpu_seconds_total': round(sum(r['seconds'] for r in recs), 1),
        'data_sha256': {
            'ff49': _sha256(FF49_ZIP),
            'sp500': _sha256(SP500_CSV),
        },
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scipy': scipy.__version__,
                     'scikit-learn': sklearn.__version__,
                     'statsmodels': statsmodels.__version__},
        'platform': platform.platform(), 'cpu_count': os.cpu_count(),
    }
    (out_dir / 'run_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')

    zpath = out_dir / ('resultats_test.zip' if args.test else 'resultats_calcul_local.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in ('replications_long.csv', 'resume_temoins.csv', 'run_info.json'):
            z.write(out_dir / f, f)

    expected = {(pa, nu, r) for pa in panels for nu in NULLS for r in range(n_rep)}
    missing = expected - {(r['panel'], r['null'], r['rep']) for r in recs}
    print()
    key = summary[summary.statistic.isin(['H1_hdep_stress_minus_calm', 'range_J',
                                          'H8_share_dependence_dominant'])]
    if len(key):
        print(key[['panel', 'null', 'statistic', 'observed', 'null_mean',
                   'p_greater', 'p_less', 'n_rep']].to_string(index=False))
    print()
    if missing:
        print(f'ATTENTION : {len(missing)} réplications manquantes. Relancez la même commande.')
    else:
        print('Terminé.')
    print(f'Fichier à renvoyer : {zpath}')


if __name__ == '__main__':
    main()
