"""Téléchargement des données de la phase B (zone euro), à lancer sur votre machine.

Le conteneur de Claude n'a pas accès aux sites de la BCE, de FRED ni de Yahoo :
ce script récupère tout en une fois, depuis la racine du dépôt ou d'ailleurs.

    pip install pandas requests yfinance
    python telecharger.py

Il produit un seul fichier à renvoyer dans le fil du projet :
``donnees_zone_euro.zip`` (quelques Mo), avec un ``manifeste.csv`` qui dit, série
par série, ce qui a réussi. Une série qui échoue n'arrête pas le script : relancez-le
plus tard ou envoyez le zip tel quel, je m'adapterai.

Sources, toutes publiques et gratuites :
- portail de données de la BCE (CISS, courbe des taux, Euribor, €STR, change) ;
- FRED (indice de stress de la Fed de St. Louis, historique complet ; PIB réel zone euro) ;
- Yahoo Finance via yfinance (indices boursiers nationaux, grandes banques et assureurs).
"""
from __future__ import annotations

import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ICI = Path(__file__).resolve().parent
SORTIE = ICI / 'donnees_zone_euro'
DEBUT = '1999-01-01'
ECB = 'https://data-api.ecb.europa.eu/service/data/{flow}/{key}'

# Segment -> {nom de colonne: (source, identifiant)}. Les cinq segments sont ceux
# du CISS de la BCE (Hollo, Kremer et Lo Duca, 2012).
SERIES = {
    'monetaire': {
        'euribor_1m': ('ecb', 'FM/D.U2.EUR.RT.MM.EURIBOR1MD_.HSTA'),
        'euribor_3m': ('ecb', 'FM/D.U2.EUR.RT.MM.EURIBOR3MD_.HSTA'),
        'euribor_6m': ('ecb', 'FM/D.U2.EUR.RT.MM.EURIBOR6MD_.HSTA'),
        'euribor_12m': ('ecb', 'FM/D.U2.EUR.RT.MM.EURIBOR1YD_.HSTA'),
        'estr': ('ecb', 'EST/B.EU000A2X2A25.WT'),
        'eonia': ('ecb', 'EON/D.EONIA_TO.RATE'),
    },
    'obligataire': {
        'aaa_2a': ('ecb', 'YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y'),
        'aaa_5a': ('ecb', 'YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_5Y'),
        'aaa_10a': ('ecb', 'YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y'),
        'tous_2a': ('ecb', 'YC/B.U2.EUR.4F.G_N_C.SV_C_YM.SR_2Y'),
        'tous_10a': ('ecb', 'YC/B.U2.EUR.4F.G_N_C.SV_C_YM.SR_10Y'),
    },
    'actions': {
        'dax': ('yahoo', '^GDAXI'),
        'cac40': ('yahoo', '^FCHI'),
        'ftse_mib': ('yahoo', 'FTSEMIB.MI'),
        'ibex35': ('yahoo', '^IBEX'),
        'aex': ('yahoo', '^AEX'),
        'euro_stoxx_50': ('yahoo', '^STOXX50E'),
    },
    'intermediaires': {
        'bnp': ('yahoo', 'BNP.PA'),
        'societe_generale': ('yahoo', 'GLE.PA'),
        'credit_agricole': ('yahoo', 'ACA.PA'),
        'deutsche_bank': ('yahoo', 'DBK.DE'),
        'commerzbank': ('yahoo', 'CBK.DE'),
        'intesa': ('yahoo', 'ISP.MI'),
        'unicredit': ('yahoo', 'UCG.MI'),
        'santander': ('yahoo', 'SAN.MC'),
        'bbva': ('yahoo', 'BBVA.MC'),
        'ing': ('yahoo', 'INGA.AS'),
        'allianz': ('yahoo', 'ALV.DE'),
        'axa': ('yahoo', 'CS.PA'),
        'generali': ('yahoo', 'G.MI'),
    },
    'change': {
        'eur_usd': ('ecb', 'EXR/D.USD.EUR.SP00.A'),
        'eur_gbp': ('ecb', 'EXR/D.GBP.EUR.SP00.A'),
        'eur_jpy': ('ecb', 'EXR/D.JPY.EUR.SP00.A'),
        'eur_chf': ('ecb', 'EXR/D.CHF.EUR.SP00.A'),
    },
    'reference': {
        'ciss_zone_euro': ('ecb', 'CISS/D.U2.Z0Z.4F.EC.SS_CIN.IDX'),
        'stlfsi4': ('fred', 'STLFSI4'),
        'pib_reel_zone_euro': ('fred', 'CLVMNACSCAB1GQEA19'),
    },
}


def _ecb(path: str) -> pd.Series:
    flow, key = path.split('/', 1)
    r = requests.get(ECB.format(flow=flow, key=key),
                     params={'format': 'csvdata', 'startPeriod': DEBUT}, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.Series(pd.to_numeric(df['OBS_VALUE'], errors='coerce').to_numpy(),
                  index=pd.to_datetime(df['TIME_PERIOD']))
    return s.sort_index().dropna()


def _ecb_ciss_all() -> pd.DataFrame:
    """Toutes les séries quotidiennes du jeu CISS pour la zone euro (composite et
    sous-indices), quelle que soit leur clé exacte."""
    r = requests.get('https://data-api.ecb.europa.eu/service/data/CISS',
                     params={'format': 'csvdata', 'startPeriod': DEBUT}, timeout=300)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df = df[df['KEY'].str.startswith('CISS.D.') & df['KEY'].str.contains(r'\.U2\.')]
    wide = df.pivot_table(index='TIME_PERIOD', columns='KEY', values='OBS_VALUE')
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


def _fred(series_id: str) -> pd.Series:
    r = requests.get('https://fred.stlouisfed.org/graph/fredgraph.csv',
                     params={'id': series_id}, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors='coerce').to_numpy(),
                  index=pd.to_datetime(df.iloc[:, 0]))
    return s.dropna()


def _yahoo(ticker: str) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start=DEBUT, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError('aucune donnée renvoyée')
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


FETCH = {'ecb': _ecb, 'fred': _fred, 'yahoo': _yahoo}


def main() -> None:
    SORTIE.mkdir(exist_ok=True)
    manifeste = []
    for segment, series in SERIES.items():
        cols = {}
        for nom, (source, ident) in series.items():
            for essai in range(3):
                try:
                    s = FETCH[source](ident)
                    cols[nom] = s
                    manifeste.append({'segment': segment, 'serie': nom, 'source': source,
                                      'identifiant': ident, 'statut': 'ok', 'n_obs': len(s),
                                      'debut': s.index.min().date(), 'fin': s.index.max().date()})
                    print(f'  ok      {segment:15s} {nom:20s} {len(s):6d} obs '
                          f'{s.index.min().date()} -> {s.index.max().date()}', flush=True)
                    break
                except Exception as e:  # noqa: BLE001 -- on veut continuer quoi qu'il arrive
                    if essai == 2:
                        manifeste.append({'segment': segment, 'serie': nom, 'source': source,
                                          'identifiant': ident, 'statut': f'échec : {e}'[:300]})
                        print(f'  ÉCHEC   {segment:15s} {nom:20s} {e}', flush=True)
                    else:
                        time.sleep(3)
        if cols:
            pd.DataFrame(cols).sort_index().to_csv(SORTIE / f'{segment}.csv')

    try:
        ciss = _ecb_ciss_all()
        ciss.to_csv(SORTIE / 'ciss_toutes_series.csv')
        manifeste.append({'segment': 'reference', 'serie': 'ciss_toutes_series',
                          'source': 'ecb', 'identifiant': 'CISS (filtre U2, quotidien)',
                          'statut': 'ok', 'n_obs': len(ciss), 'debut': ciss.index.min().date(),
                          'fin': ciss.index.max().date()})
        print(f'  ok      CISS, toutes séries zone euro : {ciss.shape[1]} séries', flush=True)
    except Exception as e:  # noqa: BLE001
        manifeste.append({'segment': 'reference', 'serie': 'ciss_toutes_series',
                          'source': 'ecb', 'identifiant': 'CISS', 'statut': f'échec : {e}'[:300]})
        print(f'  ÉCHEC   CISS, toutes séries : {e}', flush=True)

    m = pd.DataFrame(manifeste)
    m.to_csv(SORTIE / 'manifeste.csv', index=False)
    zpath = ICI / 'donnees_zone_euro.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(SORTIE.glob('*.csv')):
            z.write(f, f.name)
    n_ok = int((m['statut'] == 'ok').sum())
    print(f'\n{n_ok} séries sur {len(m)} téléchargées.')
    print(f'Fichier à renvoyer : {zpath}')
    if n_ok < len(m):
        print('Certaines séries ont échoué (voir manifeste.csv) : envoyez quand même le zip.')


if __name__ == '__main__':
    sys.exit(main())
