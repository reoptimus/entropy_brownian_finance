# Phase B : décomposition du couplage par blocs (résultats intermédiaires)

Date : 25 octobre 2026. Branche `claude/phaseB-zone-euro`. Version en rangs (scores normaux,
Ledoit-Wolf, fenêtre de 252 jours), mêmes 200 + 200 rééchantillonnages que la phase A.

La corrélation totale se décompose exactement en couplage interne à chaque bloc et en
information mutuelle entre blocs (`src/blocks.py`). Le terme entre blocs est la part systémique.
En attendant les données de la zone euro, les blocs sont des secteurs : cinq macro-secteurs pour
FF49 (approximation des cinq industries de Fama-French, la finance à part), cinq secteurs pour les
vingt titres du S&P 500.

## Écart stress moins calme, face aux témoins

| Composante | S&P 500, i.i.d. | S&P 500, blocs 21 j | FF49, i.i.d. | FF49, blocs 21 j |
|---|---|---|---|---|
| Couplage total | p < 0,005 | p = 0,030 | p < 0,005 | p = 0,010 |
| Couplage **entre** blocs | p < 0,005 | **p = 0,015** | p < 0,005 | p = 0,275 |
| Part entre blocs dans le total | p = 0,005 | **p = 0,025** | à contre-sens | p = 0,635 |
| Finance avec le reste (paire la plus forte) | p < 0,005 | p = 0,010 (consommation) | p < 0,005 | p = 0,055 (santé) |

Source : `paper/tables/block_null_200rep_summary.csv`.

## Lecture

- **Titres individuels : la contagion entre secteurs est identifiée.** Sous stress, le couplage
  entre secteurs monte au-delà des deux témoins, et sa part dans le total augmente : le stress ne
  renforce pas seulement chaque secteur, il les relie entre eux. C'est la signature systémique que
  l'indicateur zone euro doit mesurer entre segments de marché.
- **Portefeuilles sectoriels : le couplage monte surtout à l'intérieur des macro-secteurs.** Le
  terme entre blocs ne franchit pas le témoin par blocs. Agréger des titres en portefeuilles
  absorbe déjà une partie du facteur commun, et les blocs de FF49 sont eux-mêmes très corrélés.
- **Liens de la finance** : seul le lien finance-consommation sur le S&P 500 franchit les deux
  témoins ; les autres paires restent dans le bruit du témoin par blocs. Avec deux banques
  seulement dans le bloc finance du S&P 500, ce résultat reste fragile.

## Suite

Le même calcul sur les cinq segments du CISS (marché monétaire, obligations, actions,
intermédiaires financiers, change) dès réception de `donnees_zone_euro.zip`, puis la comparaison
au CISS, l'AUROC contre les récessions CEPR et le Growth-at-Risk (`scripts/euro_ciss.py`, déjà
testé de bout en bout sur données synthétiques).
