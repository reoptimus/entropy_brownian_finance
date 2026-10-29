# Phase B : CISS entropique zone euro et décomposition par blocs

Date : 27 octobre 2026. Version en rangs (scores normaux, Ledoit-Wolf, fenêtre de 252 jours),
200 + 200 rééchantillonnages (i.i.d. et blocs de 21 jours) comme en phase A. La corrélation
totale se décompose exactement en couplage interne à chaque bloc et en information mutuelle entre
blocs (`src/blocks.py`) ; le terme entre blocs est la part systémique.

## 1. Zone euro : les cinq segments du CISS

**Données.** Téléchargement de seb (`telechargement_zone_euro/telecharger.py`, 27 octobre 2026),
panel quotidien de 5 690 jours, janvier 2005 à octobre 2026, 28 séries après filtres :

| Segment | Séries |
|---|---|
| Marché monétaire | taux au jour le jour (EONIA jusqu'au 30 septembre 2019, €STR ensuite, raccord neutralisé) |
| Obligations | courbes BCE AAA 2, 5 et 10 ans, toutes notations 2 et 10 ans, écart 10 ans toutes notations moins AAA |
| Actions | DAX, CAC 40, FTSE MIB, IBEX 35, AEX (l'Euro Stoxx 50 ne commence qu'en 2007) |
| Intermédiaires | 9 banques et 3 assureurs (Crédit Agricole écarté : prix ajustés négatifs jusqu'en 2007) |
| Change | EUR/USD, EUR/GBP, EUR/JPY, EUR/CHF |

**Euribor absent.** Les quatre clés Euribor du portail BCE ont renvoyé 404. Deux variantes ont
été calculées, chacune avec ses 400 témoins :

- *avec substitut* : écart AAA 2 ans moins taux au jour le jour dans le bloc monétaire ;
- *overnight seul* : le bloc monétaire réduit au taux au jour le jour.

Le substitut partage le taux AAA 2 ans avec le bloc obligataire, ce qui lie les deux blocs par
construction : sous stress, leur information mutuelle *baisse* (−0,14 nat, hors des deux témoins)
et cet artefact suffit à masquer la hausse du couplage entre segments. **La variante overnight seul
est donc retenue comme spécification principale**, en attendant l'Euribor (à récupérer par une
autre source, par exemple l'EMMI, pour rétablir l'écart Euribor 3 mois moins overnight du CISS).

### Écart stress moins calme (stress : décile supérieur de la volatilité 21 jours du bloc actions, seuil passé seulement)

| Composante | Écart observé (nats) | p, i.i.d. | p, blocs 21 j |
|---|---|---|---|
| Couplage total | +1,61 | < 0,005 | **< 0,005** |
| **Entre segments** | +0,19 | < 0,005 | **< 0,005** |
| Part entre segments dans le total | −0,005 | à contre-sens | à contre-sens |
| Intra actions | +0,46 | < 0,005 | < 0,005 |
| Intra intermédiaires | +0,48 | < 0,005 | 0,045 |
| Intra obligations | +0,34 | < 0,005 | < 0,005 |
| Intra change | +0,14 | < 0,005 | < 0,005 |
| Obligations – change | +0,055 | < 0,005 | < 0,005 |
| Actions – change | +0,046 | < 0,005 | < 0,005 |
| Actions – intermédiaires | +0,096 | < 0,005 | 0,025 |
| Intermédiaires – change | +0,040 | < 0,005 | 0,045 |
| Obligations – intermédiaires | +0,040 | < 0,005 | 0,050 |
| Obligations – actions | +0,029 | < 0,005 | 0,055 |
| Monétaire – autres segments | ≈ 0 | n.s. | n.s. |

Source : `paper/tables/euro_null_200rep_summary_overnight_only.csv` (et sans suffixe pour la
variante avec substitut).

### Comparaison au CISS de la BCE, récessions CEPR, Growth-at-Risk

| | Couplage total | Entre segments | Volatilité 21 j | CISS BCE |
|---|---|---|---|---|
| Corrélation avec le CISS (niveaux) | 0,58 | 0,35 | 0,74 | 1 |
| AUROC récessions CEPR (2008–09, 2011–13, 2020) | **0,85** | 0,75 | 0,80 | 0,83 |
| CISS expliqué : t en plus de la volatilité (HAC) | **6,7** | 1,7 | | |
| Growth-at-Risk, quantile 10 %, seul : coef. (p) | −1,25 (0,04) | −1,57 (< 0,001) | −0,04 (0,98) | −2,56 (< 0,001) |
| Growth-at-Risk, quantile 10 %, avec CISS et croissance passée : p | 0,78 | 0,81 | | < 0,001 |

Régression hebdomadaire standardisée du CISS sur les indicateurs, 1 089 semaines ; Growth-at-Risk :
croissance du PIB réel à 4 trimestres sur la moyenne trimestrielle de l'indicateur, 79 trimestres.
Source : `paper/tables/euro_ciss_comparisons_overnight_only.json`, figure
`paper/figures/euro_ciss_entropique_overnight_only.png`.

### Lecture

- **Le stress relie les segments entre eux, et ce n'est pas un effet de volatilité ni de
  persistance.** Le couplage entre les cinq segments monte sous stress au-delà des deux témoins,
  comme entre secteurs sur les titres du S&P 500. Toutes les paires hors marché monétaire montent ;
  les liens avec le change et entre actions et intermédiaires sont les plus nets. La part entre
  segments baisse légèrement : chaque segment se resserre encore plus vite en interne.
- **Le lien souverain-banques ressort.** L'information mutuelle obligations-intermédiaires et
  obligations-actions culmine fin 2011, au plus fort de la crise des dettes souveraines (0,46 et
  0,35 nat, contre 0,10 et 0,05 fin 2006). C'est le genre de lecture que le CISS agrégé ne donne pas.
- **Le marché monétaire ne se couple pas** aux autres segments en données quotidiennes : le taux au
  jour le jour ne bouge qu'aux décisions de politique monétaire. Il faut l'Euribor (ou une prime
  de terme interbancaire) pour que ce segment porte le stress de financement des banques.
- **Indicateur coïncident, complémentaire mais pas supérieur au CISS.** Le couplage total
  identifie les récessions CEPR un peu mieux que le CISS et que la volatilité, et explique le CISS
  au-delà de la volatilité. Pour la queue basse de la croissance future, il est significatif seul
  mais n'ajoute rien une fois le CISS présent. Trois récessions et 79 trimestres : ces comparaisons
  restent indicatives.

## 2. Secteurs (S&P 500 et FF49, résultats du 25 octobre)

Cinq macro-secteurs pour FF49 (approximation des cinq industries de Fama-French, la finance à
part), cinq secteurs pour les vingt titres du S&P 500.

### Écart stress moins calme, face aux témoins

| Composante | S&P 500, i.i.d. | S&P 500, blocs 21 j | FF49, i.i.d. | FF49, blocs 21 j |
|---|---|---|---|---|
| Couplage total | p < 0,005 | p = 0,030 | p < 0,005 | p = 0,010 |
| Couplage **entre** blocs | p < 0,005 | **p = 0,015** | p < 0,005 | p = 0,275 |
| Part entre blocs dans le total | p = 0,005 | **p = 0,025** | à contre-sens | p = 0,635 |
| Finance avec le reste (paire la plus forte) | p < 0,005 | p = 0,010 (consommation) | p < 0,005 | p = 0,055 (santé) |

Source : `paper/tables/block_null_200rep_summary.csv`.

### Lecture

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

- Récupérer l'Euribor 3 mois (EMMI ou autre source) et refaire le calcul avec l'écart Euribor
  moins overnight, ce qui donne au segment monétaire son contenu de stress de financement.
- Étude d'événements contre la base de crises de l'ESRB et tests d'englobement contre SRISK et
  ΔCoVaR, comme prévu dans la feuille de route.
- Rédaction de la version banque centrale (FR et EN, LaTeX et PDF) une fois ces points tranchés.

## Reproduire

```bash
python -m scripts.euro_ciss --data data/zone_euro --n-rep 200 --jobs 4 --overnight-only  # ~20 min
python -m scripts.euro_ciss --data data/zone_euro --n-rep 200 --jobs 4                   # avec substitut
python -m scripts.euro_figure --overnight-only
python -m scripts.block_null --jobs 4                                                    # secteurs
```
