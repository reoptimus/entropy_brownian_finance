# Phase A : le canal de dépendance face au ratio d'absorption

Date : 24 octobre 2026. Code : branche `claude/phaseA-couplage`. Témoins : 200 réplications
i.i.d. et 200 par blocs de 21 jours sur chaque panel, avec les graines du calcul publié.
Les 802 écarts `h_dep` recalculés coïncident avec ceux de
`placebo_null_200rep_replications_long.csv` (écart maximal 1,1e-14).

## Verdict : on continue (go)

La famille entropique `½ log det R` franchit les deux témoins sur les deux panels, sous ses trois
formes. Le ratio d'absorption de Kritzman et al. (2011), qui est la mesure de référence de la même
famille, échoue au témoin par blocs sur les deux panels. Le canal de dépendance détecte donc
quelque chose que la mesure standard ne détecte pas, une fois l'artefact d'estimation retiré.

## Écart de régime stress moins calme, face aux témoins

p = part des réplications au moins aussi extrêmes du côté prédit (couplage sous stress).
z = (observé − moyenne du témoin) / écart type du témoin, signé : sans unité, donc comparable
d'une mesure à l'autre.

| Mesure | S&P 500, i.i.d. | S&P 500, blocs 21 j | FF49, i.i.d. | FF49, blocs 21 j |
|---|---|---|---|---|
| `h_dep` (référence du papier) | z = 6,2, p < 0,005 | z = 2,0, p = 0,030 | z = 11,9, p < 0,005 | z = 1,8, p = 0,025 |
| `h_dep` dévolatilisé (Forbes-Rigobon) | z = 5,1, p < 0,005 | z = 2,0, p = 0,040 | z = 7,9, p < 0,005 | z = 1,7, p = 0,040 |
| `h_dep` en rangs (scores normaux) | z = 9,6, p < 0,005 | **z = 2,3, p = 0,030** | z = 12,7, p < 0,005 | **z = 2,2, p = 0,010** |
| Ratio d'absorption (k = N/5) | z = 0,7, p = 0,215 | z = −0,5, p = 0,695 | z = 6,9, p < 0,005 | z = 0,4, p = 0,375 |
| Part de la 1re valeur propre | z = 5,3, p < 0,005 | z = 1,8, p = 0,040 | z = 6,7, p < 0,005 | z = 0,2, p = 0,445 |

Source : `paper/tables/coupling_null_200rep_summary.csv`. « p < 0,005 » : aucune des 200 réplications.

Trois lectures :

1. **Forbes-Rigobon.** Le couplage survit quand chaque rendement est divisé par sa volatilité
   conditionnelle (EWMA, λ = 0,94, passé seul). Il s'affaiblit d'environ 30 % mais franchit toujours
   les deux témoins : ce n'est pas un simple effet mécanique de la hausse de variance.
2. **Robustesse aux queues.** La version en rangs, invariante aux lois marginales, est la plus forte
   face au témoin par blocs sur les deux panels. Le signal ne vient pas de quelques jours extrêmes.
3. **Concurrents.** Le ratio d'absorption ne se distingue pas du témoin par blocs ; la part de la
   première valeur propre franchit sur les titres individuels mais pas sur les secteurs. Seule la
   famille entropique tient partout. Explication probable : `log det R` pèse tout le spectre, y
   compris les petites valeurs propres qui s'effondrent quand la coupe transversale se couple,
   alors que les mesures de concentration ne regardent que le haut du spectre.

## Ce que ce test ne dit pas encore

- **Pas d'alerte précoce.** Contre les dates de récession NBER, la volatilité seule (AUROC 0,84)
  bat toutes les mesures de couplage (0,58 à 0,69). Une fois la part expliquée par la volatilité
  retirée, il reste un peu d'information sur les titres individuels (0,59 à 0,60 pour les versions
  dévolatilisée et en rangs), aucune sur les secteurs. Dans les 12 mois qui précèdent une récession,
  le couplage est plutôt plus bas que d'habitude (AUROC 0,37 à 0,45) : c'est le calme d'avant la
  crise, avec seulement quatre récessions dans l'échantillon. L'indicateur doit être présenté comme
  coïncident, ce que confirmait déjà H7. Source : `paper/tables/*_coupling_auroc.csv`.
- **Face à l'indice de stress de la Fed de St. Louis, sur 1994–2026 : un apport au-delà de la
  volatilité sur les titres, pas sur les secteurs.** Historique complet du `STLFSI4` (fin 1993 à
  octobre 2026, hebdomadaire, issu du téléchargement zone euro de seb), 1 514 semaines communes avec
  le S&P 500 (jusqu'à fin 2022) et 1 697 avec FF49 (jusqu'à mi-2026). Régressions standardisées,
  erreurs HAC (4 retards) ; t de la mesure de couplage quand la volatilité 21 jours est aussi
  dans la régression :

  | Mesure | S&P 500, R² seule | S&P 500, t avec vol. | FF49, R² seule | FF49, t avec vol. |
  |---|---|---|---|---|
  | `h_dep` | 0,16 | **4,2** | 0,10 | 0,3 |
  | `h_dep` dévolatilisé | 0,16 | **4,8** | 0,09 | 0,5 |
  | `h_dep` en rangs | 0,16 | **4,7** | 0,09 | −0,1 |
  | Ratio d'absorption | 0,06 | 4,2 | 0,07 | −2,0 |
  | Part de la 1re valeur propre | 0,16 | 4,6 | 0,07 | −0,2 |
  | Volatilité 21 jours seule | 0,48 | | 0,41 | |

  Sur l'historique long, `h_dep` suit l'indice officiel aussi bien que la part de la première
  valeur propre et bien mieux que le ratio d'absorption (R² 0,16 contre 0,06 sur les titres) ;
  ensemble, `h_dep` garde t = 5,9 et le ratio d'absorption passe à −6,0. Sur les titres, le couplage
  ajoute à la volatilité (R² de 0,48 à 0,51), surtout avant 2015 (t ≈ 6 sur 1994–2014, ≈ 0 après) :
  le constat fait sur 2015–2025 seulement tenait à la fenêtre courte. Sur les portefeuilles
  sectoriels, rien n'ajoute à la volatilité sur la période complète. Les mesures restent corrélées à
  plus de 0,9, donc les régressions jointes entre mesures de couplage sont fragiles ; celles avec la
  volatilité le sont moins. Source : `paper/tables/*_coupling_encompassing.csv` (colonnes `period` :
  `full`, `pre2015`, `post2015`).
- **Tests multiples.** Cinq mesures sont testées ; les p-valeurs du témoin par blocs (0,01 à 0,04)
  sont individuellement significatives, mais une seule (la version en rangs sur FF49, p = 0,010) atteint le seuil corrigé de Bonferroni (0,05 / 5 = 0,01). Le
  résultat tient par sa cohérence sur deux panels et trois variantes, pas par une p-valeur isolée.

## Suite proposée

Phase B : construire le « CISS entropique » zone euro (cinq segments de marché, version en rangs
comme mesure principale, décomposition intra et inter-segments), le confronter au CISS de la BCE et
faire l'étude d'événements et le Growth-at-Risk. Il faudra télécharger les données du portail de la
BCE sur ta machine ou autoriser ce domaine dans l'environnement.

## Reproduire

```bash
python -m scripts.coupling_null --jobs 4          # ~2 h sur 4 cœurs, reprenable
python -m scripts.coupling_benchmark --config config/sp500.yaml
python -m scripts.coupling_benchmark --config config/default.yaml
```
