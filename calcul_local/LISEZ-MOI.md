# Calcul local : témoins calibrés à 200 réplications

Ce dossier utilise le code du dépôt (`src/`, `scripts/`, `config/`), les
données FF49 déjà suivies dans `data/` et le panel S&P 500 livré ici dans
`calcul_local/data/`. **Aucun accès réseau n'est nécessaire** une fois les
paquets Python installés. Toutes les commandes se lancent depuis la racine du
dépôt.

## Ce que ça calcule

Pour chacun des deux panels (FF49, 48 industries, 1990–2026 ; S&P 500, 20 titres,
1990–2022) :

- les statistiques observées (H1 à H6, étendue de J, typologie H8) ;
- 200 réplications du témoin **i.i.d.** ;
- 200 réplications du témoin **par blocs de 21 jours**.

Soit 802 passages du pipeline complet. C'est l'étape 3 du protocole du 29 août,
avec en plus la typologie H8 calculée sur les mêmes rééchantillonnages
(aujourd'hui elle ne repose que sur 15 à 20 réplications).

## Durée

Mesurée ici sur un cœur : ~45 s par réplication FF49, ~20 s par réplication
S&P 500, soit ~7 h de calcul au total, divisées par le nombre de processus.

| cœurs utilisés | durée approximative |
|---|---|
| 4 | ~2 h |
| 8 | ~1 h |
| 16 | ~30 min |

Mémoire : ~300 Mo par processus. 8 Go de RAM suffisent pour 8 processus.

## Étapes

1. **Python 3.10 ou plus récent.** Vérifiez avec `python --version`
   (ou `python3 --version` sur macOS/Linux).

2. **Récupérez le dépôt** et placez-vous à sa racine :

   ```bash
   git clone https://github.com/reoptimus/entropy_brownian_finance.git
   cd entropy_brownian_finance
   ```

   (ou `git pull` dans un clone existant).

3. **Installez les dépendances** (dans un environnement virtuel, recommandé) :

   ```bash
   python -m venv .venv
   # Windows :      .venv\Scripts\activate
   # macOS/Linux :  source .venv/bin/activate
   pip install -r calcul_local/requirements.txt
   ```

4. **Contrôle rapide** (~1 min, vérifie que tout s'exécute) :

   ```bash
   python calcul_local/lancer_calcul.py --test
   ```

   Il doit finir par `Terminé.` et afficher, pour le S&P 500, une valeur
   observée `H1_hdep_stress_minus_calm` de **−0,99 (phase 0 ; −1,72 avant)** et `range_J` de **18,95**
   (les chiffres du papier). Si c'est le cas, l'installation est bonne.

5. **Calcul complet** :

   ```bash
   python calcul_local/lancer_calcul.py
   ```

   Par défaut il utilise tous vos cœurs sauf un. Pour en fixer le nombre :
   `python calcul_local/lancer_calcul.py --jobs 6`. Chaque ligne affichée donne le temps
   restant estimé.

   Pour le laisser tourner la nuit sous macOS/Linux sans garder le terminal
   ouvert : `nohup python calcul_local/lancer_calcul.py > calcul.log 2>&1 &`.
   Sous Windows, laissez simplement la fenêtre ouverte et désactivez la mise
   en veille.

6. **Si le calcul s'interrompt** (veille, coupure, Ctrl+C) : relancez la même
   commande. Les réplications déjà faites sont gardées dans
   `calcul_local/resultats/checkpoint.jsonl` et ne sont pas recalculées.

## Ce qu'il faut me renvoyer

Un seul fichier : **`calcul_local/resultats/resultats_calcul_local.zip`** (quelques Mo).
Déposez-le dans le fil du projet. Il contient :

- `replications_long.csv` : chaque statistique de chaque réplication ;
- `resume_temoins.csv` : moyenne, quantiles 5–95 % et p-values par statistique ;
- `run_info.json` : commit du code, versions des paquets, empreintes SHA-256 des données, durée.

Si le script affiche `ATTENTION : … réplications manquantes`, relancez-le avant
de m'envoyer le fichier.

## Remarques

- Le résultat est déterministe : chaque réplication a sa propre graine, donc le
  nombre de cœurs ou l'ordre d'exécution ne change rien aux chiffres.
- Les graines diffèrent de celles du tableau actuel à 20 réplications : les
  valeurs individuelles ne coïncideront pas avec ce tableau, seules les
  distributions doivent être cohérentes.
