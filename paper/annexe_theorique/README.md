# Annexe théorique / Theoretical supplement

Document séparé qui rassemble les éléments théoriques retirés du papier principal
lors de son recentrage sur le « CISS entropique » destiné aux banques centrales.

- `annexe_theorique_fr.tex` / `.pdf` : version française (référence).
- `annexe_theorique_en.tex` / `.pdf` : version anglaise.
- `parts_fr/`, `parts_en/` : sources par section ; les fichiers `.tex` principaux
  en sont la concaténation (`cat parts_xx/*.tex > annexe_theorique_xx.tex`).

Compilation : `latexmk -pdf annexe_theorique_fr.tex` (idem pour `_en`).
