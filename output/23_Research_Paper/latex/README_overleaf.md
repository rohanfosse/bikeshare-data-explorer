# Guide d'utilisation — Overleaf

## Structure du projet LaTeX

```
latex/
├── main.tex          ← document principal (IEEEtran, journal, français)
├── references.bib    ← bibliographie BibTeX (21 références)
├── figures/          ← dossier des figures (à remplir, voir ci-dessous)
└── README_overleaf.md
```

## Compilation

```
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Ou dans Overleaf : compiler directement (compilation automatique BibTeX).

---

## Figures à copier

Copier les fichiers PNG suivants dans `latex/figures/` avant de compiler.
Chaque figure est générée en exécutant les notebooks correspondants.

| Nom attendu dans figures/         | Source                                                          |
|-----------------------------------|-----------------------------------------------------------------|
| `fig01_carte_nationale.png`       | `output/20_GBFS_France_Collecte/figures/01_carte_nationale.png` |
| `fig04_classement_imd.png`        | `output/23_Research_Paper/figures/fig4_classement_imd.png`      |
| `fig05_correlations.png`          | `output/22_Profil_Socioeconomique_Mobilite/figures/02_correlations_socio_IMD.png` |
| `fig06_quadrants.png`             | `output/22_Profil_Socioeconomique_Mobilite/figures/03_quadrants_socio_mobilite.png` |
| `fig07_deserts.png`               | `output/22_Profil_Socioeconomique_Mobilite/figures/05_deserts_mobilite_sociale.png` |
| `fig08_ridge.png`                 | `output/22_Profil_Socioeconomique_Mobilite/figures/06_regression_IMD.png` |
| `fig09_clustering.png`            | `output/22_Profil_Socioeconomique_Mobilite/figures/07_clustering_socio_mobilite.png` |

> La figure méthodologique (fig02) et le classement (fig04) sont générés
> directement par le notebook 23 à l'exécution.

---

## Packages requis

Le document utilise uniquement des packages standard :

- `IEEEtran` (classe, incluse dans TeX Live / MikTeX)
- `babel` (french)
- `inputenc`, `fontenc`
- `amsmath`, `amssymb`
- `graphicx`, `adjustbox`
- `booktabs`, `multirow`, `tabularx`, `array`
- `xcolor`
- `hyperref`
- `microtype`, `subcaption`, `siunitx`
- `csquotes`

Sur Overleaf, tous ces packages sont disponibles sans installation.

---

## Personnalisation

- **Auteur** : remplacer `Rohan~\textsc{[Nom]}` et l'adresse email ligne ~47.
- **Affiliations multiples** : utiliser `\IEEEauthorblockN` / `\IEEEauthorblockA`
  supplémentaires dans le bloc `\author{}`.
- **Langue** : le document est en français. Pour une soumission IEEE anglophone,
  remplacer `\usepackage[french]{babel}` par `\usepackage[english]{babel}` et
  traduire le contenu.
- **Format** : `journal` (deux colonnes). Pour une conférence IEEE,
  remplacer par `\documentclass[conference]{IEEEtran}`.

---

## Normes IEEE respectées

- Classe `IEEEtran` avec option `journal` (10 pt, deux colonnes)
- `\IEEEPARstart` pour la première lettre de l'introduction
- `\markboth` pour les en-têtes
- Style bibliographique `IEEEtran` (numérotation entre crochets)
- Tables avec `\toprule` / `\midrule` / `\bottomrule` (booktabs)
- Équations numérotées avec `\label` et `\ref`
- `figure*` / `table*` pour les éléments pleine page
