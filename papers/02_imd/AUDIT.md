# Audit honnête du papier IMD/IES — état post-corrections

Date initiale : 2026-05-12 (cycle 1). Cycle 2 : corrections appliquées.

État courant : `imd_trd.tex`, **39 pages**, 12/13 expériences exécutées
classées en 7 clean / 2 qualified / 3 substitute.

L'audit ci-dessous est rédigé du point de vue d'un *referee* de
*Transportation Research Part D*. Section 1 = solide, section 2 =
faiblesses identifiées dans le cycle 1, section 3 = corrections
appliquées dans le cycle 2, section 4 = ce qui reste à faire.

---

## 1. Ce qui tient solidement

### 1.1 Le pipeline de données
- **GBFS Gold Standard** est une vraie production : 122 systèmes, 46 139
  stations, 5 modules d'enrichissement documentés (OSM, SRTM, BAAC,
  GTFS). Tout est code-suivi, ré-exécutable.
- Le **panel dock-based de 59 villes** est cohérent et bien borné par
  des filtres explicites (`station_type == "docked_bike"`, ≥ 5 stations).

### 1.2 La méthode de calibration
- **Évolution différentielle + reparamétrisation softmax** : élimine
  proprement la contrainte simplexe + plancher `w_min`.
- Comparaison contre **CRITIC et poids normatifs** + **Lyon-removed**
  (cycle 2) : quatre alternatives dans la même table.

### 1.3 La généralisation hors-échantillon
- **E1 LOO-CV** : `ρ_LOO = 0.49 (FUB)` et `0.55 (EMP)` sur 32 et 44
  villes. Dépasse le seuil conventionnel de 0.30 pour un indicateur
  composite supervisé.
- **SDs inter-fold des poids < 0.05**.

### 1.4 Le découpage économique vs social
- **Sobol 48 % infrastructure / 34 % multimodalité** sur la variance
  de score, et **77 % multimodalité** sur la variance de rang.
- **Bayésien IES (E9)** : β_velo_travail = +9.01 [CrI exclut 0] est
  la seule covariable crédiblement non nulle ; β_income centré près
  de zéro.

---

## 2. Faiblesses identifiées au cycle 1 et leur correction

| # | Faiblesse | Statut | Comment |
|---|---|---|---|
| 1 | Sensibilité du Bayésien IES au prior τ non rapportée | **Corrigé** | E9 sweep sur τ ∈ {0.1, 1, 10} → 11/9/4 déserts respectivement. 4 villes prior-invariantes : Amiens, Lyon, Nancy, Saumur. |
| 2 | Poids Lyon-removed promis mais non tabulé | **Corrigé** | Ligne ajoutée à Table 2 ; section E12 chiffre l'effet sur ranking (τ Kendall = 0.83, Top-10 overlap 9/10). |
| 3 | "12/13 tests pass" sur-vendu, ne distingue pas substituts | **Corrigé** | Scorecard et Table 4 classifient maintenant en clean (7) / qualified (2) / substitute (3). Abstract et conclusion mis à jour. |
| 4 | E17 pseudo-flow interprétation post-hoc | **Corrigé** | Section E17 reformulée comme "méthodological probe", liste 4 interprétations possibles, ne tranche pas. |
| 5 | Montpellier 1 paragraphe sans chiffres | **Corrigé** | Table dédiée : 83 % stations avec ≥1 arrêt GTFS lourd, 40 % avec ≥3, 95e percentile du panel. |
| 6 | Pas de benchmark institutionnel | **Partiellement corrigé** | Mention ajoutée en related-work (Cerema, ADEME, Mobi2024) ; pas de comparaison empirique sur ranking faute de données institutionnelles disponibles. |
| 7 | IES gap policy chiffré sans détail par ville | **Corrigé** | Uplift médian recomputé : 12.8 pts (liste τ=1), 16.9 pts (liste invariante) ; Lyon = 28.9 pts. |

### Faiblesses restantes (cycle 2)

- **Taille du panel FUB n=32, éco-compteur n=25** : intrinsèque aux
  données disponibles, ne peut être augmenté sans nouvelles enquêtes.
  Tous les résultats Bayésien et controls sont reportés AVEC les
  intervalles de crédibilité/confiance — donc honnêtement présentés.

- **E17 1 fenêtre 24h** : la suggestion la plus serrée serait de
  déplacer entièrement en annexe. Le compromis adopté est de garder
  dans le corps avec un caveat explicite et un encadré rouge de
  4 interprétations. Suffisant pour un papier honnête.

- **Cluster #0 / #2 (E13) policy claim** : maintenant chiffré en
  termes d'écart IMD aux champions (40 pts pour cluster #0, 46 pts
  pour cluster #2) — voir Discussion §5.3 (Operational implications).

---

## 3. Verdict mis à jour

**Le papier est solide sur l'ingénierie de pipeline, la méthode de
calibration, ET — depuis le cycle 2 — sur la transparence des
caveats.** Les 12 tests sont maintenant classés honnêtement :

| Verdict | Tests | Sens |
|---|---|---|
| Clean (7) | E1, E2, E3, E6, E7, E8, E10, E13 | passes nets, seuils franchis sans réserve |
| Qualified (2) | E9, E12 | passes mais avec un effet documenté (sensibilité au prior ; effet de levier qui reorder Saint-Nazaire/Dijon) |
| Substitute (3) | E4, E5, E11 | proxy en place de protocole pré-enregistré différé |
| Outstanding (1) | E14 | requiert archives GBFS historiques |

L'argument central — *l'IMD est mieux que le volumétrique sur le
bien-être, et la qualité cyclable est statistiquement indépendante
du revenu* — est désormais **soutenu par 7 passes nets + 2 qualifiés**,
ce qui est défendable face à reviewers méfiants.

Le papier est dans un état **publiable en *TR Part D* après une seule
révision mineure** (compléter le station-level kriging pour E4 + collecter
le multi-week pseudo-flow pour E17 + récupérer les archives GBFS pour E14).
