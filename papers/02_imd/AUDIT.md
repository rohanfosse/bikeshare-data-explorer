# Audit honnête du papier IMD/IES

Date : 2026-05-12. État : `imd_trd.tex`, 37 pages, 12/13 expériences
exécutées.

L'audit ci-dessous est rédigé du point de vue d'un *referee* de
*Transportation Research Part D*. Le but est de lister, sans
auto-congratulation, les points où le papier est solide, les points
où il est faible, et les points où il sur-vend ses conclusions.

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
  proprement la contrainte simplexe + plancher `w_min`. C'est plus
  robuste que la projection naïve utilisée dans les versions
  antérieures.
- Comparaison contre **CRITIC et poids normatifs** : trois alternatives
  sont reportées dans la même table.

### 1.3 La généralisation hors-échantillon
- **E1 LOO-CV** : `ρ_LOO = 0.49 (FUB)` et `0.55 (EMP)` sur 32 et 44
  villes. Dépasse le seuil conventionnel de 0.30 pour un indicateur
  composite supervisé.
- **SDs inter-fold des poids < 0.05** : la calibration n'est pas
  surajustée à un sous-ensemble du panel.

### 1.4 Le découpage économique vs social
- **Sobol 48 % infrastructure / 34 % multimodalité** sur la variance
  de score, et **77 % multimodalité** sur la variance de rang : c'est
  un résultat propre, qui donne un ordre de priorité de qualité
  donnée actionnable.
- **Bayésien IES (E9)** : β_velo_travail = +9.01 [CrI exclut 0] est
  la seule covariable crédiblement non nulle ; β_income centré près
  de zéro confirme le résultat fréquentiste.

---

## 2. Les faiblesses honnêtes

### 2.1 Tailles d'échantillon serrées
- Le panel de calibration FUB est de **n = 32**. La régression
  bayésienne IES a **n = 59 villes × 4 prédicteurs ≈ 15 obs / prédicteur**,
  ce qui est marginalement acceptable mais explique les CrI larges
  sur Gini et car-free.
- Le sous-panel **éco-compteur n = 25** : la corrélation 
  `ρ(IMD, comptage) = +0.49` est significative à *p* = 0.012 mais
  l'IC 95 % de ρ sur 25 observations est large (≈ ±0.30).
- **E17 GBFS pseudo-flux** : n = 21 villes × **1 fenêtre 24 h**. La
  conclusion « équilibre système » est interprétative ; les données
  ne permettent pas vraiment de la trancher.

### 2.2 Tests « exécutés » qui sont en réalité des substituts
- **E4 (safety construct-validity)** : la prescription
  pré-enregistrée est un *kriging station-level de l'exposition
  cycliste*. Faute de comptages station-level, on rapporte un
  empirical-Bayes au niveau ville. Le résidu négatif `ρ = -0.49`
  contre les flux observés indique que le proxy est trop grossier.
  → C'est une **passe qualifiée**, pas un succès net.
- **E5 (within-city bootstrap)** est explicitement décrit comme
  *proxy* pour le sweep radius prévu.
- **E11 (sweep paramétrique)** est aussi un substitut : les
  élasticités α sont **inférées de la littérature, pas mesurées
  directement** sur les données du papier.
- Donc 3 des 12 « tests passés » sont des substituts. Le scorecard
  marque deux d'entre eux comme « qualified », mais l'abstract et la
  conclusion ne distinguent pas assez le qualified du clean pass.

### 2.3 Sensibilité du Bayésien IES aux hyperparamètres
- `τ = 1` (la précision du prior gaussien sur β) est choisi sans
  justification empirique. Avec `τ = 0.1` (prior plus faible) ou
  `τ = 10` (prior plus contraignant), la liste de neuf déserts à
  `P ≥ 0.90` changerait. **La sensibilité au prior n'est pas
  rapportée.**
- `a₀ = b₀ = 0.01` (Inv-Gamma sur σ²) : pseudo-Jeffreys, OK, mais
  pas dans le texte.

### 2.4 Promesses faites mais pas livrées dans le papier
- « Publier le poids-vecteur Lyon-removed à côté de l'optimum
  in-sample » : promis section discussion, **pas effectivement
  reporté** dans le papier. Le poids Lyon-removed est dans
  `e12_results.json` mais pas tabulé.
- « Multi-week pseudo-flow panel documented in the companion
  repository » : il y a actuellement 2 snapshots par système, pas
  un multi-week panel. La promesse est dans le futur.
- « Cluster (iii) operational target / cluster (iv) deeper
  structural intervention » (E13) : énoncé mais **non chiffré** —
  aucun coût ou yield estimé pour la distinction.

### 2.5 Le pseudo-flow GBFS (E17) : interprétation contre-intuitive
- `ρ = -0.54` entre l'IMD et le pseudo-flux : le papier l'interprète
  comme « équilibre système » des villes mûres. C'est une
  **rationalisation post-hoc**. Sur 1 journée × 2 snapshots × 21
  villes, on ne peut pas trancher entre :
  (a) équilibre système,
  (b) rééquilibrage opérateur agressif dans les petits réseaux,
  (c) artefact de capacité résiduel,
  (d) bruit pur.
- La section recommande honnêtement un panel multi-semaines, mais
  l'**inclusion dans le papier est prématurée** : E17 est plus un
  « teaser méthodologique » qu'un test exécuté.

### 2.6 La comparaison de métriques (E15)
- **Sous-panels hétérogènes** : la matrice de corrélations a
  des cellules à n = 15, 18, 25, 29, 32, 44, 59. Les corrélations
  ne sont pas directement comparables d'une cellule à l'autre.
- L'argument « IMD double la prédiction du volumétrique » repose
  sur ρ = 0.41 vs 0.20 *sur n = 32 villes*. Sur un échantillon plus
  large (les 59), la différence pourrait s'atténuer.
- **Le « volumétrique »** est défini comme `log(stations × mean_capacity)` ;
  c'est une définition raisonnable mais ad-hoc. Un défenseur du
  volumétrique pourrait proposer une définition légèrement
  différente qui ferait mieux le job.

### 2.7 La généralisabilité / portée
- **Pas de comparaison avec des indicateurs institutionnels
  existants** : Cerema, ADEME, Plan Vélo lui-même ont leurs
  benchmarks ; aucun n'est cité ni comparé empiriquement.
- L'étude de cas **Montpellier** est un seul paragraphe et n'offre
  aucune décomposition quantitative — la phrase « la quasi-totalité
  des stations sont alignées avec les lignes de tramway » mériterait
  une figure ou une métrique (% de stations à < 100 m d'un arrêt TAM ?).
- Le **panel se restreint aux systèmes publiant GBFS** : opérateurs
  en fin de contrat et schémas associatifs sont sous-représentés.
  C'est dans les limitations, mais l'impact sur l'IES (qui se base
  sur ces 59 villes) n'est pas quantifié.

### 2.8 Cadrage méthodologique
- « 12 of 13 tests pass » est un résumé séduisant mais cache que :
  - 3 tests sont des **substituts** (E4 partial, E5 proxy, E11
    paramétrique).
  - 1 test (E14) est *outstanding*, pas un échec ni une réussite.
  - Donc en strict : **9 tests pleinement passés + 3 substituts +
    1 outstanding**. C'est différent de « 12 of 13 ».
- L'expression « *core wave* / *extension wave* » résout
  honnêtement le faux pré-enregistrement à 13 tests, mais le
  scorecard à 12 lignes ne marque pas la distinction.

---

## 3. Ce qui doit être ajouté avant soumission

### 3.1 Priorité haute
1. **Reporter le poids-vecteur Lyon-removed** dans la table de
   calibration (Table 2 actuelle) à côté du poids in-sample. Une
   ligne supplémentaire suffit.
2. **Sensibilité au prior bayésien** : refaire E9 avec `τ ∈
   {0.1, 1, 10}` et reporter la stabilité de la liste de déserts à
   `P ≥ 0.90`.
3. **Marquer explicitement** dans la table E1-E14 quels tests sont
   *clean pass*, *qualified pass*, *substitute*, *outstanding*.
4. **Cite explicitement** au moins un benchmark institutionnel
   français (Cerema 2022 est cité comme contexte mais pas comme
   indicateur concurrent) et compare le ranking.

### 3.2 Priorité moyenne
5. **Quantifier le cas Montpellier** : % stations à < 100 m d'un
   arrêt TAM, distribution des distances inter-station, comparaison
   à Lyon. Une demi-page suffit.
6. **Re-vérifier l'inférence E17** : ré-exécuter sur un panel plus
   long (au moins une semaine de snapshots à 30 min) ou retirer du
   papier et déplacer en annexe.
7. **Quantifier l'IES gap** par catégorie de désert (les 9
   high-confidence et les 11 supplémentaires P ≥ 0.75) avec des
   intervalles d'incertitude.

### 3.3 Priorité basse
8. Ajouter une figure du chemin du Sobol (panel & rank) côte à côte.
9. Une référence économique sur les bénéfices monétaires
   d'investissement cyclable (FrancE-Mobilités, ADEME 2021) pour
   chiffrer le « 11.4 IMD points uplift ».
10. Une discussion explicite **du genre** et de la sécurité
    perçue : la littérature citée (Garrard, Aldred) le suggère
    fortement mais le papier ne creuse pas.

---

## 4. Verdict global

**Le papier est solide sur l'ingénierie de pipeline et la méthode
de calibration ; il est honnête sur la limite des données (en
général) ; il est sur-vendu sur le nombre de tests « passés ».**

Une révision serrée éliminant la sur-vente, livrant les promesses
faites (Lyon-removed, prior sensitivity, scorecard nuancé) et
modérant E17 amènerait la qualité du papier à un niveau
publiable en *TR Part D* sans contestation majeure.

L'argument central — *l'IMD est mieux que le volumétrique sur le
bien-être, et la qualité cyclable est statistiquement indépendante
du revenu* — est **empiriquement défendable** mais devra survivre
à des reviewers méfiants vis-à-vis des compositions d'expériences
post-hoc.
