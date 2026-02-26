# Donnees Socio-Demographiques de Montpellier

Ce dossier contient les donnees socio-demographiques de Montpellier traitees et nettoyees pour faciliter les etudes sociologiques, notamment sur la mobilite urbaine et l'usage du velo.

## Fichiers generes

### 1. `quartiers_montpellier.csv`
Donnees agregees pour les 7 grands quartiers de Montpellier.

**Quartiers inclus:**
- Centre
- Croix d'Argent
- Cevennes
- Hopitaux-Facultes
- Mosson
- Port Marianne
- Pres d'Arenes

### 2. `sous_quartiers_montpellier.csv`
Donnees detaillees pour les 30 sous-quartiers de Montpellier.

### 3. `tous_territoires_montpellier.csv`
Fichier consolide contenant tous les quartiers et sous-quartiers (37 territoires au total).

### 4. `transport_velo_montpellier.csv` ⭐
**Fichier cle pour les etudes sur la mobilite urbaine et le velo.**

Contient les donnees specifiques aux modes de transport et a l'equipement automobile:

#### Indicateurs de mobilite (pour se rendre au travail):
- `transport_pas_de_transport_nb/pct` - Personnes travaillant sans deplacement
- `transport_marche_a_pied_nb/pct` - Deplacements a pied
- `transport_deux_roues_velo_nb/pct` - **Usage du velo et deux-roues**
- `transport_voiture_camion_nb/pct` - Usage de la voiture
- `transport_transport_commun_nb/pct` - Usage des transports en commun

#### Equipement automobile des menages:
- `equipement_pas_de_voiture_nb/pct` - Menages sans voiture
- `equipement_une_voiture_nb/pct` - Menages avec une voiture
- `equipement_deux_voitures_ou_plus_nb/pct` - Menages multi-equipes

### 5. `socio_economique_montpellier.csv`
Indicateurs socio-economiques cles:
- Revenus fiscaux moyens par menage
- Taux de chomage
- Statut d'occupation du logement (proprietaires/locataires)
- Logements sociaux
- Categories socio-professionnelles (CSP)
- Niveaux de diplome

## Variables communes a tous les fichiers

### Identification
- `numero` - Numero du sous-quartier (uniquement pour sous-quartiers)
- `nom` - Nom du territoire
- `type` - Type: "quartier" ou "sous-quartier"

### Demographie
- `population_2009` - Population en 2009
- `population_1999` - Population en 1999
- `population_2007` - Population en 2007
- `variation_pop_1999_2009` - Variation de population 1999-2009
- `pct_population_mtp` - Pourcentage de la population de Montpellier

### Territoire
- `surface_km2` - Surface en kilometres carres
- `densite_hab_km2` - Densite d'habitants par km²

### Structure de la population par age
- `pop_0_14_ans_nb/pct`
- `pop_15_29_ans_nb/pct`
- `pop_30_44_ans_nb/pct`
- `pop_45_59_ans_nb/pct`
- `pop_60_74_ans_nb/pct`
- `pop_75_ans_et_plus_nb/pct`

### Socio-economique
- `revenu_fiscal_moyen_menage` - Revenu fiscal moyen par menage
- `nb_allocataires_rsa` - Nombre d'allocataires RSA
- `pct_chomeurs` - Pourcentage de chomeurs
- `pct_proprietaires` - Pourcentage de proprietaires
- `pct_locataires` - Pourcentage de locataires
- `pct_logements_sociaux` - Pourcentage de logements sociaux
- `nb_menages` - Nombre total de menages

### Categories socio-professionnelles (CSP)
- `csp_agriculteurs_exploitants_nb/pct`
- `csp_artisans_nb/pct`
- `csp_cadres_nb/pct`
- `csp_professions_interm_nb/pct`
- `csp_employ_nb/pct`
- `csp_ouvriers_nb/pct`
- `csp_retrait_nb/pct`

### Diplomes
- `diplome_aucun_dipl_nb/pct`
- `diplome_cep_nb/pct`
- `diplome_bepc_nb/pct`
- `diplome_cap_ou_bep_nb/pct`
- `diplome_baccalaur_nb/pct`
- `diplome_dipl_nb/pct` (diplomes superieurs)

## Utilisation pour etudes sociologiques sur le velo

### Questions de recherche possibles:

1. **Quels sont les quartiers avec le plus fort usage du velo?**
   - Analyser `transport_deux_roues_velo_pct`

2. **Quel est le profil socio-economique des quartiers "cyclables"?**
   - Croiser `transport_deux_roues_velo_pct` avec:
     - `revenu_fiscal_moyen_menage`
     - Categories socio-professionnelles
     - Niveaux de diplome

3. **Y a-t-il une correlation entre densite et usage du velo?**
   - Analyser `densite_hab_km2` vs `transport_deux_roues_velo_pct`

4. **Les menages sans voiture utilisent-ils plus le velo?**
   - Croiser `equipement_pas_de_voiture_pct` et `transport_deux_roues_velo_pct`

5. **Impact de l'age sur la mobilite a velo?**
   - Analyser la structure d'age des quartiers cyclables

6. **Concurrence velo vs transports en commun?**
   - Comparer `transport_deux_roues_velo_pct` et `transport_transport_commun_pct`

## Exemple de code Python pour analyse

```python
import pandas as pd
import matplotlib.pyplot as plt

# Charger les donnees
df_transport = pd.read_csv('transport_velo_montpellier.csv')

# Top 5 des quartiers/sous-quartiers cyclables
top_velo = df_transport.nlargest(5, 'transport_deux_roues_velo_pct')[
    ['nom', 'type', 'transport_deux_roues_velo_pct']
]
print(top_velo)

# Correlation densite / usage velo
correlation = df_transport[['densite_hab_km2', 'transport_deux_roues_velo_pct']].corr()
print(correlation)

# Visualisation
plt.scatter(
    df_transport['densite_hab_km2'],
    df_transport['transport_deux_roues_velo_pct']
)
plt.xlabel('Densite (hab/km²)')
plt.ylabel('Usage du velo (%)')
plt.title('Densite urbaine vs Usage du velo a Montpellier')
plt.show()
```

## Source des donnees

Donnees issues de: **Diagnostic Socio-Demographique de Montpellier**
- Source: Ville de Montpellier
- Annee de reference: 2009 (INSEE)
- Traitement: Script Python automatise

## Notes techniques

- Les fichiers CSV utilisent l'encodage UTF-8 avec BOM pour compatibilite Excel
- Les valeurs manquantes sont representees par NaN
- Les pourcentages sont en decimales (0.10 = 10%)
- Les nombres utilisent le point comme separateur decimal

## Regeneration des donnees

Pour regenerer ces fichiers a partir des donnees brutes:

```bash
python scripts/process_montpellier_data.py
```

---

**Date de generation:** Decembre 2025
**Outil:** Claude Code
**Format:** CSV (UTF-8 avec BOM)
