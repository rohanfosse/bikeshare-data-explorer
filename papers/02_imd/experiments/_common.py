"""Shared helpers for IMD validation experiments E1-E8.

Loads the dock-based panel, builds the component matrix and the
behavioural reference series (FUB and EMP), and exposes the
supervised differential-evolution calibration that the paper uses.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.optimize import differential_evolution

os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

_HERE = Path(__file__).resolve()
ROOT = next(
    p for p in [_HERE, *_HERE.parents]
    if (p / ".git").exists() or (p / "CITATION.cff").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_loader import compute_imd_cities, load_stations  # noqa: E402

W_MIN = 0.05
COMPONENTS = ("M_multi", "I_infra", "S_securite", "T_topo")

# Approximate NUTS-2 regional grouping for the cities in the panel.
# Used by E1's spatial-block CV. Cities not listed fall back to "OTHER".
CITY_TO_NUTS2: dict[str, str] = {
    # Île-de-France
    "Paris": "FR10",
    "Île-de-France (Cergy)": "FR10",
    # Auvergne-Rhône-Alpes (FRK)
    "Lyon": "FRK", "Saint-Étienne": "FRK", "Grenoble": "FRK",
    "Annecy": "FRK", "Valence": "FRK", "Vichy": "FRK",
    "Clermont-Ferrand": "FRK",
    # Provence-Alpes-Côte d'Azur (FRL)
    "Marseille": "FRL", "Nice": "FRL", "Avignon": "FRL", "Aix-en-Provence": "FRL",
    "Toulon": "FRL",
    # Nouvelle-Aquitaine (FRI)
    "Bordeaux": "FRI", "Pau": "FRI", "La Rochelle": "FRI", "Bayonne": "FRI",
    "Poitiers": "FRI", "Limoges": "FRI",
    # Occitanie (FRJ)
    "Toulouse": "FRJ", "Montpellier": "FRJ", "Carcassonne": "FRJ",
    "Tarbes": "FRJ", "Perpignan": "FRJ", "Nimes": "FRJ", "Béziers": "FRJ",
    # Pays de la Loire (FRG)
    "Nantes": "FRG", "Angers": "FRG", "Le Mans": "FRG",
    # Bretagne (FRH)
    "Rennes": "FRH", "Brest": "FRH", "Lorient": "FRH", "Saint-Brieuc": "FRH",
    "Vannes": "FRH",
    # Normandie (FRD)
    "Caen": "FRD", "Rouen": "FRD", "Le Havre": "FRD",
    # Hauts-de-France (FRE)
    "Lille": "FRE", "Amiens": "FRE", "Laon": "FRE",
    # Grand Est (FRF)
    "Strasbourg": "FRF", "Mulhouse": "FRF", "Metz": "FRF", "Nancy": "FRF",
    "Reims": "FRF", "Troyes": "FRF", "Épinal": "FRF", "Longwy": "FRF",
    # Bourgogne-Franche-Comté (FRC)
    "Dijon": "FRC", "Besançon": "FRC", "Belfort": "FRC",
    # Centre-Val de Loire (FRB)
    "Tours": "FRB", "Orléans": "FRB", "Bourges": "FRB",
}


@dataclass(frozen=True)
class Panel:
    """Calibration-ready panel for the IMD validation experiments."""

    cities: list[str]
    components: np.ndarray  # shape (n, 4): M, I, S, T (already normalised on [0,1])
    fub: np.ndarray         # shape (n,), NaN where missing
    emp: np.ndarray         # shape (n,), NaN where missing
    imd: np.ndarray         # shape (n,), current published IMD scaled to [0,100]
    socio: pd.DataFrame     # per-city socio-economic features

    @property
    def n(self) -> int:
        return len(self.cities)

    def with_indices(self, idx: np.ndarray) -> "Panel":
        return Panel(
            cities=[self.cities[i] for i in idx],
            components=self.components[idx],
            fub=self.fub[idx],
            emp=self.emp[idx],
            imd=self.imd[idx],
            socio=self.socio.iloc[idx].reset_index(drop=True),
        )


def load_panel() -> Panel:
    """Build the analytical panel used across all validation experiments."""
    stations = load_stations()
    imd = compute_imd_cities(stations)

    fub = pd.read_csv(
        ROOT / "data" / "external" / "mobility_sources"
        / "fub_barometre_2023_city_scores.csv"
    )
    emp = pd.read_csv(
        ROOT / "data" / "external" / "mobility_sources"
        / "emp_2019_city_modal_share.csv"
    )

    df = imd.merge(fub, on="city", how="left").merge(emp, on="city", how="left")
    cities = df["city"].astype(str).tolist()

    components = df[list(COMPONENTS)].to_numpy(dtype=float)
    fub_arr = df["fub_score_2023"].to_numpy(dtype=float)
    emp_arr = df["emp_part_velo_2019"].to_numpy(dtype=float)
    imd_arr = df["IMD"].to_numpy(dtype=float)
    socio = df[[
        "city", "revenu_median_uc", "gini_revenu",
        "part_menages_voit0", "part_velo_travail",
    ]].copy()

    return Panel(
        cities=cities,
        components=components,
        fub=fub_arr,
        emp=emp_arr,
        imd=imd_arr,
        socio=socio,
    )


def _softmax_to_simplex(z: np.ndarray, w_min: float = W_MIN) -> np.ndarray:
    """Map unconstrained z to {w: sum(w)=1, w_k >= w_min} via softmax.

    The construction w = w_min + (1 - K * w_min) * softmax(z) guarantees
    feasibility for any z when K * w_min < 1 (here K=4, w_min=0.05).
    """
    z_shift = z - np.max(z)
    soft = np.exp(z_shift)
    soft = soft / soft.sum()
    k = len(z)
    return w_min + (1.0 - k * w_min) * soft


def calibration_objective_z(
    z: np.ndarray,
    components: np.ndarray,
    fub: np.ndarray,
    emp: np.ndarray,
) -> float:
    """Return -mean Spearman against FUB and EMP under softmax weights."""
    w = _softmax_to_simplex(z)
    composite = components @ w
    mask_fub = np.isfinite(fub)
    mask_emp = np.isfinite(emp)
    rho_fub = (
        sp_stats.spearmanr(composite[mask_fub], fub[mask_fub]).statistic
        if mask_fub.sum() >= 5 else 0.0
    )
    rho_emp = (
        sp_stats.spearmanr(composite[mask_emp], emp[mask_emp]).statistic
        if mask_emp.sum() >= 5 else 0.0
    )
    return -0.5 * (rho_fub + rho_emp)


def calibrate_weights(
    components: np.ndarray,
    fub: np.ndarray,
    emp: np.ndarray,
    *,
    seed: int = 42,
    maxiter: int = 200,
    popsize: int = 15,
) -> tuple[np.ndarray, float]:
    """Run scipy DE to find weights maximising the calibration objective.

    The four-dimensional weight vector is reparameterised through softmax
    so that the simplex + lower-bound constraint is always satisfied.
    """
    bounds = [(-5.0, 5.0)] * 4
    result = differential_evolution(
        calibration_objective_z,
        bounds=bounds,
        args=(components, fub, emp),
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-7,
        polish=True,
        strategy="best1bin",
    )
    w_star = _softmax_to_simplex(result.x)
    obj_star = -result.fun
    return w_star, obj_star


def composite_score(weights: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Compute the IMD composite from weights and components."""
    return components @ weights * 100.0
