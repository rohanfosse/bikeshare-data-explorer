"""E8 -- Spatial-diffusion / autocorrelation analysis.

Tests whether IMD scores exhibit spatial dependence: do geographically
close cities have similar IMD? Computes Moran's I against two weight
matrices (k-nearest-neighbour and inverse-distance) and runs a simple
spatial-lag OLS to estimate the contagion coefficient rho.

Outputs:
    outputs/e8_results.json
    outputs/e8_moran_scatter.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression

from _common import load_panel, ROOT
from utils.data_loader import load_stations

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _city_centroids(panel_cities: list[str]) -> pd.DataFrame:
    """Compute geographic centroid (lat, lon) per city."""
    stations = load_stations()
    dock = stations[stations["station_type"] == "docked_bike"]
    centroids = (
        dock.groupby("city")[["lat", "lon"]]
        .mean()
        .reset_index()
    )
    return centroids[centroids["city"].isin(panel_cities)].reset_index(drop=True)


def _knn_weights(coords: np.ndarray, k: int = 5) -> np.ndarray:
    """K-nearest-neighbour row-normalised spatial weights matrix.

    Diagonal is zero by construction. Uses chordal distance on (lat, lon).
    """
    n = coords.shape[0]
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    w = np.zeros((n, n))
    for i in range(n):
        nn_idx = np.argpartition(d[i], k)[:k]
        w[i, nn_idx] = 1.0
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return w / row_sums


def _inv_distance_weights(coords: np.ndarray, cutoff_km: float = 200.0) -> np.ndarray:
    """Inverse-distance weights with a hard cutoff."""
    n = coords.shape[0]
    # Rough km conversion: 1 deg lat ~ 111 km, 1 deg lon ~ 78 km at 47°N
    scale = np.array([111.0, 78.0])
    coords_km = coords * scale
    d = cdist(coords_km, coords_km)
    np.fill_diagonal(d, np.inf)
    w = np.where(d < cutoff_km, 1.0 / d, 0.0)
    np.fill_diagonal(w, 0.0)
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return w / row_sums


def _morans_i(x: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """Compute Moran's I, its expectation under randomisation, and z-score."""
    n = len(x)
    x_dev = x - x.mean()
    num = (w * np.outer(x_dev, x_dev)).sum()
    denom = (x_dev ** 2).sum()
    S0 = w.sum()
    morans = (n / S0) * (num / denom)
    expected_i = -1.0 / (n - 1)

    # Variance under randomisation (Cliff-Ord)
    S1 = 0.5 * ((w + w.T) ** 2).sum()
    S2 = ((w.sum(axis=1) + w.sum(axis=0)) ** 2).sum()
    m2 = ((x - x.mean()) ** 2).sum() / n
    m4 = ((x - x.mean()) ** 4).sum() / n
    b2 = m4 / (m2 ** 2)
    a = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2)
    b = b2 * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)
    var_i = (a - b) / ((n - 1) * (n - 2) * (n - 3) * S0 ** 2) - expected_i ** 2
    var_i = max(var_i, 1e-12)
    z = (morans - expected_i) / np.sqrt(var_i)
    return float(morans), float(expected_i), float(z)


def _spatial_lag_ols(
    y: np.ndarray, w: np.ndarray, x_socio: np.ndarray | None = None,
) -> dict:
    """Spatial-lag OLS: y = rho * W y + X beta + eps (approximate, not ML)."""
    wy = w @ y
    if x_socio is not None:
        features = np.column_stack([wy.reshape(-1, 1), x_socio])
    else:
        features = wy.reshape(-1, 1)
    model = LinearRegression().fit(features, y)
    rho_lag = float(model.coef_[0])
    r2 = float(model.score(features, y))
    return {"rho_spatial_lag": rho_lag, "r2": r2,
            "n_features": int(features.shape[1])}


def main() -> None:
    log.info("Loading panel + centroids...")
    panel = load_panel()
    centroids = _city_centroids(panel.cities)
    df = pd.DataFrame({"city": panel.cities, "IMD": panel.imd}).merge(
        centroids, on="city", how="inner"
    ).reset_index(drop=True)
    log.info("  n=%d cities with valid centroids", len(df))

    coords = df[["lat", "lon"]].to_numpy()
    y_imd = df["IMD"].to_numpy()

    results: dict = {"n_cities": int(len(df))}

    # Moran's I under two weight matrices
    for w_name, w in [
        ("knn5", _knn_weights(coords, k=5)),
        ("inv_distance_200km", _inv_distance_weights(coords, cutoff_km=200.0)),
    ]:
        i_val, e_i, z = _morans_i(y_imd, w)
        # Permutation p-value
        rng = np.random.default_rng(42)
        n_perm = 999
        sim = np.empty(n_perm)
        for s in range(n_perm):
            perm = rng.permutation(y_imd)
            sim[s], _, _ = _morans_i(perm, w)
        pval = (np.abs(sim) >= np.abs(i_val)).mean()

        lag_ols = _spatial_lag_ols(y_imd, w)
        results[w_name] = {
            "morans_i": i_val,
            "expected_i": e_i,
            "z_score": z,
            "p_perm": float(pval),
            "spatial_lag_ols": lag_ols,
        }
        log.info("Weights %s: Moran's I = %+.4f (z = %.2f, p = %.3f), "
                 "spatial-lag rho = %+.3f",
                 w_name, i_val, z, pval, lag_ols["rho_spatial_lag"])

    out_json = OUT_DIR / "e8_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Moran scatter (knn5)
    w = _knn_weights(coords, k=5)
    wy = w @ y_imd
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    ax.scatter(y_imd, wy, s=34, color="#1F3A6B", alpha=0.78,
               edgecolor="white", linewidth=0.5)
    coeff = np.polyfit(y_imd, wy, 1)
    line_x = np.array([y_imd.min(), y_imd.max()])
    ax.plot(line_x, np.polyval(coeff, line_x),
            color="#7A7A7A", linewidth=0.9, linestyle="--", alpha=0.8)
    ax.axhline(wy.mean(), color="#7A7A7A", linewidth=0.4, linestyle=":",
               alpha=0.5)
    ax.axvline(y_imd.mean(), color="#7A7A7A", linewidth=0.4, linestyle=":",
               alpha=0.5)
    ax.set_xlabel("IMD of the city ($y_i$)")
    ax.set_ylabel("Mean IMD of 5 nearest cities ($W y$)")
    i_val, _, z = _morans_i(y_imd, w)
    ax.text(0.04, 0.96,
            f"Moran's I = {i_val:+.3f}\nz = {z:.2f}\n(k = 5 nearest neighbours)",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="#404040",
            bbox={"facecolor": "white", "edgecolor": "none",
                  "alpha": 0.85, "pad": 3})
    ax.grid(True, axis="both", color="#E5E5E5", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e8_moran_scatter.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e8_moran_scatter.pdf")


if __name__ == "__main__":
    main()
