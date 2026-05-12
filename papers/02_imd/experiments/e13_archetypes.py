"""E13 -- Data-driven city archetypes from the component matrix.

The median-split quadrants of Section "Four urban archetypes" rely on
an arbitrary cut of the marginal distributions. This experiment
recovers an unsupervised typology of the panel through k-means
clustering on the four-dimensional component space, with the optimal
k selected by the silhouette score on k in {2, 3, 4, 5, 6}. We then
characterise each cluster by its mean component profile, its mean
IMD, its dominant geography, and the share of cities that fall in
the published mobility-desert set.

The k-means is run with k-means++ initialisation, 200 restarts and
fixed seed for reproducibility.

Outputs:
    outputs/e13_results.json
    outputs/e13_archetypes.pdf
    outputs/e13_silhouette.pdf
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from _common import COMPONENTS, load_panel

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Reference set of mobility deserts from the paper (Section sec:results-deserts)
PUBLISHED_DESERTS = {
    "Laon", "Amiens", "Troyes", "Tarbes", "Lille", "Carcassonne",
    "Pau", "Avignon", "Valence", "Clermont-Ferrand", "Épinal", "Vichy",
}


def main() -> None:
    log.info("Loading panel...")
    panel = load_panel()
    log.info("  n=%d cities", panel.n)

    x_raw = panel.components.copy()
    scaler = StandardScaler()
    x = scaler.fit_transform(x_raw)

    ks = list(range(2, 7))
    silhouettes = []
    inertias = []
    fits = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=200, random_state=42)
        labels = km.fit_predict(x)
        sil = float(silhouette_score(x, labels))
        silhouettes.append(sil)
        inertias.append(float(km.inertia_))
        fits[k] = (km, labels, sil)
        log.info("  k=%d  silhouette=%.3f  inertia=%.2f",
                 k, sil, km.inertia_)
    k_opt_silhouette = ks[int(np.argmax(silhouettes))]
    log.info("Optimal k by silhouette = %d", k_opt_silhouette)
    # We also report k=4 to contrast the data-driven typology with the
    # median-split quadrants of Section "Four urban archetypes".
    k_opt = 4
    log.info("Reporting k=%d to mirror the four-quadrant analysis", k_opt)

    km, labels, sil = fits[k_opt]

    # Cluster characterisation
    archetypes = []
    for c in range(k_opt):
        mask = labels == c
        cluster_cities = [panel.cities[i] for i in range(panel.n) if mask[i]]
        component_means = x_raw[mask].mean(axis=0)
        imd_mean = float(panel.imd[mask].mean())
        income_mean = float(
            panel.socio["revenu_median_uc"][mask].mean()
        )
        income_med = float(
            panel.socio["revenu_median_uc"][mask].median()
        )
        share_desert = float(
            len(set(cluster_cities) & PUBLISHED_DESERTS) / max(len(cluster_cities), 1)
        )
        # Auto-label the archetype from the dominant component(s)
        sorted_idx = np.argsort(-component_means)
        dominant = [COMPONENTS[i].split("_")[0]
                    for i in sorted_idx[:2]]
        archetypes.append({
            "cluster_id": int(c),
            "n_cities": int(mask.sum()),
            "imd_mean": imd_mean,
            "income_mean": income_mean,
            "income_median": income_med,
            "component_means": {
                COMPONENTS[i]: float(component_means[i])
                for i in range(4)
            },
            "dominant_components": dominant,
            "desert_share": share_desert,
            "cities": cluster_cities,
        })

    archetypes.sort(key=lambda r: -r["imd_mean"])

    log.info("Archetypes at k=%d (sorted by mean IMD):", k_opt)
    for a in archetypes:
        log.info(
            "  Cluster #%d (n=%d, IMD=%.1f, income=%.0f)",
            a["cluster_id"], a["n_cities"], a["imd_mean"], a["income_median"],
        )
        log.info("    dominant: %s   desert share: %.2f",
                 "+".join(a["dominant_components"]), a["desert_share"])
        log.info("    example cities: %s", ", ".join(a["cities"][:5]))

    results = {
        "n_cities": int(panel.n),
        "k_reported": int(k_opt),
        "k_optimal_silhouette": int(k_opt_silhouette),
        "silhouette_at_k_reported": float(sil),
        "silhouettes_by_k": dict(zip([str(k) for k in ks], silhouettes)),
        "inertias_by_k": dict(zip([str(k) for k in ks], inertias)),
        "archetypes": archetypes,
    }
    out_json = OUT_DIR / "e13_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # ---- Figure 1: silhouette and inertia ---------------------------
    fig, ax1 = plt.subplots(figsize=(5.4, 3.0))
    ax2 = ax1.twinx()
    ax1.plot(ks, silhouettes, "o-", color="#1F3A6B",
             label="Silhouette", linewidth=1.2, markersize=6)
    ax2.plot(ks, inertias, "s--", color="#A8201A",
             label="Inertia", linewidth=1.0, markersize=5, alpha=0.7)
    ax1.set_xlabel("Number of clusters $k$")
    ax1.set_ylabel("Silhouette", color="#1F3A6B")
    ax2.set_ylabel("Inertia", color="#A8201A")
    ax1.set_xticks(ks)
    ax1.axvline(k_opt_silhouette, color="#404040", linewidth=0.6,
                linestyle=":", alpha=0.7)
    ax1.axvline(k_opt, color="#A8201A", linewidth=0.6,
                linestyle="--", alpha=0.7)
    ax1.grid(True, color="#E5E5E5", linewidth=0.5)
    fig.suptitle("Choice of the number of archetypes", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e13_silhouette.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e13_silhouette.pdf")

    # ---- Figure 2: radar-style profile of each archetype -------------
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    width = 0.78 / k_opt
    x = np.arange(4)
    colors = ["#1F3A6B", "#5B7E4F", "#B07A30", "#A8201A",
              "#7095C8", "#9A6F9C"][:k_opt]
    for i, a in enumerate(archetypes):
        vals = [a["component_means"][c] for c in COMPONENTS]
        ax.bar(
            x + (i - (k_opt - 1) / 2) * width,
            vals,
            width=width * 0.95,
            color=colors[i],
            edgecolor="white",
            linewidth=0.4,
            label=f"#{a['cluster_id']} (n={a['n_cities']}, IMD~{a['imd_mean']:.0f})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["M\nmultim.", "I\ninfra", "S\nsafety", "T\ntopo"],
                       fontsize=9)
    ax.set_ylabel("Normalised component value")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_title(f"k-means archetypes of the dock-based panel (k={k_opt})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "e13_archetypes.pdf",
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote e13_archetypes.pdf")


if __name__ == "__main__":
    main()
