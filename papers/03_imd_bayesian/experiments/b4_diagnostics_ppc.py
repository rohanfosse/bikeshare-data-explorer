"""B4-B5 -- Multi-chain MCMC diagnostics and posterior predictive checks.

Runs four MH chains from dispersed initial conditions on the
Bayesian IMD-3 model of b1. Computes for each scalar parameter
of interest:

  - the split-Rhat (Gelman-Rubin diagnostic, Vehtari et al. 2021),
  - the bulk and tail effective sample sizes (ESS),
  - the autocorrelation function of one chain,
  - the trace plot of all four chains.

Then performs posterior predictive checks on the FUB and EMP
reference series:

  - draws replicated y from the posterior,
  - computes Bayesian posterior predictive p-values on four
    test statistics (mean, sd, min, max),
  - compares the replicated distribution to the observed.

Outputs:
    outputs/b4_diagnostics.json
    outputs/b4_trace_plots.pdf
    outputs/b4_autocorrelation.pdf
    outputs/b5_ppc.pdf
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = next(
    p for p in [HERE, *HERE.parents]
    if (p / ".git").exists() or (p / "CITATION.cff").exists()
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "papers" / "02_imd" / "experiments"))
sys.path.insert(0, str(ROOT / "papers" / "03_imd_bayesian" / "experiments"))

from _common import load_panel  # noqa: E402
from utils.data_loader import load_stations  # noqa: E402

# Import the B1 model machinery
import importlib.util
B1_PATH = ROOT / "papers" / "03_imd_bayesian" / "experiments" / "b1_bayesian_imd.py"
spec = importlib.util.spec_from_file_location("b1", B1_PATH)
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

N_CHAINS = 4
N_BURN = 3000
N_KEEP = 4000


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def split_rhat(chains: np.ndarray) -> float:
    """Split-Rhat (Vehtari et al. 2021).

    chains shape: (n_chains, n_iter). Splits each chain in half so
    that the effective number of chains is 2 * n_chains.
    """
    n_chains, n_iter = chains.shape
    half = n_iter // 2
    if half < 4:
        return float("nan")
    splits = np.concatenate([chains[:, :half], chains[:, half:2*half]], axis=0)
    m, n = splits.shape  # m = 2*n_chains, n = half
    chain_means = splits.mean(axis=1)
    chain_vars = splits.var(axis=1, ddof=1)
    overall_mean = chain_means.mean()
    B = n * np.var(chain_means, ddof=1)        # between-chain
    W = chain_vars.mean()                       # within-chain
    if W <= 0:
        return float("nan")
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def effective_sample_size(chains: np.ndarray) -> float:
    """Bulk ESS via auto-correlation truncation (Geyer's IPS).

    Simple implementation: sum auto-correlation lags until
    consecutive pair sum is negative.
    """
    n_chains, n_iter = chains.shape
    total_n = n_chains * n_iter
    # Concatenate chains for global AC
    x = chains.flatten() - chains.flatten().mean()
    var = x.var()
    if var <= 0:
        return float("nan")
    max_lag = min(2000, n_iter // 2)
    ac = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            ac[lag] = 1.0
        else:
            ac[lag] = np.mean(x[:-lag] * x[lag:]) / var
    # Geyer's monotone positive sequence
    rho_sum = ac[0]
    for k in range(0, max_lag - 1, 2):
        pair = ac[k+1] + (ac[k+2] if k+2 < max_lag else 0.0)
        if pair < 0:
            break
        rho_sum += 2 * (ac[k+1] + (ac[k+2] if k+2 < max_lag else 0.0))
    if rho_sum <= 0:
        return float("nan")
    return float(total_n / rho_sum)


def autocorrelation(x: np.ndarray, max_lag: int = 200) -> np.ndarray:
    x = x - x.mean()
    var = x.var()
    if var <= 0:
        return np.full(max_lag, np.nan)
    n = len(x)
    ac = np.zeros(max_lag)
    for lag in range(max_lag):
        if lag == 0:
            ac[lag] = 1.0
        else:
            if n - lag < 5:
                ac[lag] = np.nan
            else:
                ac[lag] = np.mean(x[:-lag] * x[lag:]) / var
    return ac


# ---------------------------------------------------------------------------
# Multi-chain MH driver
# ---------------------------------------------------------------------------

def mh_one_chain(
    component_city_means: np.ndarray,
    fub: np.ndarray,
    emp: np.ndarray,
    seed: int,
    init_disperse: float = 1.0,
    n_burn: int = N_BURN,
    n_keep: int = N_KEEP,
) -> dict:
    rng = np.random.default_rng(seed)
    state = {
        "z": rng.normal(0, init_disperse, size=b1.K),
        "alpha": rng.normal(0, 0.5, size=2),
        "beta": rng.normal(0.3, 0.3, size=2),
        "log_sigma": rng.normal(0, 0.2, size=2),
    }
    scales = {k: float(v) for k, v in b1.PROPOSAL_SCALE_INIT.items()}
    # Swap b1's RNG so the global sampler uses this chain's seed
    b1_rng_backup = b1.RNG
    b1.RNG = rng
    try:
        log_p_current = b1.log_posterior(state, component_city_means, fub, emp)
        samples = {k: [] for k in state}
        accepts = {k: 0 for k in state}
        proposals = {k: 0 for k in state}
        total_iter = n_burn + n_keep
        adapt_every = 200
        for it in range(total_iter):
            for key in ("z", "alpha", "beta", "log_sigma"):
                prop_state = {k: v.copy() if hasattr(v, "copy") else v
                              for k, v in state.items()}
                prop_state[key] = state[key] + rng.normal(0, scales[key],
                                                            size=state[key].shape)
                proposals[key] += 1
                try:
                    log_p_prop = b1.log_posterior(prop_state, component_city_means, fub, emp)
                except Exception:
                    continue
                if np.log(rng.uniform()) < log_p_prop - log_p_current:
                    state = prop_state
                    log_p_current = log_p_prop
                    accepts[key] += 1
            if it < n_burn and (it + 1) % adapt_every == 0:
                for k in scales:
                    rate = accepts[k] / max(proposals[k], 1)
                    if rate < 0.17:
                        scales[k] *= 0.7
                    elif rate < 0.34:
                        scales[k] *= 0.9
                    elif rate > 0.6:
                        scales[k] *= 1.3
                    elif rate > 0.4:
                        scales[k] *= 1.1
                    accepts[k] = 0
                    proposals[k] = 0
            if it >= n_burn:
                for k in state:
                    samples[k].append(state[k].copy())
    finally:
        b1.RNG = b1_rng_backup
    return {k: np.array(v) for k, v in samples.items()}


def main() -> None:
    log.info("Loading panel and stations...")
    panel = load_panel()
    stations = load_stations()
    dock = b1.normalise_components(stations)
    city_means = dock.groupby("city")[["M_norm", "I_norm", "T_norm"]].mean()
    cmm = city_means.reindex(panel.cities).fillna(city_means.median())
    component_city_means = cmm.to_numpy()
    fub = b1.standardise(panel.fub)
    emp = b1.standardise(np.log1p(panel.emp))

    log.info("Running %d chains, %d burn + %d keep each...",
             N_CHAINS, N_BURN, N_KEEP)
    chain_results = []
    for c in range(N_CHAINS):
        log.info("  chain %d ...", c + 1)
        seed = 2026 + c
        ch = mh_one_chain(component_city_means, fub, emp,
                          seed=seed, init_disperse=1.5)
        chain_results.append(ch)

    # Stack chains for each parameter
    z_all = np.stack([c["z"] for c in chain_results], axis=0)        # (n_chains, n_iter, K)
    alpha_all = np.stack([c["alpha"] for c in chain_results], axis=0)
    beta_all = np.stack([c["beta"] for c in chain_results], axis=0)
    log_sigma_all = np.stack([c["log_sigma"] for c in chain_results], axis=0)

    # Compute w_M from z for each draw
    w_all = np.zeros((N_CHAINS, N_KEEP, b1.K))
    for ci in range(N_CHAINS):
        for it in range(N_KEEP):
            w_all[ci, it] = b1.softmax_with_floor(z_all[ci, it])

    # Diagnostics for each scalar
    diagnostics = {}
    for name, arr in [
        ("w_M", w_all[..., 0]),
        ("w_I", w_all[..., 1]),
        ("w_T", w_all[..., 2]),
        ("alpha_FUB", alpha_all[..., 0]),
        ("alpha_EMP", alpha_all[..., 1]),
        ("beta_FUB", beta_all[..., 0]),
        ("beta_EMP", beta_all[..., 1]),
        ("log_sigma_FUB", log_sigma_all[..., 0]),
        ("log_sigma_EMP", log_sigma_all[..., 1]),
    ]:
        rhat = split_rhat(arr)
        ess = effective_sample_size(arr)
        diagnostics[name] = {"split_rhat": rhat, "ess_bulk": ess,
                              "n_total": int(N_CHAINS * N_KEEP)}
        log.info("  %-15s  R-hat = %.3f   ESS = %.0f   "
                 "(total = %d)", name, rhat, ess, N_CHAINS * N_KEEP)

    # Posterior predictive checks for FUB and EMP
    log.info("\nPosterior predictive checks...")
    n_draw = 1000
    # Flatten chains
    z_flat = z_all.reshape(-1, b1.K)
    alpha_flat = alpha_all.reshape(-1, 2)
    beta_flat = beta_all.reshape(-1, 2)
    log_sigma_flat = log_sigma_all.reshape(-1, 2)
    sigma_flat = np.exp(log_sigma_flat)
    idx = np.random.default_rng(2026).choice(
        z_flat.shape[0], size=n_draw, replace=False,
    )
    z_sub = z_flat[idx]
    alpha_sub = alpha_flat[idx]
    beta_sub = beta_flat[idx]
    sigma_sub = sigma_flat[idx]

    ppc = {}
    for j, (name, y) in enumerate([("FUB", fub), ("EMP", emp)]):
        mask = np.isfinite(y)
        y_obs = y[mask]
        rep = np.zeros((n_draw, mask.sum()))
        for d in range(n_draw):
            w = b1.softmax_with_floor(z_sub[d])
            imd_city = component_city_means @ w
            imd_std = (imd_city - imd_city.mean()) / imd_city.std()
            mu = alpha_sub[d, j] + beta_sub[d, j] * imd_std[mask]
            rep[d] = np.random.default_rng(d + j).normal(mu, sigma_sub[d, j])
        # Test statistics
        ppc_data = {}
        for stat_name, stat in [
            ("mean", np.mean),
            ("sd", np.std),
            ("min", np.min),
            ("max", np.max),
        ]:
            obs_val = float(stat(y_obs))
            rep_vals = np.array([stat(rep[d]) for d in range(n_draw)])
            bayes_p = float((rep_vals >= obs_val).mean())
            ppc_data[stat_name] = {
                "observed": obs_val,
                "replicate_mean": float(rep_vals.mean()),
                "replicate_q025": float(np.percentile(rep_vals, 2.5)),
                "replicate_q975": float(np.percentile(rep_vals, 97.5)),
                "bayes_p_value": bayes_p,
            }
            log.info("  %s  %s  observed = %.3f   "
                     "replicate = %.3f [%.3f, %.3f]   "
                     "Bayes p = %.3f",
                     name, stat_name, obs_val,
                     rep_vals.mean(),
                     np.percentile(rep_vals, 2.5),
                     np.percentile(rep_vals, 97.5),
                     bayes_p)
        ppc[name] = ppc_data

    results = {
        "n_chains": N_CHAINS,
        "n_burn": N_BURN,
        "n_keep_per_chain": N_KEEP,
        "diagnostics": diagnostics,
        "posterior_predictive": ppc,
    }
    out_json = OUT_DIR / "b4_diagnostics.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_json)

    # Figure: trace plots for w_M, beta_FUB, beta_EMP
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 6.5), sharex=True)
    colors = ["#1F3A6B", "#A8201A", "#5B7E4F", "#B07A30"]
    for ax, (name, arr) in zip(axes, [
        ("w_M", w_all[..., 0]),
        ("beta_FUB", beta_all[..., 0]),
        ("beta_EMP", beta_all[..., 1]),
    ]):
        for c in range(N_CHAINS):
            ax.plot(arr[c], color=colors[c], linewidth=0.5, alpha=0.7)
        ax.set_ylabel(name)
        rhat = diagnostics[name]["split_rhat"]
        ess = diagnostics[name]["ess_bulk"]
        ax.text(0.99, 0.02, f"R-hat = {rhat:.3f}   ESS = {ess:.0f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#202020",
                bbox={"facecolor": "white", "edgecolor": "#D0D0D0",
                      "alpha": 0.9, "pad": 3})
        ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    axes[-1].set_xlabel("MCMC iteration (post-burn)")
    fig.suptitle("B4: trace plots over four MH chains", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b4_trace_plots.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote b4_trace_plots.pdf")

    # Figure: autocorrelation
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, (name, arr) in zip(axes, [
        ("w_M", w_all[..., 0]),
        ("beta_FUB", beta_all[..., 0]),
        ("beta_EMP", beta_all[..., 1]),
    ]):
        ac = autocorrelation(arr[0], max_lag=200)
        ax.bar(np.arange(len(ac)), ac, color="#1F3A6B", width=1.0,
                edgecolor="none")
        ax.axhline(0, color="#404040", linewidth=0.6)
        ax.set_xlabel("Lag")
        ax.set_title(name, fontsize=10)
        ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    axes[0].set_ylabel("Autocorrelation")
    fig.suptitle("B4: autocorrelation (chain 1)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b4_autocorrelation.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote b4_autocorrelation.pdf")

    # Figure: PPC
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.2))
    for j, (name, y) in enumerate([("FUB", fub), ("EMP", emp)]):
        mask = np.isfinite(y)
        y_obs = y[mask]
        for s, (stat_name, stat) in enumerate([
            ("mean", np.mean), ("sd", np.std),
            ("min", np.min), ("max", np.max),
        ]):
            ax = axes[j, s]
            obs_val = stat(y_obs)
            rep_vals = []
            for d in range(n_draw):
                w = b1.softmax_with_floor(z_sub[d])
                imd_city = component_city_means @ w
                imd_std = (imd_city - imd_city.mean()) / imd_city.std()
                mu = alpha_sub[d, j] + beta_sub[d, j] * imd_std[mask]
                rep = np.random.default_rng(d + j + 100).normal(mu, sigma_sub[d, j])
                rep_vals.append(stat(rep))
            ax.hist(rep_vals, bins=30, color="#1F3A6B",
                     edgecolor="white", linewidth=0.4, alpha=0.85)
            ax.axvline(obs_val, color="#A8201A", linewidth=1.2,
                        label=f"obs = {obs_val:.2f}")
            bp = (np.array(rep_vals) >= obs_val).mean()
            ax.text(0.98, 0.95, f"Bayes p = {bp:.2f}",
                     transform=ax.transAxes, ha="right", va="top",
                     fontsize=8, color="#202020",
                     bbox={"facecolor": "white", "edgecolor": "#D0D0D0",
                           "alpha": 0.9, "pad": 3})
            ax.set_title(f"{name}: {stat_name}", fontsize=9)
            ax.grid(True, axis="y", color="#E5E5E5", linewidth=0.5)
    fig.suptitle("B5: posterior predictive checks on FUB and EMP",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "b5_ppc.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote b5_ppc.pdf")


if __name__ == "__main__":
    main()
