"""SCHOTT -> SciGlass matching Version 2: adaptive coverage, weighted composition, uncertainty.

    python match_schott_sciglass_v2.py

Выход: output_v2/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Reuse loaders from v1
from match_schott_sciglass import (
    DEFAULT_SCHOTT_XLSX,
    DEFAULT_SCIGLASS_ZIP,
    K_NEIGHBORS,
    OXIDE_MOL_COLS,
    OXIDE_SUM_TARGET,
    OXIDE_SUM_TOL,
    PROPERTY_MAP,
    classify_glass_family,
    load_schott_catalog,
    setup_logging,
)

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "output_v2"

FEATURES_MODE_A = ["ND300", "NUD300"]
FEATURES_MODE_B = ["ND300", "NUD300", "DENSITY", "TG"]

# Явные + остаточные / групповые оксиды SciGlass
RESIDUAL_OXIDE_COLS = [
    "RO", "R2O", "R2O3", "RO2", "RO3", "RO4", "R2O5",
    "RF", "RF2", "RF3", "RF4", "RF5",
    "RHal", "RHal2", "RHal3", "RHal4", "RHal5", "RHaln",
    "RmOn", "RmNn",
]

RECON_METHODS = ("exp_weight", "idw", "cluster_kmeans")


def all_oxide_columns(available: list[str]) -> list[str]:
    primary = [c for c in OXIDE_MOL_COLS if c in available]
    residual = [c for c in RESIDUAL_OXIDE_COLS if c in available]
    return primary + [c for c in residual if c not in primary]


def load_sciglass_extended(zip_path: Path) -> pd.DataFrame:
    """Загрузка SciGlass с расширенным набором оксидов."""
    ox_cols = list(dict.fromkeys(OXIDE_MOL_COLS + RESIDUAL_OXIDE_COLS))
    props = list(dict.fromkeys(FEATURES_MODE_B))
    usecols = ["KOD", "GLASNO", *props, *ox_cols]

    with zipfile.ZipFile(zip_path) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                usecols=lambda c: c in usecols,
                low_memory=False,
            )

    df["sciglass_id"] = df["KOD"] * 100_000_000 + df["GLASNO"]
    for col in props + ox_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col in props:
                df.loc[df[col] == 0, col] = np.nan
    return df


def schott_to_sg_features(schott: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=schott.index)
    for sh, (sg, _) in PROPERTY_MAP.items():
        if sg in feature_list and sh in schott.columns:
            out[sg] = schott[sh].values
    return out


def filter_sciglass_pool(
    sciglass: pd.DataFrame,
    pool_features: list[str],
    mode: Literal["strict", "adaptive"],
    min_props_adaptive: int = 3,
) -> tuple[pd.DataFrame, np.ndarray]:
    """STRICT: все pool_features; ADAPTIVE: >= min_props из pool_features (обычно 4)."""
    feat = sciglass[pool_features].copy()
    valid = feat.notna()
    n_valid = valid.sum(axis=1).to_numpy()

    if mode == "strict":
        mask = n_valid == len(pool_features)
    else:
        mask = n_valid >= min(min_props_adaptive, len(pool_features))

    return sciglass.loc[mask].copy(), mask


def fit_zscore_params(sciglass: pd.DataFrame, feature_list: list[str]) -> dict[str, tuple[float, float]]:
    params = {}
    for col in feature_list:
        s = sciglass[col].dropna()
        params[col] = (float(s.mean()), float(s.std(ddof=0)) or 1.0)
    return params


def z_transform(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / (std if std > 1e-12 else 1.0)


def adaptive_distance_matrix(
    schott_row: pd.Series,
    sciglass_feat: pd.DataFrame,
    feature_list: list[str],
    z_params: dict[str, tuple[float, float]],
    n_target: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    d(x,y) = sqrt(mean((x_j-y_j)^2)) * sqrt(n_target / n_used)
    только по пересечению ненулевых z-признаков (векторизовано).
    """
    n_sg = len(sciglass_feat)
    f = len(feature_list)
    q = np.full(f, np.nan)
    for j, col in enumerate(feature_list):
        if col in schott_row.index and pd.notna(schott_row[col]):
            m, s = z_params[col]
            q[j] = z_transform(float(schott_row[col]), m, s)

    if np.all(np.isnan(q)):
        return np.full(n_sg, np.nan), np.zeros(n_sg, dtype=int)

    Z = np.zeros((n_sg, f))
    M = np.zeros((n_sg, f), dtype=bool)
    for j, col in enumerate(feature_list):
        m, s = z_params[col]
        coldata = sciglass_feat[col]
        M[:, j] = coldata.notna().to_numpy()
        fill = coldata.fillna(m).to_numpy()
        Z[:, j] = z_transform(fill, m, s)

    valid_q = ~np.isnan(q)
    valid = M & valid_q[np.newaxis, :]
    diff2 = (Z - q[np.newaxis, :]) ** 2
    diff2[~valid] = np.nan
    n_used = valid.sum(axis=1)
    mse = np.nanmean(diff2, axis=1)
    penalty = np.sqrt(n_target / np.maximum(n_used, 1))
    distances = np.sqrt(mse) * penalty
    distances[n_used == 0] = np.nan
    return distances, n_used


def composition_completeness(comp: pd.Series, oxide_cols: list[str]) -> float:
    ox = comp.reindex(oxide_cols).fillna(0)
    if (ox < -1e-6).any():
        return 0.0
    total = float(ox.sum())
    return float(np.clip(total / OXIDE_SUM_TARGET, 0.0, 1.0))


def is_physically_plausible(comp: pd.Series, oxide_cols: list[str]) -> bool:
    ox = comp.reindex(oxide_cols).fillna(0)
    if (ox < -1e-6).any():
        return False
    total = float(ox.sum())
    return abs(total - OXIDE_SUM_TARGET) <= OXIDE_SUM_TOL


def weighted_composition(
    neighbor_comps: pd.DataFrame,
    weights: np.ndarray,
    oxide_cols: list[str],
) -> pd.Series:
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() + 1e-12)
    mat = neighbor_comps[oxide_cols].fillna(0).to_numpy()
    return pd.Series(mat.T @ w, index=oxide_cols)


def reconstruct_exp(
    comps: pd.DataFrame,
    distances: np.ndarray,
    oxide_cols: list[str],
) -> tuple[pd.Series, float]:
    d = np.maximum(distances, 1e-9)
    alpha = 1.0 / (np.median(d) + 1e-6)
    w = np.exp(-alpha * d)
    return weighted_composition(comps, w, oxide_cols), float(alpha)


def reconstruct_idw(
    comps: pd.DataFrame,
    distances: np.ndarray,
    oxide_cols: list[str],
    eps: float = 1e-6,
) -> pd.Series:
    w = 1.0 / (distances + eps)
    return weighted_composition(comps, w, oxide_cols)


def reconstruct_cluster(
    comps: pd.DataFrame,
    distances: np.ndarray,
    oxide_cols: list[str],
    families: list[str],
) -> tuple[pd.Series, int]:
    """KMeans на составах соседей; кластер ближайшего соседа."""
    X = comps[oxide_cols].fillna(0).to_numpy()
    n = len(X)
    if n < 3:
        w = 1.0 / (distances + 1e-6)
        return weighted_composition(comps, w, oxide_cols), 1

    labels: np.ndarray
    k = 0
    try:
        import hdbscan

        labels = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1).fit_predict(X)
        k = len(set(labels)) - (1 if -1 in labels else 0)
    except ImportError:
        k = int(np.clip(2 + (n > 10), 2, min(4, n - 1)))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)

    nearest_cluster = labels[0]
    if nearest_cluster < 0:
        nearest_cluster = int(pd.Series(labels[labels >= 0]).mode().iloc[0]) if (labels >= 0).any() else 0
    idx = np.where(labels == nearest_cluster)[0]
    w = 1.0 / (distances[idx] + 1e-6)
    comp = weighted_composition(comps.iloc[idx], w, oxide_cols)
    return comp, k


def family_entropy(families: list[str]) -> float:
    if not families:
        return np.nan
    counts = pd.Series(families).value_counts(normalize=True)
    return float(scipy_entropy(counts.to_numpy()))


def composition_variance(comps: pd.DataFrame, oxide_cols: list[str]) -> float:
    if len(comps) < 2:
        return 0.0
    return float(comps[oxide_cols].fillna(0).var(ddof=1).mean())


def pick_top_neighbors(
    sciglass_pool: pd.DataFrame,
    distances: np.ndarray,
    k: int = K_NEIGHBORS,
) -> pd.DataFrame:
    valid = np.isfinite(distances)
    if not valid.any():
        return sciglass_pool.iloc[0:0].copy()
    idx = np.argsort(distances)
    idx = idx[valid[idx]][:k]
    out = sciglass_pool.iloc[idx].copy()
    out["_distance"] = distances[idx]
    out["_rank"] = np.arange(1, len(out) + 1)
    return out


def process_glass(
    glass_name: str,
    schott_props: pd.Series,
    sciglass_pool: pd.DataFrame,
    feature_list: list[str],
    z_params: dict[str, tuple[float, float]],
    oxide_cols: list[str],
    filter_mode: Literal["strict", "adaptive"],
    property_mode: Literal["A", "B"],
) -> dict[str, Any]:
    dist, n_used = adaptive_distance_matrix(
        schott_props,
        sciglass_pool[feature_list],
        feature_list,
        z_params,
        n_target=len(FEATURES_MODE_B),
    )
    neighbors = pick_top_neighbors(sciglass_pool, dist, K_NEIGHBORS)
    if neighbors.empty:
        return {"glass_name": glass_name, "error": "no_neighbors"}

    d = neighbors["_distance"].to_numpy()
    comps = neighbors[oxide_cols]
    families = [classify_glass_family(neighbors.iloc[i], oxide_cols) for i in range(len(neighbors))]

    recon = {}
    for method in RECON_METHODS:
        if method == "exp_weight":
            comp, alpha = reconstruct_exp(comps, d, oxide_cols)
            recon[method] = {"composition": comp, "alpha": alpha}
        elif method == "idw":
            recon[method] = {"composition": reconstruct_idw(comps, d, oxide_cols)}
        else:
            comp, nk = reconstruct_cluster(comps, d, oxide_cols, families)
            recon[method] = {"composition": comp, "n_clusters": nk}

    # Основной результат — IDW (устойчивее mean)
    main_comp = recon["idw"]["composition"]
    comp_var = composition_variance(comps, oxide_cols)
    d_mean = float(np.mean(d))
    unc = comp_var * d_mean

    return {
        "glass_name": glass_name,
        "filter_mode": filter_mode,
        "property_mode": property_mode,
        "n_pool": len(sciglass_pool),
        "n_neighbors": len(neighbors),
        "distance_first": float(d[0]),
        "distance_mean_top20": d_mean,
        "composition_variance": comp_var,
        "family_entropy": family_entropy(families),
        "effective_neighbors": int(np.sum(d < d_mean * 2)),
        "uncertainty_score": unc,
        "composition_completeness": composition_completeness(main_comp, oxide_cols),
        "physically_plausible": is_physically_plausible(main_comp, oxide_cols),
        "matched_sciglass_id": int(neighbors.iloc[0]["sciglass_id"]),
        "glass_family_mode": pd.Series(families).mode().iloc[0] if families else "uncertain",
        "predicted_composition": main_comp,
        "reconstructions": recon,
        "neighbors": neighbors,
        "n_used_mean": float(neighbors[feature_list].notna().sum(axis=1).mean()) if feature_list else np.nan,
    }


def comp_to_str(comp: pd.Series, threshold: float = 0.5) -> str:
    parts = [f"{k}={v:.2f}" for k, v in comp.items() if v > threshold]
    return "; ".join(parts)


def run_scenario(
    schott: pd.DataFrame,
    sciglass: pd.DataFrame,
    feature_list: list[str],
    filter_mode: Literal["strict", "adaptive"],
    property_mode: Literal["A", "B"],
    oxide_cols: list[str],
    pool_features: list[str] | None = None,
) -> list[dict[str, Any]]:
    pool_features = pool_features or FEATURES_MODE_B
    pool, _ = filter_sciglass_pool(sciglass, pool_features, filter_mode)
    z_params = fit_zscore_params(pool, feature_list)
    schott_feat = schott_to_sg_features(schott, feature_list)

    results = []
    for idx, row in schott.iterrows():
        props = schott_feat.loc[idx]
        if property_mode == "A":
            if props[feature_list].isna().any():
                continue
        else:
            if props[feature_list].isna().any():
                continue
        res = process_glass(
            row["glass_name"],
            props,
            pool,
            feature_list,
            z_params,
            oxide_cols,
            filter_mode,
            property_mode,
        )
        results.append(res)
    return results


def build_output_tables(
    results: list[dict[str, Any]],
    oxide_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidates = []
    uncertainty = []
    neighbors_rows = []

    for r in results:
        if "error" in r:
            continue
        comp = r["predicted_composition"]
        cand_row = {
            "glass_name": r["glass_name"],
            "filter_mode": r["filter_mode"],
            "property_mode": r["property_mode"],
            "matched_sciglass_id": r["matched_sciglass_id"],
            "distance": r["distance_first"],
            "predicted_composition": comp_to_str(comp),
            "composition_std": json.dumps(
                r["neighbors"][oxide_cols].fillna(0).std(ddof=1).round(3).to_dict()
            ),
            "glass_family": r["glass_family_mode"],
            "composition_completeness": r["composition_completeness"],
            "physically_plausible": r["physically_plausible"],
            "recon_method": "idw",
            "n_neighbors_used": r["n_neighbors"],
        }
        for method in RECON_METHODS:
            cand_row[f"composition_{method}"] = comp_to_str(
                r["reconstructions"][method]["composition"]
            )
        candidates.append(cand_row)
        unc_row = {
            "glass_name": r["glass_name"],
            "filter_mode": r["filter_mode"],
            "property_mode": r["property_mode"],
            "distance_first": r["distance_first"],
            "distance_mean_top20": r["distance_mean_top20"],
            "composition_variance": r["composition_variance"],
            "family_entropy": r["family_entropy"],
            "effective_neighbors": r["effective_neighbors"],
            "uncertainty_score": r["uncertainty_score"],
            "composition_completeness": r["composition_completeness"],
            "physically_plausible": r["physically_plausible"],
            "high_uncertainty": r["uncertainty_score"] > 0.05 or not r["physically_plausible"],
        }
        for method in RECON_METHODS:
            unc_row[f"comp_{method}"] = comp_to_str(r["reconstructions"][method]["composition"])
        uncertainty.append(unc_row)
        for _, nrow in r["neighbors"].iterrows():
            neighbors_rows.append({
                "glass_name": r["glass_name"],
                "filter_mode": r["filter_mode"],
                "property_mode": r["property_mode"],
                "rank": int(nrow["_rank"]),
                "sciglass_id": int(nrow["sciglass_id"]),
                "distance": float(nrow["_distance"]),
                **{f"oxide_{c}": nrow[c] for c in oxide_cols if c in nrow.index},
            })

    return (
        pd.DataFrame(candidates),
        pd.DataFrame(uncertainty),
        pd.DataFrame(neighbors_rows),
    )


def composition_shift(comp_a: pd.Series, comp_b: pd.Series, oxide_cols: list[str]) -> float:
    a = comp_a.reindex(oxide_cols).fillna(0).to_numpy()
    b = comp_b.reindex(oxide_cols).fillna(0).to_numpy()
    return float(np.sqrt(np.mean((a - b) ** 2)))


def plot_diagnostics(
    schott: pd.DataFrame,
    sciglass_pool: pd.DataFrame,
    results: list[dict[str, Any]],
    uncertainty_df: pd.DataFrame,
    feature_list: list[str],
    output_dir: Path,
    tag: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    z_params = fit_zscore_params(sciglass_pool, feature_list)
    schott_feat = schott_to_sg_features(schott, feature_list).dropna(how="any", subset=feature_list)

    def _z_matrix(df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in feature_list:
            m, s = z_params[c]
            fill = df[c].fillna(m).to_numpy(dtype=float)
            cols.append(z_transform(fill, m, s))
        return np.column_stack(cols)

    Z_sg = _z_matrix(sciglass_pool) if len(sciglass_pool) else np.zeros((0, len(feature_list)))
    Z_sh = _z_matrix(schott_feat) if len(schott_feat) else np.zeros((0, len(feature_list)))
    Z_all = np.vstack([Z_sg, Z_sh]) if len(Z_sg) + len(Z_sh) else np.zeros((0, len(feature_list)))
    Z_all = np.nan_to_num(Z_all, nan=0.0)

    n_comp = min(2, Z_all.shape[1], max(Z_all.shape[0] - 1, 1))
    pca = PCA(n_components=n_comp, random_state=42)
    xy = pca.fit_transform(Z_all)
    if n_comp < 2:
        xy = np.column_stack([xy[:, 0], np.zeros(len(xy))])
    n_sg = len(Z_sg)

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(xy[:n_sg, 0], xy[:n_sg, 1], s=3, alpha=0.12, c="#348ABD", label="SciGlass")
    ax.scatter(xy[n_sg:, 0], xy[n_sg:, 1], s=60, c="#A60628", edgecolors="k", label="SCHOTT")
    ax.set_title(f"PCA свойств ({tag})")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / f"pca_{tag}.png", dpi=150)
    plt.close(fig)

    if len(uncertainty_df):
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(
            uncertainty_df["distance_first"],
            uncertainty_df["composition_variance"],
            c=uncertainty_df["uncertainty_score"],
            cmap="YlOrRd",
            s=50,
        )
        ax.set_xlabel("distance_first")
        ax.set_ylabel("composition_variance")
        ax.set_title(f"distance vs variance ({tag})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"distance_vs_variance_{tag}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(uncertainty_df["uncertainty_score"].dropna(), bins=30, color="#348ABD", edgecolor="white")
        ax.set_xlabel("uncertainty_score")
        ax.set_title(f"Распределение uncertainty ({tag})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"uncertainty_hist_{tag}.png", dpi=150)
        plt.close(fig)

    # heatmap top glasses by oxide (subset)
    ok = [r for r in results if "predicted_composition" in r][:25]
    if ok:
        oxide_cols = [c for c in all_oxide_columns(sciglass_pool.columns) if c in ok[0]["predicted_composition"].index]
        mat = pd.DataFrame(
            [r["predicted_composition"].reindex(oxide_cols).fillna(0) for r in ok],
            index=[r["glass_name"] for r in ok],
        )
        fig, ax = plt.subplots(figsize=(14, max(6, len(mat) * 0.35)))
        sns.heatmap(mat, ax=ax, cmap="YlGnBu", linewidths=0.2)
        ax.set_title(f"Состав (IDW), топ стёкол ({tag})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"composition_heatmap_{tag}.png", dpi=150)
        plt.close(fig)


def print_v2_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print("ОТЧЁТ Version 2")
    print("=" * 64)
    print("\n(1) ЗАПИСИ SciGlass В ПУЛЕ ПОИСКА")
    for key, val in report["pool_sizes"].items():
        print(f"  {key}: {val:,}")
    print(f"\n  Baseline STRICT (v1): {report['baseline_strict_v1']:,}")
    print(f"  Прирост ADAPTIVE vs STRICT: {report['adaptive_vs_strict_ratio']:.2f}x")

    print("\n(2) ПРИЗНАКИ")
    print(f"  MODE_A: {', '.join(report['features_mode_a'])}")
    print(f"  MODE_B: {', '.join(report['features_mode_b'])}")
    print(f"  Оксиды (+ остаточные): {len(report['oxide_columns'])} колонок")

    print("\n(3) ПОКРЫТИЕ / УСТОЙЧИВОСТЬ")
    for row in report["scenario_summary"]:
        print(
            f"  {row['label']}: pool={row['pool']:,}, "
            f"median_unc={row['median_uncertainty']:.4f}, "
            f"plausible={row['pct_plausible']:.1f}%"
        )
    print(f"\n  composition_shift (MODE_A vs MODE_B, median): {report['median_composition_shift']:.3f}")

    print("\n(4) ВЫХОД")
    print(f"  {report['output_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SCHOTT-SciGlass v2")
    parser.add_argument("--schott-xlsx", type=Path, default=DEFAULT_SCHOTT_XLSX)
    parser.add_argument("--sciglass-zip", type=Path, default=DEFAULT_SCIGLASS_ZIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    setup_logging()
    args.output.mkdir(parents=True, exist_ok=True)

    schott = load_schott_catalog(args.schott_xlsx)
    sciglass = load_sciglass_extended(args.sciglass_zip)
    oxide_cols = all_oxide_columns(sciglass.columns.tolist())

    pool_strict_b, _ = filter_sciglass_pool(sciglass, FEATURES_MODE_B, "strict")
    pool_adapt_b, _ = filter_sciglass_pool(sciglass, FEATURES_MODE_B, "adaptive")
    n_total = len(sciglass)

    report: dict[str, Any] = {
        "sciglass_total": n_total,
        "baseline_strict_v1": len(pool_strict_b),
        "pool_sizes": {
            "strict_mode_b": len(pool_strict_b),
            "adaptive_mode_b": len(pool_adapt_b),
        },
        "features_mode_a": FEATURES_MODE_A,
        "features_mode_b": FEATURES_MODE_B,
        "oxide_columns": oxide_cols,
        "adaptive_vs_strict_ratio": len(pool_adapt_b) / max(len(pool_strict_b), 1),
        "output_dir": str(args.output.resolve()),
        "scenario_summary": [],
    }

    all_candidates = []
    all_uncertainty = []
    all_neighbors = []

    scenarios = [
        ("adaptive_B", "adaptive", FEATURES_MODE_B, "B"),
        ("strict_B", "strict", FEATURES_MODE_B, "B"),
        ("adaptive_A", "adaptive", FEATURES_MODE_A, "A"),
    ]

    comp_by_glass_ab: dict[str, dict[str, pd.Series]] = {}

    for label, fmode, flist, pmode in scenarios:
        logger.info("Сценарий: %s", label)
        results = run_scenario(schott, sciglass, flist, fmode, pmode, oxide_cols)
        cand, unc, neigh = build_output_tables(results, oxide_cols)
        cand["scenario"] = label
        unc["scenario"] = label
        neigh["scenario"] = label
        all_candidates.append(cand)
        all_uncertainty.append(unc)
        all_neighbors.append(neigh)

        pool, _ = filter_sciglass_pool(sciglass, FEATURES_MODE_B, fmode)
        report["scenario_summary"].append({
            "label": label,
            "pool": len(pool),
            "median_uncertainty": float(unc["uncertainty_score"].median()) if len(unc) else np.nan,
            "pct_plausible": float(100 * unc["physically_plausible"].mean()) if len(unc) and "physically_plausible" in unc else 0,
        })
        plot_diagnostics(schott, pool, results, unc, flist, args.output, label)

        if label.startswith("adaptive_A"):
            for r in results:
                if "predicted_composition" in r:
                    comp_by_glass_ab.setdefault(r["glass_name"], {})["A"] = r["predicted_composition"]
        if label.startswith("adaptive_B"):
            for r in results:
                if "predicted_composition" in r:
                    comp_by_glass_ab.setdefault(r["glass_name"], {})["B"] = r["predicted_composition"]

    shifts = []
    for name, comps in comp_by_glass_ab.items():
        if "A" in comps and "B" in comps:
            shifts.append(composition_shift(comps["A"], comps["B"], oxide_cols))
    report["median_composition_shift"] = float(np.median(shifts)) if shifts else np.nan

    candidates = pd.concat(all_candidates, ignore_index=True)
    uncertainty = pd.concat(all_uncertainty, ignore_index=True)
    neighbors = pd.concat(all_neighbors, ignore_index=True)

    # Основной вывод — adaptive + MODE_B + idw
    main = candidates[
        (candidates["scenario"] == "adaptive_B")
    ].copy()
    main.to_csv(args.output / "composition_candidates.csv", index=False)
    uncertainty[uncertainty["scenario"] == "adaptive_B"].to_csv(
        args.output / "composition_uncertainty.csv", index=False
    )
    neighbors[neighbors["scenario"] == "adaptive_B"].to_csv(
        args.output / "neighbor_analysis.csv", index=False
    )

    report["composition_shift"] = {
        "n_glasses": len(shifts),
        "median": report["median_composition_shift"],
        "mean": float(np.mean(shifts)) if shifts else np.nan,
    }
    (args.output / "match_report_v2.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print_v2_report(report)
    logger.info("Сохранено в %s", args.output)


if __name__ == "__main__":
    main()
