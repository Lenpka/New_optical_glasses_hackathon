"""SCHOTT -> SciGlass Version 3: neighbor stability, plausibility, local inverse models.

Не усредняет составы соседей. Основной состав — лучший сосед (MODE_B) или локальная модель.

    python match_schott_sciglass_v3.py

Выход: output_v3/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

from match_schott_sciglass import (
    DEFAULT_SCHOTT_XLSX,
    DEFAULT_SCIGLASS_ZIP,
    K_NEIGHBORS,
    OXIDE_MOL_COLS,
    OXIDE_SUM_TARGET,
    OXIDE_SUM_TOL,
    classify_glass_family,
    load_schott_catalog,
    setup_logging,
)
from match_schott_sciglass_v2 import (
    FEATURES_MODE_A,
    FEATURES_MODE_B,
    adaptive_distance_matrix,
    all_oxide_columns,
    comp_to_str,
    composition_variance,
    family_entropy,
    filter_sciglass_pool,
    fit_zscore_params,
    load_sciglass_extended,
    pick_top_neighbors,
    schott_to_sg_features,
    z_transform,
)

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "output_v3"

FEATURE_LABELS = {
    "ND300": "nd",
    "NUD300": "vd",
    "DENSITY": "density",
    "TG": "tg",
}

JACCARD_STABLE = 0.8
JACCARD_MODERATE = 0.5
JACCARD_UNSTABLE = 0.3


def is_primary_plausible(row: pd.Series, primary: list[str]) -> bool:
    """Физичность одной реальной записи SciGlass (только молярные оксиды)."""
    ox = row.reindex(primary).fillna(0)
    if (ox < -1e-6).any():
        return False
    total = float(ox.sum())
    return abs(total - OXIDE_SUM_TARGET) <= OXIDE_SUM_TOL


def neighbor_plausibility_metrics(
    neighbors: pd.DataFrame,
    primary: list[str],
) -> dict[str, Any]:
    flags = [is_primary_plausible(neighbors.iloc[i], primary) for i in range(len(neighbors))]
    n = len(flags)
    top5 = flags[: min(5, n)]
    return {
        "best_neighbor_plausible": bool(flags[0]) if n else False,
        "plausible_ratio_top5": float(np.mean(top5)) if top5 else np.nan,
        "plausible_ratio_top20": float(np.mean(flags)) if n else np.nan,
        "n_plausible_top20": int(sum(flags)),
    }


def jaccard_topk(ids_a: set[int], ids_b: set[int]) -> float:
    if not ids_a and not ids_b:
        return np.nan
    union = ids_a | ids_b
    if not union:
        return np.nan
    return len(ids_a & ids_b) / len(union)


def jaccard_label(j: float) -> str:
    if np.isnan(j):
        return "unknown"
    if j > JACCARD_STABLE:
        return "stable"
    if j >= JACCARD_MODERATE:
        return "moderate"
    if j >= JACCARD_UNSTABLE:
        return "weak"
    return "ill_posed"


def per_neighbor_sq_distances(
    schott_props: pd.Series,
    neighbor_row: pd.Series,
    feature_list: list[str],
    z_params: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Квадраты z-расстояний по каждому признаку (пересечение доступных)."""
    out: dict[str, float] = {}
    for col in feature_list:
        if col not in schott_props.index or pd.isna(schott_props[col]):
            continue
        if col not in neighbor_row.index or pd.isna(neighbor_row[col]):
            continue
        m, s = z_params[col]
        q = z_transform(float(schott_props[col]), m, s)
        y = z_transform(float(neighbor_row[col]), m, s)
        out[col] = float((q - y) ** 2)
    return out


def distance_attribution_topk(
    schott_props: pd.Series,
    neighbors: pd.DataFrame,
    feature_list: list[str],
    z_params: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Средняя доля d_j^2 в сумме квадратов по top-k соседям."""
    accum = {c: [] for c in feature_list}
    for i in range(len(neighbors)):
        sq = per_neighbor_sq_distances(schott_props, neighbors.iloc[i], feature_list, z_params)
        total = sum(sq.values())
        if total <= 0:
            continue
        for c, v in sq.items():
            accum[c].append(v / total)
    return {c: float(np.mean(accum[c])) if accum[c] else 0.0 for c in feature_list}


def project_composition_sum100(pred: np.ndarray) -> np.ndarray:
    pred = np.maximum(pred, 0.0)
    s = pred.sum()
    if s <= 1e-9:
        return np.full_like(pred, OXIDE_SUM_TARGET / len(pred))
    return pred / s * OXIDE_SUM_TARGET


def fit_local_pls(
    X: np.ndarray,
    Y: np.ndarray,
    x_query: np.ndarray,
    n_comp: int = 3,
) -> tuple[np.ndarray, float]:
    n_comp = min(n_comp, X.shape[0] - 1, X.shape[1])
    if n_comp < 1:
        return Y[0], np.nan
    pls = PLSRegression(n_components=n_comp)
    pls.fit(X, Y)
    pred = pls.predict(x_query.reshape(1, -1)).ravel()
    pred = project_composition_sum100(pred)
    cv_err = float(np.mean((Y - pls.predict(X)) ** 2))
    return pred, cv_err


def fit_local_xgb(
    X: np.ndarray,
    Y: np.ndarray,
    x_query: np.ndarray,
) -> tuple[np.ndarray, float]:
    try:
        from xgboost import XGBRegressor
    except ImportError:
        return fit_local_pls(X, Y, x_query)[0], np.nan

    preds = []
    cv = []
    for j in range(Y.shape[1]):
        m = XGBRegressor(
            n_estimators=40,
            max_depth=2,
            learning_rate=0.1,
            reg_alpha=1.0,
            reg_lambda=2.0,
            subsample=0.9,
            random_state=42,
            verbosity=0,
        )
        m.fit(X, Y[:, j])
        preds.append(float(m.predict(x_query.reshape(1, -1))[0]))
        cv.append(float(mean_squared_error(Y[:, j], m.predict(X))))
    pred = project_composition_sum100(np.array(preds))
    return pred, float(np.mean(cv))


def fit_local_gpr(
    X: np.ndarray,
    Y: np.ndarray,
    x_query: np.ndarray,
) -> tuple[np.ndarray, float]:
    """GPR по каждому оксиду (медленно; n=20)."""
    preds = []
    cv = []
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.5)
    for j in range(Y.shape[1]):
        try:
            gpr = GaussianProcessRegressor(kernel=kernel, random_state=42, normalize_y=True)
            gpr.fit(X, Y[:, j])
            p = float(gpr.predict(x_query.reshape(1, -1))[0])
            cv.append(float(mean_squared_error(Y[:, j], gpr.predict(X))))
        except Exception:
            p = float(np.mean(Y[:, j]))
            cv.append(np.nan)
        preds.append(p)
    pred = project_composition_sum100(np.array(preds))
    return pred, float(np.nanmean(cv))


def find_neighbors(
    glass_name: str,
    schott_props: pd.Series,
    pool: pd.DataFrame,
    feature_list: list[str],
    z_params: dict[str, tuple[float, float]],
    k: int = K_NEIGHBORS,
) -> pd.DataFrame:
    dist, _ = adaptive_distance_matrix(
        schott_props,
        pool[feature_list],
        feature_list,
        z_params,
        n_target=len(FEATURES_MODE_B),
    )
    return pick_top_neighbors(pool, dist, k)


def _properties_to_query_series(
    nd: float,
    vd: float,
    density: float | None,
    tg: float | None,
    pool: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, list[str]]:
    """Свойства запроса в формате SciGlass; при отсутствии ρ/Tg — медиана пула."""
    imputed: list[str] = []
    d_val = density
    t_val = tg
    if d_val is None or (isinstance(d_val, float) and np.isnan(d_val)):
        d_val = float(pool["DENSITY"].median())
        imputed.append("density")
    if t_val is None or (isinstance(t_val, float) and np.isnan(t_val)):
        t_val = float(pool["TG"].median())
        imputed.append("tg")

    props_a = pd.Series({"ND300": nd, "NUD300": vd})
    props_b = pd.Series({"ND300": nd, "NUD300": vd, "DENSITY": d_val, "TG": t_val})
    return props_a, props_b, imputed


def recover_from_properties(
    nd: float,
    vd: float,
    *,
    density: float | None = None,
    tg: float | None = None,
    label: str = "user_query",
    sciglass_zip: Path = DEFAULT_SCIGLASS_ZIP,
    k_neighbors: int = K_NEIGHBORS,
    use_gpr: bool = False,
) -> dict[str, Any]:
    """Восстановить состав по целевым свойствам (реальные соседи SciGlass, без усреднения).

    Возвращает словарь с основным составом (лучший сосед MODE_B), соседями, Jaccard и uncertainty.
    """
    sciglass = load_sciglass_extended(sciglass_zip)
    primary = [c for c in OXIDE_MOL_COLS if c in sciglass.columns]
    oxide_cols = all_oxide_columns(sciglass.columns.tolist())
    pool, _ = filter_sciglass_pool(sciglass, FEATURES_MODE_B, "adaptive")
    z_a = fit_zscore_params(pool, FEATURES_MODE_A)
    z_b = fit_zscore_params(pool, FEATURES_MODE_B)

    pa, pb, imputed = _properties_to_query_series(nd, vd, density, tg, pool)
    name = label

    na = find_neighbors(name, pa, pool, FEATURES_MODE_A, z_a, k=k_neighbors)
    nb = find_neighbors(name, pb, pool, FEATURES_MODE_B, z_b, k=k_neighbors)
    if na.empty or nb.empty:
        raise ValueError("Не найдены соседи в SciGlass для заданных свойств")

    ids_a = set(na["sciglass_id"].astype(int))
    ids_b = set(nb["sciglass_id"].astype(int))
    j = jaccard_topk(ids_a, ids_b)
    pl_b = neighbor_plausibility_metrics(nb, primary)

    best = nb.iloc[0]
    best_comp = primary_composition_row(best, primary)

    X = np.column_stack([
        z_transform(nb[c].fillna(z_b[c][0]).to_numpy(), *z_b[c]) for c in FEATURES_MODE_B
    ])
    Y = nb[primary].fillna(0).to_numpy()
    xq = np.array([z_transform(float(pb[c]), *z_b[c]) for c in FEATURES_MODE_B])
    comp_pls, err_pls = fit_local_pls(X, Y, xq)
    comp_xgb, err_xgb = fit_local_xgb(X, Y, xq)
    if use_gpr:
        comp_gpr, err_gpr = fit_local_gpr(X, Y, xq)
    else:
        comp_gpr, err_gpr = comp_pls.copy(), np.nan

    d_nb = nb["_distance"].to_numpy()
    comp_var = composition_variance(nb, primary)
    unc = comp_var * float(np.mean(d_nb))

    neighbors_table = []
    for rank, (_, nrow) in enumerate(nb.iterrows(), start=1):
        comp_row = primary_composition_row(nrow, primary)
        neighbors_table.append({
            "rank": rank,
            "sciglass_id": int(nrow["sciglass_id"]),
            "distance": float(nrow["_distance"]),
            "composition": comp_to_str(comp_row),
            "plausible": is_primary_plausible(nrow, primary),
            "in_mode_a_topk": int(nrow["sciglass_id"]) in ids_a,
            "nd": float(nrow["ND300"]) if pd.notna(nrow.get("ND300")) else None,
            "vd": float(nrow["NUD300"]) if pd.notna(nrow.get("NUD300")) else None,
        })

    return {
        "query": {
            "label": label,
            "nd": nd,
            "vd": vd,
            "density": float(pb["DENSITY"]),
            "tg": float(pb["TG"]),
            "density_input": density,
            "tg_input": tg,
            "imputed_fields": imputed,
        },
        "primary_composition": comp_to_str(best_comp),
        "composition_source": "best_neighbor_mode_b",
        "matched_sciglass_id": int(best["sciglass_id"]),
        "distance_first": float(d_nb[0]),
        "distance_mean_topk": float(np.mean(d_nb)),
        "jaccard_topk": j,
        "jaccard_label": jaccard_label(j),
        "uncertainty_score": unc,
        "composition_variance_neighbors": comp_var,
        "best_neighbor_plausible": pl_b["best_neighbor_plausible"],
        "plausible_ratio_topk": pl_b["plausible_ratio_top20"],
        "composition_pls": comp_to_str(pd.Series(comp_pls, index=primary)),
        "composition_xgb": comp_to_str(pd.Series(comp_xgb, index=primary)),
        "local_pls_cv_mse": err_pls,
        "local_xgb_cv_mse": err_xgb,
        "disclaimer": (
            "Состав взят из реальной записи SciGlass (ближайший сосед). "
            "Локальные модели PLS/XGB — вспомогательная оценка, не синтез стекла."
        ),
        "neighbors": neighbors_table,
    }


def recover_from_schott_name(
    glass_name: str,
    *,
    schott_xlsx: Path = DEFAULT_SCHOTT_XLSX,
    sciglass_zip: Path = DEFAULT_SCIGLASS_ZIP,
    k_neighbors: int = K_NEIGHBORS,
    use_gpr: bool = False,
) -> dict[str, Any]:
    """Восстановить состав по марке стекла из каталога SCHOTT."""
    schott = load_schott_catalog(schott_xlsx)
    hit = schott[schott["glass_name"].astype(str).str.strip().str.lower() == glass_name.strip().lower()]
    if hit.empty:
        partial = schott[schott["glass_name"].astype(str).str.contains(glass_name, case=False, na=False)]
        if len(partial) == 1:
            hit = partial
        elif len(partial) > 1:
            names = partial["glass_name"].tolist()[:15]
            raise ValueError(f"Неоднозначное имя «{glass_name}». Варианты: {', '.join(names)}")
        else:
            raise ValueError(f"Стекло «{glass_name}» не найдено в каталоге SCHOTT")

    row = hit.iloc[0]
    if pd.isna(row.get("nd")) or pd.isna(row.get("vd")):
        raise ValueError(f"У стекла {row['glass_name']} нет nd/vd в каталоге")

    out = recover_from_properties(
        float(row["nd"]),
        float(row["vd"]),
        density=float(row["density"]) if pd.notna(row.get("density")) else None,
        tg=float(row["tg"]) if pd.notna(row.get("tg")) else None,
        label=str(row["glass_name"]),
        sciglass_zip=sciglass_zip,
        k_neighbors=k_neighbors,
        use_gpr=use_gpr,
    )
    out["schott_catalog"] = {
        "glass_name": str(row["glass_name"]),
        "nd": float(row["nd"]),
        "vd": float(row["vd"]),
        "density": float(row["density"]) if pd.notna(row.get("density")) else None,
        "tg": float(row["tg"]) if pd.notna(row.get("tg")) else None,
    }
    return out


def primary_composition_row(row: pd.Series, primary: list[str]) -> pd.Series:
    return row.reindex(primary).fillna(0)


def run_v3(
    schott: pd.DataFrame,
    sciglass: pd.DataFrame,
    primary: list[str],
    oxide_cols: list[str],
    use_gpr: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pool, _ = filter_sciglass_pool(sciglass, FEATURES_MODE_B, "adaptive")
    z_a = fit_zscore_params(pool, FEATURES_MODE_A)
    z_b = fit_zscore_params(pool, FEATURES_MODE_B)

    schott_feat_a = schott_to_sg_features(schott, FEATURES_MODE_A)
    schott_feat_b = schott_to_sg_features(schott, FEATURES_MODE_B)

    stability_rows = []
    plaus_rows = []
    attrib_rows = []
    local_rows = []
    candidate_rows = []
    neighbor_rows = []

    all_attrib = {c: [] for c in FEATURES_MODE_B}

    for idx, srow in schott.iterrows():
        name = srow["glass_name"]
        pa = schott_feat_a.loc[idx]
        pb = schott_feat_b.loc[idx]
        if pa[FEATURES_MODE_A].isna().any() or pb[FEATURES_MODE_B].isna().any():
            continue

        na = find_neighbors(name, pa, pool, FEATURES_MODE_A, z_a)
        nb = find_neighbors(name, pb, pool, FEATURES_MODE_B, z_b)
        if na.empty or nb.empty:
            continue

        ids_a = set(na["sciglass_id"].astype(int))
        ids_b = set(nb["sciglass_id"].astype(int))
        j = jaccard_topk(ids_a, ids_b)
        stable_ids = ids_a & ids_b

        pl_a = neighbor_plausibility_metrics(na, primary)
        pl_b = neighbor_plausibility_metrics(nb, primary)

        plaus_rows.append({
            "glass_name": name,
            **{f"mode_a_{k}": v for k, v in pl_a.items()},
            **{f"mode_b_{k}": v for k, v in pl_b.items()},
        })

        stability_rows.append({
            "glass_name": name,
            "jaccard_top20": j,
            "jaccard_label": jaccard_label(j),
            "n_intersection": len(stable_ids),
            "n_union": len(ids_a | ids_b),
            "n_mode_a": len(ids_a),
            "n_mode_b": len(ids_b),
            "stable_neighbor_count": len(stable_ids),
        })

        attr = distance_attribution_topk(pb, nb, FEATURES_MODE_B, z_b)
        row_attr = {"glass_name": name, **{FEATURE_LABELS.get(c, c): attr[c] for c in FEATURES_MODE_B}}
        attrib_rows.append(row_attr)
        for c in FEATURES_MODE_B:
            all_attrib[c].append(attr[c])

        # Лучший сосед MODE_B — основной состав (без усреднения)
        best = nb.iloc[0]
        best_comp = primary_composition_row(best, primary)

        # Локальные модели на top-20 MODE_B
        X = np.column_stack([
            z_transform(nb[c].fillna(z_b[c][0]).to_numpy(), *z_b[c]) for c in FEATURES_MODE_B
        ])
        Y = nb[primary].fillna(0).to_numpy()
        xq = np.array([z_transform(float(pb[c]), *z_b[c]) for c in FEATURES_MODE_B])

        comp_pls, err_pls = fit_local_pls(X, Y, xq)
        comp_xgb, err_xgb = fit_local_xgb(X, Y, xq)
        if use_gpr:
            comp_gpr, err_gpr = fit_local_gpr(X, Y, xq)
        else:
            comp_gpr, err_gpr = comp_pls.copy(), np.nan

        d_nb = nb["_distance"].to_numpy()
        comp_var = composition_variance(nb, primary)
        unc = comp_var * float(np.mean(d_nb))
        families = [classify_glass_family(nb.iloc[i], oxide_cols) for i in range(len(nb))]

        candidate_rows.append({
            "glass_name": name,
            "composition_source": "best_neighbor_mode_b",
            "matched_sciglass_id": int(best["sciglass_id"]),
            "distance_first": float(d_nb[0]),
            "distance_mean_top20": float(np.mean(d_nb)),
            "jaccard_mode_a_b": j,
            "jaccard_label": jaccard_label(j),
            "predicted_composition": comp_to_str(best_comp),
            "composition_pls": comp_to_str(pd.Series(comp_pls, index=primary)),
            "composition_xgb": comp_to_str(pd.Series(comp_xgb, index=primary)),
            "composition_gpr": comp_to_str(pd.Series(comp_gpr, index=primary)),
            "local_pls_cv_mse": err_pls,
            "local_xgb_cv_mse": err_xgb,
            "local_gpr_cv_mse": err_gpr,
            "composition_variance_neighbors": comp_var,
            "uncertainty_score": unc,
            "family_entropy": family_entropy(families),
            "best_neighbor_plausible": pl_b["best_neighbor_plausible"],
            "plausible_ratio_top20": pl_b["plausible_ratio_top20"],
            "plausible_ratio_top5": pl_b["plausible_ratio_top5"],
            "n_neighbors_used": len(nb),
            "stable_neighbor_count": len(stable_ids),
        })

        for rank, (_, nrow) in enumerate(nb.iterrows(), start=1):
            neighbor_rows.append({
                "glass_name": name,
                "property_mode": "B",
                "rank": rank,
                "sciglass_id": int(nrow["sciglass_id"]),
                "distance": float(nrow["_distance"]),
                "primary_plausible": is_primary_plausible(nrow, primary),
                "in_mode_a_top20": int(nrow["sciglass_id"]) in ids_a,
                "in_stable_set": int(nrow["sciglass_id"]) in stable_ids,
            })

    stability_df = pd.DataFrame(stability_rows)
    plaus_df = pd.DataFrame(plaus_rows)
    attrib_df = pd.DataFrame(attrib_rows)
    local_df = pd.DataFrame(candidate_rows)
    candidates_df = local_df.copy()

    mean_attrib = {FEATURE_LABELS.get(c, c): float(np.mean(all_attrib[c])) for c in FEATURES_MODE_B}
    total = sum(mean_attrib.values()) or 1.0
    mean_attrib_pct = {k: 100.0 * v / total for k, v in mean_attrib.items()}

    report: dict[str, Any] = {
        "n_schott_processed": len(stability_df),
        "pool_adaptive": len(pool),
        "primary_oxide_columns": primary,
        "mean_distance_attribution_fraction": mean_attrib,
        "mean_distance_attribution_percent": mean_attrib_pct,
        "jaccard_summary": {},
        "plausibility_summary": {},
        "local_model_summary": {},
    }

    if len(stability_df):
        j = stability_df["jaccard_top20"]
        report["jaccard_summary"] = {
            "median": float(j.median()),
            "mean": float(j.mean()),
            "pct_stable_gt_0.8": float(100 * (j > JACCARD_STABLE).mean()),
            "pct_moderate_0.5_0.8": float(100 * ((j >= JACCARD_MODERATE) & (j <= JACCARD_STABLE)).mean()),
            "pct_ill_posed_lt_0.3": float(100 * (j < JACCARD_UNSTABLE).mean()),
        }

    if len(plaus_df):
        report["plausibility_summary"] = {
            "mode_b_best_neighbor_plausible_pct": float(
                100 * plaus_df["mode_b_best_neighbor_plausible"].mean()
            ),
            "mode_b_plausible_ratio_top20_mean": float(plaus_df["mode_b_plausible_ratio_top20"].mean()),
            "mode_b_plausible_ratio_top5_mean": float(plaus_df["mode_b_plausible_ratio_top5"].mean()),
            "mode_a_plausible_ratio_top20_mean": float(plaus_df["mode_a_plausible_ratio_top20"].mean()),
        }

    if len(local_df):
        report["local_model_summary"] = {
            "median_uncertainty_best_neighbor": float(local_df["uncertainty_score"].median()),
            "median_local_pls_cv_mse": float(local_df["local_pls_cv_mse"].median()),
            "median_local_xgb_cv_mse": float(local_df["local_xgb_cv_mse"].median()),
        }

    return stability_df, plaus_df, attrib_df, candidates_df, pd.DataFrame(neighbor_rows), report


def plot_v3_figures(
    stability_df: pd.DataFrame,
    attrib_pct: dict[str, float],
    candidates_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if len(stability_df):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(stability_df["jaccard_top20"].dropna(), bins=25, color="#348ABD", edgecolor="white")
        for x, c in [(0.3, "red"), (0.5, "orange"), (0.8, "green")]:
            ax.axvline(x, color=c, ls="--", lw=1.2, label=f"J={x}")
        ax.set_xlabel("Jaccard(top20 MODE_A, top20 MODE_B)")
        ax.set_title("Устойчивость множества соседей")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "jaccard_histogram.png", dpi=150)
        plt.close(fig)

    if attrib_pct:
        fig, ax = plt.subplots(figsize=(7, 5))
        keys = list(attrib_pct.keys())
        vals = [attrib_pct[k] for k in keys]
        ax.barh(keys, vals, color="#348ABD")
        ax.set_xlabel("Доля в mean(d_j^2), %")
        ax.set_title("Вклад признаков в расстояние (MODE_B, top-20)")
        fig.tight_layout()
        fig.savefig(fig_dir / "distance_attribution.png", dpi=150)
        plt.close(fig)

    if len(stability_df) and len(candidates_df):
        merged = stability_df.merge(
            candidates_df[["glass_name", "uncertainty_score"]], on="glass_name"
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(merged["jaccard_top20"], merged["uncertainty_score"], alpha=0.6, s=40)
        ax.set_xlabel("Jaccard overlap")
        ax.set_ylabel("uncertainty (var × mean distance)")
        ax.set_title("Обусловленность: overlap vs uncertainty")
        fig.tight_layout()
        fig.savefig(fig_dir / "jaccard_vs_uncertainty.png", dpi=150)
        plt.close(fig)


def print_v3_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print("ОТЧЁТ Version 3 — обусловленность обратной задачи")
    print("=" * 64)

    js = report.get("jaccard_summary", {})
    print("\n(1) JACCARD MODE_A vs MODE_B (top-20)")
    if js:
        print(f"  median J = {js['median']:.3f}")
        print(f"  stable (J>0.8): {js['pct_stable_gt_0.8']:.1f}%")
        print(f"  moderate (0.5–0.8): {js['pct_moderate_0.5_0.8']:.1f}%")
        print(f"  ill-posed (J<0.3): {js['pct_ill_posed_lt_0.3']:.1f}%")

    ps = report.get("plausibility_summary", {})
    print("\n(2) ФИЗИЧНОСТЬ СОСЕДЕЙ (молярные оксиды, sum~100, без усреднения)")
    if ps:
        print(f"  best_neighbor plausible (MODE_B): {ps['mode_b_best_neighbor_plausible_pct']:.1f}%")
        print(f"  plausible_ratio_top20 (MODE_B): {ps['mode_b_plausible_ratio_top20_mean']:.2f}")
        print(f"  plausible_ratio_top5 (MODE_B): {ps['mode_b_plausible_ratio_top5_mean']:.2f}")

    ap = report.get("mean_distance_attribution_percent", {})
    print("\n(3) ВКЛАД ПРИЗНАКОВ В d^2 (среднее по стёклам, MODE_B)")
    for k, v in sorted(ap.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.1f}%")

    print("\n(4) ЛОКАЛЬНЫЕ МОДЕЛИ (top-20, sum x_i = 100)")
    lm = report.get("local_model_summary", {})
    if lm:
        print(f"  median uncertainty (best neighbor): {lm['median_uncertainty_best_neighbor']:.4f}")
        print(f"  median LOO-style CV MSE PLS: {lm['median_local_pls_cv_mse']:.4f}")
        print(f"  median LOO-style CV MSE XGB: {lm['median_local_xgb_cv_mse']:.4f}")

    print(f"\n(5) ВЫХОД: {report['output_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SCHOTT-SciGlass v3")
    parser.add_argument("--schott-xlsx", type=Path, default=DEFAULT_SCHOTT_XLSX)
    parser.add_argument("--sciglass-zip", type=Path, default=DEFAULT_SCIGLASS_ZIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--enable-gpr", action="store_true", help="GPR (медленно, 19 оксидов x 122)")
    args = parser.parse_args()

    setup_logging()
    args.output.mkdir(parents=True, exist_ok=True)

    schott = load_schott_catalog(args.schott_xlsx)
    sciglass = load_sciglass_extended(args.sciglass_zip)
    primary = [c for c in OXIDE_MOL_COLS if c in sciglass.columns]
    oxide_cols = all_oxide_columns(sciglass.columns.tolist())

    stability_df, plaus_df, attrib_df, candidates_df, neighbors_df, report = run_v3(
        schott, sciglass, primary, oxide_cols, use_gpr=args.enable_gpr
    )
    report["output_dir"] = str(args.output.resolve())

    stability_df.to_csv(args.output / "neighbor_stability.csv", index=False)
    plaus_df.to_csv(args.output / "neighbor_plausibility.csv", index=False)
    attrib_df.to_csv(args.output / "distance_attribution.csv", index=False)
    candidates_df.to_csv(args.output / "composition_candidates.csv", index=False)
    candidates_df.to_csv(args.output / "composition_local_models.csv", index=False)
    neighbors_df.to_csv(args.output / "neighbor_analysis.csv", index=False)

    (args.output / "match_report_v3.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    plot_v3_figures(
        stability_df,
        report.get("mean_distance_attribution_percent", {}),
        candidates_df,
        args.output,
    )

    print_v3_report(report)
    logger.info("Сохранено в %s", args.output)


if __name__ == "__main__":
    main()
