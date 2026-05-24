"""Сопоставление каталога SCHOTT со SciGlass по свойствам (inverse composition).

Запуск:
    python match_schott_sciglass.py
    python match_schott_sciglass.py --schott-xlsx "schott-optical-glass-overview-excel-format-en 202501113.xlsx"

Не предсказывает свойства — ищет ближайшие реальные стекла SciGlass и агрегирует их составы.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SCHOTT_XLSX = PROJECT_DIR / "schott-optical-glass-overview-excel-format-en 202501113.xlsx"
DEFAULT_SCIGLASS_ZIP = Path(r"C:/Users/user/AppData/Local/GlassPy/GlassPy/data/select_SciGK.csv.zip")
DEFAULT_OUTPUT = PROJECT_DIR / "output" / "schott_match"

# SCHOTT (каталог) -> SciGlass (select_SciGK)
PROPERTY_MAP = {
    "nd": ("ND300", "Показатель преломления nd"),
    "vd": ("NUD300", "Число Аббе vd"),
    "density": ("DENSITY", "Плотность"),
    "tg": ("TG", "Температура стеклования Tg"),
}

# Стартовый набор признаков (не все 186 колонок)
MATCH_FEATURES = ["ND300", "NUD300", "DENSITY", "TG"]

OXIDE_MOL_COLS = [
    "SIO2", "AL2O3", "B2O3", "CAO", "K2O", "NA2O", "PBO", "Li2O", "MgO", "SRO", "BAO",
    "ZNO", "P2O5", "GEO2", "ZRO2", "TIO2", "TEO2", "RO", "FemOn",
]

K_NEIGHBORS = 20
OXIDE_SUM_TARGET = 100.0
OXIDE_SUM_TOL = 2.0


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        stream=sys.stdout,
    )


def load_schott_catalog(xlsx_path: Path, sheet: str = "Preferred glasses") -> pd.DataFrame:
    """Парсинг каталога SCHOTT (строка 3 — заголовки свойств)."""
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
    header_row = 3
    headers = raw.iloc[header_row].tolist()
    df = raw.iloc[header_row + 1 :].copy()
    # Уникальные имена колонок (в Excel бывают дубликаты)
    clean_headers: list[str] = []
    seen: dict[str, int] = {}
    for i, h in enumerate(headers):
        name = str(h).strip() if pd.notna(h) else f"col_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}__{seen[name]}"
        else:
            seen[name] = 0
        clean_headers.append(name)
    df.columns = clean_headers

    col_map: dict[str, str] = {}
    for col in df.columns:
        cl = str(col).split("__")[0].strip().lower()
        if cl == "glass" and "glass_name" not in col_map.values():
            col_map[col] = "glass_name"
        elif cl == "nd" and "nd" not in col_map.values():
            col_map[col] = "nd"
        elif cl == "vd" and "vd" not in col_map.values():
            col_map[col] = "vd"
        elif cl == "density" and "density" not in col_map.values():
            col_map[col] = "density"
        elif cl == "tg" and "tg" not in col_map.values():
            col_map[col] = "tg"

    df = df.rename(columns=col_map)
    keep = [c for c in ["glass_name", "nd", "vd", "density", "tg"] if c in df.columns]
    df = df[keep].copy()

    df["glass_name"] = df["glass_name"].astype(str).str.strip()
    df = df[df["glass_name"].notna() & (df["glass_name"] != "") & (df["glass_name"] != "nan")]
    df = df[~df["glass_name"].str.lower().eq("glass")]

    for col in ["nd", "vd", "density", "tg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.reset_index(drop=True)


def load_sciglass(
    zip_path: Path,
    oxide_cols: list[str] | None = None,
) -> pd.DataFrame:
    """SciGlass select_SciGK: свойства + мольные оксиды."""
    if oxide_cols is None:
        oxide_cols = OXIDE_MOL_COLS

    usecols = ["KOD", "GLASNO", *MATCH_FEATURES, *oxide_cols]
    with zipfile.ZipFile(zip_path) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, sep="\t", usecols=lambda c: c in usecols, low_memory=False)

    df["sciglass_id"] = df["KOD"] * 100_000_000 + df["GLASNO"]

    for col in MATCH_FEATURES + oxide_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if col in MATCH_FEATURES:
                df.loc[df[col] == 0, col] = np.nan

    return df


def build_feature_matrix(
    schott: pd.DataFrame,
    sciglass: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, dict[str, str]]:
    """Очистка, общие признаки, z-score по SciGlass."""
    schott_cols = {sh: sg for sh, (sg, _) in PROPERTY_MAP.items() if sh in schott.columns}
    missing_sg = [c for c in feature_cols if c not in sciglass.columns]
    if missing_sg:
        raise KeyError(f"В SciGlass нет колонок: {missing_sg}")

    schott_feat = pd.DataFrame(index=schott.index)
    for sh_col, sg_col in schott_cols.items():
        if sg_col in feature_cols:
            schott_feat[sg_col] = schott[sh_col].values

    sciglass_feat = sciglass[feature_cols].copy()

    schott_clean = schott_feat.dropna(how="any")
    sciglass_clean = sciglass_feat.dropna(how="any")

    scaler = StandardScaler()
    scaler.fit(sciglass_clean)

    schott_z = pd.DataFrame(
        scaler.transform(schott_clean),
        index=schott_clean.index,
        columns=feature_cols,
    )
    sciglass_z = pd.DataFrame(
        scaler.transform(sciglass_clean),
        index=sciglass_clean.index,
        columns=feature_cols,
    )

    mapping_report = {
        f"SCHOTT.{sh_col}": sg_col
        for sh_col, sg_col in schott_cols.items()
        if sg_col in feature_cols
    }
    return schott_z, sciglass_z, scaler, mapping_report


def _mahalanobis_distances(
    query: np.ndarray,
    database: np.ndarray,
) -> np.ndarray | None:
    """Mahalanobis distance; None если ковариация вырождена."""
    try:
        lw = LedoitWolf().fit(database)
        prec = lw.precision_
        diff = database - query
        d_db = np.sqrt(np.maximum(np.sum(diff @ prec * diff, axis=1), 0))
        return d_db
    except Exception as exc:
        logger.warning("Mahalanobis недоступен: %s", exc)
        return None


def classify_glass_family(row: pd.Series, oxide_cols: list[str]) -> str:
    """Эвристическая классификация по долям оксидов (мольн.)."""
    ox = {c: float(row.get(c, 0) or 0) for c in oxide_cols}
    sio2 = ox.get("SIO2", 0)
    b2o3 = ox.get("B2O3", 0)
    pbo = ox.get("PBO", 0)
    p2o5 = ox.get("P2O5", 0)
    tio2 = ox.get("TIO2", 0)
    la_like = ox.get("GEO2", 0) + ox.get("ZRO2", 0)
    al2o3 = ox.get("AL2O3", 0)

    if p2o5 > 25:
        return "phosphate"
    if pbo > 15 or tio2 > 12:
        return "flint" if pbo >= tio2 else "heavy flint"
    if la_like > 8:
        return "lanthanum"
    if b2o3 > 8 and sio2 > 75:
        return "borosilicate"
    if al2o3 > 15 and sio2 > 55:
        return "aluminosilicate"
    if sio2 > 60:
        return "crown"
    return "uncertain"


def composition_plausibility(comp: pd.Series) -> dict[str, Any]:
    ox = comp.fillna(0)
    neg = (ox < -1e-6).any()
    total = ox.sum()
    ok_sum = abs(total - OXIDE_SUM_TARGET) <= OXIDE_SUM_TOL
    return {
        "oxide_sum_pct": total,
        "oxide_sum_ok": bool(ok_sum),
        "has_negative": bool(neg),
        "physically_plausible": bool(ok_sum and not neg),
    }


def match_one_glass(
    schott_row: pd.Series,
    schott_z_row: np.ndarray,
    sciglass: pd.DataFrame,
    sciglass_z: pd.DataFrame,
    feature_cols: list[str],
    oxide_cols: list[str],
    k: int = K_NEIGHBORS,
) -> pd.DataFrame:
    """ТОП-k соседей SciGlass для одного стекла SCHOTT."""
    idx_sg = sciglass_z.index.to_numpy()
    Z = sciglass_z.to_numpy()
    q = schott_z_row.reshape(1, -1)

    # Euclidean / kNN в z-пространстве
    nn = NearestNeighbors(n_neighbors=min(k, len(Z)), metric="euclidean")
    nn.fit(Z)
    dist_eucl, idx_eucl = nn.kneighbors(q)

    # Cosine (на z-признаках)
    cos_dist = pairwise_distances(q, Z, metric="cosine").ravel()
    order_cos = np.argsort(cos_dist)[:k]

    # Mahalanobis
    maha = _mahalanobis_distances(q.ravel(), Z)

    rows = []
    for rank, j in enumerate(idx_eucl[0], start=1):
        sg_idx = idx_sg[j]
        sg_row = sciglass.loc[sg_idx]
        comp = sg_row[oxide_cols].fillna(0)
        plaus = composition_plausibility(comp)
        family = classify_glass_family(sg_row, oxide_cols)

        rows.append({
            "glass_name": schott_row["glass_name"],
            "rank": rank,
            "sciglass_id": int(sg_row["sciglass_id"]),
            "distance_euclidean_z": float(dist_eucl[0, rank - 1]),
            "distance_cosine": float(cos_dist[j]),
            "distance_mahalanobis": float(maha[j]) if maha is not None else np.nan,
            "glass_family": family,
            **{f"oxide_{c}": comp[c] for c in oxide_cols},
            **{f"sg_{c}": sg_row[c] for c in feature_cols if c in sg_row.index},
            **plaus,
        })

    return pd.DataFrame(rows)


def aggregate_matches(matches: pd.DataFrame, oxide_cols: list[str]) -> dict[str, Any]:
    """Средний состав, std и 95% ДИ по ТОП-k (только физически правдоподобные, иначе все)."""
    sub = matches[matches["physically_plausible"]] if "physically_plausible" in matches.columns else matches
    if sub.empty:
        sub = matches
        uncertain = True
    else:
        uncertain = len(sub) < len(matches) / 2

    comp = sub[[f"oxide_{c}" for c in oxide_cols]].rename(columns=lambda x: x.replace("oxide_", ""))
    mean_c = comp.mean()
    std_c = comp.std(ddof=1) if len(comp) > 1 else comp.iloc[0] * 0
    n = len(comp)
    se = std_c / np.sqrt(n) if n > 1 else std_c
    ci_lo = mean_c - 1.96 * se
    ci_hi = mean_c + 1.96 * se

    plaus_mean = composition_plausibility(mean_c)
    family_mode = sub["glass_family"].mode().iloc[0] if len(sub) else "uncertain"

    return {
        "predicted_composition": mean_c.to_dict(),
        "composition_std": std_c.to_dict(),
        "composition_ci95_lo": ci_lo.to_dict(),
        "composition_ci95_hi": ci_hi.to_dict(),
        "n_neighbors_used": int(n),
        "high_uncertainty": uncertain,
        "glass_family_mode": family_mode,
        "best_distance_euclidean_z": float(matches["distance_euclidean_z"].iloc[0]),
        "best_sciglass_id": int(matches["sciglass_id"].iloc[0]),
        **plaus_mean,
    }


def plot_property_space(
    schott: pd.DataFrame,
    sciglass: pd.DataFrame,
    schott_z: pd.DataFrame,
    sciglass_z: pd.DataFrame,
    feature_cols: list[str],
    output_dir: Path,
) -> None:
    """PCA и UMAP: положение SCHOTT среди SciGlass."""
    import matplotlib.pyplot as plt

    try:
        import umap
    except ImportError:
        umap = None

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    Z_all = sciglass_z.to_numpy()
    pca = PCA(n_components=2, random_state=42)
    sg_pca = pca.fit_transform(Z_all)
    sh_pca = pca.transform(schott_z.to_numpy())

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(sg_pca[:, 0], sg_pca[:, 1], s=4, alpha=0.15, c="#348ABD", label="SciGlass")
    ax.scatter(sh_pca[:, 0], sh_pca[:, 1], s=80, c="#A60628", edgecolors="k", label="SCHOTT", zorder=5)
    for i, name in enumerate(schott.loc[schott_z.index, "glass_name"]):
        ax.annotate(name, (sh_pca[i, 0], sh_pca[i, 1]), fontsize=7, alpha=0.85)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("PCA: пространство свойств (z-score)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "pca_schott_sciglass.png", dpi=150)
    plt.close(fig)

    if umap is not None and len(Z_all) > 500:
        idx = np.random.default_rng(42).choice(len(Z_all), size=min(25000, len(Z_all)), replace=False)
        Z_sub = Z_all[idx]
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.2)
        sg_um = reducer.fit_transform(Z_sub)
        sh_um = reducer.transform(schott_z.to_numpy())

        fig, ax = plt.subplots(figsize=(12, 9))
        ax.scatter(sg_um[:, 0], sg_um[:, 1], s=4, alpha=0.2, c="#348ABD", label="SciGlass (subset)")
        ax.scatter(sh_um[:, 0], sh_um[:, 1], s=80, c="#A60628", edgecolors="k", label="SCHOTT", zorder=5)
        ax.set_title("UMAP: пространство свойств (z-score)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "umap_schott_sciglass.png", dpi=150)
        plt.close(fig)
    else:
        logger.warning("UMAP пропущен (нет umap-learn или мало данных)")


def run_matching(
    schott_xlsx: Path,
    sciglass_zip: Path,
    output_dir: Path,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    if feature_cols is None:
        feature_cols = MATCH_FEATURES

    schott = load_schott_catalog(schott_xlsx)
    sciglass = load_sciglass(sciglass_zip)
    oxide_cols = [c for c in OXIDE_MOL_COLS if c in sciglass.columns]

    schott_z, sciglass_z, scaler, mapping_report = build_feature_matrix(
        schott, sciglass, feature_cols
    )

    schott_used = schott.loc[schott_z.index].copy()
    sciglass_used = sciglass.loc[sciglass_z.index].copy()

    all_matches: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for idx in schott_z.index:
        row = schott_used.loc[idx]
        z_row = schott_z.loc[idx].to_numpy()
        matches = match_one_glass(
            row, z_row, sciglass_used, sciglass_z, feature_cols, oxide_cols, k=K_NEIGHBORS
        )
        all_matches.append(matches)
        agg = aggregate_matches(matches, oxide_cols)
        comp_str = "; ".join(f"{k}={v:.2f}" for k, v in agg["predicted_composition"].items() if v > 0.5)
        summaries.append({
            "glass_name": row["glass_name"],
            "matched_sciglass_id": agg["best_sciglass_id"],
            "distance": agg["best_distance_euclidean_z"],
            "predicted_composition": comp_str,
            "composition_std": json.dumps({k: round(v, 3) for k, v in agg["composition_std"].items()}),
            "glass_family": agg["glass_family_mode"],
            "oxide_sum_pct": agg.get("oxide_sum_pct", np.nan),
            "physically_plausible": agg.get("physically_plausible", False),
            "high_uncertainty": agg["high_uncertainty"],
            "n_neighbors_for_mean": agg["n_neighbors_used"],
        })

    matches_long = pd.concat(all_matches, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    output_dir.mkdir(parents=True, exist_ok=True)
    matches_long.to_csv(output_dir / "schott_sciglass_top20_matches.csv", index=False)
    summary_df.to_csv(output_dir / "schott_composition_summary.csv", index=False)

    meta = {
        "property_mapping": mapping_report,
        "features_used": feature_cols,
        "schott_raw_rows": len(schott),
        "schott_after_clean": len(schott_used),
        "sciglass_raw_rows": len(sciglass),
        "sciglass_after_clean": len(sciglass_used),
        "oxide_columns": oxide_cols,
        "k_neighbors": K_NEIGHBORS,
    }
    (output_dir / "match_report.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_property_space(schott_used, sciglass_used, schott_z, sciglass_z, feature_cols, output_dir)

    return {
        **meta,
        "summary": summary_df,
        "matches_long": matches_long,
    }


def print_report(result: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("(1) СОПОСТАВЛЕНИЕ КОЛОНОК SCHOTT -> SciGlass")
    print("=" * 60)
    for schott_col, sci_col in result["property_mapping"].items():
        desc = PROPERTY_MAP.get(schott_col.replace("SCHOTT.", ""), ("", ""))[1]
        print(f"  {schott_col:20s} -> {sci_col:12s}  ({desc})")

    print("\n" + "=" * 60)
    print("(2) СТЕКЛА ПОСЛЕ ОЧИСТКИ")
    print("=" * 60)
    print(f"  SCHOTT (каталог):     {result['schott_raw_rows']:>6} -> {result['schott_after_clean']:>6}")
    print(f"  SciGlass (база):      {result['sciglass_raw_rows']:>6} -> {result['sciglass_after_clean']:>6}")

    print("\n" + "=" * 60)
    print("(3) ПРИЗНАКИ В ПОИСКЕ (z-score по SciGlass)")
    print("=" * 60)
    for f in result["features_used"]:
        print(f"  - {f}")
    print(f"  kNN k={result['k_neighbors']}, cosine similarity, Mahalanobis (Ledoit-Wolf)")
    print(f"  Оксиды для состава: {', '.join(result['oxide_columns'][:8])}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="SCHOTT -> SciGlass composition matching")
    parser.add_argument("--schott-xlsx", type=Path, default=DEFAULT_SCHOTT_XLSX)
    parser.add_argument("--sciglass-zip", type=Path, default=DEFAULT_SCIGLASS_ZIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    result = run_matching(args.schott_xlsx, args.sciglass_zip, args.output)
    print_report(result)
    logger.info("Результаты: %s", args.output.resolve())


if __name__ == "__main__":
    main()
