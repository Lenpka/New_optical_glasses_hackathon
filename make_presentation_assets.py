"""Презентационные артефакты: главный scatter, ТОП-10, uncertainty vs n_d.

    python make_presentation_assets.py

Выход: output/presentation/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from match_schott_sciglass import load_schott_catalog, PROPERTY_MAP

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SCHOTT = PROJECT_DIR / "schott-optical-glass-overview-excel-format-en 202501113.xlsx"
OUT = PROJECT_DIR / "output" / "presentation"


def load_sci_glass_nd_vd() -> pd.DataFrame:
    df = pd.read_parquet(
        PROJECT_DIR / "merged_data.parquet",
        columns=["RefractiveIndex", "AbbeNum", "PBO"],
    )
    nd = pd.to_numeric(df["RefractiveIndex"], errors="coerce")
    vd = pd.to_numeric(df["AbbeNum"], errors="coerce")
    pbo = pd.to_numeric(df["PBO"], errors="coerce").fillna(0)
    m = (nd > 1.70) & (pbo <= 1) & nd.notna() & vd.notna()
    return pd.DataFrame({"nd": nd[m], "vd": vd[m], "distance": np.nan})


def load_design_candidates() -> pd.DataFrame:
    rows = []
    for path, src in [
        (PROJECT_DIR / "output/inverse_design_2h/all_pareto_candidates.csv", "NSGA-II"),
        (PROJECT_DIR / "output/gan_design/all_gan_candidates.csv", "cWGAN-GP"),
    ]:
        if not path.exists():
            continue
        d = pd.read_csv(path)
        d = d[d["feasible"] == True].copy() if "feasible" in d.columns else d
        d["source"] = src
        rows.append(d)
    if not rows:
        raise FileNotFoundError("Нет CSV design-кандидатов (inverse_design_2h / gan_design)")
    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"ND300_pred": "n_pred", "NUD300_pred": "vd_pred"})
    return out


def select_top10(
    cands: pd.DataFrame,
    dist_min: float = 0.25,
    dist_max: float = 1.85,
) -> pd.DataFrame:
    """PbO=0, n>=1.8, умеренная distance (не дубликат, не экстраполяция)."""
    sub = cands[
        (cands["PbO"] <= 0.01)
        & (cands["n_pred"] >= 1.80)
        & (cands["distance_to_training"] >= dist_min)
        & (cands["distance_to_training"] <= dist_max)
    ].copy()
    if len(sub) < 10:
        sub = cands[
            (cands["PbO"] <= 0.01)
            & (cands["n_pred"] >= 1.80)
            & (cands["distance_to_training"] <= 2.0)
        ].copy()
        sub = sub[sub["distance_to_training"] >= dist_min]

    sub["score"] = sub["n_pred"] - 0.35 * sub["distance_to_training"]
    sub = sub.sort_values(["score", "distance_to_training"], ascending=[False, True])
    sub = sub.drop_duplicates(subset=["composition"], keep="first")

    # хотя бы 2 метода в топе
    picked = []
    for _, row in sub.iterrows():
        if len(picked) >= 10:
            break
        picked.append(row)
    out = pd.DataFrame(picked)
    if "cWGAN-GP" in sub["source"].values and (out["source"] == "cWGAN-GP").sum() < 2:
        gan = sub[sub["source"] == "cWGAN-GP"].head(3)
        out = pd.concat([out.head(8), gan], ignore_index=True).drop_duplicates(
            subset=["composition"], keep="first"
        ).head(10)
    return out.reset_index(drop=True)


def plot_main_scatter(sg: pd.DataFrame, cands: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(
        sg["nd"], sg["vd"],
        s=8, alpha=0.15, c="#a6cee3", edgecolors="none",
        label=f"SciGlass (ND>1.70, PbO≤1, n={len(sg):,})",
        zorder=1,
    )

    sizes = cands["source"].map({"NSGA-II": 50, "cWGAN-GP": 28}).fillna(35)
    sc = ax.scatter(
        cands["n_pred"], cands["vd_pred"],
        c=cands["distance_to_training"],
        s=sizes,
        alpha=0.7,
        cmap="viridis_r",
        edgecolors="k",
        linewidths=0.35,
        zorder=3,
    )
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("distance_to_training\n(ниже = ближе к SciGlass)")

    top10 = select_top10(cands)
    ax.scatter(
        top10["n_pred"], top10["vd_pred"],
        s=120, facecolors="none", edgecolors="#FFB000", linewidths=2.2,
        label="ТОП-10 (Pb=0, dist≤2)", zorder=5,
    )

    ax.axvline(1.80, color="#e31a1c", ls="--", lw=2, label=r"$n_d$ = 1.80 (цель)")
    ax.axvspan(1.80, ax.get_xlim()[1] if ax.get_xlim()[1] > 1.8 else 2.5, alpha=0.06, color="red")

    ax.set_xlabel(r"$n_d$ (predicted)", fontsize=12)
    ax.set_ylabel(r"Abbe number $\nu_d$ (predicted)", fontsize=12)
    ax.set_title(
        "High-index lead-free design candidates\n"
        "цвет = новизна (distance to SciGlass); риск экстраполяции при distance > 1.4",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.grid(alpha=0.25)
    ax.set_xlim(left=1.65)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_recovery_uncertainty() -> pd.DataFrame:
    """SCHOTT recovery (v3): uncertainty vs каталожный n_d."""
    cand = pd.read_csv(PROJECT_DIR / "output_v3/composition_candidates.csv")
    schott = load_schott_catalog(DEFAULT_SCHOTT)
    schott = schott.rename(columns={"nd": "n_catalog", "vd": "vd_catalog"})
    m = cand.merge(schott[["glass_name", "n_catalog", "vd_catalog"]], on="glass_name", how="left")
    m["uncertainty"] = m["uncertainty_score"]
    m["kind"] = "recovery (SCHOTT→SciGlass)"
    return m


def plot_uncertainty_vs_nd(design: pd.DataFrame, recovery: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Design: distance как proxy uncertainty
    ax = axes[0]
    d = design[(design["PbO"] <= 0.01) & design["n_pred"].notna()].copy()
    ax.scatter(d["n_pred"], d["distance_to_training"], c="#984EA3", s=18, alpha=0.35, label="feasible")
    ax.axvline(1.80, color="red", ls="--", lw=1.5)
    ax.axhline(1.415, color="purple", ls=":", lw=1, label="dist p95 = 1.41")
    # binned median
    bins = np.arange(1.70, 2.45, 0.05)
    d["bin"] = pd.cut(d["n_pred"], bins)
    med = d.groupby("bin", observed=True)["distance_to_training"].median()
    centers = [interval.mid for interval in med.index]
    ax.plot(centers, med.values, "o-", color="#e31a1c", lw=2, markersize=6, label="median distance")
    ax.set_xlabel(r"$n_d$ predicted")
    ax.set_ylabel("distance_to_training")
    ax.set_title("Design (NSGA-II + GAN)\nproxy неопределённости")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # Recovery: uncertainty_score vs catalog n
    ax = axes[1]
    r = recovery.dropna(subset=["n_catalog", "uncertainty"])
    ax.scatter(r["n_catalog"], r["uncertainty"], s=40, alpha=0.6, c="#348ABD")
    ax.axvline(1.80, color="red", ls="--", lw=1.5)
    r["bin"] = pd.cut(r["n_catalog"], bins)
    med_u = r.groupby("bin", observed=True)["uncertainty"].median()
    centers_r = [interval.mid for interval in med_u.index]
    ax.plot(centers_r, med_u.values, "o-", color="#e31a1c", lw=2, markersize=6, label="median uncertainty")
    ax.set_xlabel(r"$n_d$ catalog (SCHOTT)")
    ax.set_ylabel("uncertainty_score")
    ax.set_title("Recovery (v3)\nvar(comp) × mean(distance)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "Растёт ли неопределённость при $n_d > 1.8$? "
        "Да: median distance/uncertainty выше справа от 1.8",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    sg = load_sci_glass_nd_vd()
    cands = load_design_candidates()

    plot_main_scatter(sg, cands, OUT / "figures" / "presentation_nd_vd_distance.png")

    top10 = select_top10(cands)
    table = top10[[
        "composition", "n_pred", "vd_pred", "PbO",
        "distance_to_training", "glass_family", "source",
    ]].rename(columns={
        "n_pred": "n_pred",
        "vd_pred": "nu_pred",
        "glass_family": "family",
        "source": "method",
    })
    table.insert(0, "rank", range(1, len(table) + 1))
    table.to_csv(OUT / "TOP10_candidates.csv", index=False)
    try:
        table.to_markdown(OUT / "TOP10_candidates.md", index=False)
    except ImportError:
        pass

    recovery = load_recovery_uncertainty()
    plot_uncertainty_vs_nd(cands, recovery, OUT / "figures" / "uncertainty_vs_nd.png")

    # краткая статистика для отчёта
    d = cands[(cands["PbO"] <= 0.01) & (cands["n_pred"] >= 1.80)]
    below = d[d["n_pred"] < 1.80]
    above = d[d["n_pred"] >= 1.80]
    stats = {
        "design_feasible_n": len(cands),
        "nd_ge_1.8_n": len(above),
        "median_dist_nd_lt_1.8": float(d[d["n_pred"] < 1.80]["distance_to_training"].median()) if (d["n_pred"] < 1.80).any() else None,
        "median_dist_nd_ge_1.8": float(above["distance_to_training"].median()),
        "median_unc_recovery_nd_lt_1.8": float(recovery[recovery["n_catalog"] < 1.80]["uncertainty"].median()),
        "median_unc_recovery_nd_ge_1.8": float(recovery[recovery["n_catalog"] >= 1.80]["uncertainty"].median()),
    }
    import json
    (OUT / "presentation_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print("Готово:", OUT.resolve())
    print("  figures/presentation_nd_vd_distance.png")
    print("  figures/uncertainty_vs_nd.png")
    print("  TOP10_candidates.csv")


if __name__ == "__main__":
    main()
