"""Единый стиль презентационных графиков по оптическим стёклам.

    python viz_presentation.py          # все 5 рисунков
    python make_presentation_assets.py  # презентация + TOP10
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

from match_schott_sciglass import (
    DEFAULT_SCIGLASS_ZIP,
    DEFAULT_SCHOTT_XLSX,
    MATCH_FEATURES,
    build_feature_matrix,
    load_schott_catalog,
    load_sciglass,
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SCHOTT = PROJECT_DIR / "schott-optical-glass-overview-excel-format-en 202501113.xlsx"

# --- Пороги (из отчётов, не подгоняются под картинку) ---
ND_TARGET = 1.80
DIST_P95 = 1.415
DIST_TOP_MAX = 1.85
DIST_TOP_MIN = 0.25

# --- Палитра (Okabe–Ito, colorblind-friendly) ---
C_SG = "#BDBDBD"
C_SG_EDGE = "none"
C_SCHOTT = "#D55E00"
C_FEASIBLE = "#0072B2"
C_TOP = "#E69F00"
C_TOP_EDGE = "#000000"
C_MEDIAN = "#CC6677"
C_TARGET = "#882255"
C_FAINT = "#CCCCCC"
C_EXTRAP = "#AA4499"

# Шрифты
FONT_TITLE = 17
FONT_LABEL = 14
FONT_TICK = 12
FONT_LEGEND = 11
FONT_ANNOT = 11
DPI = 220

KEY_SCHOTT_GLASSES = [
    "N-BK7",
    "N-SF6",
    "N-SF11",
    "N-SF57",
    "N-LASF31",
    "N-LASF41",
    "SF6",
    "F2",
    "N-BAF10",
    "N-KZFS2",
    "N-LAK22",
    "N-BALF4",
]


def apply_presentation_theme() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titlesize": FONT_TITLE,
            "axes.titleweight": "bold",
            "axes.labelsize": FONT_LABEL,
            "axes.labelweight": "medium",
            "axes.titlepad": 12,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_LEGEND,
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "grid.alpha": 0.22,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
        }
    )


def _style_axes(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, axis="both")
    else:
        ax.grid(False)


def _add_nd_target_line(ax: plt.Axes, ymax: float | None = None) -> None:
    ax.axvline(
        ND_TARGET,
        color=C_TARGET,
        ls=(0, (6, 4)),
        lw=2.0,
        zorder=2,
        label=rf"цель $n_d$ = {ND_TARGET:.2f}",
    )
    y1, y2 = ax.get_ylim()
    y_hi = ymax if ymax is not None else y2
    ax.axvspan(ND_TARGET, ax.get_xlim()[1], color=C_TARGET, alpha=0.06, zorder=0)


def _distance_cmap() -> mcolors.LinearSegmentedColormap:
    """Спокойная последовательная шкала: близко → далеко."""
    return mcolors.LinearSegmentedColormap.from_list(
        "dist_cb",
        ["#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#882255"],
        N=256,
    )


def _format_distance_colorbar(fig: plt.Figure, mappable, label: str) -> None:
    cbar = fig.colorbar(mappable, ax=mappable.axes, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=FONT_LABEL - 1, labelpad=8)
    cbar.ax.tick_params(labelsize=FONT_TICK - 1)
    ticks = [0.5, 1.0, 1.5, 2.0]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1f}" for t in ticks])


def select_top10(
    cands: pd.DataFrame,
    dist_min: float = DIST_TOP_MIN,
    dist_max: float = DIST_TOP_MAX,
) -> pd.DataFrame:
    sub = cands[
        (cands["PbO"] <= 0.01)
        & (cands["n_pred"] >= ND_TARGET)
        & (cands["distance_to_training"] >= dist_min)
        & (cands["distance_to_training"] <= dist_max)
    ].copy()
    if len(sub) < 10:
        sub = cands[
            (cands["PbO"] <= 0.01)
            & (cands["n_pred"] >= ND_TARGET)
            & (cands["distance_to_training"] <= 2.0)
        ].copy()
        sub = sub[sub["distance_to_training"] >= dist_min]

    sub["score"] = sub["n_pred"] - 0.35 * sub["distance_to_training"]
    sub = sub.sort_values(["score", "distance_to_training"], ascending=[False, True])
    sub = sub.drop_duplicates(subset=["composition"], keep="first")

    picked: list[pd.Series] = []
    for _, row in sub.iterrows():
        if len(picked) >= 10:
            break
        picked.append(row)
    out = pd.DataFrame(picked)
    if "cWGAN-GP" in sub["source"].values and (out["source"] == "cWGAN-GP").sum() < 2:
        gan = sub[sub["source"] == "cWGAN-GP"].head(3)
        out = (
            pd.concat([out.head(8), gan], ignore_index=True)
            .drop_duplicates(subset=["composition"], keep="first")
            .head(10)
        )
    return out.reset_index(drop=True)


def plot_high_index_candidates(
    sg: pd.DataFrame,
    cands: pd.DataFrame,
    out_path: Path,
) -> None:
    """Главный слайд: фокус на n_d > 1.80."""
    apply_presentation_theme()
    top10 = select_top10(cands)
    top_comp = set(top10["composition"])

    focus = cands[
        (cands["PbO"] <= 0.01)
        & (cands["n_pred"] >= 1.75)
        & cands["n_pred"].notna()
        & cands["vd_pred"].notna()
    ].copy()
    focus_rest = focus[~focus["composition"].isin(top_comp)]
    focus_top = focus[focus["composition"].isin(top_comp)]

    sg_win = sg[(sg["nd"] >= 1.72) & (sg["vd"] >= 5)].copy()

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.scatter(
        sg_win["nd"],
        sg_win["vd"],
        s=14,
        alpha=0.35,
        c=C_SG,
        edgecolors=C_SG_EDGE,
        rasterized=True,
        zorder=1,
    )

    cmap = _distance_cmap()
    vmin, vmax = 0.35, 2.2
    if len(focus_rest):
        sc = ax.scatter(
            focus_rest["n_pred"],
            focus_rest["vd_pred"],
            c=focus_rest["distance_to_training"],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=42,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
    else:
        sc = ax.scatter([], [], c=[], cmap=cmap, vmin=vmin, vmax=vmax)

    if len(focus_top):
        ax.scatter(
            focus_top["n_pred"],
            focus_top["vd_pred"],
            s=200,
            c=C_TOP,
            edgecolors=C_TOP_EDGE,
            linewidths=1.8,
            zorder=6,
            label=f"ТОП-10 (PbO=0, {DIST_TOP_MIN:.2f}≤dist≤{DIST_TOP_MAX:.2f})",
        )

    _add_nd_target_line(ax)

    x_lo = 1.78
    x_hi = max(2.15, float(focus["n_pred"].max()) + 0.03) if len(focus) else 2.15
    y_lo = max(5, float(np.nanpercentile(focus["vd_pred"], 2)) - 3) if len(focus) else 10
    y_hi = min(65, float(np.nanpercentile(focus["vd_pred"], 98)) + 5) if len(focus) else 55
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    ax.set_xlabel(r"Показатель преломления $n_d$ (модель)")
    ax.set_ylabel(r"Число Аббе $\nu_d$ (модель)")
    ax.set_title("Бессвинцовые кандидаты: зона $n_d > 1{,}80$")

    handles = [
        Patch(facecolor=C_SG, edgecolor="none", label="SciGlass (фон, ND>1.70)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=C_TOP,
            markeredgecolor=C_TOP_EDGE,
            markeredgewidth=1.5,
            markersize=11,
            label=f"ТОП-10",
        ),
        Line2D([0], [0], color=C_TARGET, ls=(0, (6, 4)), lw=2, label=rf"$n_d$ = {ND_TARGET:.2f}"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.95)
    _style_axes(ax)

    if len(focus_rest):
        _format_distance_colorbar(
            fig,
            sc,
            "Расстояние до обучающей выборки\n(ниже — ближе к известным стёклам)",
        )

    ax.text(
        0.98,
        0.04,
        f"порог риска dist ≈ {DIST_P95:.2f} (p95)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_ANNOT,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#dddddd"),
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _binned_median(
    x: pd.Series,
    y: pd.Series,
    bins: np.ndarray,
) -> tuple[list[float], list[float]]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    df["bin"] = pd.cut(df["x"], bins)
    med = df.groupby("bin", observed=True)["y"].median()
    centers = [float(interval.mid) for interval in med.index]
    return centers, med.tolist()


def plot_uncertainty_vs_nd(
    design: pd.DataFrame,
    recovery: pd.DataFrame,
    out_path: Path,
) -> None:
    apply_presentation_theme()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharex=True)

    bins = np.arange(1.68, 2.42, 0.06)
    xlim = (1.68, 2.28)

    # --- Design ---
    ax = axes[0]
    d = design[(design["PbO"] <= 0.01) & design["n_pred"].notna()].copy()
    ax.scatter(
        d["n_pred"],
        d["distance_to_training"],
        s=22,
        alpha=0.22,
        c=C_FEASIBLE,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )
    centers, meds = _binned_median(d["n_pred"], d["distance_to_training"], bins)
    ax.plot(centers, meds, "o-", color=C_MEDIAN, lw=2.8, ms=7, zorder=4, label="медиана distance")
    _add_nd_target_line(ax)
    ax.axhline(DIST_P95, color=C_EXTRAP, ls=":", lw=1.8, label=rf"dist p95 = {DIST_P95:.2f}")
    ax.set_xlim(*xlim)
    ax.set_ylim(0, min(3.5, float(d["distance_to_training"].quantile(0.995)) + 0.2))
    ax.set_ylabel("distance (прокси неопределённости)")
    ax.set_title("Проектирование\n(NSGA-II + cWGAN-GP)")
    ax.legend(loc="upper left", framealpha=0.95)
    _style_axes(ax, grid=True)

    # --- Recovery ---
    ax = axes[1]
    r = recovery.dropna(subset=["n_catalog", "uncertainty"])
    ax.scatter(
        r["n_catalog"],
        r["uncertainty"],
        s=36,
        alpha=0.45,
        c=C_SCHOTT,
        edgecolors="white",
        linewidths=0.3,
        zorder=1,
    )
    centers_r, meds_r = _binned_median(r["n_catalog"], r["uncertainty"], bins)
    ax.plot(
        centers_r,
        meds_r,
        "o-",
        color=C_MEDIAN,
        lw=2.8,
        ms=7,
        zorder=4,
        label="медиана uncertainty",
    )
    _add_nd_target_line(ax)
    ax.set_xlim(*xlim)
    ax.set_ylabel("uncertainty_score")
    ax.set_title("Восстановление состава\n(SCHOTT → SciGlass, v3)")
    ax.legend(loc="upper left", framealpha=0.95)
    _style_axes(ax, grid=True)

    for ax in axes:
        ax.set_xlabel(r"Показатель преломления $n_d$")

    fig.suptitle(
        "При росте $n_d$ неопределённость растёт",
        fontsize=FONT_TITLE + 1,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pareto_candidates(
    candidates: pd.DataFrame,
    top20: pd.DataFrame,
    out_path: Path,
) -> None:
    apply_presentation_theme()
    if not len(candidates):
        return

    top_comp = set(top20["composition"]) if len(top20) else set()
    rest = candidates[~candidates["composition"].isin(top_comp)]
    tops = candidates[candidates["composition"].isin(top_comp)]

    fig, ax = plt.subplots(figsize=(10.5, 7))
    cmap = _distance_cmap()
    vmin, vmax = 0.4, 3.0

    if len(rest):
        sc = ax.scatter(
            rest["ND300_pred"],
            rest["NUD300_pred"],
            c=rest["distance_to_training"],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=36,
            alpha=0.55,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
    else:
        sc = ax.scatter([], [], c=[], cmap=cmap)

    if len(tops):
        ax.scatter(
            tops["ND300_pred"],
            tops["NUD300_pred"],
            s=160,
            facecolors=C_TOP,
            edgecolors=C_TOP_EDGE,
            linewidths=1.6,
            zorder=5,
            label="ТОП-20 (PbO=0, $n_d$>1.80, feasible)",
        )

    _add_nd_target_line(ax)
    ax.set_xlim(1.78, max(2.12, float(candidates["ND300_pred"].max()) + 0.02))
    vd_lo = max(8, float(candidates["NUD300_pred"].quantile(0.02)) - 2)
    vd_hi = min(60, float(candidates["NUD300_pred"].quantile(0.98)) + 3)
    ax.set_ylim(vd_lo, vd_hi)

    ax.set_xlabel(r"$n_d$ (предсказание модели)")
    ax.set_ylabel(r"$\nu_d$ (предсказание модели)")
    ax.set_title("NSGA-II: компромисс $n_d$, $\\nu_d$ и новизна")

    handles = [
        Line2D([0], [0], color=C_TARGET, ls=(0, (6, 4)), lw=2, label=rf"цель $n_d$ = {ND_TARGET:.2f}"),
    ]
    if len(tops):
        handles.insert(
            0,
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=C_TOP,
                markeredgecolor=C_TOP_EDGE,
                markeredgewidth=1.5,
                markersize=11,
                label="ТОП-20",
            ),
        )
    ax.legend(handles=handles, loc="upper right", framealpha=0.95)
    _style_axes(ax)
    if len(rest):
        _format_distance_colorbar(fig, sc, "Расстояние до обучающей выборки")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_design_evaluation(
    candidates: pd.DataFrame,
    top20: pd.DataFrame,
    dist_threshold: float,
    out_path: Path,
) -> None:
    """Две панели: свойства и риск экстраполяции."""
    apply_presentation_theme()
    top_comp = set(top20["composition"]) if len(top20) else set()
    feas = candidates[candidates["feasible"] == True].copy()
    infeas = candidates[candidates["feasible"] != True].copy()
    top = feas[feas["composition"].isin(top_comp)]
    feas_rest = feas[~feas["composition"].isin(top_comp)]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6))

    # --- Левая: n_d vs nu_d ---
    ax = axes[0]
    if len(infeas):
        ax.scatter(
            infeas["ND300_pred"],
            infeas["NUD300_pred"],
            s=22,
            c=C_FAINT,
            alpha=0.35,
            edgecolors="none",
            label="Pareto (не feasible)",
            rasterized=True,
            zorder=1,
        )
    if len(feas_rest):
        ax.scatter(
            feas_rest["ND300_pred"],
            feas_rest["NUD300_pred"],
            s=55,
            c=C_FEASIBLE,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            label=f"Feasible (n={len(feas)})",
            zorder=3,
        )
    if len(top):
        ax.scatter(
            top["ND300_pred"],
            top["NUD300_pred"],
            s=150,
            c=C_TOP,
            edgecolors=C_TOP_EDGE,
            linewidths=1.6,
            label="ТОП-20",
            zorder=5,
        )
    _add_nd_target_line(ax)
    ax.set_xlim(1.78, max(2.15, float(candidates["ND300_pred"].max()) + 0.02))
    ax.set_xlabel(r"$n_d$ (предсказание)")
    ax.set_ylabel(r"$\nu_d$ (предсказание)")
    ax.set_title("Пространство свойств")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=FONT_LEGEND - 1)
    _style_axes(ax)

    # --- Правая: n_d vs distance ---
    ax = axes[1]
    if len(infeas):
        ax.scatter(
            infeas["ND300_pred"],
            infeas["distance_to_training"],
            s=20,
            c=C_FAINT,
            alpha=0.3,
            edgecolors="none",
            rasterized=True,
            zorder=1,
        )
    if len(feas_rest):
        ax.scatter(
            feas_rest["ND300_pred"],
            feas_rest["distance_to_training"],
            s=55,
            c=C_FEASIBLE,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
    if len(top):
        ax.scatter(
            top["ND300_pred"],
            top["distance_to_training"],
            s=150,
            c=C_TOP,
            edgecolors=C_TOP_EDGE,
            linewidths=1.6,
            zorder=5,
        )
    _add_nd_target_line(ax)
    ax.axhline(
        dist_threshold,
        color=C_EXTRAP,
        ls=":",
        lw=2,
        label=rf"порог dist p95 = {dist_threshold:.2f}",
    )
    y_max = min(4.0, float(candidates["distance_to_training"].quantile(0.99)) + 0.3)
    ax.set_ylim(0, y_max)
    ax.set_xlim(1.78, max(2.15, float(candidates["ND300_pred"].max()) + 0.02))
    ax.set_xlabel(r"$n_d$ (предсказание)")
    ax.set_ylabel("Расстояние до обучающей выборки")
    ax.set_title("Риск экстраполяции surrogate")
    ax.legend(loc="upper left", framealpha=0.95, fontsize=FONT_LEGEND - 1)
    _style_axes(ax)

    ax.text(
        0.98,
        0.95,
        "выше порога →\nэкстраполяция",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_ANNOT,
        color=C_EXTRAP,
        fontweight="medium",
    )

    fig.suptitle(
        "NSGA-II (2 ч): кандидаты в пространстве свойств",
        fontsize=FONT_TITLE + 1,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _pick_schott_labels(schott: pd.DataFrame, names: Iterable[str]) -> list[str]:
    catalog = set(schott["glass_name"].astype(str))
    picked = [n for n in names if n in catalog]
    if len(picked) < 6 and "nd" in schott.columns:
        hi = schott.nlargest(3, "nd")["glass_name"].astype(str).tolist()
        lo = schott.nsmallest(2, "nd")["glass_name"].astype(str).tolist()
        for n in hi + lo:
            if n not in picked:
                picked.append(n)
    return picked[:14]


def plot_pca_property_space(
    schott: pd.DataFrame,
    schott_z: pd.DataFrame,
    sciglass_z: pd.DataFrame,
    out_path: Path,
) -> None:
    apply_presentation_theme()
    Z_all = sciglass_z.to_numpy()
    pca = PCA(n_components=2, random_state=42)
    sg_pca = pca.fit_transform(Z_all)
    sh_pca = pca.transform(schott_z.to_numpy())

    rng = np.random.default_rng(42)
    n_sg = len(sg_pca)
    if n_sg > 30000:
        idx = rng.choice(n_sg, size=30000, replace=False)
        sg_plot = sg_pca[idx]
    else:
        sg_plot = sg_pca

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.scatter(
        sg_plot[:, 0],
        sg_plot[:, 1],
        s=10,
        alpha=0.28,
        c=C_SG,
        edgecolors="none",
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        sh_pca[:, 0],
        sh_pca[:, 1],
        s=70,
        c=C_SCHOTT,
        edgecolors="white",
        linewidths=0.8,
        zorder=4,
        label="SCHOTT (каталог)",
    )

    label_names = _pick_schott_labels(schott.loc[schott_z.index], KEY_SCHOTT_GLASSES)
    name_to_idx = {
        str(schott.loc[schott_z.index[i], "glass_name"]): i
        for i in range(len(sh_pca))
    }
    offsets = [(6, 6), (-6, 8), (8, -8), (-10, -6), (0, 10), (10, 0)]
    for k, name in enumerate(label_names):
        if name not in name_to_idx:
            continue
        i = name_to_idx[name]
        dx, dy = offsets[k % len(offsets)]
        ax.annotate(
            name,
            (sh_pca[i, 0], sh_pca[i, 1]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=FONT_ANNOT,
            color="#333333",
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6, shrinkA=2, shrinkB=2),
            zorder=6,
        )

    pc1 = pca.explained_variance_ratio_[0] * 100
    pc2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({pc1:.0f}% дисперсии)")
    ax.set_ylabel(f"PC2 ({pc2:.0f}% дисперсии)")
    ax.set_title("PCA: SCHOTT в пространстве свойств SciGlass")
    ax.legend(loc="best", framealpha=0.95)
    _style_axes(ax, grid=True)

    ax.text(
        0.02,
        0.02,
        "4 свойства: $n_d$, $\\nu_d$, $\\rho$, $T_g$ (z-score)",
        transform=ax.transAxes,
        fontsize=FONT_ANNOT - 1,
        color="#555555",
        va="bottom",
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_sci_glass_nd_vd() -> pd.DataFrame:
    df = pd.read_parquet(
        PROJECT_DIR / "merged_data.parquet",
        columns=["RefractiveIndex", "AbbeNum", "PBO"],
    )
    nd = pd.to_numeric(df["RefractiveIndex"], errors="coerce")
    vd = pd.to_numeric(df["AbbeNum"], errors="coerce")
    pbo = pd.to_numeric(df["PBO"], errors="coerce").fillna(0)
    m = (nd > 1.70) & (pbo <= 1) & nd.notna() & vd.notna()
    return pd.DataFrame({"nd": nd[m], "vd": vd[m]})


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
    return out.rename(columns={"ND300_pred": "n_pred", "NUD300_pred": "vd_pred"})


def load_recovery_uncertainty() -> pd.DataFrame:
    from match_schott_sciglass import load_schott_catalog

    cand = pd.read_csv(PROJECT_DIR / "output_v3/composition_candidates.csv")
    schott = load_schott_catalog(DEFAULT_SCHOTT)
    schott = schott.rename(columns={"nd": "n_catalog", "vd": "vd_catalog"})
    m = cand.merge(schott[["glass_name", "n_catalog", "vd_catalog"]], on="glass_name", how="left")
    m["uncertainty"] = m["uncertainty_score"]
    return m


def regenerate_all_figures(
    project_dir: Path | None = None,
) -> dict[str, Path]:
    """Пересобрать все 5 презентационных рисунков из существующих CSV."""
    root = project_dir or PROJECT_DIR
    paths: dict[str, Path] = {}

    pres = root / "output" / "presentation" / "figures"
    pres.mkdir(parents=True, exist_ok=True)

    sg = load_sci_glass_nd_vd()
    cands = load_design_candidates()
    recovery = load_recovery_uncertainty()

    p1 = pres / "presentation_nd_vd_distance.png"
    plot_high_index_candidates(sg, cands, p1)
    paths["main_scatter"] = p1

    p2 = pres / "uncertainty_vs_nd.png"
    plot_uncertainty_vs_nd(cands, recovery, p2)
    paths["uncertainty"] = p2

    nsga_csv = root / "output" / "inverse_design_2h" / "all_pareto_candidates.csv"
    top20_csv = root / "output" / "inverse_design_2h" / "top_20_lead_free_high_n.csv"
    report_json = root / "output" / "inverse_design_2h" / "inverse_design_report.json"

    if nsga_csv.exists():
        pareto = pd.read_csv(nsga_csv)
        top20 = pd.read_csv(top20_csv) if top20_csv.exists() else pareto.head(0)
        dist_thr = DIST_P95
        if report_json.exists():
            import json

            dist_thr = float(
                json.loads(report_json.read_text(encoding="utf-8")).get(
                    "distance_threshold_p95", DIST_P95
                )
            )

        p3 = root / "output" / "inverse_design_2h" / "figures" / "pareto_nd_vd.png"
        plot_pareto_candidates(pareto, top20, p3)
        paths["pareto"] = p3

        p4 = root / "output" / "inverse_design_2h" / "figures" / "design_evaluation_summary.png"
        plot_design_evaluation(pareto, top20, dist_thr, p4)
        paths["design_eval"] = p4

    schott_xlsx = DEFAULT_SCHOTT_XLSX
    sciglass_zip = DEFAULT_SCIGLASS_ZIP
    if schott_xlsx.exists() and sciglass_zip.exists():
        schott = load_schott_catalog(schott_xlsx)
        sciglass = load_sciglass(sciglass_zip)
        schott_z, sciglass_z, _, _ = build_feature_matrix(schott, sciglass, MATCH_FEATURES)
        p5 = root / "output" / "schott_match" / "figures" / "pca_schott_sciglass.png"
        plot_pca_property_space(schott.loc[schott_z.index], schott_z, sciglass_z, p5)
        paths["pca"] = p5

    return paths


def main() -> None:
    paths = regenerate_all_figures()
    print("Презентационные графики обновлены:")
    for k, p in paths.items():
        print(f"  [{k}] {p}")


if __name__ == "__main__":
    main()
