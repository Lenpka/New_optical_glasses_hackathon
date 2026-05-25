"""Презентационные артефакты: главный scatter, ТОП-10, uncertainty vs n_d, все 5 слайдов.

    python make_presentation_assets.py

Выход: output/presentation/ (+ pareto, design_eval, PCA в своих папках)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from viz_presentation import (
    DIST_TOP_MAX,
    DIST_TOP_MIN,
    ND_TARGET,
    load_design_candidates,
    load_recovery_uncertainty,
    load_sci_glass_nd_vd,
    plot_high_index_candidates,
    plot_uncertainty_vs_nd,
    regenerate_all_figures,
    select_top10,
)

PROJECT_DIR = Path(__file__).resolve().parent
OUT = PROJECT_DIR / "output" / "presentation"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)

    cands = load_design_candidates()
    top10 = select_top10(cands)
    table = top10[
        [
            "composition",
            "n_pred",
            "vd_pred",
            "PbO",
            "distance_to_training",
            "glass_family",
            "source",
        ]
    ].rename(
        columns={
            "vd_pred": "nu_pred",
            "glass_family": "family",
            "source": "method",
        }
    )
    table.insert(0, "rank", range(1, len(table) + 1))
    table.to_csv(OUT / "TOP10_candidates.csv", index=False)
    try:
        table.to_markdown(OUT / "TOP10_candidates.md", index=False)
    except ImportError:
        pass

    paths = regenerate_all_figures()

    recovery = load_recovery_uncertainty()
    d = cands[(cands["PbO"] <= 0.01) & (cands["n_pred"] >= ND_TARGET)]
    above = d[d["n_pred"] >= ND_TARGET]
    stats = {
        "design_feasible_n": len(cands),
        "nd_ge_1.8_n": len(above),
        "median_dist_nd_lt_1.8": float(
            d[d["n_pred"] < ND_TARGET]["distance_to_training"].median()
        )
        if (d["n_pred"] < ND_TARGET).any()
        else None,
        "median_dist_nd_ge_1.8": float(above["distance_to_training"].median()),
        "median_unc_recovery_nd_lt_1.8": float(
            recovery[recovery["n_catalog"] < ND_TARGET]["uncertainty"].median()
        ),
        "median_unc_recovery_nd_ge_1.8": float(
            recovery[recovery["n_catalog"] >= ND_TARGET]["uncertainty"].median()
        ),
        "top10_dist_range": [DIST_TOP_MIN, DIST_TOP_MAX],
    }
    (OUT / "presentation_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Готово:", OUT.resolve())
    for name, p in paths.items():
        print(f"  [{name}] {p}")
    print("  TOP10_candidates.csv")


if __name__ == "__main__":
    main()
