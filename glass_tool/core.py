"""Ядро: восстановление состава (SciGlass) и генерация (cWGAN-GP)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from match_schott_sciglass import DEFAULT_SCHOTT_XLSX, DEFAULT_SCIGLASS_ZIP, load_schott_catalog
from match_schott_sciglass_v3 import recover_from_properties, recover_from_schott_name

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_GAN_DIR = PROJECT_DIR / "output" / "gan_design"
DEFAULT_FORWARD_DIR = PROJECT_DIR / "output" / "inverse_design_2h" / "forward_models"
DEFAULT_DATA = PROJECT_DIR / "merged_data.parquet"


@dataclass
class ToolPaths:
    """Пути к данным и обученным моделям."""

    sciglass_zip: Path = DEFAULT_SCIGLASS_ZIP
    schott_xlsx: Path = DEFAULT_SCHOTT_XLSX
    merged_data: Path = PROJECT_DIR / "merged_data.parquet"
    gan_dir: Path = DEFAULT_GAN_DIR
    forward_models: Path = DEFAULT_FORWARD_DIR


def check_environment(paths: ToolPaths | None = None) -> dict[str, Any]:
    """Проверка, что всё необходимое для работы инструмента на месте."""
    p = paths or ToolPaths()
    checks = {
        "sciglass_zip": p.sciglass_zip.exists(),
        "schott_xlsx": p.schott_xlsx.exists(),
        "merged_data": p.merged_data.exists(),
        "gan_generator": (p.gan_dir / "checkpoints" / "generator.pt").exists(),
        "gan_condition_stats": (p.gan_dir / "condition_stats.json").exists(),
        "forward_models": p.forward_models.exists(),
    }
    checks["ready_recover"] = checks["sciglass_zip"]
    checks["ready_generate"] = all(
        [
            checks["gan_generator"],
            checks["gan_condition_stats"],
            checks["forward_models"],
            checks["merged_data"],
        ]
    )
    return checks


def recover_composition(
    *,
    nd: float | None = None,
    vd: float | None = None,
    density: float | None = None,
    tg: float | None = None,
    glass_name: str | None = None,
    k_neighbors: int = 20,
    paths: ToolPaths | None = None,
) -> dict[str, Any]:
    """Восстановить состав: по свойствам или по имени из каталога SCHOTT."""
    p = paths or ToolPaths()
    if not p.sciglass_zip.exists():
        raise FileNotFoundError(
            f"Нет базы SciGlass: {p.sciglass_zip}\n"
            "Укажите путь: --sciglass-zip или переменную SCIGLASS_ZIP"
        )

    if glass_name:
        return recover_from_schott_name(
            glass_name,
            schott_xlsx=p.schott_xlsx,
            sciglass_zip=p.sciglass_zip,
            k_neighbors=k_neighbors,
        )

    if nd is None or vd is None:
        raise ValueError("Укажите --nd и --vd или --glass «имя из SCHOTT»")

    return recover_from_properties(
        float(nd),
        float(vd),
        density=density,
        tg=tg,
        sciglass_zip=p.sciglass_zip,
        k_neighbors=k_neighbors,
    )


def generate_compositions(
    nd: float,
    vd: float,
    *,
    n_samples: int = 300,
    top_k: int = 20,
    pb_free_only: bool = True,
    min_nd: float = 1.75,
    paths: ToolPaths | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Сгенерировать кандидаты cWGAN-GP под целевые n_d и ν_d."""
    import torch
    from gan_glass_design import (
        Generator,
        generate_compositions as gan_generate_raw,
        load_forward_models,
        score_and_filter,
    )
    from inverse_glass_design import (
        build_distance_model,
        filter_design_space,
        load_sciglass_df,
        oxide_columns,
    )

    p = paths or ToolPaths()
    ckpt = p.gan_dir / "checkpoints" / "generator.pt"
    stats_path = p.gan_dir / "condition_stats.json"

    for label, path in [
        ("GAN", ckpt),
        ("condition_stats", stats_path),
        ("forward_models", p.forward_models),
        ("merged_data", p.merged_data),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Нет {label}: {path}")

    cond_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    df = load_sciglass_df(p.merged_data)
    oxide_cols = oxide_columns(df)
    design_df = filter_design_space(df, oxide_cols)
    models = load_forward_models(p.forward_models, oxide_cols)
    _, _, dist_threshold = build_distance_model(design_df, oxide_cols)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pbo_idx = oxide_cols.index("PBO") if "PBO" in oxide_cols else None
    G = Generator(len(oxide_cols), pbo_idx).to(dev)
    G.load_state_dict(torch.load(ckpt, map_location=dev))
    G.eval()

    generated = gan_generate_raw(
        G,
        cond_stats,
        oxide_cols,
        nd_targets=[float(nd)],
        nud_targets=[float(vd)],
        n_per_target=max(1, n_samples),
        device=dev,
    )
    scored = score_and_filter(generated, oxide_cols, models, design_df, dist_threshold)

    sub = scored.copy()
    if pb_free_only:
        sub = sub[sub["PbO"] <= 0.01]
    if min_nd:
        sub = sub[sub["ND300_pred"] >= min_nd]

    top = sub.sort_values(
        ["feasible", "ND300_pred", "distance_to_training"],
        ascending=[False, False, True],
    ).head(top_k)

    rows = []
    for _, r in top.iterrows():
        rows.append({
            "rank": int(r.get("gan_rank", 0)),
            "composition": r["composition"],
            "n_pred": round(float(r["ND300_pred"]), 4),
            "vd_pred": round(float(r["NUD300_pred"]), 2),
            "tg_pred": round(float(r["TG_pred"]), 1),
            "density_pred": round(float(r["DENSITY_pred"]), 3),
            "pbo": round(float(r["PbO"]), 4),
            "distance": round(float(r["distance_to_training"]), 3),
            "feasible": bool(r["feasible"]),
            "family": r.get("glass_family", ""),
        })

    return {
        "target": {"nd": nd, "vd": vd},
        "n_generated": len(generated),
        "n_scored": len(scored),
        "n_feasible_all": int(scored["feasible"].sum()) if len(scored) else 0,
        "distance_threshold_p95": float(dist_threshold),
        "disclaimer": (
            "Кандидаты сгенерированы cWGAN-GP и оценены surrogate-моделями. "
            "Проверяйте distance_to_training: значения > порога p95 — риск экстраполяции."
        ),
        "candidates": rows,
    }


def list_schott_glasses(paths: ToolPaths | None = None, limit: int = 50) -> list[str]:
    p = paths or ToolPaths()
    if not p.schott_xlsx.exists():
        return []
    schott = load_schott_catalog(p.schott_xlsx)
    names = schott["glass_name"].astype(str).tolist()
    return names[:limit]


def format_recover_text(result: dict[str, Any]) -> str:
    q = result.get("query", {})
    lines = [
        "=" * 60,
        "ВОССТАНОВЛЕНИЕ СОСТАВА (SciGlass, без синтеза)",
        "=" * 60,
        f"  Запрос: {q.get('label', '—')}",
        f"  n_d = {q.get('nd')},  ν_d = {q.get('vd')}",
        f"  ρ = {q.get('density')},  T_g = {q.get('tg')}",
    ]
    if q.get("imputed_fields"):
        lines.append(f"  (оценено по медиане пула: {', '.join(q['imputed_fields'])})")
    if result.get("schott_catalog"):
        sc = result["schott_catalog"]
        lines.append(f"  Каталог SCHOTT: {sc['glass_name']}")

    lines += [
        "",
        "ОСНОВНОЙ СОСТАВ (лучший сосед MODE_B):",
        f"  {result['primary_composition']}",
        "",
        f"  SciGlass ID: {result['matched_sciglass_id']}",
        f"  distance: {result['distance_first']:.4f}  |  Jaccard: {result['jaccard_topk']:.3f} ({result['jaccard_label']})",
        f"  uncertainty: {result['uncertainty_score']:.4f}",
        f"  сосед физичен: {result['best_neighbor_plausible']}",
        "",
        "Вспомогательно (локальные модели, не основной ответ):",
        f"  PLS: {result.get('composition_pls', '—')}",
        "",
        "ТОП-5 соседей:",
    ]
    for n in result.get("neighbors", [])[:5]:
        lines.append(
            f"  #{n['rank']}  d={n['distance']:.3f}  id={n['sciglass_id']}  "
            f"n_d={n.get('nd')}  plausible={n['plausible']}"
        )
        lines.append(f"       {n['composition'][:90]}...")
    lines.append("")
    lines.append(result.get("disclaimer", ""))
    return "\n".join(lines)


def format_generate_text(result: dict[str, Any]) -> str:
    lines = [
        "=" * 60,
        "ГЕНЕРАЦИЯ СОСТАВОВ (cWGAN-GP)",
        "=" * 60,
        f"  Цель: n_d = {result['target']['nd']},  ν_d = {result['target']['vd']}",
        f"  Сгенерировано: {result['n_generated']}, feasible: {result['n_feasible_all']}",
        f"  Порог distance (p95): {result['distance_threshold_p95']:.3f}",
        "",
        "Лучшие кандидаты:",
    ]
    for c in result.get("candidates", []):
        flag = "OK" if c["feasible"] else "—"
        lines.append(
            f"  [{flag}] n={c['n_pred']}  ν_d={c['vd_pred']}  dist={c['distance']}  PbO={c['pbo']}"
        )
        lines.append(f"       {c['composition'][:95]}...")
    lines.append("")
    lines.append(result.get("disclaimer", ""))
    return "\n".join(lines)
