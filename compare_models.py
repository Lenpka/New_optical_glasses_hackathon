"""Сравнительные графики всех моделей и типов данных (plots.py).

    python compare_models.py
    python compare_models.py --output figures/comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from plots import (
    FEATURE_SET_LABELS,
    MODEL_LABELS,
    MODEL_ORDER,
    configure_theme,
    plot_model_comparison_suite,
)
from train_refractive_index_models import (
    FEATURE_SETS,
    PROJECT_DIR,
    TARGET,
    default_models_dir,
    load_merged_data,
    prepare_dataset,
)

logger = logging.getLogger(__name__)

TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURE_SET_DIRS = {
    "elements": default_models_dir("elements"),
    "oxides": default_models_dir("oxides"),
    "oxides_wt": default_models_dir("oxides_wt"),
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def evaluate_saved_models(
    merged_data: pd.DataFrame,
    feature_set: str,
    models_dir: Path,
) -> tuple[pd.DataFrame, list[dict], dict[str, Any]]:
    """Предсказания на тесте и метрики для всех .joblib в папке."""
    config_path = models_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Нет config.json в {models_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_cols = config["feature_columns"]
    candidates = FEATURE_SETS.get(feature_set, feature_cols)

    ml_df, cols, _meta = prepare_dataset(
        merged_data,
        candidates,
        feature_set=feature_set,
    )
    use_cols = [c for c in feature_cols if c in cols]

    X = ml_df[use_cols]
    y = ml_df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_test_arr = y_test.to_numpy()

    rows: list[dict] = []
    parity_panels: list[dict] = []
    best_row: dict | None = None

    model_files = config.get("model_files", {})
    for model_name in MODEL_ORDER:
        if model_name not in model_files:
            continue
        path = models_dir / model_files[model_name]
        if not path.exists():
            logger.warning("Нет файла модели: %s", path)
            continue

        model = joblib.load(path)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)

        row = {
            "feature_set": feature_set,
            "model": model_name,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
        }
        rows.append(row)

        label = FEATURE_SET_LABELS.get(feature_set, feature_set)
        mlabel = MODEL_LABELS.get(model_name, model_name)
        parity_panels.append({
            "y_true": y_test_arr,
            "y_pred": y_pred,
            "title": f"{label}\n{mlabel}",
        })

        if best_row is None or rmse < best_row["RMSE"]:
            best_row = {
                **row,
                "y_true": y_test_arr,
                "y_pred": y_pred,
                "title": f"{label} | {mlabel}",
            }

    if not rows:
        raise RuntimeError(f"Не удалось оценить модели в {models_dir}")

    metrics = pd.DataFrame(rows)
    assert best_row is not None
    return metrics, parity_panels, best_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Сравнение ML-моделей и графики")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_DIR / "merged_data.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "figures" / "comparison",
    )
    parser.add_argument("--show", action="store_true", help="Показать графики в окне")
    args = parser.parse_args()

    setup_logging()
    configure_theme()

    logger.info("Загрузка данных: %s", args.data)
    merged_data = load_merged_data(args.data)

    all_metrics: list[pd.DataFrame] = []
    parity_by_type: dict[str, list[dict]] = {}
    best_per_type: list[dict] = []

    for feature_set, models_dir in FEATURE_SET_DIRS.items():
        if not models_dir.exists():
            logger.warning("Пропуск %s: нет папки %s", feature_set, models_dir)
            continue
        logger.info("Оценка: %s (%s)", feature_set, models_dir)
        metrics, panels, best = evaluate_saved_models(
            merged_data, feature_set, models_dir
        )
        all_metrics.append(metrics)
        parity_by_type[feature_set] = panels
        best_per_type.append({
            "y_true": best["y_true"],
            "y_pred": best["y_pred"],
            "title": best["title"],
            "color": "#CF4457" if feature_set == "elements" else "#348ABD",
        })
        logger.info(
            "  лучшая: %s (RMSE=%.4g, R2=%.4f)",
            best["model"],
            best["RMSE"],
            best["R2"],
        )

    if not all_metrics:
        raise SystemExit(
            "Нет обученных моделей. Запустите:\n"
            "  python train_refractive_index_models.py --features elements\n"
            "  python train_refractive_index_models.py --features oxides\n"
            "  python train_refractive_index_models.py --features oxides_wt"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_all = pd.concat(all_metrics, ignore_index=True)
    metrics_all.to_csv(args.output / "metrics_all_types.csv", index=False)

    global_best_idx = metrics_all["RMSE"].idxmin()
    gb = metrics_all.loc[global_best_idx]
    fs_best = str(gb["feature_set"])
    model_best = str(gb["model"])

    _, panels_fs, _ = evaluate_saved_models(
        merged_data, fs_best, FEATURE_SET_DIRS[fs_best]
    )
    global_best_panel = None
    for p in panels_fs:
        if MODEL_LABELS.get(model_best, model_best) in p["title"]:
            global_best_panel = {
                "y_true": p["y_true"],
                "y_pred": p["y_pred"],
                "title": (
                    f"Лучшая из всех: {FEATURE_SET_LABELS.get(fs_best, fs_best)} + "
                    f"{MODEL_LABELS.get(model_best, model_best)} "
                    f"(RMSE={gb['RMSE']:.4g}, R2={gb['R2']:.4f})"
                ),
            }
            break
    if global_best_panel is None:
        raise RuntimeError(f"Не найдена панель для {fs_best}/{model_best}")

    logger.info(
        "Глобальный лидер: %s / %s — RMSE=%.4g, R2=%.4f",
        fs_best,
        model_best,
        gb["RMSE"],
        gb["R2"],
    )

    saved = plot_model_comparison_suite(
        metrics_all,
        parity_by_type,
        best_per_type,
        global_best_panel,
        output_dir=args.output,
        show=args.show,
    )

    logger.info("Графики сохранены в %s:", args.output.resolve())
    for name, path in saved.items():
        logger.info("  %s: %s", name, path.name)
    logger.info("Таблица метрик: metrics_all_types.csv")


if __name__ == "__main__":
    main()
