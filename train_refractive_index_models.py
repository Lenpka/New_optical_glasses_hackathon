"""Обучение моделей: состав → RefractiveIndex.

Использование:

    python train_refractive_index_models.py
    python train_refractive_index_models.py --features elements
    python train_refractive_index_models.py --features oxides      # мольные: SIO2, AL2O3, ...
    python train_refractive_index_models.py --features oxides_wt   # весовые: WSIO2, WAL2O3, ...
    python train_refractive_index_models.py --data merged_data.parquet

Лог по умолчанию: <models-dir>/train.log (консоль + файл).

Ожидается merged_data из review.ipynb (сохраните: merged_data.to_parquet(...)).

Модели: LinearRegression, Ridge, RandomForest, XGBoost, LightGBM, CatBoost.
Зависимости: pip install -r requirements-ml.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "merged_data.parquet"

TARGET = "RefractiveIndex"
N_MIN = 1.35
N_MAX = 4.0

ELEMENT_FEATURES = [
    "H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh",
    "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Th", "U",
]

# Мольные доли оксидов (колонки SIO2, AL2O3, ... в SciGlass / merged_data)
OXIDE_MOL_FEATURES = [
    "SIO2", "AL2O3", "B2O3", "CAO", "K2O", "NA2O", "PBO", "Li2O", "MgO", "SRO", "BAO",
    "ZNO", "P2O5", "GEO2", "ZRO2", "TIO2", "TEO2", "RO", "FemOn",
]

# Весовые доли оксидов — колонки W* из df_props
OXIDE_WT_FEATURES = [
    "WSIO2", "WAL2O3", "WB2O3", "WCAO", "WK2O", "WNA2O", "WPBO", "WLi2O", "WMgO",
    "WSRO", "WBAO", "WZNO", "WGEO2", "WZRO2", "WTIO2", "WTEO2", "WRO", "WFemOn", "WP2O5",
]

FEATURE_SETS = {
    "elements": ELEMENT_FEATURES,
    "oxides": OXIDE_MOL_FEATURES,
    "oxides_wt": OXIDE_WT_FEATURES,
}

logger = logging.getLogger(__name__)


def resolve_feature_list(features: str) -> tuple[str, list[str]]:
    """Имя набора признаков → (feature_set_name, columns)."""
    key = features.lower()
    if key not in FEATURE_SETS:
        raise ValueError(f"Неизвестный --features: {features}. Доступно: {list(FEATURE_SETS)}")
    return key, FEATURE_SETS[key]


def default_models_dir(feature_set: str) -> Path:
    return PROJECT_DIR / ("models" if feature_set == "elements" else f"models_{feature_set}")


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Настроить логи в консоль и опционально в файл."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        root.addHandler(file_handler)
        logger.info("Лог-файл: %s", log_file.resolve())

    for name in ("xgboost", "lightgbm", "catboost", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)


def load_merged_data(path: Path) -> pd.DataFrame:
    """Загрузить объединённый датасет (parquet, csv или pickle)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Файл данных не найден: {path}\n"
            "Сохраните merged_data из ноутбука:\n"
            '  merged_data.to_parquet("merged_data.parquet")'
        )
    suffix = path.suffix.lower()
    logger.info("Чтение данных: %s (%s)", path.name, suffix or "no ext")
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path, sep="\t" if path.name.endswith(".tsv") else ",")
    elif suffix in {".pkl", ".pickle"}:
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Неподдерживаемый формат: {path.suffix}")
    logger.info("Загружено: %s строк x %s колонок", f"{len(df):,}", df.shape[1])
    return df


def prepare_dataset(
    merged_data: pd.DataFrame,
    candidate_features: list[str],
    *,
    feature_set: str = "elements",
    n_min: float = N_MIN,
    n_max: float = N_MAX,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Фильтрация n и подготовка матрицы признаков."""
    feature_cols = [c for c in candidate_features if c in merged_data.columns]
    missing_feats = [c for c in candidate_features if c not in merged_data.columns]

    if TARGET not in merged_data.columns:
        raise KeyError(f"В данных нет целевой колонки '{TARGET}'")

    n_raw = merged_data[TARGET].replace(0, np.nan)
    valid_n = n_raw.notna() & (n_raw > n_min) & (n_raw < n_max)

    ml_df = merged_data.loc[valid_n, feature_cols + [TARGET]].copy()
    ml_df[TARGET] = n_raw.loc[valid_n]
    ml_df[feature_cols] = ml_df[feature_cols].fillna(0)
    ml_df = ml_df.dropna(subset=[TARGET])

    if not feature_cols:
        raise ValueError(
            f"Ни один признак из набора '{feature_set}' не найден в данных. "
            f"Ожидались, например: {candidate_features[:5]}..."
        )

    meta = {
        "feature_set": feature_set,
        "n_before_filter": int(n_raw.notna().sum()),
        "n_after_filter": len(ml_df),
        "n_features": len(feature_cols),
        "missing_features": missing_feats,
        "n_min": n_min,
        "n_max": n_max,
        "target": TARGET,
    }
    logger.info(
        "Набор признаков: %s (%s колонок)",
        feature_set,
        meta["n_features"],
    )
    logger.info(
        "Фильтр %s: %s -> %s (%.2f < n < %.2f)",
        TARGET,
        f"{meta['n_before_filter']:,}",
        f"{meta['n_after_filter']:,}",
        n_min,
        n_max,
    )
    if missing_feats:
        logger.warning("Признаки отсутствуют в данных: %s", ", ".join(missing_feats))
    return ml_df, feature_cols, meta


def build_models(*, random_state: int = 42) -> dict[str, Any]:
    """Словарь имя → нетренированная модель."""
    rs = random_state
    return {
        "linear_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "ridge": Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=24,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=rs,
        ),
        "xgboost": XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=rs,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=500,
            max_depth=-1,
            num_leaves=63,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=rs,
            verbose=-1,
        ),
        "catboost": CatBoostRegressor(
            iterations=500,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_seed=rs,
            verbose=0,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, Any], dict[str, np.ndarray], pd.DataFrame, Any, Any]:
    """Обучить модели, вернуть fitted models, предсказания и метрики."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Train/test split: train=%s, test=%s (test_size=%.2f, seed=%s)",
        f"{len(X_train):,}",
        f"{len(X_test):,}",
        test_size,
        random_state,
    )

    fitted: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    results: list[dict[str, Any]] = []

    for name, model in build_models(random_state=random_state).items():
        t0 = time.perf_counter()
        logger.info("Обучение: %s ...", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.perf_counter() - t0
        fitted[name] = model
        predictions[name] = y_pred
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        results.append({
            "model": name,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
        })
        logger.info(
            "  %s - R2=%.4f, RMSE=%.4g, MAE=%.4g, время=%.1f s",
            name,
            r2,
            rmse,
            mae,
            elapsed,
        )

    metrics_df = pd.DataFrame(results).sort_values("RMSE")
    best = metrics_df.iloc[0]
    logger.info(
        "Лучшая модель: %s (RMSE=%.4g, R2=%.4f)",
        best["model"],
        best["RMSE"],
        best["R2"],
    )
    return fitted, predictions, metrics_df, y_test, X_test


def save_artifacts(
    models_dir: Path,
    fitted_models: dict[str, Any],
    feature_cols: list[str],
    meta: dict[str, Any],
    metrics_df: pd.DataFrame,
) -> None:
    """Сохранить модели, список признаков и метрики."""
    models_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Сохранение артефактов в %s", models_dir.resolve())

    for name, model in fitted_models.items():
        path = models_dir / f"{name}.joblib"
        joblib.dump(model, path)
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("  модель: %s (%.2f MB)", path.name, size_mb)

    config = {
        **meta,
        "feature_columns": feature_cols,
        "model_files": {name: f"{name}.joblib" for name in fitted_models},
    }
    config_path = models_dir / "config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_path = models_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("  config: %s", config_path.name)
    logger.info("  metrics: %s", metrics_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Обучение моделей RefractiveIndex по составу (элементы или оксиды)",
    )
    parser.add_argument(
        "--features",
        choices=list(FEATURE_SETS),
        default="elements",
        help="elements | oxides (SIO2, AL2O3, ...) | oxides_wt (WSIO2, WAL2O3, ...)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Путь к merged_data (по умолчанию {DEFAULT_DATA_PATH.name})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Папка для .joblib (по умолчанию: models или models_oxides)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-min", type=float, default=N_MIN)
    parser.add_argument("--n-max", type=float, default=N_MAX)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Уровень логирования",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Путь к лог-файлу (по умолчанию: <models-dir>/train.log)",
    )
    parser.add_argument(
        "--no-log-file",
        action="store_true",
        help="Не писать лог в файл, только в консоль",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_set, candidate_features = resolve_feature_list(args.features)
    models_dir = args.models_dir or default_models_dir(feature_set)

    log_file = None if args.no_log_file else (args.log_file or models_dir / "train.log")
    setup_logging(level=args.log_level, log_file=log_file)

    logger.info("=== Обучение %s | признаки: %s ===", TARGET, feature_set)
    logger.debug(
        "Параметры: data=%s, models_dir=%s, test_size=%s, seed=%s",
        args.data,
        models_dir,
        args.test_size,
        args.random_state,
    )

    t_start = time.perf_counter()
    merged_data = load_merged_data(args.data)

    ml_df, feature_cols, meta = prepare_dataset(
        merged_data,
        candidate_features,
        feature_set=feature_set,
        n_min=args.n_min,
        n_max=args.n_max,
    )

    X = ml_df[feature_cols]
    y = ml_df[TARGET]

    fitted, _preds, metrics_df, _y_test, _X_test = train_and_evaluate(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    save_artifacts(models_dir, fitted, feature_cols, meta, metrics_df)

    logger.info("Метрики на тесте:\n%s", metrics_df.round(4).to_string(index=False))
    logger.info("Готово за %.1f с", time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
