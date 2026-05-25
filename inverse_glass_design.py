"""Inverse design of high-index lead-free optical glasses (SciGlass surrogate + NSGA-II).

    python inverse_glass_design.py
    python inverse_glass_design.py --data merged_data.parquet --generations 100

Пайплайн:
  SciGlass -> forward models (composition -> properties) -> NSGA-II -> фильтрация -> кандидаты

Выход: output/inverse_design/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from match_schott_sciglass import classify_glass_family, setup_logging

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "merged_data.parquet"
DEFAULT_OUTPUT = PROJECT_DIR / "output" / "inverse_design"

PROPERTY_MAP = {
    "ND300": "RefractiveIndex",
    "NUD300": "AbbeNum",
    "DENSITY": "Density293K",
    "TG": "Tg",
    "TMELT": "Tmelt",
    "TLIQ": "Tliquidus",
    "TSOFT": "Tsoft",
}

# Tg ≈ (2/3) * Tmelt в Kelvin (эмпирика SciGlass: median ≈ 0.67)
TG_TMELT_RATIO = 2.0 / 3.0

TARGETS = ["ND300", "NUD300", "DENSITY", "TG"]

OXIDE_FEATURES = [
    "SIO2", "AL2O3", "B2O3", "CAO", "K2O", "NA2O", "PBO", "Li2O", "MgO", "SRO", "BAO",
    "ZNO", "P2O5", "GEO2", "ZRO2", "TIO2", "TEO2", "Bi2O3", "WO3", "RO", "FemOn",
    "R2O3", "R2O5", "RO2",
]

PLOT_OXIDES = {
    "SIO2": "SIO2",
    "B2O3": "B2O3",
    "TIO2": "TIO2",
    "ZRO2": "ZRO2",
    "La2O3_proxy_R2O3": "R2O3",
    "BAO": "BAO",
    "ZNO": "ZNO",
    "Nb2O5_proxy_R2O5": "R2O5",
    "Bi2O3": "Bi2O3",
    "WO3": "WO3",
    "GEO2": "GEO2",
}

ANOMALY_OXIDES = ["TIO2", "Bi2O3", "WO3", "R2O3", "R2O5"]

MODEL_BUILDERS = {
    "xgboost": lambda rs: XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=rs,
    ),
    "lightgbm": lambda rs: LGBMRegressor(
        n_estimators=400,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=rs,
        verbose=-1,
    ),
    "catboost": lambda rs: CatBoostRegressor(
        iterations=400,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=rs,
        verbose=0,
        allow_writing_files=False,
        thread_count=-1,
    ),
}


def load_sciglass_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Нет файла данных: {path}")
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    logger.info("Загружено %s строк", f"{len(df):,}")
    return df


def oxide_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in OXIDE_FEATURES if c in df.columns]


def tg_from_tmelt_celsius(tmelt_c: pd.Series) -> pd.Series:
    """Tg [C] из Tmelt [C]: Tg_K = (2/3) * Tm_K."""
    tm_k = pd.to_numeric(tmelt_c, errors="coerce") + 273.15
    return TG_TMELT_RATIO * tm_k - 273.15


def build_tg_training_target(df: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    """
    Целевой Tg для обучения: измеренный; если нет — из Tmelt по правилу 2/3 (Kelvin).
    """
    tg_col = PROPERTY_MAP["TG"]
    tm_col = PROPERTY_MAP["TMELT"]
    tg = pd.to_numeric(df[tg_col], errors="coerce") if tg_col in df.columns else pd.Series(np.nan, index=df.index)
    tm = pd.to_numeric(df[tm_col], errors="coerce") if tm_col in df.columns else pd.Series(np.nan, index=df.index)

    imputed = tg_from_tmelt_celsius(tm)
    out = tg.copy()
    fill_mask = tg.isna() & imputed.notna()
    out.loc[fill_mask] = imputed.loc[fill_mask]

    stats = {
        "tg_measured": int(tg.notna().sum()),
        "tg_imputed_from_tmelt": int(fill_mask.sum()),
        "tg_total_train": int(out.notna().sum()),
    }
    return out, stats


def prepare_xy(
    df: pd.DataFrame,
    target_key: str,
    oxide_cols: list[str],
    tg_augmented: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray | None]:
    sample_weight = None
    if target_key == "TG" and tg_augmented is not None:
        y = tg_augmented
        tg_meas = pd.to_numeric(df[PROPERTY_MAP["TG"]], errors="coerce")
        sample_weight = tg_meas.loc[y.index].notna().astype(float)
        sample_weight = sample_weight.replace(0.0, 0.4).replace(1.0, 1.0).to_numpy()
    else:
        col = PROPERTY_MAP[target_key]
        y = pd.to_numeric(df[col], errors="coerce")
    valid = y.notna()
    X = df.loc[valid, oxide_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    y = y.loc[valid]
    if sample_weight is not None:
        sample_weight = sample_weight[valid.to_numpy()]
    return X, y, sample_weight


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_forward_models(
    df: pd.DataFrame,
    oxide_cols: list[str],
    output_dir: Path,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """composition -> ND300, NUD300, DENSITY, TG; выбор лучшей модели на test."""
    models_dir = output_dir / "forward_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    tg_augmented, tg_stats = build_tg_training_target(df)
    if tg_stats["tg_imputed_from_tmelt"]:
        tg_meas = pd.to_numeric(df[PROPERTY_MAP["TG"]], errors="coerce")
        tm = pd.to_numeric(df[PROPERTY_MAP["TMELT"]], errors="coerce")
        both = tg_meas.notna() & tm.notna()
        if both.sum() > 50:
            rule = tg_from_tmelt_celsius(tm.loc[both])
            err = float(np.mean(np.abs(rule - tg_meas.loc[both])))
            logger.info(
                "Tg augmentation: measured=%s, imputed=%s, total=%s; |rule-measured| mean=%.1f C (n=%s)",
                f"{tg_stats['tg_measured']:,}",
                f"{tg_stats['tg_imputed_from_tmelt']:,}",
                f"{tg_stats['tg_total_train']:,}",
                err,
                both.sum(),
            )

    best_models: dict[str, Any] = {}
    metrics_rows = []

    for target in TARGETS:
        tg_aug = tg_augmented if target == "TG" else None
        X, y, sw = prepare_xy(df, target, oxide_cols, tg_augmented=tg_aug)
        if len(X) < 500:
            logger.warning("Мало данных для %s: %s", target, len(X))
            continue

        tg_meas = pd.to_numeric(df[PROPERTY_MAP["TG"]], errors="coerce") if target == "TG" else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state,
        )
        sw_train = None
        if sw is not None:
            sw_s = pd.Series(sw, index=X.index)
            sw_train = sw_s.loc[y_train.index].to_numpy()

        if target == "TG" and tg_meas is not None:
            eval_mask = tg_meas.loc[y_test.index].notna().to_numpy()
        else:
            eval_mask = np.ones(len(y_test), dtype=bool)

        best_rmse = np.inf
        best_name = None
        best_model = None

        for name, builder in MODEL_BUILDERS.items():
            model = builder(random_state)
            fit_kw: dict[str, Any] = {}
            if sw_train is not None:
                fit_kw["sample_weight"] = sw_train
            model.fit(X_train, y_train, **fit_kw)
            pred = model.predict(X_test)
            if eval_mask.any():
                m = eval_metrics(y_test.to_numpy()[eval_mask], pred[eval_mask])
            else:
                m = eval_metrics(y_test.to_numpy(), pred)
            m.update({"target": target, "model": name, "n_train": len(X_train)})
            if target == "TG":
                m["n_test_measured_only"] = int(eval_mask.sum())
            metrics_rows.append(m)
            suffix = f" (test n={m.get('n_test_measured_only', len(y_test))} measured)" if target == "TG" else ""
            logger.info(
                "%s / %s: R2=%.3f RMSE=%.4f MAE=%.4f%s",
                target, name, m["r2"], m["rmse"], m["mae"], suffix,
            )
            if m["rmse"] < best_rmse:
                best_rmse = m["rmse"]
                best_name = name
                best_model = model

        assert best_model is not None
        best_models[target] = {
            "model": best_model,
            "model_name": best_name,
            "features": oxide_cols,
        }
        joblib.dump(best_model, models_dir / f"{target}_{best_name}.joblib")
        logger.info("Лучшая модель %s: %s (RMSE=%.2f)", target, best_name, best_rmse)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "forward_model_metrics.csv", index=False)

    summary: dict[str, Any] = {"tg_augmentation": tg_stats}
    for target in TARGETS:
        sub = metrics_df[metrics_df["target"] == target]
        if len(sub):
            best = sub.loc[sub["rmse"].idxmin()]
            summary[target] = {
                "best_model": best["model"],
                "r2": float(best["r2"]),
                "rmse": float(best["rmse"]),
                "mae": float(best["mae"]),
            }
            if target == "TG" and "n_test_measured_only" in best:
                summary[target]["n_test_measured_only"] = int(best["n_test_measured_only"])
    (output_dir / "forward_model_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return best_models, metrics_df


def filter_design_space(
    df: pd.DataFrame,
    oxide_cols: list[str],
    nd_min: float = 1.70,
    pbo_max: float = 1.0,
) -> pd.DataFrame:
    nd = pd.to_numeric(df[PROPERTY_MAP["ND300"]], errors="coerce")
    pbo = pd.to_numeric(df["PBO"], errors="coerce").fillna(0) if "PBO" in df.columns else 0
    mask = (nd > nd_min) & (pbo <= pbo_max)
    sub = df.loc[mask].copy()
    logger.info(
        "Design space: ND300>%.2f, PBO<=%.1f -> %s записей",
        nd_min, pbo_max, f"{len(sub):,}",
    )
    return sub


def plot_oxide_distributions(design_df: pd.DataFrame, output_dir: Path) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ncols = 3
    keys = list(PLOT_OXIDES.keys())
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, label in zip(axes, keys):
        col = PLOT_OXIDES[label]
        if col not in design_df.columns:
            ax.set_visible(False)
            continue
        vals = pd.to_numeric(design_df[col], errors="coerce").dropna()
        vals = vals[vals > 0]
        ax.hist(vals, bins=40, color="#348ABD", edgecolor="white", alpha=0.85)
        ax.set_title(label)
        ax.set_xlabel("mol %")
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    fig.suptitle("Распределения оксидов (ND>1.70, PbO<=1 mol%)", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "oxide_distributions_design_space.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def composition_bounds(design_df: pd.DataFrame, oxide_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """p2-p98 по обучающему подпространству."""
    X = design_df[oxide_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    lo = X.quantile(0.02).to_numpy()
    hi = X.quantile(0.98).to_numpy()
    lo = np.maximum(lo, 0.0)
    hi = np.maximum(hi, lo + 0.5)
    if "PBO" in oxide_cols:
        lo[oxide_cols.index("PBO")] = 0.0
        hi[oxide_cols.index("PBO")] = 0.0
    if "SIO2" in oxide_cols:
        lo[oxide_cols.index("SIO2")] = max(lo[oxide_cols.index("SIO2")], 5.0)
    return lo, hi


def decode_composition(x: np.ndarray, oxide_cols: list[str], pbo_idx: int | None) -> np.ndarray:
    comp = np.clip(x, 0.0, None)
    if pbo_idx is not None:
        comp[pbo_idx] = 0.0
    s = comp.sum()
    if s <= 1e-9:
        comp = np.ones_like(comp) / len(comp) * 100
    else:
        comp = comp / s * 100.0
    return comp


def predict_properties(comp: np.ndarray, models: dict[str, Any], oxide_cols: list[str]) -> dict[str, float]:
    row = pd.DataFrame([comp], columns=oxide_cols)
    out = {}
    for target, info in models.items():
        out[target] = float(info["model"].predict(row)[0])
    return out


def comp_to_str(comp: dict[str, float], threshold: float = 0.5) -> str:
    parts = [f"{k}={v:.2f}" for k, v in sorted(comp.items()) if v > threshold]
    return "; ".join(parts)


class GlassInverseProblem(ElementwiseProblem):
    def __init__(
        self,
        models: dict[str, Any],
        oxide_cols: list[str],
        xl: np.ndarray,
        xu: np.ndarray,
        pbo_idx: int | None,
        nn: NearestNeighbors,
        dist_scaler: StandardScaler,
        tliq_model: Any | None = None,
    ):
        super().__init__(n_var=len(oxide_cols), n_obj=5, n_constr=3, xl=xl, xu=xu)
        self.models = models
        self.oxide_cols = oxide_cols
        self.pbo_idx = pbo_idx
        self.tliq_model = tliq_model
        self.nn = nn
        self.dist_scaler = dist_scaler

    def _evaluate(self, x, out, *args, **kwargs):
        comp = decode_composition(x, self.oxide_cols, self.pbo_idx)
        props = predict_properties(comp, self.models, self.oxide_cols)
        xs = self.dist_scaler.transform(comp.reshape(1, -1))
        dist = float(self.nn.kneighbors(xs)[0][0, 0])

        # minimize: -ND, -NUD, DENSITY, -TG, distance_to_training
        out["F"] = [
            -props["ND300"],
            -props["NUD300"],
            props["DENSITY"],
            -props["TG"],
            dist,
        ]

        sio2_idx = self.oxide_cols.index("SIO2") if "SIO2" in self.oxide_cols else 0
        g_tg = 450.0 - props["TG"]
        g_sio2 = 5.0 - comp[sio2_idx]
        if self.tliq_model is not None:
            tliq = float(self.tliq_model.predict(pd.DataFrame([comp], columns=self.oxide_cols))[0])
            g_tliq = tliq - 1600.0
        else:
            g_tliq = -1.0  # satisfied if no model
        out["G"] = [g_tg, g_tliq, g_sio2]


def run_nsga2(
    problem: GlassInverseProblem,
    design_df: pd.DataFrame,
    oxide_cols: list[str],
    xl: np.ndarray,
    xu: np.ndarray,
    pop_size: int = 200,
    n_gen: int = 120,
    seed: int = 42,
) -> Any:
    class SciGlassSampling(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            rng = np.random.default_rng(seed)
            comps = design_df[oxide_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
            idx = rng.integers(0, len(comps), size=n_samples)
            X = comps.iloc[idx].to_numpy()
            return np.clip(X, xl, xu)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=SciGlassSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    res = minimize(
        problem,
        algorithm,
        get_termination("n_gen", n_gen),
        seed=seed,
        verbose=False,
    )
    return res


def build_distance_model(
    design_df: pd.DataFrame,
    oxide_cols: list[str],
    round_decimals: int = 1,
) -> tuple[NearestNeighbors, StandardScaler, float]:
    """NN по уникальным составам; порог = p95 расстояний design->unique."""
    X = design_df[oxide_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    Xr = np.round(X, round_decimals)
    _, uniq_idx = np.unique(Xr, axis=0, return_index=True)
    X_uniq = X[uniq_idx]

    scaler = StandardScaler()
    Xs_uniq = scaler.fit_transform(X_uniq)
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto")
    nn.fit(Xs_uniq)

    # порог: p95 расстояния до ближайшего *другого* состава в SciGlass
    d2 = nn.kneighbors(Xs_uniq)[0][:, 1]
    threshold = float(np.percentile(d2, 95))
    logger.info(
        "Distance model: %s unique comps, dist threshold (p95) = %.4f",
        f"{len(X_uniq):,}",
        threshold,
    )
    nn_query = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nn_query.fit(Xs_uniq)
    return nn_query, scaler, threshold


def distance_to_training(comp: np.ndarray, nn: NearestNeighbors, scaler: StandardScaler) -> float:
    xs = scaler.transform(comp.reshape(1, -1))
    dist, _ = nn.kneighbors(xs)
    return float(dist[0, 0])


def oxide_limits(design_df: pd.DataFrame, oxide_cols: list[str]) -> dict[str, tuple[float, float]]:
    limits = {}
    for ox in ANOMALY_OXIDES:
        if ox not in oxide_cols:
            continue
        vals = pd.to_numeric(design_df[ox], errors="coerce").fillna(0)
        limits[ox] = (0.0, float(vals.quantile(0.99)))
    return limits


def candidates_from_pareto(
    res: Any,
    models: dict[str, Any],
    oxide_cols: list[str],
    pbo_idx: int | None,
    nn: NearestNeighbors,
    scaler: StandardScaler,
    ox_limits: dict[str, tuple[float, float]],
    dist_threshold: float,
    design_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Преобразовать Pareto-решения в таблицу кандидатов."""
    rows = []
    for x in res.X:
        comp_v = decode_composition(x, oxide_cols, pbo_idx)
        comp = dict(zip(oxide_cols, comp_v))
        props = predict_properties(comp_v, models, oxide_cols)
        dist = distance_to_training(comp_v, nn, scaler)

        anomaly = any(comp.get(ox, 0) > ox_limits.get(ox, (0, np.inf))[1] for ox in ANOMALY_OXIDES)
        feasible = (
            props["TG"] >= 450
            and comp.get("SIO2", 0) >= 5
            and comp.get("PBO", 0) <= 0.01
            and dist <= max(dist_threshold, 3.0)
            and not anomaly
            and props["ND300"] >= 1.75
        )

        row_series = pd.Series(comp)
        rows.append({
            "composition": comp_to_str(comp),
            **{f"oxide_{k}": v for k, v in comp.items()},
            "ND300_pred": props["ND300"],
            "NUD300_pred": props["NUD300"],
            "DENSITY_pred": props["DENSITY"],
            "TG_pred": props["TG"],
            "PbO": comp.get("PBO", 0),
            "glass_family": classify_glass_family(row_series, oxide_cols),
            "pareto_rank": 0,
            "distance_to_training": dist,
            "feasible": feasible,
            "anomaly_oxide": anomaly,
        })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(
            ["feasible", "ND300_pred", "distance_to_training"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        df["pareto_rank"] = np.arange(1, len(df) + 1)
    return df


def select_top20(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[
        (df["PbO"] <= 0.01)
        & (df["ND300_pred"] > 1.80)
        & (df["feasible"])
    ].copy()
    sub = sub.sort_values(["ND300_pred", "distance_to_training"], ascending=[False, True])
    return sub.head(20)


def plot_pareto(candidates: pd.DataFrame, output_dir: Path) -> None:
    from viz_presentation import plot_pareto_candidates

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if not len(candidates):
        return
    top20 = select_top20(candidates)
    plot_pareto_candidates(candidates, top20, fig_dir / "pareto_nd_vd.png")


def print_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 64)
    print("INVERSE DESIGN — отчёт")
    print("=" * 64)
    print(f"\nDesign space: {report['design_space_n']:,} записей (ND>1.70, PbO<=1)")
    print("\nForward models (лучшие):")
    for t, m in report.get("forward_models", {}).items():
        print(f"  {t}: {m['best_model']}  R2={m['r2']:.3f}  RMSE={m['rmse']:.4f}")
    print(f"\nNSGA-II: pop={report['nsga_pop']}, gen={report['nsga_gen']}, pareto={report['n_pareto']}")
    print(f"Feasible candidates: {report['n_feasible']}")
    print(f"Top ND300_pred: {report.get('max_nd_pred', 'n/a')}")
    print(f"\nВыход: {report['output_dir']}")


def load_cached_models(output_dir: Path, oxide_cols: list[str]) -> dict[str, Any] | None:
    models_dir = output_dir / "forward_models"
    if not models_dir.exists():
        return None
    models = {}
    for target in TARGETS:
        hits = list(models_dir.glob(f"{target}_*.joblib"))
        if not hits:
            return None
        p = hits[0]
        name = p.stem.split("_", 1)[1]
        models[target] = {"model": joblib.load(p), "model_name": name, "features": oxide_cols}
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description="Inverse design optical glasses")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pop-size", type=int, default=200)
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-models", action="store_true", help="Пропустить обучение, если модели уже есть")
    args = parser.parse_args()

    setup_logging()
    args.output.mkdir(parents=True, exist_ok=True)

    df = load_sciglass_df(args.data)
    oxide_cols = oxide_columns(df)
    if not oxide_cols:
        raise ValueError("Нет колонок оксидов в данных")

    # ЭТАП 1: прямые модели
    models = load_cached_models(args.output, oxide_cols) if args.reuse_models else None
    if models is None:
        models, _ = train_forward_models(df, oxide_cols, args.output, args.seed)
    else:
        logger.info("Переиспользуем сохранённые forward models")

    # Опционально TLiq для ограничения
    tliq_model = None
    if "TLIQ" in PROPERTY_MAP and PROPERTY_MAP["TLIQ"] in df.columns:
        X_t, y_t, _ = prepare_xy(df, "TLIQ", oxide_cols)
        if len(X_t) > 1000:
            tliq_model = MODEL_BUILDERS["lightgbm"](args.seed)
            tliq_model.fit(X_t, y_t)
            joblib.dump(tliq_model, args.output / "forward_models" / "TLIQ_lightgbm.joblib")

    # ЭТАП 2: design space
    design_df = filter_design_space(df, oxide_cols)
    plot_oxide_distributions(design_df, args.output)
    design_df.to_csv(args.output / "design_space_high_n_pb_free.csv", index=False)

    xl, xu = composition_bounds(design_df, oxide_cols)
    pbo_idx = oxide_cols.index("PBO") if "PBO" in oxide_cols else None
    ox_limits = oxide_limits(design_df, oxide_cols)
    nn, scaler, dist_threshold = build_distance_model(design_df, oxide_cols)

    # ЭТАП 3: NSGA-II
    problem = GlassInverseProblem(
        models, oxide_cols, xl, xu, pbo_idx, nn, scaler, tliq_model
    )
    logger.info("NSGA-II: pop=%s gen=%s", args.pop_size, args.generations)
    res = run_nsga2(
        problem, design_df, oxide_cols, xl, xu,
        args.pop_size, args.generations, args.seed,
    )

    # ЭТАП 4: фильтрация
    candidates = candidates_from_pareto(
        res, models, oxide_cols, pbo_idx, nn, scaler, ox_limits,
        dist_threshold=dist_threshold, design_df=design_df,
    )
    top100 = candidates.head(100)
    top20 = select_top20(candidates)

    top100.to_csv(args.output / "top_100_candidates.csv", index=False)
    top20.to_csv(args.output / "top_20_lead_free_high_n.csv", index=False)
    candidates.to_csv(args.output / "all_pareto_candidates.csv", index=False)

    plot_pareto(candidates, args.output)

    summary_path = args.output / "forward_model_summary.json"
    forward_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    report = {
        "design_space_n": len(design_df),
        "forward_models": forward_summary,
        "nsga_pop": args.pop_size,
        "nsga_gen": args.generations,
        "n_pareto": len(candidates),
        "n_feasible": int(candidates["feasible"].sum()) if len(candidates) else 0,
        "max_nd_pred": float(candidates["ND300_pred"].max()) if len(candidates) else None,
        "n_top20": len(top20),
        "distance_threshold_p95": dist_threshold,
        "oxide_limits_p99": ox_limits,
        "output_dir": str(args.output.resolve()),
    }
    (args.output / "inverse_design_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print_report(report)
    logger.info("Готово: %s", args.output)


if __name__ == "__main__":
    main()
