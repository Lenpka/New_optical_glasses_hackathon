"""Графики для анализа данных хакатона.

Импорт в ноутбуке (из папки проекта):

    from plots import plot_distribution, plot_boxplot, plot_parity

    plot_distribution(df, "RefractiveIndex")
    plot_boxplot(df, "AbbeNumber", by="Glass_Class")
    plot_parity(y_true, y_pred)

Повторная настройка стиля (опционально):

    from plots import configure_theme
    configure_theme()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, linregress, pearsonr, skew

__all__ = [
    "FEATURE_SET_LABELS",
    "MODEL_LABELS",
    "MODEL_ORDER",
    "configure_theme",
    "format_axis_label",
    "plot_boxplot",
    "plot_distribution",
    "plot_metrics_comparison",
    "plot_model_comparison_suite",
    "plot_parity",
    "plot_parity_grid",
]

FEATURE_SET_LABELS = {
    "elements": "Элементы (AtMol)",
    "oxides": "Оксиды (мольн.)",
    "oxides_wt": "Оксиды (вес.)",
}

MODEL_LABELS = {
    "linear_regression": "Linear",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}

MODEL_ORDER = [
    "linear_regression",
    "ridge",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
]

DEFAULT_COLORS = ["#348ABD", "#7A68A6", "#A60628", "#467821", "#CF4457"]

DEFAULT_THEME: dict[str, Any] = {
    "context": "notebook",
    "style": "whitegrid",
    "palette": "deep",
    "font_scale": 1.2,
    "rc": {
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "figure.figsize": (10, 6),
        "axes.titlepad": 15,
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
    },
}


def configure_theme(**overrides: Any) -> None:
    """Применить стиль seaborn/matplotlib (без plt.show и tight_layout)."""
    params = {**DEFAULT_THEME, **overrides}
    rc = params.pop("rc", {})
    sns.set_theme(**params)
    plt.rcParams.update(rc)


def format_axis_label(name: str, unit: Optional[str] = None) -> str:
    """Подпись оси с единицей измерения. unit может быть LaTeX, напр. r'$\\mu$m$'."""
    if not unit:
        return name
    unit = unit.strip()
    if unit.startswith("$"):
        return f"{name}, {unit}"
    return f"{name} ({unit})"


def _resolve_series(
    data: Union[pd.DataFrame, pd.Series, None],
    column: Union[str, pd.Series, np.ndarray, None],
    default_name: str = "значение",
) -> tuple[pd.Series, str]:
    if isinstance(column, str):
        if data is None:
            raise ValueError("Для строкового column нужен DataFrame/Series в data.")
        series = data[column]
        colname = column
    elif isinstance(column, pd.Series):
        series = column
        colname = getattr(column, "name", default_name)
    else:
        series = pd.Series(column, name=default_name)
        colname = default_name
    return series, colname


def _resolve_xy_pair(
    y_true: Union[pd.Series, np.ndarray, list[float], None] = None,
    y_pred: Union[pd.Series, np.ndarray, list[float], None] = None,
    *,
    data: Optional[pd.DataFrame] = None,
    true_column: Optional[str] = None,
    pred_column: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    if data is not None:
        if true_column is None or pred_column is None:
            raise ValueError("Укажите true_column и pred_column при передаче data.")
        true = data[true_column].to_numpy(dtype=float)
        pred = data[pred_column].to_numpy(dtype=float)
        return true, pred, true_column, pred_column
    if y_true is None or y_pred is None:
        raise ValueError("Передайте (y_true, y_pred) или data с true_column/pred_column.")
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return true, pred, "Истинное значение", "Предсказанное значение"


def _descriptive_stats(x: pd.Series) -> dict[str, Any]:
    x = x.dropna()
    return {
        "count": len(x),
        "mean": x.mean(),
        "std": x.std(),
        "median": x.median(),
        "q25": x.quantile(0.25),
        "q75": x.quantile(0.75),
        "min": x.min(),
        "max": x.max(),
        "skewness": skew(x),
        "kurtosis": kurtosis(x, fisher=True),
        "iqr": x.quantile(0.75) - x.quantile(0.25),
    }


def _format_descriptive_text(
    stats: Mapping[str, Any],
    *,
    prefix: str = "",
    median_label: str = "Медиана",
    q1_label: str = "1-й квартиль",
    q3_label: str = "3-й квартиль",
) -> str:
    head = f"{prefix}\n" if prefix else ""
    return head + "\n".join(
        [
            f"N: {stats['count']:,}",
            f"μ: {stats['mean']:,.4g}",
            f"σ: {stats['std']:,.4g}",
            f"{median_label}: {stats['median']:,.4g}",
            f"{q1_label}–{q3_label}: [{stats['q25']:,.4g}, {stats['q75']:,.4g}]",
            f"IQR: {stats['iqr']:,.4g}",
            f"Мин–макс: [{stats['min']:,.4g}, {stats['max']:,.4g}]",
            f"Асимметрия: {stats['skewness']:,.3f}",
            f"Экс. эксцесс: {stats['kurtosis']:,.3f}",
        ]
    )


def _parity_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    residuals = y_pred - y_true
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    bias = float(np.mean(residuals))
    mape = (
        float(np.mean(np.abs(residuals / y_true)) * 100)
        if len(y_true) and not np.any(y_true == 0)
        else np.nan
    )
    pearson_r = pearsonr(y_true, y_pred)[0] if len(y_true) > 1 else np.nan
    slope, intercept, _, _, _ = linregress(y_true, y_pred) if len(y_true) > 1 else (np.nan,) * 5
    return {
        "count": len(y_true),
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "bias": bias,
        "pearson_r": pearson_r,
        "slope": float(slope),
        "intercept": float(intercept),
        "max_abs_error": float(np.max(np.abs(residuals))) if len(residuals) else np.nan,
    }


def _format_parity_text(stats: Mapping[str, Any]) -> str:
    lines = [
        f"N: {stats['count']:,}",
        f"R²: {stats['r2']:,.4f}",
        f"RMSE: {stats['rmse']:,.4g}",
        f"MAE: {stats['mae']:,.4g}",
        f"Смещение: {stats['bias']:,.4g}",
        f"r (Пирсон): {stats['pearson_r']:,.4f}",
        f"Макс. |ошибка|: {stats['max_abs_error']:,.4g}",
        f"Наклон: {stats['slope']:,.4g}",
        f"Сдвиг: {stats['intercept']:,.4g}",
    ]
    if not np.isnan(stats["mape"]):
        lines.insert(4, f"MAPE: {stats['mape']:,.2f}%")
    return "\n".join(lines)


def _add_stat_box(
    ax: plt.Axes,
    text: str,
    position: tuple[float, float],
    *,
    ha: str = "right",
    va: str = "top",
    facecolor: str = "white",
    fontsize: int = 15,
) -> None:
    ax.text(
        position[0],
        position[1],
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        transform=ax.transAxes,
        bbox=dict(facecolor=facecolor, edgecolor="gray", alpha=0.9, pad=0.5),
    )


def _get_or_create_axes(
    ax: Optional[plt.Axes],
    figsize: tuple[float, float],
) -> tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, main_ax = plt.subplots(figsize=figsize)
        return fig, main_ax
    return ax.figure, ax


def _finalize_figure(
    fig: plt.Figure,
    *,
    show: bool,
    return_value: Optional[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    if show:
        fig.tight_layout()
        plt.show()
    return return_value if return_value is not None else None


def plot_distribution(
    data: Union[pd.DataFrame, pd.Series],
    column: Union[str, pd.Series],
    *,
    title: Optional[str] = None,
    bins: int = 50,
    figsize: tuple[float, float] = (14, 10),
    kde: bool = True,
    stats_position: tuple[float, float] = (0.98, 0.99),
    hist_colors: Optional[list[str]] = None,
    kde_colors: Optional[list[str]] = None,
    box_facecolor: str = "white",
    show_stats: bool = True,
    show_violin: bool = False,
    violin_color: str = "#FFD23F",
    rug: bool = False,
    return_stats: bool = False,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
    mean_color: str = "#348ABD",
    median_color: str = "#A60628",
    q1_color: str = "#467821",
    q3_color: str = "#CF4457",
    mean_label: str = "Среднее",
    median_label: str = "Медиана",
    q1_label: str = "1-й квартиль",
    q3_label: str = "3-й квартиль",
    hist_label: str = "Гистограмма",
    kde_label: str = "Плотность (KDE)",
    xlabel: Optional[str] = None,
    ylabel: str = "Плотность",
    xlabel_unit: Optional[str] = None,
    ylabel_unit: Optional[str] = None,
    legend_loc: str = "center right",
    **kwargs: Any,
) -> Optional[Mapping[str, Any]]:
    """Гистограмма + KDE с вертикальными линиями статистик и русской легендой."""
    if hist_colors is None:
        hist_colors = DEFAULT_COLORS.copy()
    if kde_colors is None:
        kde_colors = ["#FFD23F", "#0057B7"]

    series, colname = _resolve_series(data, column)
    x = series.dropna()
    stats = _descriptive_stats(x)

    fig, main_ax = _get_or_create_axes(ax, figsize)

    hist_kwargs = {k: v for k, v in kwargs.items() if k in {"alpha", "edgecolor", "linewidth"}}
    sns.histplot(
        x,
        bins=bins,
        stat="density",
        kde=False,
        color=hist_colors[0],
        label=hist_label,
        ax=main_ax,
        **hist_kwargs,
    )

    if kde:
        sns.kdeplot(
            x,
            color=kde_colors[0],
            linewidth=2.5,
            label=kde_label,
            ax=main_ax,
        )

    if rug:
        sns.rugplot(x, color=hist_colors[1], alpha=0.35, height=0.03, ax=main_ax)

    for value, color, label, linestyle in [
        (stats["mean"], mean_color, mean_label, "--"),
        (stats["median"], median_color, median_label, "-"),
        (stats["q25"], q1_color, q1_label, ":"),
        (stats["q75"], q3_color, q3_label, ":"),
    ]:
        main_ax.axvline(
            value,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=f"{label}: {value:,.4g}",
        )

    if show_stats:
        _add_stat_box(
            main_ax,
            _format_descriptive_text(stats, median_label=median_label, q1_label=q1_label, q3_label=q3_label),
            stats_position,
            facecolor=box_facecolor,
        )

    main_ax.set_title(title or f"Распределение: {colname}", fontsize=20, pad=12)
    main_ax.set_xlabel(format_axis_label(xlabel or colname, xlabel_unit))
    main_ax.set_ylabel(format_axis_label(ylabel, ylabel_unit))
    main_ax.grid(True, alpha=0.25, linestyle="--")
    main_ax.legend(loc=legend_loc, framealpha=0.95, fontsize=16)

    if show_violin:
        pos = main_ax.get_position()
        divider = fig.add_axes(
            [pos.x0, pos.y0 - 0.08, pos.width, 0.08],
            sharex=main_ax,
        )
        sns.violinplot(x=x, ax=divider, color=violin_color, linewidth=1, cut=0)
        divider.set_axis_off()

    return _finalize_figure(fig, show=show, return_value=stats if return_stats else None)


def plot_boxplot(
    data: Union[pd.DataFrame, pd.Series],
    column: Union[str, pd.Series],
    *,
    by: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (14, 10),
    stats_position: tuple[float, float] = (0.98, 0.99),
    palette: Optional[list[str]] = None,
    box_color: str = "#348ABD",
    box_facecolor: str = "white",
    show_stats: bool = True,
    show_mean: bool = True,
    show_reference_lines: bool = True,
    show_strip: bool = True,
    strip_alpha: float = 0.35,
    show_outliers: bool = True,
    return_stats: bool = False,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
    mean_color: str = "#FFD23F",
    median_color: str = "#A60628",
    mean_label: str = "Среднее",
    median_label: str = "Медиана",
    q1_label: str = "1-й квартиль",
    q3_label: str = "3-й квартиль",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel_unit: Optional[str] = None,
    ylabel_unit: Optional[str] = None,
    legend_loc: str = "center right",
    **kwargs: Any,
) -> Optional[Mapping[str, Any]]:
    """Boxplot с точками, линиями статистик и блоком описательных метрик."""
    series, colname = _resolve_series(data, column)
    fig, main_ax = _get_or_create_axes(ax, figsize)

    if palette is None:
        palette = DEFAULT_COLORS

    all_stats: dict[str, Any] = {}

    if by is None:
        x = series.dropna()
        stats = _descriptive_stats(x)
        all_stats[colname] = stats

        sns.boxplot(
            y=x,
            color=box_color,
            ax=main_ax,
            showfliers=show_outliers,
            width=0.45,
            **{k: v for k, v in kwargs.items() if k in {"linewidth", "fliersize"}},
        )

        if show_strip:
            sns.stripplot(
                y=x,
                color=box_color,
                alpha=strip_alpha,
                size=3,
                jitter=0.18,
                ax=main_ax,
            )

        if show_reference_lines:
            for value, color, label, linestyle in [
                (stats["mean"], mean_color, mean_label, "--"),
                (stats["median"], median_color, median_label, "-"),
                (stats["q25"], palette[3], q1_label, ":"),
                (stats["q75"], palette[4], q3_label, ":"),
            ]:
                main_ax.axhline(
                    value,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    label=f"{label}: {value:,.4g}",
                )

        if show_stats:
            _add_stat_box(
                main_ax,
                _format_descriptive_text(stats, median_label=median_label, q1_label=q1_label, q3_label=q3_label),
                stats_position,
                facecolor=box_facecolor,
            )

        ylab = format_axis_label(ylabel or colname, ylabel_unit)
        main_ax.set_ylabel(ylab)
        main_ax.set_xlabel("")
        main_ax.set_xticks([])
        plot_title = title or f"Boxplot: {colname}"
    else:
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Для группировки by нужен DataFrame в data.")
        df = data[[column, by]].dropna()
        groups = df.groupby(by, sort=False)[column]
        stat_blocks: list[str] = []

        sns.boxplot(
            data=df,
            x=by,
            y=column,
            palette=palette,
            ax=main_ax,
            showfliers=show_outliers,
            **{k: v for k, v in kwargs.items() if k in {"linewidth", "fliersize"}},
        )

        if show_strip:
            sns.stripplot(
                data=df,
                x=by,
                y=column,
                color="black",
                alpha=strip_alpha,
                size=2.5,
                jitter=0.22,
                ax=main_ax,
            )

        if show_mean:
            means = groups.mean()
            for idx, (name, mean_val) in enumerate(means.items()):
                main_ax.scatter(
                    idx,
                    mean_val,
                    color=mean_color,
                    s=90,
                    zorder=5,
                    marker="D",
                    label=mean_label if idx == 0 else None,
                )

        for name, group in groups:
            stats = _descriptive_stats(group)
            all_stats[str(name)] = stats
            stat_blocks.append(
                _format_descriptive_text(stats, prefix=str(name), median_label=median_label, q1_label=q1_label, q3_label=q3_label)
            )

        if show_stats:
            _add_stat_box(
                main_ax,
                "\n\n".join(stat_blocks),
                stats_position,
                facecolor=box_facecolor,
                fontsize=12,
            )

        ylab = format_axis_label(ylabel or colname, ylabel_unit)
        xlab = format_axis_label(xlabel or by, xlabel_unit)
        main_ax.set_ylabel(ylab)
        main_ax.set_xlabel(xlab)
        plot_title = title or f"Boxplot: {colname} по {by}"

    main_ax.set_title(plot_title, fontsize=20, pad=12)
    main_ax.grid(True, alpha=0.25, linestyle="--", axis="y")
    if show_reference_lines or (by is not None and show_mean):
        main_ax.legend(loc=legend_loc, framealpha=0.95, fontsize=14)

    return _finalize_figure(fig, show=show, return_value=all_stats if return_stats else None)


def plot_parity(
    y_true: Union[pd.Series, np.ndarray, list[float], None] = None,
    y_pred: Union[pd.Series, np.ndarray, list[float], None] = None,
    *,
    data: Optional[pd.DataFrame] = None,
    true_column: Optional[str] = None,
    pred_column: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12, 12),
    stats_position: tuple[float, float] = (0.02, 0.98),
    box_facecolor: str = "white",
    show_stats: bool = True,
    show_diagonal: bool = True,
    show_regression: bool = True,
    show_margin: bool = False,
    margin_pct: float = 10.0,
    equal_aspect: bool = True,
    alpha: float = 0.45,
    point_color: str = "#348ABD",
    diagonal_color: str = "#A60628",
    regression_color: str = "#467821",
    margin_color: str = "#7A68A6",
    point_label: str = "Образцы",
    diagonal_label: str = "Идеал (y = x)",
    regression_label: str = "Линейная регрессия",
    margin_label: str = "±{pct:.0f}%",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel_unit: Optional[str] = None,
    ylabel_unit: Optional[str] = None,
    legend_loc: str = "lower right",
    return_stats: bool = False,
    show: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs: Any,
) -> Optional[Mapping[str, Any]]:
    """Parity plot: предсказанное vs истинное с метриками качества модели."""
    true, pred, default_xlabel, default_ylabel = _resolve_xy_pair(
        y_true,
        y_pred,
        data=data,
        true_column=true_column,
        pred_column=pred_column,
    )
    stats = _parity_metrics(true, pred)

    fig, main_ax = _get_or_create_axes(ax, figsize)

    scatter_kwargs = {k: v for k, v in kwargs.items() if k in {"s", "edgecolor", "linewidths"}}
    main_ax.scatter(
        true,
        pred,
        alpha=alpha,
        color=point_color,
        label=point_label,
        **scatter_kwargs,
    )

    lo = float(min(true.min(), pred.min()))
    hi = float(max(true.max(), pred.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    lo -= pad
    hi += pad

    if show_diagonal:
        main_ax.plot(
            [lo, hi],
            [lo, hi],
            color=diagonal_color,
            linestyle="--",
            linewidth=2.5,
            label=diagonal_label,
        )

    if show_regression and stats["count"] > 1 and not np.isnan(stats["slope"]):
        x_line = np.array([lo, hi])
        y_line = stats["slope"] * x_line + stats["intercept"]
        main_ax.plot(
            x_line,
            y_line,
            color=regression_color,
            linewidth=2.5,
            label=f"{regression_label}: y = {stats['slope']:,.3g}x + {stats['intercept']:,.3g}",
        )

    if show_margin:
        pct = margin_pct / 100.0
        x_band = np.array([lo, hi])
        main_ax.plot(
            x_band,
            x_band * (1 + pct),
            color=margin_color,
            linestyle=":",
            linewidth=2,
            label=margin_label.format(pct=margin_pct) + " (верх)",
        )
        main_ax.plot(
            x_band,
            x_band * (1 - pct),
            color=margin_color,
            linestyle=":",
            linewidth=2,
            label=margin_label.format(pct=margin_pct) + " (низ)",
        )

    if show_stats:
        _add_stat_box(
            main_ax,
            _format_parity_text(stats),
            stats_position,
            ha="left",
            va="top",
            facecolor=box_facecolor,
        )

    xlab = format_axis_label(xlabel or default_xlabel, xlabel_unit)
    ylab = format_axis_label(ylabel or default_ylabel, ylabel_unit)
    main_ax.set_xlabel(xlab)
    main_ax.set_ylabel(ylab)
    main_ax.set_title(title or "Parity plot", fontsize=20, pad=12)
    main_ax.set_xlim(lo, hi)
    main_ax.set_ylim(lo, hi)
    if equal_aspect:
        main_ax.set_aspect("equal", adjustable="box")
    main_ax.grid(True, alpha=0.25, linestyle="--")
    main_ax.legend(loc=legend_loc, framealpha=0.95, fontsize=14)

    return _finalize_figure(fig, show=show, return_value=stats if return_stats else None)


def _parity_on_axes(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str,
    show_stats: bool = True,
    alpha: float = 0.35,
    point_color: str = "#348ABD",
) -> dict[str, Any]:
    """Компактный parity plot на заданных осях."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    stats = _parity_metrics(y_true, y_pred)

    ax.scatter(y_true, y_pred, alpha=alpha, s=12, color=point_color, rasterized=True)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    lo -= pad
    hi += pad

    ax.plot([lo, hi], [lo, hi], color="#A60628", linestyle="--", linewidth=1.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_xlabel("Истинный n", fontsize=9)
    ax.set_ylabel("Предсказанный n", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linestyle="--")

    if show_stats:
        short = f"R2={stats['r2']:.3f}\nRMSE={stats['rmse']:.3g}\nMAE={stats['mae']:.3g}"
        ax.text(
            0.03,
            0.97,
            short,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85, pad=0.3),
        )
    return stats


def plot_metrics_comparison(
    metrics: pd.DataFrame,
    *,
    metrics_to_plot: tuple[str, ...] = ("RMSE", "MAE", "R2"),
    figsize: tuple[float, float] = (18, 5),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Столбчатые диаграммы метрик: модели x тип данных."""
    required = {"feature_set", "model", *metrics_to_plot}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"В metrics нет колонок: {missing}")

    plot_df = metrics.copy()
    plot_df["model"] = pd.Categorical(
        plot_df["model"],
        categories=[m for m in MODEL_ORDER if m in plot_df["model"].unique()],
        ordered=True,
    )
    plot_df["feature_label"] = plot_df["feature_set"].map(
        lambda x: FEATURE_SET_LABELS.get(x, x)
    )
    plot_df["model_label"] = plot_df["model"].map(
        lambda m: MODEL_LABELS.get(str(m), str(m))
    )

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figsize, sharey=False)
    if len(metrics_to_plot) == 1:
        axes = [axes]

    palette = sns.color_palette("deep", n_colors=plot_df["feature_label"].nunique())

    for ax, metric in zip(axes, metrics_to_plot):
        sns.barplot(
            data=plot_df,
            x="model_label",
            y=metric,
            hue="feature_label",
            ax=ax,
            palette=palette,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.set_title(f"Сравнение моделей: {metric}", fontsize=14)
        ax.tick_params(axis="x", rotation=35, labelsize=9)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        if metric == "R2":
            ax.set_ylim(0, min(1.05, plot_df[metric].max() * 1.08))
        ax.legend(title="Тип данных", fontsize=9, title_fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Метрики на тестовой выборке", fontsize=18, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    return fig


def plot_parity_grid(
    panels: list[dict[str, Any]],
    *,
    ncols: int = 3,
    figsize_per_ax: tuple[float, float] = (4.2, 4.2),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """Сетка parity plots. Каждая панель: y_true, y_pred, title."""
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, panel in zip(axes_flat, panels):
        _parity_on_axes(
            ax,
            panel["y_true"],
            panel["y_pred"],
            title=panel.get("title", ""),
            show_stats=panel.get("show_stats", True),
            point_color=panel.get("color", "#348ABD"),
        )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, y=1.01)
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    return fig


def plot_model_comparison_suite(
    metrics: pd.DataFrame,
    parity_panels: dict[str, list[dict[str, Any]]],
    best_panels: list[dict[str, Any]],
    global_best_panel: dict[str, Any],
    *,
    output_dir: Union[str, Path],
    show: bool = False,
) -> dict[str, Path]:
    """Сохранить полный набор сравнительных графиков."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    p = out / "01_metrics_bars.png"
    fig = plot_metrics_comparison(metrics, save_path=p, show=show)
    plt.close(fig)
    saved["metrics_bars"] = p

    # Тепловая карта RMSE
    pivot = metrics.pivot(index="model", columns="feature_set", values="RMSE")
    pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])
    pivot.columns = [FEATURE_SET_LABELS.get(c, c) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd_r", ax=ax, linewidths=0.5)
    ax.set_title("RMSE: модель x тип данных", fontsize=14)
    ax.set_ylabel("")
    ax.set_xlabel("Тип данных")
    fig.tight_layout()
    p = out / "02_metrics_heatmap_rmse.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved["metrics_heatmap"] = p

    # Все модели x все типы данных
    all_panels: list[dict[str, Any]] = []
    for fs_key in parity_panels:
        all_panels.extend(parity_panels[fs_key])
    p = out / "03_parity_all_models.png"
    fig = plot_parity_grid(
        all_panels,
        ncols=6,
        figsize_per_ax=(3.2, 3.2),
        save_path=p,
        show=show,
        suptitle="Parity: все модели x все типы данных",
    )
    plt.close(fig)
    saved["parity_all"] = p

    # По одному ряду на тип данных
    for fs_key, panels in parity_panels.items():
        label = FEATURE_SET_LABELS.get(fs_key, fs_key)
        p = out / f"04_parity_{fs_key}.png"
        fig = plot_parity_grid(
            panels,
            ncols=3,
            figsize_per_ax=(4.5, 4.5),
            save_path=p,
            show=show,
            suptitle=f"Parity: {label}",
        )
        plt.close(fig)
        saved[f"parity_{fs_key}"] = p

    # Лучшая модель в каждом типе данных
    p = out / "05_parity_best_per_type.png"
    fig = plot_parity_grid(
        best_panels,
        ncols=3,
        figsize_per_ax=(5, 5),
        save_path=p,
        show=show,
        suptitle="Лучшая модель в каждом типе данных",
    )
    plt.close(fig)
    saved["parity_best_per_type"] = p

    # Абсолютный лидер
    p = out / "06_parity_global_best.png"
    plot_parity(
        global_best_panel["y_true"],
        global_best_panel["y_pred"],
        title=global_best_panel.get("title", "Лучшая модель (все типы)"),
        show_margin=True,
        margin_pct=5,
        show=False,
    )
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved["parity_global_best"] = p

    return saved


configure_theme()
