"""Regression figures for inline notebooks and optional CLI exports."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _finish(figure, save_path, dpi):
    figure.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    # Trả figure đã đóng để notebook hiển thị đúng một lần bằng display.
    plt.close(figure)
    return figure


def _paired_values(y_true, y_pred):
    from src.evaluation.metrics import RegressionEvaluator
    RegressionEvaluator.calculate_mse(y_true, y_pred)
    return np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)


def plot_actual_vs_predicted(y_true, y_pred, model_name="MLP PyTorch",
                             mse=None, r2=None, save_path=None, dpi=300):
    """Return a scatter figure; save only when a path is explicitly supplied."""
    actual, predicted = _paired_values(y_true, y_pred)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(actual, predicted, alpha=0.3, s=12, rasterized=True)
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    axis.plot([low, high], [low, high], "r--", label="Ideal prediction")
    axis.set(xlabel="Actual price (USD)", ylabel="Predicted price (USD)",
             title=f"Actual vs. predicted: {model_name}")
    if mse is not None and r2 is not None:
        axis.text(0.03, 0.97, f"MSE = {mse:,.0f} USD²\nR² = {r2:.4f}",
                  transform=axis.transAxes, va="top")
    axis.legend(loc="lower right")
    return _finish(figure, save_path, dpi)


def plot_loss_and_mae_curves(history, save_path=None, dpi=300):
    """Accept a metric dictionary or a history-compatible object and return learning curves."""
    values = history.history if hasattr(history, "history") else history
    keys = ("loss", "val_loss", "mae", "val_mae")
    arrays = {key: np.asarray(values[key], dtype=float) for key in keys}
    if any(array.ndim != 1 or not len(array) or not np.isfinite(array).all()
           for array in arrays.values()):
        raise ValueError("Training history must contain nonempty finite vectors.")
    if len({len(array) for array in arrays.values()}) != 1:
        raise ValueError("Training history vectors must have equal lengths.")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = np.arange(1, len(arrays["loss"]) + 1)
    for axis, key, unit in zip(axes, ("loss", "mae"), ("MSE (USD²)", "MAE (USD)")):
        axis.plot(epochs, arrays[key], label="Train")
        axis.plot(epochs, arrays[f"val_{key}"], label="Validation")
        axis.set(xlabel="Epoch", ylabel=unit, title=f"Training and validation {key}")
        axis.legend()
        axis.grid(alpha=0.2)
    return _finish(figure, save_path, dpi)


def plot_model_comparison_bar(summary_df, save_path=None, dpi=300):
    """Return separate axes for metrics with different units."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, key, unit in zip(axes.flat, ("MSE", "RMSE", "MAE", "R2_Score"),
                               ("USD²", "USD", "USD", "unitless")):
        values = summary_df[key].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            plt.close(figure)
            raise ValueError("Comparison metrics must be finite.")
        bars = axis.bar(summary_df["Model"], values, color="steelblue")
        axis.bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
        axis.set(title=key, ylabel=unit)
        axis.tick_params(axis="x", labelrotation=18)
        axis.margins(y=0.15)
    return _finish(figure, save_path, dpi)


def plot_residuals_distribution(y_true, y_pred, model_name="MLP PyTorch",
                                save_path=None, dpi=300):
    """Describe residuals without asserting a normal distribution."""
    actual, predicted = _paired_values(y_true, y_pred)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(actual - predicted, bins=40, color="steelblue", edgecolor="white")
    axis.axvline(0, color="red", linestyle="--")
    axis.set(xlabel="Residual: actual - predicted (USD)", ylabel="Count",
             title=f"Residual distribution: {model_name}")
    return _finish(figure, save_path, dpi)


def plot_feature_distributions(df, columns=None, save_path=None, dpi=300):
    """Plot histograms for all numeric features to assess skewness and scale."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    numeric_df = df.select_dtypes(include=[np.number])
    if columns is not None:
        numeric_df = numeric_df[[c for c in columns if c in numeric_df.columns]]
    if numeric_df.empty:
        raise ValueError("No numeric features found to plot distributions.")

    n_cols = 3
    n_rows = int(np.ceil(len(numeric_df.columns) / n_cols))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, col in enumerate(numeric_df.columns):
        series = numeric_df[col].dropna()
        axes[idx].hist(series, bins=35, color="steelblue", edgecolor="white", alpha=0.85)
        axes[idx].set_title(col, fontsize=11, fontweight="bold")
        axes[idx].grid(alpha=0.2)
        axes[idx].tick_params(labelsize=9)

    for idx in range(len(numeric_df.columns), len(axes)):
        axes[idx].set_visible(False)

    figure.suptitle("Phân phối tần suất các đặc trưng số (Feature Distributions)", fontsize=14, y=1.01)
    return _finish(figure, save_path, dpi)


def plot_geographical_distribution(df, target_col="median_house_value", save_path=None, dpi=300):
    """Plot California geographical map with coordinates, prices, and population."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    for req in ("longitude", "latitude"):
        if req not in df.columns:
            raise ValueError(f"DataFrame must contain '{req}' column.")

    figure, axis = plt.subplots(figsize=(10, 7))
    has_target = target_col in df.columns
    c_val = df[target_col] if has_target else "steelblue"
    s_val = (df["population"] / 100).clip(lower=10, upper=300) if "population" in df.columns else 20

    scatter = axis.scatter(
        df["longitude"], df["latitude"],
        c=c_val, cmap="coolwarm" if has_target else None,
        s=s_val, alpha=0.4, label="Dân số (kích thước điểm)" if "population" in df.columns else None
    )
    axis.set_xlabel("Kinh độ (Longitude)", fontsize=11)
    axis.set_ylabel("Vĩ độ (Latitude)", fontsize=11)
    axis.set_title("Bản đồ phân bố không gian và mức giá nhà California", fontsize=13, fontweight="bold")
    axis.grid(alpha=0.2)

    if has_target:
        cbar = figure.colorbar(scatter, ax=axis)
        cbar.set_label("Giá nhà trung vị (USD)", fontsize=10)
    if "population" in df.columns:
        axis.legend(loc="upper right")

    return _finish(figure, save_path, dpi)


def plot_correlation_heatmap(df, target_col="median_house_value", save_path=None, dpi=300):
    """Plot Pearson correlation heatmap across numeric features."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        raise ValueError("At least two numeric features are required for a correlation heatmap.")

    corr = numeric_df.corr(method="pearson")
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                linewidths=0.5, ax=axis, cbar_kws={"label": "Hệ số tương quan Pearson"})
    axis.set_title("Ma trận tương quan Pearson giữa các đặc trưng", fontsize=13, fontweight="bold")
    axis.tick_params(axis="x", labelrotation=35)
    return _finish(figure, save_path, dpi)


def plot_target_distribution(df_or_series, target_col="median_house_value", save_path=None, dpi=300):
    """Plot target distribution histogram and boxplot highlighting price ceiling."""
    if isinstance(df_or_series, pd.DataFrame):
        if target_col not in df_or_series.columns:
            raise ValueError(f"Column '{target_col}' not found in DataFrame.")
        series = df_or_series[target_col].dropna()
    elif isinstance(df_or_series, (pd.Series, np.ndarray)):
        series = pd.Series(df_or_series).dropna()
    else:
        raise TypeError("Input must be a pandas DataFrame, Series, or NumPy array.")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    axes[0].hist(series, bins=45, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(500001, color="red", linestyle="--", linewidth=1.5, label="Trần giá ($500,001)")
    axes[0].set_xlabel("Giá nhà trung vị (USD)", fontsize=11)
    axes[0].set_ylabel("Số lượng mẫu (Count)", fontsize=11)
    axes[0].set_title("Phân phối biến mục tiêu (Histogram)", fontsize=12, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    # Boxplot
    axes[1].boxplot(series, patch_artist=True,
                    boxprops=dict(facecolor="lightblue", color="steelblue"),
                    medianprops=dict(color="red", linewidth=2))
    axes[1].set_ylabel("Giá nhà trung vị (USD)", fontsize=11)
    axes[1].set_title("Biểu đồ hộp biến mục tiêu (Boxplot)", fontsize=12, fontweight="bold")
    axes[1].grid(alpha=0.2)

    figure.suptitle("Phân tích phân phối giá nhà trung vị (Target Distribution)", fontsize=13, y=1.02)
    return _finish(figure, save_path, dpi)


def plot_categorical_distribution(df, cat_col="ocean_proximity", target_col="median_house_value", save_path=None, dpi=300):
    """Plot counts and price comparison across categorical groups."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if cat_col not in df.columns:
        raise ValueError(f"Column '{cat_col}' not found in DataFrame.")

    has_target = target_col in df.columns
    n_axes = 2 if has_target else 1
    figure, axes = plt.subplots(1, n_axes, figsize=(7 * n_axes, 5))
    axes = np.atleast_1d(axes)

    # Bar chart counts
    counts = df[cat_col].value_counts(dropna=False)
    bars = axes[0].bar(counts.index.astype(str), counts.values, color="steelblue", edgecolor="white")
    axes[0].bar_label(bars, fmt="%d", padding=3, fontsize=9)
    axes[0].set_xlabel(cat_col, fontsize=11)
    axes[0].set_ylabel("Số lượng mẫu (Count)", fontsize=11)
    axes[0].set_title(f"Tần suất số lượng mẫu theo {cat_col}", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[0].grid(alpha=0.2)

    # Boxplot by category if target present
    if has_target:
        categories = counts.index.dropna().tolist()
        group_data = [df[df[cat_col] == c][target_col].dropna().values for c in categories]
        axes[1].boxplot(group_data, patch_artist=True,
                        boxprops=dict(facecolor="lightcyan", color="teal"),
                        medianprops=dict(color="darkred", linewidth=1.8))
        axes[1].set_xticks(range(1, len(categories) + 1))
        axes[1].set_xticklabels([str(c) for c in categories])
        axes[1].set_xlabel(cat_col, fontsize=11)
        axes[1].set_ylabel(f"{target_col} (USD)", fontsize=11)
        axes[1].set_title(f"Phân phối giá nhà theo {cat_col}", fontsize=12, fontweight="bold")
        axes[1].tick_params(axis="x", labelrotation=20)
        axes[1].grid(alpha=0.2)

    figure.suptitle(f"Phân tích đặc trưng phân loại '{cat_col}'", fontsize=13, y=1.02)
    return _finish(figure, save_path, dpi)
