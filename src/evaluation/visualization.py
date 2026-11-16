"""Regression figures for inline notebooks and optional CLI exports."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


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
