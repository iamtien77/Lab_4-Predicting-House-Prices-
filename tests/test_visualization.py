"""Inline plotting must remain file-free unless export is explicitly requested."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest
from src.evaluation.visualization import (
    plot_actual_vs_predicted, plot_loss_and_mae_curves,
    plot_model_comparison_bar, plot_residuals_distribution,
)


def plot_cases():
    actual = np.array([100000, 150000, 200000])
    predicted = np.array([110000, 140000, 205000])
    history = {"loss": [100, 80], "val_loss": [120, 90], "mae": [8, 6], "val_mae": [9, 7]}
    summary = pd.DataFrame({
        "Model": ["Linear Regression", "MLP PyTorch"],
        "MSE": [100, 80], "RMSE": [10, np.sqrt(80)],
        "MAE": [8, 7], "R2_Score": [0.5, 0.6],
    })
    return [
        (plot_actual_vs_predicted, (actual, predicted), 1),
        (plot_loss_and_mae_curves, (history,), 2),
        (plot_model_comparison_bar, (summary,), 4),
        (plot_residuals_distribution, (actual, predicted), 1),
    ]


def test_inline_figures_do_not_write_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for function, args, axes_count in plot_cases():
        figure = function(*args, save_path=None)
        assert isinstance(figure, Figure)
        assert len(figure.axes) == axes_count
        assert not plt.fignum_exists(figure.number)
        figure.canvas.draw()
    assert list(tmp_path.rglob("*")) == []


def test_explicit_exports_write_valid_png_files(tmp_path):
    for index, (function, args, _) in enumerate(plot_cases()):
        path = tmp_path / "nested" / f"figure_{index}.png"
        figure = function(*args, save_path=path, dpi=50)
        assert isinstance(figure, Figure)
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_history_rejects_nonfinite_and_mismatched_vectors():
    for history in (
        {"loss": [1], "val_loss": [np.nan], "mae": [1], "val_mae": [1]},
        {"loss": [1, 2], "val_loss": [1], "mae": [1], "val_mae": [1]},
    ):
        with pytest.raises(ValueError):
            plot_loss_and_mae_curves(history)


def test_prediction_plots_reject_mismatched_samples():
    for function in (plot_actual_vs_predicted, plot_residuals_distribution):
        with pytest.raises(ValueError):
            function([1, 2], [1])
