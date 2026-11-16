"""CLI orchestration; notebooks call the same reusable modules in src."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import ProjectConfig, config
from src.data.loader import load_housing_data
from src.data.preprocessing import (
    fit_preprocessor, save_processed_splits, split_data_8_1_1, transform_features,
)
from src.evaluation.metrics import RegressionEvaluator
from src.evaluation.visualization import (
    plot_actual_vs_predicted, plot_loss_and_mae_curves,
    plot_model_comparison_bar, plot_residuals_distribution,
)
from src.models.baseline_models import BaselineRegressionSuite
from src.models.mlp_model import build_mlp_model, predict_mlp, train_mlp_model
from src.models.persistence import save_baseline_models, save_mlp_model, save_preprocessor
from src.utils.logger import setup_logger


def _positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return result


def _positive_float(value):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Value must be positive and finite.")
    return result


def parse_args(argv=None):
    """Parse overrides forwarded to the actual training operations."""
    parser = argparse.ArgumentParser(description="Train and evaluate California Housing regressors.")
    parser.add_argument("--epochs", type=_positive_int, default=config.MLP_EPOCHS)
    parser.add_argument("--batch-size", type=_positive_int, default=config.MLP_BATCH_SIZE)
    parser.add_argument("--lr", type=_positive_float, default=config.MLP_LEARNING_RATE)
    parser.add_argument("--random-state", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--apply-log", action="store_true")
    parser.add_argument("--create-interactions", action="store_true")
    parser.add_argument("--data-path", type=Path, default=config.RAW_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=config.BASE_DIR,
                        help="Root for CLI data/processed, results and logs.")
    args = parser.parse_args(argv)
    if not 0 <= args.random_state < 2**32:
        parser.error("--random-state must be in [0, 2**32).")
    return args


def set_random_seed(seed):
    """Seed the CPU experiment and request deterministic PyTorch operations."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def demo_inference(model, preprocessor, X_sample, y_sample):
    """Return up to three predictions from raw held-out rows without refitting."""
    if not X_sample.index.equals(y_sample.index):
        raise ValueError("Sample features and target indexes must match.")
    sample = X_sample.iloc[:3]
    predictions = predict_mlp(model, transform_features(preprocessor, sample))
    actual = y_sample.iloc[:len(sample)].to_numpy()
    frame = pd.DataFrame({"Actual_USD": actual, "Predicted_USD": predictions,
                          "Absolute_Error_USD": np.abs(actual - predictions)}, index=sample.index)
    print("Minh họa dự đoán trên tối đa ba mẫu test (đơn vị USD):")
    print(frame.to_string())
    return frame


def run_pipeline(args=None):
    """Split raw data first, train baselines then MLP, and evaluate test last."""
    args = parse_args([]) if args is None else args
    settings = ProjectConfig(BASE_DIR=Path(args.output_dir).resolve())
    settings.create_directories()
    logger = setup_logger(log_file=settings.LOG_FILE)
    logger.info("Chỉ số: MSE (USD²), RMSE/MAE (USD), R² (không đơn vị).")
    set_random_seed(args.random_state)
    data, target = load_housing_data(args.data_path)
    # Chia dữ liệu thô trước mọi phép học thống kê và tạo đặc trưng.
    splits = split_data_8_1_1(data.drop(columns=target), data[target],
                              random_state=args.random_state)
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    logger.info("Số mẫu train/validation/test: %d/%d/%d", len(X_train), len(X_val), len(X_test))
    save_processed_splits(*splits, target_col=target, output_dir=settings.PROCESSED_DATA_DIR)
    preprocessor = fit_preprocessor(X_train, apply_log=args.apply_log,
                                     create_interactions=args.create_interactions)
    train_features = transform_features(preprocessor, X_train)
    val_features = transform_features(preprocessor, X_val)
    save_preprocessor(preprocessor, settings.PREPROCESSOR_PATH)
    suite = BaselineRegressionSuite(random_state=args.random_state,
                                    dt_max_depth=settings.DT_MAX_DEPTH,
                                    rf_n_estimators=settings.RF_N_ESTIMATORS)
    suite.fit_all(train_features, y_train)
    save_baseline_models(suite.models, settings.BASELINES_PATH)
    model = build_mlp_model(train_features.shape[1], learning_rate=args.lr)
    model, history = train_mlp_model(model, train_features, y_train, val_features, y_val,
                                      epochs=args.epochs, batch_size=args.batch_size)
    logger.info("Best validation checkpoint: epoch %d, RMSE %.2f USD",
                model.best_epoch, np.sqrt(model.best_val_loss))
    save_mlp_model(model, settings.MLP_MODEL_PATH)
    history_frame = pd.DataFrame(history)
    history_frame.insert(0, "epoch", np.arange(1, len(history_frame) + 1))
    history_frame.to_csv(settings.METRICS_DIR / "training_history.csv", index=False)
    validation_predictions = suite.predict_all(val_features)
    validation_predictions["MLP PyTorch"] = predict_mlp(model, val_features)
    validation_summary = RegressionEvaluator.summarize_models([
        RegressionEvaluator.evaluate_predictions(y_val, prediction, name)
        for name, prediction in validation_predictions.items()
    ])
    validation_summary.to_csv(settings.METRICS_DIR / "validation_summary.csv", index=False)
    # Chỉ mở test khi toàn bộ cấu hình và quá trình huấn luyện đã cố định.
    test_features = transform_features(preprocessor, X_test)
    predictions = suite.predict_all(test_features)
    predictions["MLP PyTorch"] = predict_mlp(model, test_features)
    summary = RegressionEvaluator.summarize_models([
        RegressionEvaluator.evaluate_predictions(y_test, prediction, name)
        for name, prediction in predictions.items()
    ])
    summary.to_csv(settings.METRICS_SUMMARY_CSV, index=False)
    logger.info("Kết quả đánh giá cuối trên test:\n%s", summary.to_string(index=False))
    if not args.skip_plots:
        plot_loss_and_mae_curves(history, settings.LOSS_CURVE_PLOT)
        plot_model_comparison_bar(summary, settings.MODEL_COMPARISON_PLOT)
        for name, prediction in predictions.items():
            slug = name.lower().replace(" ", "_")
            row = summary.loc[summary["Model"] == name].iloc[0]
            plot_actual_vs_predicted(y_test, prediction, name, row.MSE, row.R2_Score,
                                      settings.FIGURES_DIR / f"{slug}_actual_vs_predicted.png")
            plot_residuals_distribution(y_test, prediction, name,
                                         settings.FIGURES_DIR / f"{slug}_residuals.png")
    demo = demo_inference(model, preprocessor, X_test, y_test)
    demo.to_csv(settings.METRICS_DIR / "sample_predictions.csv", index=False)
    logger.info("Hoàn thành pipeline; không dùng kết quả test để chỉnh mô hình.")
    return {"summary": summary, "validation_summary": validation_summary,
            "history": history, "predictions": predictions, "splits": splits,
            "preprocessor": preprocessor, "model": model, "baselines": suite.models}


if __name__ == "__main__":
    run_pipeline(parse_args())
