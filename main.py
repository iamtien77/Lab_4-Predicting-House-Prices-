"""CLI orchestration; notebooks call the same reusable modules in src."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd

from src.config import config
from src.data.loader import fetch_dataset_info, load_housing_data
from src.data.preprocessing import (
    fit_preprocessor, save_processed_splits, split_data_8_1_1, transform_features,
)
from src.evaluation.metrics import RegressionEvaluator
from src.evaluation.visualization import (
    plot_actual_vs_predicted, plot_categorical_distribution,
    plot_correlation_heatmap, plot_feature_distributions,
    plot_geographical_distribution, plot_loss_and_mae_curves,
    plot_model_comparison_bar, plot_residuals_distribution,
    plot_target_distribution,
)
from src.models.baseline_models import BaselineRegressionSuite
from src.models.mlp_model import build_mlp_model, predict_mlp, train_mlp_model
from src.models.persistence import (
    file_sha256, load_mlp_model, load_preprocessor, publish_cli_run,
    save_baseline_models, save_mlp_model, save_preprocessor,
)
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
                        help="Root for isolated runs/ and results/latest_run.json.")
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
    # CLI exports PNGs without a desktop/Tk runtime. Importing main in notebooks
    # does not change their inline backend because this runs only on invocation.
    import matplotlib
    matplotlib.use("Agg")
    args = parse_args([]) if args is None else args
    output_root = Path(args.output_dir).resolve()
    started_at = datetime.now(timezone.utc)
    run_id = f"{started_at:%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}"
    run_dir = output_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    settings = replace(config, BASE_DIR=run_dir)
    settings.create_directories()
    logger = setup_logger(log_file=settings.LOG_FILE)
    logger.info("Chỉ số: MSE (USD²), RMSE/MAE (USD), R² (không đơn vị).")
    set_random_seed(args.random_state)
    source_hash = file_sha256(args.data_path)
    source_files = [config.BASE_DIR / "main.py", config.BASE_DIR / "requirements.txt",
                    *sorted((config.BASE_DIR / "src").rglob("*.py"))]
    source_hashes = {path.relative_to(config.BASE_DIR).as_posix(): file_sha256(path)
                     for path in source_files}
    data, target = load_housing_data(args.data_path)
    # Xác định và khám phá tổng quan dữ liệu thô ban đầu (Data Understanding)
    raw_info = fetch_dataset_info(data, target)
    missing_cols = {k: v for k, v in raw_info["missing"].items() if v > 0}
    target_stats = raw_info["target_summary"]
    logger.info("Xác định dữ liệu thô: %d dòng, %d cột | Số dòng trùng lặp: %d",
                raw_info["shape"][0], raw_info["shape"][1], raw_info["duplicate_rows"])
    logger.info("Kiểm tra giá trị khuyết thiếu ban đầu: %s",
                f"{missing_cols}" if missing_cols else "Không có giá trị khuyết")
    logger.info("Biến mục tiêu '%s': min=%.0f USD, trung vị=%.0f USD, max=%.0f USD (đạt trần giá: %d mẫu)",
                target, target_stats["min"], target_stats["50%"], target_stats["max"],
                raw_info["target_max_count"])
    logger.info("Chia tập 80/10/10; phân tầng target=%s; số bins=%d/%d.",
                settings.USE_STRATIFIED_SPLIT, settings.N_STRATIFY_BINS_TRAIN,
                settings.N_STRATIFY_BINS_TEMP)
    # Chia dữ liệu thô trước mọi phép học thống kê và tạo đặc trưng.
    splits = split_data_8_1_1(data.drop(columns=target), data[target],
                              random_state=args.random_state,
                              stratify=settings.USE_STRATIFIED_SPLIT,
                              n_bins_train=settings.N_STRATIFY_BINS_TRAIN,
                              n_bins_temp=settings.N_STRATIFY_BINS_TEMP)
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
                                    rf_n_estimators=settings.RF_N_ESTIMATORS,
                                    fit_intercept=settings.LR_FIT_INTERCEPT)
    suite.fit_all(train_features, y_train)
    save_baseline_models(suite.models, settings.BASELINES_PATH)
    model = build_mlp_model(train_features.shape[1], learning_rate=args.lr)
    model, history = train_mlp_model(model, train_features, y_train, val_features, y_val,
                                      epochs=args.epochs, batch_size=args.batch_size,
                                      verbose=settings.MLP_VERBOSE)
    logger.info("Điểm lưu (checkpoint) tốt nhất trên tập kiểm định: epoch %d, RMSE %.2f USD",
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
        train_frame = X_train.assign(**{target: y_train})
        plot_feature_distributions(train_frame, save_path=settings.FIGURES_DIR / "eda_feature_distributions.png")
        plot_geographical_distribution(train_frame, target_col=target, save_path=settings.FIGURES_DIR / "eda_geographical_scatter.png")
        plot_correlation_heatmap(train_frame, target_col=target, save_path=settings.FIGURES_DIR / "eda_correlation_heatmap.png")
        plot_target_distribution(train_frame, target_col=target, save_path=settings.FIGURES_DIR / "eda_target_distribution.png")
        plot_categorical_distribution(train_frame, cat_col="ocean_proximity", target_col=target, save_path=settings.FIGURES_DIR / "eda_categorical_distribution.png")
        logger.info("Đã xuất 5 biểu đồ EDA trực quan vào: %s", settings.FIGURES_DIR)

        plot_loss_and_mae_curves(history, settings.LOSS_CURVE_PLOT)
        plot_model_comparison_bar(summary, settings.MODEL_COMPARISON_PLOT)
        for name, prediction in predictions.items():
            slug = name.lower().replace(" ", "_")
            row = summary.loc[summary["Model"] == name].iloc[0]
            plot_actual_vs_predicted(y_test, prediction, name, row.MSE, row.R2_Score,
                                      settings.FIGURES_DIR / f"{slug}_actual_vs_predicted.png")
            plot_residuals_distribution(y_test, prediction, name,
                                         settings.FIGURES_DIR / f"{slug}_residuals.png")
    restored_model = load_mlp_model(settings.MLP_MODEL_PATH)
    restored_preprocessor = load_preprocessor(settings.PREPROCESSOR_PATH)
    demo = demo_inference(restored_model, restored_preprocessor, X_test, y_test)
    np.testing.assert_allclose(demo["Predicted_USD"], predictions["MLP PyTorch"][:len(demo)],
                               rtol=1e-5, atol=1e-3)
    demo.to_csv(settings.METRICS_DIR / "sample_predictions.csv", index=False)
    if file_sha256(args.data_path) != source_hash:
        raise ValueError("Raw dataset changed during training; this run will not be published.")
    effective_config = {**settings.to_dict(), "RANDOM_STATE": args.random_state,
                        "MLP_EPOCHS": args.epochs, "MLP_BATCH_SIZE": args.batch_size,
                        "MLP_LEARNING_RATE": args.lr, "apply_log": args.apply_log,
                        "create_interactions": args.create_interactions, "skip_plots": args.skip_plots}
    current_source_files = [config.BASE_DIR / "main.py", config.BASE_DIR / "requirements.txt",
                            *sorted((config.BASE_DIR / "src").rglob("*.py"))]
    current_hashes = {path.relative_to(config.BASE_DIR).as_posix(): file_sha256(path)
                      for path in current_source_files}
    if current_hashes != source_hashes:
        raise ValueError("Source code changed during training; this run will not be published.")
    manifest = publish_cli_run(output_root, run_dir, {
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data": {"path": str(Path(args.data_path).resolve()), "sha256": source_hash},
        "parameters": effective_config,
        "feature_names": list(preprocessor.get_feature_names_out()),
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "best_epoch": model.best_epoch,
        "package_versions": {name: version(name) for name in ("numpy", "pandas", "torch", "scikit-learn")},
        "source_sha256": source_hashes,
    })
    logger.info("Đã công bố bộ kết quả đồng bộ: %s", run_dir)
    logger.info("Hoàn thành pipeline; không dùng kết quả test để chỉnh mô hình.")
    return {"run_dir": run_dir, "manifest": manifest,
            "raw_info": raw_info, "summary": summary, "validation_summary": validation_summary,
            "history": history, "predictions": predictions, "splits": splits,
            "preprocessor": preprocessor, "model": model, "baselines": suite.models}


if __name__ == "__main__":
    run_pipeline(parse_args())
