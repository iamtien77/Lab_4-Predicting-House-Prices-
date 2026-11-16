"""Small end-to-end training tests with real CLI overrides and artifacts."""
import numpy as np
import pandas as pd
import pytest

import main
from src.config import config
from src.data.preprocessing import transform_features
from src.models.mlp_model import predict_mlp
from src.models.persistence import load_preprocessor, load_baseline_models, load_mlp_model


def test_pipeline_smoke_and_inference_round_trip(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    size = 200
    data = pd.DataFrame({
        "longitude": rng.uniform(-124, -114, size),
        "latitude": rng.uniform(32, 42, size),
        "housing_median_age": rng.integers(1, 50, size).astype(float),
        "total_rooms": rng.uniform(500, 3000, size),
        "total_bedrooms": rng.uniform(100, 400, size),
        "population": rng.uniform(300, 1500, size),
        "households": rng.uniform(80, 300, size),
        "median_income": rng.uniform(1, 10, size),
        "ocean_proximity": np.where(np.arange(size) % 2, "INLAND", "NEAR BAY"),
    })
    data.loc[::5, "total_bedrooms"] = np.nan
    data[config.TARGET_COLUMN] = 50000 + 20000 * data["median_income"] + rng.normal(0, 1000, size)
    source = tmp_path / "housing.csv"
    data.to_csv(source, index=False)
    original_bytes = source.read_bytes()
    output = tmp_path / "run"
    args = main.parse_args([
        "--data-path", str(source), "--output-dir", str(output),
        "--epochs", "1", "--batch-size", "64", "--lr", "0.002",
        "--random-state", "42", "--apply-log", "--create-interactions",
    ])
    events = []
    observed = {}
    actual_split = main.split_data_8_1_1
    actual_fit = main.fit_preprocessor
    actual_transform = main.transform_features
    actual_baseline_fit = main.BaselineRegressionSuite.fit_all
    actual_train = main.train_mlp_model

    def split_spy(X, y, **kwargs):
        events.append("split")
        splits = actual_split(X, y, **kwargs)
        observed["splits"] = splits
        return splits

    def preprocessing_spy(X, **kwargs):
        assert events == ["split"]
        pd.testing.assert_frame_equal(X, observed["splits"][0])
        assert kwargs == {"apply_log": True, "create_interactions": True}
        events.append("fit_preprocessor")
        return actual_fit(X, **kwargs)

    def transform_spy(processor, X):
        test_index = observed["splits"][2].index
        if len(X.index.intersection(test_index)):
            assert "mlp_finished" in events
            events.append("test_transform")
        return actual_transform(processor, X)

    def baseline_spy(self, X, y):
        assert "mlp_started" not in events
        result = actual_baseline_fit(self, X, y)
        events.append("baselines_finished")
        return result

    def training_spy(model, X, y, X_val, y_val, **kwargs):
        assert "baselines_finished" in events
        assert kwargs["epochs"] == 1 and kwargs["batch_size"] == 64
        assert model.learning_rate == pytest.approx(0.002)
        assert X.shape == (160, 13) and X_val.shape == (20, 13)
        events.append("mlp_started")
        result = actual_train(model, X, y, X_val, y_val, **kwargs)
        events.append("mlp_finished")
        return result

    monkeypatch.setattr(main, "split_data_8_1_1", split_spy)
    monkeypatch.setattr(main, "fit_preprocessor", preprocessing_spy)
    monkeypatch.setattr(main, "transform_features", transform_spy)
    monkeypatch.setattr(main.BaselineRegressionSuite, "fit_all", baseline_spy)
    monkeypatch.setattr(main, "train_mlp_model", training_spy)
    result = main.run_pipeline(args)
    output = result["run_dir"]
    assert source.read_bytes() == original_bytes
    assert events.index("split") < events.index("fit_preprocessor")
    assert events.index("baselines_finished") < events.index("mlp_started")
    assert events.index("mlp_finished") < events.index("test_transform")
    summary = result["summary"]
    assert set(summary["Model"]) == {"Linear Regression", "Decision Tree", "Random Forest", "MLP PyTorch"}
    assert np.isfinite(summary[["MSE", "RMSE", "MAE", "R2_Score"]].to_numpy()).all()
    assert len(result["history"]["loss"]) == 1
    for prediction in result["predictions"].values():
        assert prediction.shape == (20,) and np.isfinite(prediction).all()
    for name, raw_X, raw_y in zip(("train", "val", "test"), result["splits"][:3], result["splits"][3:]):
        saved = pd.read_csv(output / "data" / "processed" / f"{name}.csv")
        pd.testing.assert_frame_equal(saved.drop(columns=config.TARGET_COLUMN), raw_X.reset_index(drop=True))
        np.testing.assert_allclose(saved[config.TARGET_COLUMN], raw_y)
    assert sum(pd.read_csv(output / "data" / "processed" / f"{name}.csv")["total_bedrooms"].isna().sum()
               for name in ("train", "val", "test")) == 40
    metrics = output / "results" / "metrics"
    for name in ("model_performance_summary.csv", "validation_summary.csv", "training_history.csv",
                 "sample_predictions.csv"):
        assert (metrics / name).is_file()
    figures = list((output / "results" / "figures").glob("*.png"))
    assert len(figures) == 15
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in figures)

    # Nạp toàn bộ artifact của cùng lần chạy; mẫu mới không được fit lại.
    artifacts = output / "results" / "saved_models"
    restored_processor = load_preprocessor(artifacts / "preprocessor.joblib")
    restored_baselines = load_baseline_models(artifacts / "baseline_models.joblib")
    restored_mlp = load_mlp_model(artifacts / "mlp_house_price_model.pt")
    sample = result["splits"][2].iloc[:3].copy()
    sample["ocean_proximity"] = "UNSEEN_CATEGORY"
    sample["total_bedrooms"] = np.nan
    expected = transform_features(result["preprocessor"], sample)
    actual = transform_features(restored_processor, sample.iloc[:, ::-1])
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(predict_mlp(restored_mlp, actual),
                               predict_mlp(result["model"], expected), rtol=1e-5, atol=1e-5)
    for name, model in restored_baselines.items():
        np.testing.assert_allclose(model.predict(actual), result["baselines"][name].predict(expected))
    with pytest.raises(ValueError):
        transform_features(restored_processor, sample.drop(columns="longitude"))
    with pytest.raises(FileNotFoundError):
        load_preprocessor(artifacts / "absent.joblib")

    # A subsequent --skip-plots run must not inherit the first run's figures.
    monkeypatch.undo()
    args.skip_plots = True
    second = main.run_pipeline(args)
    assert second["run_dir"] != output
    assert len(list((output / "results/figures").glob("*.png"))) == 15
    assert not list((second["run_dir"] / "results/figures").glob("*.png"))
    from src.models.persistence import load_cli_run
    latest = load_cli_run(args.output_dir)
    assert latest["run_dir"] == second["run_dir"]
    assert latest["manifest"]["parameters"]["skip_plots"] is True


@pytest.mark.parametrize("option,value", [
    ("--epochs", "0"), ("--batch-size", "-1"), ("--lr", "nan"), ("--random-state", "-1"),
])
def test_cli_rejects_invalid_overrides(option, value):
    with pytest.raises(SystemExit):
        main.parse_args([option, value])
