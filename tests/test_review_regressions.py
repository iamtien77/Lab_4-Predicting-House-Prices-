"""Regression coverage for notebook features and isolated CLI runs."""
import json
import numpy as np
import pytest

import main
from src.config import config
from src.data.preprocessing import fit_preprocessor, transform_features
from tests.test_data import make_data


def test_notebook_preprocessing_supports_interactions():
    notebook = json.loads((config.BASE_DIR / "notebooks/01_eda_and_preprocessing.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "CREATE_INTERACTIONS = False" in "".join(cell["source"]))
    X, _ = make_data()
    X["ocean_proximity"] = np.resize(["INLAND", "NEAR BAY", "ISLAND", "NEAR OCEAN", "<1H OCEAN"], len(X))
    namespace = {"X_train": X.iloc[:80], "X_val": X.iloc[80:], "np": np,
                 "fit_preprocessor": fit_preprocessor, "transform_features": transform_features}
    exec(source.replace("CREATE_INTERACTIONS = False", "CREATE_INTERACTIONS = True"), namespace)
    assert namespace["train_features"].shape == (80, 16)


def test_failed_cli_run_preserves_last_success_and_detects_tampering(tmp_path, monkeypatch):
    X, y = make_data(100)
    source = tmp_path / "housing.csv"
    X.assign(**{config.TARGET_COLUMN: y}).to_csv(source, index=False)
    args = main.parse_args(["--data-path", str(source), "--output-dir", str(tmp_path),
                            "--epochs", "1", "--skip-plots"])
    completed = main.run_pipeline(args)
    pointer = tmp_path / "results/latest_run.json"
    before = pointer.read_bytes()
    run_dir = completed["run_dir"]
    original_preprocessor = (run_dir / "results/saved_models/preprocessor.joblib").read_bytes()

    def fail_training(*args, **kwargs):
        raise RuntimeError("Simulated interrupted training")

    monkeypatch.setattr(main, "train_mlp_model", fail_training)
    args.random_state = 7
    with pytest.raises(RuntimeError, match="interrupted"):
        main.run_pipeline(args)
    assert pointer.read_bytes() == before
    assert (run_dir / "results/saved_models/preprocessor.joblib").read_bytes() == original_preprocessor

    from src.models.persistence import load_cli_run
    restored = load_cli_run(tmp_path)
    assert restored["run_dir"] == run_dir
    assert not list((run_dir / "results/figures").glob("*.png"))
    artifact = run_dir / "results/saved_models/preprocessor.joblib"
    artifact.write_bytes(original_preprocessor + b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        load_cli_run(tmp_path)


def test_split_configuration_is_used(monkeypatch):
    from src.data.preprocessing import split_data_8_1_1
    X, y = make_data()
    monkeypatch.setattr(config, "USE_STRATIFIED_SPLIT", False)
    # Constant targets cannot be quantile-stratified: config must disable it.
    splits = split_data_8_1_1(X, y * 0)
    assert [len(part) for part in splits[:3]] == [80, 10, 10]


def test_model_defaults_follow_active_configuration(monkeypatch):
    from src.models.baseline_models import BaselineRegressionSuite
    from src.models.mlp_model import build_mlp_model
    monkeypatch.setattr(config, "LR_FIT_INTERCEPT", False)
    monkeypatch.setattr(config, "DT_MAX_DEPTH", 2)
    monkeypatch.setattr(config, "RF_N_ESTIMATORS", 3)
    monkeypatch.setattr(config, "RANDOM_STATE", 17)
    monkeypatch.setattr(config, "MLP_LEARNING_RATE", 0.002)
    suite = BaselineRegressionSuite()
    assert suite.models["Linear Regression"].fit_intercept is False
    assert suite.models["Decision Tree"].max_depth == 2
    assert suite.models["Random Forest"].n_estimators == 3
    assert suite.models["Random Forest"].random_state == 17
    assert build_mlp_model(3).learning_rate == 0.002


def test_source_change_during_training_prevents_publication(tmp_path, monkeypatch):
    X, y = make_data()
    source = tmp_path / "housing.csv"
    X.assign(**{config.TARGET_COLUMN: y}).to_csv(source, index=False)
    args = main.parse_args(["--data-path", str(source), "--output-dir", str(tmp_path),
                            "--epochs", "1", "--skip-plots"])
    real_hash = main.file_sha256
    calls = 0

    def changed_source_hash(path):
        nonlocal calls
        if path == config.BASE_DIR / "main.py":
            calls += 1
            if calls > 1:
                return "changed-during-training"
        return real_hash(path)

    monkeypatch.setattr(main, "file_sha256", changed_source_hash)
    with pytest.raises(ValueError, match="Source code changed"):
        main.run_pipeline(args)
    assert not (tmp_path / "results/latest_run.json").exists()
    assert not list((tmp_path / "runs").glob("*/manifest.json"))
