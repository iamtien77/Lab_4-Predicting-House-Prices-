"""Persist trusted local estimators and their fitted preprocessing objects."""

from pathlib import Path
import hashlib
import json
import os
from uuid import uuid4

import joblib
import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


def _prepare_path(filepath):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _require_file(filepath):
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {path}")
    return path


def save_mlp_model(model, filepath):
    """Save weights, target statistics and architecture, never a Python model."""
    from src.models.mlp_model import HousePriceMLP
    if Path(filepath).suffix != ".pt":
        raise ValueError("MLP artifacts must use the .pt extension.")
    if not isinstance(model, HousePriceMLP) or not model.is_fitted:
        raise ValueError("The PyTorch model must be trained before saving.")
    torch.save({"format_version": 1, "input_dim": model.input_dim,
                "hidden_units": [32, 16], "learning_rate": model.learning_rate,
                "target_unit": "USD", "best_epoch": model.best_epoch,
                "best_val_loss": model.best_val_loss,
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}},
               _prepare_path(filepath))


def load_mlp_model(filepath):
    """Restore a validated weights-only checkpoint on CPU for inference."""
    from src.models.mlp_model import build_mlp_model
    checkpoint = torch.load(_require_file(filepath), map_location="cpu", weights_only=True)
    required = {"format_version", "input_dim", "hidden_units", "learning_rate",
                "target_unit", "best_epoch", "best_val_loss", "state_dict"}
    if not isinstance(checkpoint, dict) or set(checkpoint) != required:
        raise ValueError("Invalid MLP checkpoint schema.")
    if (checkpoint["format_version"] != 1 or checkpoint["hidden_units"] != [32, 16]
            or checkpoint["target_unit"] != "USD"):
        raise ValueError("Unsupported MLP checkpoint format or architecture.")
    epoch, loss = checkpoint["best_epoch"], checkpoint["best_val_loss"]
    if (isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1
            or not isinstance(loss, (int, float)) or not np.isfinite(loss) or loss < 0):
        raise ValueError("Invalid checkpoint training metadata.")
    model = build_mlp_model(checkpoint["input_dim"], checkpoint["learning_rate"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if (any(not torch.isfinite(value).all() for value in model.state_dict().values())
            or model.target_scale <= 0):
        raise ValueError("Checkpoint weights and target statistics must be finite with positive scale.")
    model.best_epoch, model.best_val_loss = epoch, loss
    model.is_fitted = True
    model.eval()
    return model


def _validate_baselines(models):
    names = {"Linear Regression", "Decision Tree", "Random Forest"}
    if not isinstance(models, dict) or set(models) != names:
        raise ValueError("Expected the complete three-model baseline mapping.")
    for model in models.values():
        check_is_fitted(model)
        if not callable(getattr(model, "predict", None)):
            raise TypeError("Baseline artifacts must be fitted predictors.")


def save_baseline_models(baseline_models, filepath):
    """Save all fitted baselines from the same training run."""
    _validate_baselines(baseline_models)
    joblib.dump(baseline_models, _prepare_path(filepath))


def load_baseline_models(filepath):
    """Load and validate trusted local baseline estimators."""
    models = joblib.load(_require_file(filepath))
    _validate_baselines(models)
    return models


def _validate_preprocessor(preprocessor):
    if not isinstance(preprocessor, (ColumnTransformer, Pipeline)):
        raise TypeError("Expected a ColumnTransformer or preprocessing Pipeline.")
    check_is_fitted(preprocessor)
    if not callable(getattr(preprocessor, "transform", None)):
        raise TypeError("Preprocessor must support transform.")


def save_preprocessor(preprocessor, filepath):
    """Save the complete fitted preprocessing object, including optional features."""
    _validate_preprocessor(preprocessor)
    joblib.dump(preprocessor, _prepare_path(filepath))


def load_preprocessor(filepath):
    """Load a trusted local preprocessor and verify its fitted state."""
    preprocessor = joblib.load(_require_file(filepath))
    _validate_preprocessor(preprocessor)
    return preprocessor


def file_sha256(filepath):
    """Fingerprint local data and artifacts without loading them into memory."""
    digest = hashlib.sha256()
    with Path(filepath).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    """Publish a complete JSON file with a same-directory atomic replacement."""
    path = _prepare_path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _contained_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or Path(relative).is_absolute():
        raise ValueError("Artifact paths must remain inside their run directory.")
    return path


def verify_cli_run(run_dir):
    """Verify a completed run before deserializing any estimator."""
    run_dir = Path(run_dir).resolve()
    manifest = json.loads(_require_file(run_dir / "manifest.json").read_text(encoding="utf-8"))
    required_files = {
        "data/processed/train.csv", "data/processed/val.csv", "data/processed/test.csv",
        "results/saved_models/preprocessor.joblib", "results/saved_models/baseline_models.joblib",
        "results/saved_models/mlp_house_price_model.pt", "results/metrics/training_history.csv",
        "results/metrics/validation_summary.csv", "results/metrics/model_performance_summary.csv",
        "results/metrics/sample_predictions.csv",
    }
    if (manifest.get("format_version") != 1 or manifest.get("status") != "complete"
            or not required_files.issubset(manifest.get("files", {}))):
        raise ValueError("Invalid or incomplete CLI run manifest.")
    for relative, expected in manifest["files"].items():
        path = _contained_path(run_dir, relative)
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Artifact mismatch: {relative}. Use a complete, unchanged run.")
    return manifest


def publish_cli_run(output_root, run_dir, metadata):
    """Publish only a complete verified run; older runs are never overwritten."""
    output_root, run_dir = Path(output_root).resolve(), Path(run_dir).resolve()
    if not run_dir.is_relative_to(output_root / "runs"):
        raise ValueError("CLI runs must be stored below output_root/runs.")
    files = {}
    for folder in (run_dir / "data/processed", run_dir / "results"):
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                files[path.relative_to(run_dir).as_posix()] = file_sha256(path)
    manifest = {**metadata, "format_version": 1, "status": "complete",
                "run_id": run_dir.name, "files": files}
    _atomic_json(run_dir / "manifest.json", manifest)
    verify_cli_run(run_dir)
    _atomic_json(output_root / "results/latest_run.json", {
        "run_dir": run_dir.relative_to(output_root).as_posix(),
        "manifest_sha256": file_sha256(run_dir / "manifest.json"),
    })
    return manifest


def load_cli_run(output_root, *, run_dir=None):
    """Load a trusted CLI run as a verified preprocessor/model set.

    Without run_dir, resolve the latest successfully published run. Explicit
    run_dir is relative to output_root and enables reproducible older runs.
    Low-level loaders remain available for notebooks with their own manifests.
    """
    output_root = Path(output_root).resolve()
    if run_dir is None:
        pointer = json.loads(_require_file(output_root / "results/latest_run.json").read_text(encoding="utf-8"))
        root = _contained_path(output_root, pointer["run_dir"])
        if file_sha256(_require_file(root / "manifest.json")) != pointer["manifest_sha256"]:
            raise ValueError("Run manifest mismatch.")
    else:
        root = _contained_path(output_root, run_dir)
    if not root.is_relative_to(output_root / "runs"):
        raise ValueError("Expected an isolated CLI run below runs/.")
    manifest = verify_cli_run(root)
    models = root / "results/saved_models"
    preprocessor = load_preprocessor(models / "preprocessor.joblib")
    model = load_mlp_model(models / "mlp_house_price_model.pt")
    baselines = load_baseline_models(models / "baseline_models.joblib")
    dimension = len(preprocessor.get_feature_names_out())
    if model.input_dim != dimension or any(item.n_features_in_ != dimension for item in baselines.values()):
        raise ValueError("Model/preprocessor feature dimension mismatch.")
    return {"run_dir": root, "manifest": manifest, "preprocessor": preprocessor,
            "model": model, "baselines": baselines}
