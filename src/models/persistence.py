"""Persist trusted local estimators and their fitted preprocessing objects."""

from pathlib import Path

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
