"""Lưu trữ và nạp các mô hình ước lượng cục bộ an toàn cùng đối tượng tiền xử lý đã khớp."""

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
    """Lưu trọng số mô hình, thống kê chuẩn hóa target và thông tin kiến trúc (không lưu đối tượng Python model thô)."""
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
    """Khôi phục checkpoint an toàn chỉ gồm trọng số lên CPU phục vụ suy luận."""
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
    """Lưu toàn bộ các mô hình baseline đã fit trong cùng phiên huấn luyện."""
    _validate_baselines(baseline_models)
    joblib.dump(baseline_models, _prepare_path(filepath))


def load_baseline_models(filepath):
    """Nạp và xác thực các mô hình baseline cục bộ tin cậy."""
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
    """Lưu toàn bộ đối tượng tiền xử lý đã khớp, bao gồm cả kỹ thuật tạo đặc trưng tùy chọn."""
    _validate_preprocessor(preprocessor)
    joblib.dump(preprocessor, _prepare_path(filepath))


def load_preprocessor(filepath):
    """Nạp bộ tiền xử lý cục bộ và xác thực trạng thái đã fit."""
    preprocessor = joblib.load(_require_file(filepath))
    _validate_preprocessor(preprocessor)
    return preprocessor


def file_sha256(filepath):
    """Tính mã băm SHA256 cho dữ liệu và artifact cục bộ theo khối mà không nạp toàn bộ vào bộ nhớ."""
    digest = hashlib.sha256()
    with Path(filepath).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path, value):
    """Xuất bản tệp JSON hoàn chỉnh bằng cơ chế thay thế nguyên tử trong cùng thư mục."""
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
    """Xác thực tính toàn vẹn của một phiên chạy CLI hoàn tất trước khi giải tuần tự hóa mô hình."""
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
    """Chỉ xuất bản phiên chạy đã xác thực đầy đủ; không bao giờ ghi đè lên các lần chạy trước đó."""
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
    """Nạp một phiên chạy CLI tin cậy dưới dạng tập hợp bộ tiền xử lý và mô hình đã được xác thực.

    Khi không truyền run_dir, giải quyết theo lần chạy hoàn tất mới nhất. run_dir
    tường minh tương đối với output_root và cho phép tái lập các lần chạy cũ.
    Các loader cấp thấp vẫn sẵn sàng phục vụ các notebook có manifest riêng.
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
