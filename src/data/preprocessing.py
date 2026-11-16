"""Mô-đun chia tách dữ liệu thô độc lập và khớp các phép biến đổi tiền xử lý chỉ trên tập Train."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted
from src.config import config
from src.data.loader import validate_features, validate_target
from src.features.feature_engineering import HousingFeatureEngineer


def _validate_pair(X, y):
    """Kiểm tra tính tương thích giữa tập đặc trưng X và nhãn mục tiêu y (cùng số dòng, chỉ mục trùng khớp)."""
    validate_features(X)
    validate_target(y)
    if not X.index.is_unique or not y.index.is_unique or not X.index.equals(y.index):
        raise ValueError("X/y must have identical unique indexes in the same order.")


def _stratification_labels(y, bins, test_size):
    """Tạo nhãn phân tầng dựa trên phân vị của biến mục tiêu y để đảm bảo phân phối đồng đều giữa các tập."""
    message = "Stratification is not feasible; reduce bins or select stratify=False."
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 2:
        raise ValueError(message)
    labels = pd.qcut(y, bins, duplicates="drop", labels=False)
    counts = labels.value_counts()
    test_count = math.ceil(len(y) * test_size)
    if labels.isna().any() or len(counts) < 2 or counts.min() < 2:
        raise ValueError(message)
    if min(test_count, len(y) - test_count) < len(counts):
        raise ValueError(message)
    return labels


def split_data_8_1_1(X, y, random_state=None, stratify=None, n_bins_train=None, n_bins_temp=None):
    """
    Phân chia dữ liệu thô thành 3 tập độc lập: Train (80%), Validation (10%) và Test (10%).
    Bảo toàn định dạng thô ban đầu, không áp dụng bất kỳ bước tiền xử lý hay chuẩn hóa nào trước khi chia.
    """
    _validate_pair(X, y)
    random_state = config.RANDOM_STATE if random_state is None else random_state
    stratify = config.USE_STRATIFIED_SPLIT if stratify is None else stratify
    n_bins_train = config.N_STRATIFY_BINS_TRAIN if n_bins_train is None else n_bins_train
    n_bins_temp = config.N_STRATIFY_BINS_TEMP if n_bins_temp is None else n_bins_temp
    labels = _stratification_labels(y, n_bins_train, 0.2) if stratify else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=labels
    )
    labels_temp = _stratification_labels(y_temp, n_bins_temp, 0.5) if stratify else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=labels_temp
    )
    print(f"[Chia tập dữ liệu] Tỷ lệ 80/10/10 thành công: Train = {len(X_train):,} dòng, Validation = {len(X_val):,} dòng, Test = {len(X_test):,} dòng.")
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_preprocessor(X_train, *, apply_log=False, create_interactions=False):
    """
    Khớp (fit) bộ tiền xử lý gồm điền khuyết, tạo đặc trưng tùy chọn, mã hóa one-hot và chuẩn hóa z-score.
    QUY TẮC BẮT BUỘC: Bộ biến đổi CHỈ được học thống kê (fit) trên tập Train để ngăn ngừa Data Leakage.
    """
    validate_features(X_train, require_complete_columns=True)
    # Pipeline xử lý biến số: Điền trung vị -> Tạo đặc trưng tùy chọn -> Chuẩn hóa z-score
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("features", HousingFeatureEngineer(apply_log=apply_log, create_interactions=create_interactions)),
        ("scaler", StandardScaler()),
    ])
    # Pipeline xử lý biến phân loại: Điền giá trị thường gặp nhất -> Mã hóa One-Hot
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=None)),
    ])
    # Kết hợp xử lý theo từng nhóm cột thông qua ColumnTransformer
    preprocessor = ColumnTransformer(
        [("numeric", numeric, config.NUMERIC_FEATURE_NAMES),
         ("categorical", categorical, config.CATEGORICAL_FEATURE_NAMES)],
        remainder="drop", sparse_threshold=0,
    )
    preprocessor.fit(X_train)
    print(f"[Tiền xử lý] Đã fit thành công ColumnTransformer trên tập Train (số đặc trưng sau fit: {len(preprocessor.get_feature_names_out())}).")
    return preprocessor


def transform_features(preprocessor, X):
    """
    Chuyển đổi các đặc trưng thô bằng bộ tiền xử lý đã fit sẵn từ tập Train.
    Trả về ma trận số thực dense ndarray phục vụ việc đưa vào mô hình học máy.
    """
    check_is_fitted(preprocessor)
    validate_features(X)
    result = np.asarray(preprocessor.transform(X), dtype=float)
    expected_shape = (len(X), len(preprocessor.get_feature_names_out()))
    if result.shape != expected_shape or not np.isfinite(result).all():
        raise ValueError("Transformed features have invalid shape or non-finite values.")
    return result


def save_processed_splits(X_train, X_val, X_test, y_train, y_val, y_test,
                          target_col, output_dir=None):
    """
    Lưu các tập dữ liệu thô đã chia ra các tệp CSV riêng biệt (train.csv, val.csv, test.csv).
    Bảo toàn giá trị khuyết (NaN) và giữ nguyên đơn vị USD của biến mục tiêu, không scale trước khi lưu.
    """
    if target_col != config.TARGET_COLUMN:
        raise ValueError(f"Expected target name {config.TARGET_COLUMN}.")
    pairs = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    for X, y in pairs:
        _validate_pair(X, y)
    combined_index = X_train.index.append(X_val.index).append(X_test.index)
    if not combined_index.is_unique:
        raise ValueError("Raw split indexes must be disjoint before saving.")
    directory = Path(output_dir) if output_dir is not None else config.PROCESSED_DATA_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for name, (X, y) in zip(("train", "val", "test"), pairs):
        frame = X.copy()
        frame[target_col] = y
        frame.to_csv(directory / f"{name}.csv", index=False)
    print(f"[Lưu dữ liệu] Đã lưu thành công các tập train.csv, val.csv, test.csv tại thư mục: '{directory}'.")
