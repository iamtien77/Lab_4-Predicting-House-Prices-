"""Đọc và kiểm tra tính hợp lệ của tập dữ liệu California Housing cục bộ."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import config


def validate_features(df, *, require_complete_columns=False):
    """
    Xác thực các đặc trưng thô, đảm bảo đủ cột và đúng kiểu dữ liệu.
    Không làm thay đổi giá trị hoặc thứ tự các dòng quan sát.
    """
    expected = config.NUMERIC_FEATURE_NAMES + config.CATEGORICAL_FEATURE_NAMES
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Expected a non-empty feature DataFrame.")
    if not df.columns.is_unique or set(df.columns) != set(expected):
        raise ValueError(f"Invalid schema; missing={sorted(set(expected) - set(df.columns))}, "
                         f"extra={sorted(set(df.columns) - set(expected))}; columns must be unique.")
    for column in config.NUMERIC_FEATURE_NAMES:
        if not pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column]):
            raise ValueError(f"Column {column} must be numeric.")
        if np.isinf(df[column].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"Column {column} contains infinity.")
    for column in config.CATEGORICAL_FEATURE_NAMES:
        if not df[column].dropna().map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Column {column} must contain strings or missing values.")
    if require_complete_columns and df.isna().all().any():
        raise ValueError("Training columns cannot be entirely missing.")


def validate_target(y):
    """
    Kiểm tra biến mục tiêu là kiểu số hữu hạn, không chứa giá trị khuyết và không tự ý điền giá trị.
    """
    if not isinstance(y, pd.Series) or y.empty or not pd.api.types.is_numeric_dtype(y):
        raise ValueError("Target must be a non-empty numeric Series.")
    if pd.api.types.is_bool_dtype(y) or not np.isfinite(y.to_numpy(dtype=float, na_value=np.nan)).all():
        raise ValueError("Target must contain only finite numeric values.")


def load_housing_data(filepath=None):
    """
    Đọc tệp dữ liệu CSV thô và trả về DataFrame cùng tên cột mục tiêu.
    Chỉ đọc tệp cục bộ, không tự động tải dữ liệu từ mạng hay điền khuyết.
    """
    path = Path(filepath) if filepath is not None else config.RAW_CSV_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Housing CSV not found: {path}")
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise ValueError(f"Invalid housing CSV: {path}") from exc
    target = config.TARGET_COLUMN
    if target not in df:
        raise ValueError(f"Missing target column: {target}")
    validate_target(df[target])
    validate_features(df.drop(columns=target))
    print(f"[Nạp dữ liệu] Đã đọc thành công tệp CSV từ '{path}' ({len(df):,} dòng, {df.shape[1]} cột).")
    return df, target


def fetch_dataset_info(df, target_col):
    """
    Thống kê tổng quan dữ liệu mà không làm biến đổi hay tự ý loại bỏ bất kỳ dòng quan sát nào.
    """
    validate_target(df[target_col])
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_summary": df[target_col].describe().to_dict(),
        "target_max_count": int(df[target_col].eq(df[target_col].max()).sum()),
        "categorical_counts": {column: df[column].value_counts(dropna=False).to_dict()
                                for column in config.CATEGORICAL_FEATURE_NAMES},
    }
