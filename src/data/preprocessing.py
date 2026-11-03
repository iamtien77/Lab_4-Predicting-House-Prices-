"""Bộ khung preprocessing dùng chung — Huyền & My (Task 1). Chưa triển khai.
Chia tập trước; imputation, encoding và scaling chỉ học từ train.
"""
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer


def split_data_8_1_1(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42,
    stratify: bool = True, n_bins_train: int = 10, n_bins_temp: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Trả (X_train, X_val, X_test, y_train, y_val, y_test) CHƯA biến đổi.

    1. Kiểm số dòng và index X/y trùng khớp, index duy nhất, target hữu hạn.
    2. Chia train/temp với test_size=0.20; chia val/test với test_size=0.50,
       cùng random_state. CSV hiện tại: 16512 / 2064 / 2064 dòng.
    3. stratify=True: qcut(y, n_bins_train, duplicates='drop', labels=False)
       cho lần 1; qcut(y_temp, n_bins_temp, ...) cho lần 2. Kiểm bin hợp lệ,
       đủ mẫu mỗi bin và đủ mẫu mỗi phía cho số lớp trước khi split.
       Không khả thi (kể cả target hằng): ValueError hướng dẫn giảm bins hoặc
       chọn stratify=False; không âm thầm đổi chiến lược.
    4. Giữ index gốc; kiểm giao rỗng giữa ba tập, bảo toàn đủ dòng và X/y.
       Số dòng lẻ làm tròn theo train_test_split.

    8:1:1 là lựa chọn dự án; PDF minh họa train/test 80/20. qcut không tự ngăn
    leakage: imputer/encoder/scaler vẫn phải fit chỉ trên train.
    """
    pass


def fit_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """Fit và trả ColumnTransformer trên X_train THÔ; chưa có implementation.

    - Chỉ nhận 9 feature theo config, không có target hoặc cột đã encode.
      Kiểm schema, số hữu hạn (cho phép NaN); reject cột train thiếu hoàn toàn.
    - Numeric: SimpleImputer(strategy='median') -> StandardScaler.
    - Categorical: SimpleImputer(strategy='most_frequent') ->
      OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop=None).
    - Chọn cột bằng tên, thứ tự numeric rồi categorical theo config;
      ColumnTransformer(remainder='drop', sparse_threshold=0).
    - Fit duy nhất trên train. feature_names_in_, get_feature_names_out() và
      fitted transformers lưu schema, thứ tự cột và thống kê.
    - Không biến đổi target. Feature engineering tùy chọn chưa nối vào luồng
      cơ bản; nếu bật sau này phải nằm trong artifact preprocessing chung.

    Không get_dummies riêng trên val/test, không học median/mode khi transform.
    Output dense cho Keras; số chiều phụ thuộc categories có trong train.
    """
    pass


def transform_features(preprocessor: ColumnTransformer, X: pd.DataFrame) -> np.ndarray:
    """Kiểm schema rồi transform bằng preprocessor đã fit, trả dense (N, D).

    Nhận feature thô, không có target. Cho phép đảo thứ tự cột vì chọn bằng tên;
    thiếu/dư cột báo ValueError. Category mới thành toàn 0 trong nhánh one-hot;
    không refit/đổi số chiều. NotFittedError nếu chưa fit.
    Kiểm output hữu hạn, giữ số dòng và D bằng số feature_names_out.
    Numeric được scale; one-hot giữ 0/1, không yêu cầu mọi cột có std=1.
    """
    pass


def save_processed_splits(
    X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
    target_col: str, output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Artifact: train.csv, val.csv, test.csv là RAW SPLITS có target.

    Ghép X/y khi index đã kiểm trùng khớp; bảo toàn thứ tự dòng và NaN.
    Ghi index=False vào output_dir hoặc config.PROCESSED_DATA_DIR.
    Không ghi feature đã impute/encode/scale, không sinh .npy.
    Bên đọc CSV tách config.TARGET_COLUMN rồi gọi transform_features với
    preprocessor.joblib của chính lần chia này; không fit lại.
    Index CSV mới không chứng minh split disjoint; phải kiểm trước khi lưu.
    """
    pass
