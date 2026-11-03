"""Bộ khung nạp California Housing local. Phụ trách: Huyền & My (Task 1).
Chưa triển khai. Không tải mạng hoặc chuyển dataset khi đọc file lỗi.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd


def load_housing_data(filepath: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, str]:
    """Đặc tả: đọc filepath hoặc config.RAW_CSV_PATH, trả (df, target_col).

    - Schema: config.NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES +
      TARGET_COLUMN ('median_house_value', USD).
    - Thiếu file: FileNotFoundError; rỗng/sai schema/sai kiểu số/Inf: ValueError.
      Cột thiếu hoặc dư đều báo rõ; không tự đổi dataset hay tên target.
    - Chấp nhận missing feature để imputer học sau split; target phải hữu hạn,
      không thiếu. Không điền target bằng median, không tự bỏ dòng.
    - Giữ thứ tự dòng và index duy nhất phục vụ kiểm tra các tập không giao nhau.
    - Không impute, encode, scale, tạo feature hoặc sửa CSV gốc ở loader.
    - 20640 dòng là thông tin mô tả, không là điều kiện bắt buộc; test có thể
      dùng fixture nhỏ cùng schema.
    """
    pass


def fetch_dataset_info(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """Đặc tả: trả shape, dtypes, missing từng cột, dòng trùng, target.describe().
    Chỉ mô tả, không thay đổi df. EDA chọn feature chỉ trên train sau split.
    Đếm target ở giá trị lớn nhất và tần suất categorical để ghi hạn chế;
    không tự loại nhóm hiếm hoặc các mẫu giá cao.
    """
    pass
