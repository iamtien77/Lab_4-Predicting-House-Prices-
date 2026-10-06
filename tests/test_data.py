"""
Kiểm Thử Tự Động Module Dữ Liệu & Phân Chia Tập 8:1:1 (Unit Tests - Data)
Phụ trách: Bạn Huyền và Bạn My (Task 1)
Nhiệm vụ:
- Kiểm tra tính đúng đắn của hàm nạp dữ liệu (Data Loader).
- Kiểm tra tỷ lệ phân chia tập dữ liệu 8:1:1 (Train 80%, Validation 10%, Test 10%).
- Kiểm tra tính chất chuẩn hóa StandardScaler và đảm bảo không có rò rỉ dữ liệu (Data Leakage).
"""

import pytest
import numpy as np
import pandas as pd


def test_data_loader() -> None:
    """
    Kiểm thử hàm nạp dữ liệu: kiểm tra DataFrame không rỗng và chứa đủ các cột cần thiết.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Gọi load_boston_housing_data().
    # BƯỚC 2: assert df is not None
    # BƯỚC 3: assert len(df) > 0, "DataFrame không được rỗng"
    # BƯỚC 4: assert target_col in df.columns, f"Cột mục tiêu {target_col} phải tồn tại"
    # BƯỚC 5: assert df.shape[1] >= 10, "Dataset phải có đủ số cột đặc trưng"
    """
    pass


def test_split_data_8_1_1() -> None:
    """
    Kiểm thử hàm phân chia tập dữ liệu theo tỷ lệ chuẩn 8:1:1 (80% Train - 10% Val - 10% Test).

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Tạo dữ liệu giả lập (Mock Data) với 100 dòng:
    #   X_mock = pd.DataFrame(np.random.randn(100, 5), columns=[f'feat_{i}' for i in range(5)])
    #   y_mock = pd.Series(np.random.uniform(10, 50, size=100), name='price')

    # BƯỚC 2: Gọi split_data_8_1_1(X_mock, y_mock, random_state=42, stratify=False):
    #   X_train, X_val, X_test, y_train, y_val, y_test = split_data_8_1_1(...)

    # BƯỚC 3: Kiểm định số lượng mẫu tương ứng:
    #   assert len(X_train) == 80, f"Tập Train phải chiếm 80 mẫu, thực tế: {len(X_train)}"
    #   assert len(X_val) == 10,   f"Tập Val phải chiếm 10 mẫu, thực tế: {len(X_val)}"
    #   assert len(X_test) == 10,  f"Tập Test phải chiếm 10 mẫu, thực tế: {len(X_test)}"

    # BƯỚC 4: Kiểm định sự tương thích giữa X và y:
    #   assert len(X_train) == len(y_train)
    #   assert len(X_val) == len(y_val)
    #   assert len(X_test) == len(y_test)

    # BƯỚC 5: Đảm bảo tổng số mẫu bằng 100%:
    #   assert len(X_train) + len(X_val) + len(X_test) == 100
    """
    pass


def test_scaling_no_leakage() -> None:
    """
    Kiểm thử quy trình chuẩn hóa StandardScaler: Mean ~ 0 và Std ~ 1 trên Train,
    không làm rò rỉ thông tin phân bố của Val và Test.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Tạo 3 DataFrame giả lập đại diện cho Train (80 dòng), Val (10 dòng), Test (10 dòng).
    # BƯỚC 2: Gọi scale_features(X_train, X_val, X_test).
    # BƯỚC 3: Kiểm tra kỳ vọng toán học trên tập Train sau chuẩn hóa:
    #   assert np.allclose(np.mean(X_train_scaled, axis=0), 0.0, atol=1e-5), "Mean của Train phải xấp xỉ 0"
    #   assert np.allclose(np.std(X_train_scaled, axis=0), 1.0, atol=1e-5), "Std của Train phải xấp xỉ 1"
    # BƯỚC 4: Kiểm tra không chứa NaN:
    #   assert not np.isnan(X_train_scaled).any()
    #   assert not np.isnan(X_val_scaled).any()
    #   assert not np.isnan(X_test_scaled).any()
    """
    pass