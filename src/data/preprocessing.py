"""
Module Tiền Xử Lý Dữ Liệu & Phân Chia Tập Dữ Liệu 8:1:1
Phụ trách: Bạn Huyền và Bạn My (Task 1)
Nhiệm vụ: 
- Phân chia tập dữ liệu chính xác theo tỷ lệ 8:1:1 (80% Train, 10% Validation, 10% Test).
- Chuẩn hóa đặc trưng bằng StandardScaler (fit độc quyền trên Train để chống rò rỉ dữ liệu).
- Xử lý giá trị khuyết thiếu và lưu trữ artifact đã tiền xử lý.
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def handle_missing_values(
    df: pd.DataFrame, 
    strategy: str = "median"
) -> pd.DataFrame:
    """
    Xử lý giá trị khuyết thiếu (NaN) trên toàn bộ DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame đầu vào có thể chứa giá trị NaN (ví dụ: total_bedrooms trong California housing).
    strategy : str
        Chiến lược điền khuyết thiếu ('median', 'mean', hoặc 'drop'). Mặc định là 'median' vì bền vững với ngoại lai.

    Returns:
    --------
    pd.DataFrame
        DataFrame sạch đã được điền đủ tất cả các ô thiếu.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Tạo bản sao sâu (deep copy) của DataFrame: clean_df = df.copy().
    # BƯỚC 2: Kiểm tra tổng số ô khuyết: missing_count = clean_df.isna().sum().
    # BƯỚC 3: Đối với các cột số (numeric columns):
    #   - Nếu strategy == 'median': tính giá trị trung vị từng cột (clean_df[col].median()) và fillna.
    #   - Nếu strategy == 'mean': tính giá trị trung bình từng cột (clean_df[col].mean()) và fillna.
    # BƯỚC 4: Đối với các cột phân loại (categorical columns):
    #   - Điền giá trị xuất hiện nhiều nhất (mode) hoặc gán nhãn 'Missing'.
    # BƯỚC 5: Kiểm tra xác nhận lại: assert clean_df.isna().sum().sum() == 0.
    # BƯỚC 6: Trả về clean_df.
    """
    pass


def split_data_8_1_1(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    stratify: bool = True,
    n_bins_train: int = 10,
    n_bins_temp: int = 5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Phân chia dữ liệu theo tỷ lệ chuẩn 8:1:1 (80% Train - 10% Validation - 10% Test)
    tuân thủ nghiêm ngặt code mẫu Phase 03 (phase_03_train_valid_test_split.py).

    Parameters:
    -----------
    X : pd.DataFrame
        Tập ma trận đặc trưng đầu vào (Features).
    y : pd.Series
        Vector biến mục tiêu giá nhà (Target).
    random_state : int
        Hạt giống ngẫu nhiên cố định để đảm bảo kết quả phân chia có thể tái lập (mặc định 42).
    stratify : bool
        Nếu True, thực hiện phân tầng dựa trên phân vị giá (qcut) để giữ đồng nhất phân bố giá giữa các tập.
    n_bins_train : int
        Số lượng khoảng phân vị cho lần chia đầu tiên (Train vs Temp, mặc định 10).
    n_bins_temp : int
        Số lượng khoảng phân vị cho lần chia thứ hai (Val vs Test, mặc định 5).

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]
        (X_train, X_val, X_test, y_train, y_val, y_test)
        - X_train, y_train: Chiếm 80% tổng số mẫu (Train set).
        - X_val, y_val:     Chiếm 10% tổng số mẫu (Validation set để kiểm tra loss qua epoch).
        - X_test, y_test:   Chiếm 10% tổng số mẫu (Test set độc lập để đánh giá cuối cùng).

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Kiểm tra tính hợp lệ của dữ liệu đầu vào:
    #   - assert len(X) == len(y), đảm bảo số lượng dòng của features và target khớp nhau.
    #   - total_samples = len(X).

    # BƯỚC 2: Chuẩn bị nhãn phân tầng nếu stratify == True:
    #   - Sử dụng pd.qcut(y, q=n_bins_train, duplicates='drop', labels=False) để gán nhãn phân vị cho từng mẫu.
    #   - Nếu stratify == False: gán stratify_labels_1 = None.

    # BƯỚC 3: LẦN CHIA THỨ NHẤT - Tách Train (80%) và Temp (20%):
    #   - Gọi sklearn.model_selection.train_test_split:
    #       X_train, X_temp, y_train, y_temp = train_test_split(
    #           X, y,
    #           test_size=0.20,             # 20% cho phần tạm (sau đó sẽ chia đôi thành 10% Val + 10% Test)
    #           random_state=random_state,
    #           stratify=stratify_labels_1
    #       )
    #   - Lúc này: X_train chiếm 80% (khoảng 404/506 mẫu với Boston Housing).
    #              X_temp chiếm 20% (khoảng 102/506 mẫu).

    # BƯỚC 4: Chuẩn bị nhãn phân tầng cho tập Temp:
    #   - Nếu stratify == True:
    #       temp_bin = pd.qcut(y_temp, q=n_bins_temp, duplicates='drop', labels=False)
    #   - Nếu stratify == False: temp_bin = None.

    # BƯỚC 5: LẦN CHIA THỨ HAI - Chia đôi tập Temp (20%) thành Validation (10%) và Test (10%):
    #   - test_size = 0.50 (tức là lấy 50% của 20% tổng thể = 10% tổng thể):
    #       X_val, X_test, y_val, y_test = train_test_split(
    #           X_temp, y_temp,
    #           test_size=0.50,             # 50% của 20% = 10%
    #           random_state=random_state,
    #           stratify=temp_bin
    #       )
    #   - Lúc này: X_val chiếm 10% (khoảng 51 mẫu).
    #              X_test chiếm 10% (khoảng 51 mẫu).

    # BƯỚC 6: Kiểm tra và in bảng tóm tắt tỷ lệ (Verification & Summary):
    #   - Tính tỷ lệ phần trăm thực tế:
    #       train_pct = len(X_train) / total_samples * 100
    #       val_pct   = len(X_val) / total_samples * 100
    #       test_pct  = len(X_test) / total_samples * 100
    #   - Đảm bảo: len(X_train) + len(X_val) + len(X_test) == total_samples.
    #   - Thống kê mean, median của y_train, y_val, y_test để xác nhận phân bố giá không bị lệch.

    # BƯỚC 7: Trả về bộ 6 đối tượng: (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    pass


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Chuẩn hóa đặc trưng bằng StandardScaler theo nguyên tắc phòng chống rò rỉ dữ liệu (Data Leakage).

    Parameters:
    -----------
    X_train : pd.DataFrame
        Ma trận đặc trưng tập huấn luyện (80%).
    X_val : pd.DataFrame
        Ma trận đặc trưng tập kiểm định (10%).
    X_test : pd.DataFrame
        Ma trận đặc trưng tập kiểm thử (10%).

    Returns:
    --------
    Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]
        (X_train_scaled, X_val_scaled, X_test_scaled, fitted_scaler)

    Logic chi tiết cần cài đặt:
    ---------------------------
    # NGUYÊN TẮC CỐT LÕI VỀ DATA LEAKAGE:
    # - Bộ scaler CHỈ ĐƯỢC PHÉP học thông số thống kê (Mean μ và Độ lệch chuẩn σ) từ tập X_train.
    # - Tuyệt đối KHÔNG gọi fit hoặc fit_transform trên tập X_val hay X_test.

    # BƯỚC 1: Khởi tạo scaler = StandardScaler().
    # BƯỚC 2: Học μ, σ và biến đổi tập Train:
    #   - X_train_scaled = scaler.fit_transform(X_train)
    #   - Lúc này mean của X_train_scaled xấp xỉ 0, std xấp xỉ 1.

    # BƯỚC 3: Dùng μ, σ đã học từ Train để biến đổi tập Validation và Test:
    #   - X_val_scaled  = scaler.transform(X_val)
    #   - X_test_scaled = scaler.transform(X_test)

    # BƯỚC 4: Kiểm tra tính toàn vẹn của mảng sau chuẩn hóa:
    #   - assert not np.isnan(X_train_scaled).any()
    #   - assert not np.isnan(X_val_scaled).any()
    #   - assert not np.isnan(X_test_scaled).any()
    #   - assert X_train_scaled.shape[1] == X_val_scaled.shape[1] == X_test_scaled.shape[1]

    # BƯỚC 5: Trả về (X_train_scaled, X_val_scaled, X_test_scaled, scaler).
    """
    pass


def save_processed_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    target_col: str,
    output_dir: Optional[str] = None
) -> None:
    """
    Lưu trữ 3 tập dữ liệu đã chia (8:1:1) thành các file CSV riêng biệt trong data/processed/.

    Parameters:
    -----------
    X_train, X_val, X_test : pd.DataFrame
        Đặc trưng của từng tập.
    y_train, y_val, y_test : pd.Series
        Biến mục tiêu tương ứng.
    target_col : str
        Tên cột mục tiêu.
    output_dir : Optional[str]
        Thư mục lưu trữ (mặc định lấy từ config.PROCESSED_DATA_DIR).

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Ghép lại features và target cho từng tập:
    #   - train_df = X_train.copy(); train_df[target_col] = y_train
    #   - val_df   = X_val.copy();   val_df[target_col]   = y_val
    #   - test_df  = X_test.copy();  test_df[target_col]  = y_test

    # BƯỚC 2: Xuất ra định dạng CSV (index=False):
    #   - train_df.to_csv(output_dir / "train.csv", index=False)
    #   - val_df.to_csv(output_dir / "val.csv", index=False)
    #   - test_df.to_csv(output_dir / "test.csv", index=False)

    # BƯỚC 3: Ghi log xác nhận số lượng dòng và dung lượng các file đã lưu thành công.
    """
    pass