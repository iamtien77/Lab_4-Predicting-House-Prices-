"""
Module Nạp Dữ Liệu (Data Loader Module)
Phụ trách: Bạn Huyền và Bạn My (Task 1)
Nhiệm vụ: Nạp dữ liệu Boston Housing an toàn đa nguồn (Local CSV -> OpenML -> CMU StatLib),
chuẩn hóa định dạng cột và trích xuất thông tin tổng quan của dữ liệu.
"""

from typing import Tuple, Dict, Any, Optional
import pandas as pd


def load_boston_housing_data(
    filepath: Optional[str] = None,
    use_fallback: bool = True
) -> Tuple[pd.DataFrame, str]:
    """
    Nạp dữ liệu giá nhà an toàn với cơ chế fallback 3 cấp độ.

    Parameters:
    -----------
    filepath : Optional[str]
        Đường dẫn file CSV cục bộ (ưu tiên đọc file này trước).
    use_fallback : bool
        Nếu True, khi không đọc được file cục bộ sẽ tự động thử tải từ OpenML hoặc URL dự phòng.

    Returns:
    --------
    Tuple[pd.DataFrame, str]
        - DataFrame chứa dữ liệu đầy đủ (bao gồm các đặc trưng và biến mục tiêu).
        - Tên cột mục tiêu được xác định (ví dụ: 'MEDV' hoặc 'median_house_value').

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Kiểm tra nguồn dữ liệu cục bộ (Local File)
    # - Nếu filepath được truyền vào hoặc config.RAW_CSV_PATH tồn tại trên đĩa:
    #     + Dùng pd.read_csv() để đọc file.
    #     + Kiểm tra số lượng dòng và cột (ví dụ: Boston Housing thường có 506 dòng x 14 cột,
    #       California Housing có 20,640 dòng x 10 cột).
    #     + Ghi nhận tên cột mục tiêu: nếu có 'MEDV' thì chọn 'MEDV', nếu có 'median_house_value' thì chọn nó.
    #     + Trả về (df, target_col).

    # BƯỚC 2: Cơ chế dự phòng cấp 1 - Tải qua sklearn.datasets.fetch_openml
    # - Nếu không có file local và use_fallback == True:
    #     + Dùng fetch_openml(name='boston', version=1, as_frame=True) để lấy dataset.
    #     + Ghép dữ liệu data và target thành một DataFrame duy nhất:
    #       df = data.frame
    #       target_col = 'MEDV'
    #     + Xử lý trường hợp cảnh báo đạo đức / phân biệt chủng tộc của dataset Boston trên phiên bản scikit-learn mới.

    # BƯỚC 3: Cơ chế dự phòng cấp 2 - Tải trực tiếp từ StatLib CMU Archive URL
    # - Nếu OpenML thất bại do mạng:
    #     + Đọc từ URL: "http://lib.stat.cmu.edu/datasets/boston" bằng pd.read_csv với separator khoảng trắng.
    #     + Đọc định dạng dữ liệu xen kẽ 2 dòng mỗi quan sát của StatLib.
    #     + Gán lại danh sách 14 tên cột chuẩn: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV.

    # BƯỚC 4: Kiểm tra tính toàn vẹn (Integrity Assertion)
    # - Đảm bảo DataFrame không rỗng (len(df) > 0).
    # - Đảm bảo cột target_col tồn tại trong danh sách cột của df.
    # - Trả về (df, target_col).
    """
    pass


def fetch_dataset_info(
    df: pd.DataFrame, 
    target_col: str
) -> Dict[str, Any]:
    """
    Trích xuất thông tin tổng quan, cấu trúc schema và thống kê cơ bản của dataset (tương tự Phase 2).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu đã nạp.
    target_col : str
        Tên cột mục tiêu cần phân tích.

    Returns:
    --------
    Dict[str, Any]
        Từ điển chứa thông tin: số dòng, số cột, kiểu dữ liệu, số ô NaN, min, max, mean, std của target.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Đếm số lượng quan sát và số lượng cột:
    #   - n_rows = len(df)
    #   - n_cols = df.shape[1]

    # BƯỚC 2: Phân loại kiểu cột:
    #   - numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    #   - categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # BƯỚC 3: Thống kê số lượng ô khuyết thiếu (Missing values):
    #   - total_missing = int(df.isna().sum().sum())
    #   - missing_per_column = df.isna().sum().to_dict()

    # BƯỚC 4: Thống kê mô tả biến mục tiêu:
    #   - target_stats = df[target_col].describe().to_dict()

    # BƯỚC 5: Đóng gói toàn bộ thông tin vào Dict và trả về để logger hoặc màn hình console hiển thị.
    """
    pass