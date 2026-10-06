"""
Module Kỹ Thuật Đặc Trưng (Feature Engineering Module)
Phụ trách: Bạn Huyền và Bạn My (Task 1)
Nhiệm vụ: 
- Tạo đặc trưng mở rộng, biến đổi hàm Log1p cho biến phân phối lệch (Skewed Distribution).
- Tạo biến tương tác có ý nghĩa kinh tế - xã hội (Domain-specific Interaction Features).
- Tùy chọn mở rộng đa thức (Polynomial Features) nhằm tăng khả năng biểu diễn phi tuyến cho mô hình.
"""

from typing import List, Optional
import pandas as pd
import numpy as np


class BostonFeatureEngineer:
    """
    Bộ biến đổi đặc trưng cho bài toán dự đoán giá nhà theo kiến trúc chuẩn Scikit-learn (fit / transform).
    Tham khảo từ Phase 05 trong code mẫu dự án.
    """

    def __init__(
        self,
        apply_log: bool = True,
        create_interactions: bool = True,
        skewed_cols: Optional[List[str]] = None
    ):
        """
        Khởi tạo các tham số cấu hình bộ kỹ thuật đặc trưng.

        Parameters:
        -----------
        apply_log : bool
            Có thực hiện biến đổi log1p cho các cột có độ lệch cao hay không.
        create_interactions : bool
            Có tạo các biến tương tác giữa các đặc trưng hay không.
        skewed_cols : Optional[List[str]]
            Danh sách các cột cần áp dụng biến đổi log1p (mặc định: ['CRIM', 'ZN']).

        Logic chi tiết cần cài đặt:
        ---------------------------
        # Lưu các cờ cấu hình: self.apply_log, self.create_interactions.
        # Khởi tạo danh sách skewed_cols mặc định nếu chưa truyền vào:
        #   - Đối với Boston: ['CRIM', 'ZN'] (phân bố lệch phải nặng).
        #   - Đối với California: ['total_rooms', 'total_bedrooms', 'population', 'households'].
        # Khởi tạo thuộc tính lưu metadata các cột đã sinh: self.generated_feature_names = [].
        """
        pass

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BostonFeatureEngineer":
        """
        Học các thông số thống kê cần thiết từ tập huấn luyện (nếu có).

        Parameters:
        -----------
        X : pd.DataFrame
            Ma trận đặc trưng tập huấn luyện.
        y : Optional[pd.Series]
            Biến mục tiêu (nếu cần tính target encoding).

        Returns:
        --------
        self : BostonFeatureEngineer
            Trả về chính đối tượng đã fit.

        Logic chi tiết cần cài đặt:
        ---------------------------
        # BƯỚC 1: Lưu danh sách cột gốc: self.original_cols_ = list(X.columns).
        # BƯỚC 2: Kiểm tra các cột trong skewed_cols có thực sự tồn tại trong X hay không.
        # BƯỚC 3: Trả về self để hỗ trợ cơ chế gọi hàm dạng chuỗi (method chaining).
        """
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Biến đổi ma trận đặc trưng X bằng cách áp dụng log transform và sinh đặc trưng mới.

        Parameters:
        -----------
        X : pd.DataFrame
            Ma trận đặc trưng đầu vào (Train, Val hoặc Test).

        Returns:
        --------
        pd.DataFrame
            DataFrame đã được bổ sung các đặc trưng mới.

        Logic chi tiết cần cài đặt:
        ---------------------------
        # BƯỚC 1: Tạo bản sao sâu: df_trans = X.copy().

        # BƯỚC 2: Biến đổi Log1p cho các cột phân phối lệch (Skewed Distribution):
        #   - Nếu self.apply_log == True:
        #       + Duyệt qua từng cột trong self.skewed_cols.
        #       + Áp dụng np.log1p(df_trans[col]) để đưa phân bố về gần dạng chuẩn (Gaussian-like).
        #       + Đặt tên cột mới: f"log_{col}" hoặc thay thế trực tiếp.

        # BƯỚC 3: Tạo các đặc trưng tương tác kinh tế (Domain Interaction Ratios):
        #   - Nếu self.create_interactions == True:
        #       + Với Boston Housing:
        #           - Tỷ lệ phòng trên tầng lớp lao động: 'RM_to_LSTAT' = df_trans['RM'] / (df_trans['LSTAT'] + 1e-5).
        #           - Tỷ lệ thuế trên chỉ số tiếp cận cao tốc: 'TAX_to_RAD' = df_trans['TAX'] / (df_trans['RAD'] + 1e-5).
        #           - Tỷ lệ học sinh/giáo viên nhân với thuế: 'PTRATIO_x_TAX' = df_trans['PTRATIO'] * df_trans['TAX'].
        #           - Mức độ tội phạm trên khoảng cách trung tâm việc làm: 'CRIM_to_DIS' = df_trans['CRIM'] / (df_trans['DIS'] + 1e-5).
        #       + Với California Housing (nếu áp dụng):
        #           - 'rooms_per_household' = df_trans['total_rooms'] / (df_trans['households'] + 1e-5).
        #           - 'bedrooms_ratio' = df_trans['total_bedrooms'] / (df_trans['total_rooms'] + 1e-5).
        #           - 'people_per_household' = df_trans['population'] / (df_trans['households'] + 1e-5).

        # BƯỚC 4: Kiểm tra và xử lý các giá trị vô cực hoặc NaN phát sinh:
        #   - Thay thế np.inf, -np.inf bằng np.nan.
        #   - Điền np.nan bằng giá trị 0 hoặc median của cột.
        #   - assert df_trans.isna().sum().sum() == 0.

        # BƯỚC 5: Cập nhật self.generated_feature_names_ và trả về df_trans.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Kết hợp fit và transform trên cùng tập dữ liệu (thường áp dụng cho X_train).

        Parameters:
        -----------
        X : pd.DataFrame
            Ma trận đặc trưng tập huấn luyện.
        y : Optional[pd.Series]
            Biến mục tiêu tương ứng.

        Returns:
        --------
        pd.DataFrame
            Ma trận đặc trưng đã biến đổi.

        Logic chi tiết cần cài đặt:
        ---------------------------
        # Gọi self.fit(X, y) sau đó gọi và trả về self.transform(X).
        """
        pass