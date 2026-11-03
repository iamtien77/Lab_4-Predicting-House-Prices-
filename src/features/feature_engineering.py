"""Bộ khung feature engineering TÙY CHỌN. Phụ trách: Huyền & My (Task 1).
Chưa triển khai và chưa dùng trong pipeline cơ bản. Hoàn thiện baseline trước.
"""

from typing import List, Optional
import pandas as pd


class HousingFeatureEngineer:
    """Đặc tả mở rộng California Housing; fit/transform cùng quy tắc ở mọi tập.

    Không dùng target để tạo feature. Nếu triển khai tùy chọn này, đặt trong
    chuỗi preprocessing và lưu toàn bộ chuỗi; không gọi riêng trong notebook.
    """

    def __init__(self, apply_log: bool = False, create_interactions: bool = False,
                 skewed_cols: Optional[List[str]] = None):
        """Đặc tả: lưu cờ, mặc định tắt; các cột log gợi ý total_rooms,
        total_bedrooms, population, households. Quyết định bật dựa train/validation.
        """
        pass

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "HousingFeatureEngineer":
        """Đặc tả: kiểm schema, lưu thứ tự cột từ train; y chỉ để tương thích,
        không sử dụng. Nếu có thống kê cần học thì chỉ học từ X_train.
        """
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Đặc tả: trả bản sao, bảo toàn index; không học thống kê mới.

        Nếu bật, tính ratios từ số gốc đã impute: rooms_per_household,
        bedrooms_ratio, people_per_household. Reject mẫu số <=0; không che lỗi
        bằng epsilon tùy ý hoặc median của val/test. Sau đó mới log1p các cột
        được chọn (kiểm miền giá trị). log1p không bảo đảm phân phối chuẩn.
        Trả cùng schema đầu ra đã xác lập khi fit, không NaN/Inf.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Đặc tả: fit(X, y) rồi transform(X); chỉ dùng trên train."""
        pass
