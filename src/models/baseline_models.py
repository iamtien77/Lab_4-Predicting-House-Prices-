"""
Module Các Mô Hình Hồi Quy Baseline Đối Chuẩn (Task 6 Baseline Models Module)
Phụ trách: Bạn Tường và Bạn Trí (Task 2)
Nhiệm vụ:
- Xây dựng bộ 3 thuật toán học máy hồi quy kinh điển đối chuẩn theo yêu cầu Task 6 của đề bài:
    1. Linear Regression (Hồi quy tuyến tính)
    2. Decision Tree Regressor (Cây quyết định hồi quy)
    3. Random Forest Regressor (Rừng ngẫu nhiên hồi quy)
- Huấn luyện đồng thời trên tập Train (80%) và sinh mảng dự đoán trên tập Test (10%).
"""

from typing import Dict, Any, Optional
import numpy as np


class BaselineRegressionSuite:
    """
    Bộ đóng gói 3 mô hình học máy hồi quy baseline (Scikit-Learn).
    Cung cấp giao diện đồng nhất fit_all() và predict_all() phục vụ so sánh trực tiếp với MLP.
    """

    def __init__(
        self,
        random_state: int = 42,
        dt_max_depth: int = 5,
        rf_n_estimators: int = 100
    ):
        """
        Khởi tạo các mô hình Baseline với siêu tham số cấu hình.

        Parameters:
        -----------
        random_state : int
            Hạt giống ngẫu nhiên cố định (mặc định 42).
        dt_max_depth : int
            Độ sâu tối đa cho DecisionTreeRegressor (mặc định 5 nhằm khống chế overfitting).
        rf_n_estimators : int
            Số lượng cây trong rừng RandomForestRegressor (mặc định 100).

        Logic chi tiết cần cài đặt:
        ---------------------------
        # BƯỚC 1: Import các lớp mô hình từ sklearn:
        #   - from sklearn.linear_model import LinearRegression
        #   - from sklearn.tree import DecisionTreeRegressor
        #   - from sklearn.ensemble import RandomForestRegressor

        # BƯỚC 2: Khởi tạo từ điển chứa 3 đối tượng mô hình:
        #   self.models = {
        #       'Linear Regression': LinearRegression(fit_intercept=True),
        #       'Decision Tree': DecisionTreeRegressor(max_depth=dt_max_depth, random_state=random_state),
        #       'Random Forest': RandomForestRegressor(n_estimators=rf_n_estimators, random_state=random_state)
        #   }

        # BƯỚC 3: Khởi tạo cờ kiểm tra trạng thái huấn luyện: self.is_fitted = False.
        """
        pass

    def fit_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, Any]:
        """
        Huấn luyện đồng loạt cả 3 mô hình Baseline trên tập Train (80%).

        Parameters:
        -----------
        X_train : np.ndarray
            Ma trận đặc trưng tập huấn luyện đã chuẩn hóa.
        y_train : np.ndarray
            Mảng giá trị mục tiêu tập huấn luyện.

        Returns:
        --------
        Dict[str, Any]
            Từ điển chứa 3 mô hình đã được fit hoàn chỉnh.

        Logic chi tiết cần cài đặt:
        ---------------------------
        # BƯỚC 1: Duyệt qua từng cặp (name, model) trong self.models.items():
        #   - In log / thông báo bắt đầu huấn luyện model 'name'.
        #   - Gọi model.fit(X_train, y_train).
        # BƯỚC 2: Cập nhật self.is_fitted = True.
        # BƯỚC 3: Trả về self.models.
        """
        pass

    def predict_all(
        self,
        X: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Thực hiện dự đoán giá nhà trên mảng X (thường là X_test) cho toàn bộ 3 mô hình Baseline.

        Parameters:
        -----------
        X : np.ndarray
            Ma trận đặc trưng đầu vào cần dự đoán.

        Returns:
        --------
        Dict[str, np.ndarray]
            Từ điển chứa kết quả dự đoán của từng mô hình:
            {'Linear Regression': y_pred_lr, 'Decision Tree': y_pred_dt, 'Random Forest': y_pred_rf}

        Logic chi tiết cần cài đặt:
        ---------------------------
        # BƯỚC 1: Kiểm tra assert self.is_fitted, "Các mô hình baseline chưa được huấn luyện!"
        # BƯỚC 2: Khởi tạo từ điển rỗng: predictions = {}.
        # BƯỚC 3: Duyệt qua từng (name, model) trong self.models.items():
        #   - y_pred = model.predict(X).flatten()
        #   - predictions[name] = y_pred
        # BƯỚC 4: Trả về từ điển predictions.
        """
        pass

    def get_model(self, name: str) -> Any:
        """
        Trích xuất một mô hình cụ thể theo tên (ví dụ: 'Random Forest').

        Parameters:
        -----------
        name : str
            Tên mô hình trong danh sách.

        Returns:
        --------
        Any
            Đối tượng mô hình Scikit-Learn tương ứng.

        Logic chi tiết cần cài đặt:
        ---------------------------
        # Kiểm tra name có trong self.models hay không, nếu có trả về self.models[name].
        """
        pass