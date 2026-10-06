"""
Module Đánh Giá Chỉ Số Hồi Quy (Regression Metrics Module)
Phụ trách: Bạn Tiến và Bạn Thịnh (Task 3)
Nhiệm vụ:
- Tính toán 4 chỉ số cốt lõi đánh giá bài toán hồi quy:
    1. MSE (Mean Squared Error)
    2. RMSE (Root Mean Squared Error)
    3. MAE (Mean Absolute Error)
    4. R² Score (Hệ số xác định - R-squared)
- Tổng hợp bảng so sánh đối chuẩn hiệu năng đa mô hình (MLP vs 3 Baselines).
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


class RegressionEvaluator:
    """
    Bộ đánh giá chỉ số hiệu năng bài toán hồi quy (Regression Metrics Evaluator).
    Hỗ trợ tính toán thủ công theo công thức toán học kết hợp đối chiếu với thư viện Scikit-learn.
    """

    @staticmethod
    def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Tính Mean Squared Error (MSE): Sai số toàn phương trung bình.
        Công thức: MSE = (1/N) * sum((y_i - y_hat_i)^2)

        Parameters:
        -----------
        y_true : np.ndarray
            Mảng giá trị thực tế.
        y_pred : np.ndarray
            Mảng giá trị dự đoán từ mô hình.

        Returns:
        --------
        float
            Giá trị MSE.

        Logic chi tiết cần cài đặt:
        ---------------------------
        # BƯỚC 1: Đảm bảo cả hai mảng có cùng kích thước và 1 chiều:
        #   y_true = np.asarray(y_true).flatten()
        #   y_pred = np.asarray(y_pred).flatten()
        # BƯỚC 2: Tính sai phân từng mẫu: error = y_true - y_pred.
        # BƯỚC 3: Bình phương và lấy giá trị trung bình: mse = np.mean(error ** 2).
        # BƯỚC 4: Trả về float(mse).
        """
        pass

    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Tính Root Mean Squared Error (RMSE): Căn bậc hai của sai số toàn phương trung bình.
        Công thức: RMSE = sqrt(MSE)

        Parameters:
        -----------
        y_true : np.ndarray
            Mảng giá trị thực tế.
        y_pred : np.ndarray
            Mảng giá trị dự đoán từ mô hình.

        Returns:
        --------
        float
            Giá trị RMSE (cùng đơn vị đo với giá nhà).

        Logic chi tiết cần cài đặt:
        ---------------------------
        # BƯỚC 1: Tính mse = RegressionEvaluator.calculate_mse(y_true, y_pred).
        # BƯỚC 2: Lấy căn bậc hai: rmse = np.sqrt(mse).
        # BƯỚC 3: Trả về float(rmse).
        """
        pass

    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Tính Mean Absolute Error (MAE): Sai số tuyệt đối trung bình.
        Công thức: MAE = (1/N) * sum(|y_i - y_hat_i|)

        Parameters:
        -----------
        y_true : np.ndarray
            Mảng giá trị thực tế.
        y_pred : np.ndarray
            Mảng giá trị dự đoán từ mô hình.

        Returns:
        --------
        float
            Giá trị MAE.

        Logic chi tiết cần cài đặt:
        ---------------------------
        # BƯỚC 1: y_t = np.asarray(y_true).flatten(); y_p = np.asarray(y_pred).flatten().
        # BƯỚC 2: Tính sai số tuyệt đối: abs_error = np.abs(y_t - y_p).
        # BƯỚC 3: Lấy trung bình: mae = np.mean(abs_error).
        # BƯỚC 4: Trả về float(mae).
        """
        pass

    @staticmethod
    def calculate_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Tính Hệ số xác định R² (R-squared Score): Tỷ lệ phương sai biến mục tiêu được giải thích bởi mô hình.
        Công thức: R² = 1 - (sum((y_i - y_hat_i)^2) / sum((y_i - y_mean)^2))

        Parameters:
        -----------
        y_true : np.ndarray
            Mảng giá trị thực tế.
        y_pred : np.ndarray
            Mảng giá trị dự đoán từ mô hình.

        Returns:
        --------
        float
            Hệ số R² (tiến gần 1.0 biểu thị mô hình dự đoán rất tốt).

        Logic chi tiết cần cài đặt:
        ---------------------------
        # BƯỚC 1: y_t = np.asarray(y_true).flatten(); y_p = np.asarray(y_pred).flatten().
        # BƯỚC 2: Tính tổng bình phương phần dư (Sum of Squared Residuals - SS_res):
        #   ss_res = np.sum((y_t - y_p) ** 2)
        # BƯỚC 3: Tính tổng bình phương tổng thể (Total Sum of Squares - SS_tot):
        #   y_mean = np.mean(y_t)
        #   ss_tot = np.sum((y_t - y_mean) ** 2)
        # BƯỚC 4: Tránh chia cho 0: nếu ss_tot == 0, trả về 0.0.
        # BƯỚC 5: r2 = 1.0 - (ss_res / ss_tot).
        # BƯỚC 6: Trả về float(r2).
        """
        pass

    @classmethod
    def evaluate_predictions(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Đánh giá đầy đủ cả 4 chỉ số (MSE, RMSE, MAE, R²) cho một mô hình cụ thể.

        Parameters:
        -----------
        y_true : np.ndarray
            Mảng nhãn thực tế.
        y_pred : np.ndarray
            Mảng nhãn dự đoán.
        model_name : str
            Tên mô hình (ví dụ: 'MLP Keras', 'Random Forest').

        Returns:
        --------
        Dict[str, Any]
            Từ điển chứa tên mô hình và 4 chỉ số đo lường.

        Logic chi tiết cần cài đặt:
        ---------------------------
        # BƯỚC 1: Gọi tính mse = cls.calculate_mse(y_true, y_pred).
        # BƯỚC 2: Gọi tính rmse = cls.calculate_rmse(y_true, y_pred).
        # BƯỚC 3: Gọi tính mae = cls.calculate_mae(y_true, y_pred).
        # BƯỚC 4: Gọi tính r2 = cls.calculate_r2_score(y_true, y_pred).
        # BƯỚC 5: Đóng gói và trả về:
        #   {
        #       'Model': model_name,
        #       'MSE': round(mse, 4),
        #       'RMSE': round(rmse, 4),
        #       'MAE': round(mae, 4),
        #       'R2_Score': round(r2, 4)
        #   }
        """
        pass

    @classmethod
    def summarize_models(
        cls,
        evaluation_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Tổng hợp kết quả đánh giá của tất cả các mô hình thành bảng DataFrame trực quan,
        sắp xếp theo R² Score giảm dần hoặc MSE tăng dần.

        Parameters:
        -----------
        evaluation_results : List[Dict[str, Any]]
            Danh sách các dict chứa chỉ số của từng mô hình.

        Returns:
        --------
        pd.DataFrame
            Bảng tổng hợp đối chuẩn hiệu năng các mô hình (phục vụ Task 6).

        Logic chi tiết cần cài đặt:
        ---------------------------
        # BƯỚC 1: Chuyển list các dict thành pd.DataFrame(evaluation_results).
        # BƯỚC 2: Sắp xếp DataFrame theo R2_Score giảm dần (hoặc MSE tăng dần):
        #   summary_df = summary_df.sort_values(by='R2_Score', ascending=False).reset_index(drop=True).
        # BƯỚC 3: Trả về summary_df.
        """
        pass