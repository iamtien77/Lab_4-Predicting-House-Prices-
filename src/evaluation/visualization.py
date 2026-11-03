"""
Module Trực Quan Hóa Đồ Thị & Báo Cáo Hiệu Năng (Visualization Module)
Phụ trách: Bạn Tiến và Bạn Thịnh (Task 3)
Nhiệm vụ:
- Vẽ biểu đồ phân tán giá trị thực tế vs dự đoán (Actual vs Predicted Scatter Plot) có đường chéo lý tưởng y = x.
- Vẽ đồ thị tiến trình học tập qua 100 epochs (Learning Curves: Train Loss vs Val Loss, Train MAE vs Val MAE).
- Vẽ biểu đồ cột đối chuẩn so sánh đa mô hình (Model Comparison Bar Chart cho Task 6).
- Vẽ biểu đồ phân bố phần dư (Residuals Distribution Plot).
"""

from typing import Optional, List, Union
from pathlib import Path
import numpy as np
import pandas as pd


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "MLP Keras",
    mse: Optional[float] = None,
    r2: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> None:
    """
    Vẽ biểu đồ phân tán so sánh giá thực tế vs giá dự đoán (Actual vs Predicted)
    kèm đường tham chiếu lý tưởng y = x nét đứt màu đỏ (theo yêu cầu đề bài Lab 4).

    Parameters:
    -----------
    y_true : np.ndarray
        Mảng giá trị thực tế của tập kiểm thử.
    y_pred : np.ndarray
        Mảng giá trị mô hình dự đoán.
    model_name : str
        Tên mô hình để hiển thị tiêu đề biểu đồ.
    mse : Optional[float]
        Chỉ số MSE để chú thích lên biểu đồ.
    r2 : Optional[float]
        Chỉ số R² để chú thích lên biểu đồ.
    save_path : Optional[Union[str, Path]]
        Đường dẫn file ảnh xuất ra (ví dụ: results/figures/actual_vs_predicted.png).
    dpi : int
        Độ phân giải ảnh (mặc định 300 dpi cho chất lượng cao).

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Import matplotlib.pyplot as plt và seaborn as sns.
    # BƯỚC 2: Khởi tạo figure: fig, ax = plt.subplots(figsize=(8, 6)).
    # BƯỚC 3: Vẽ biểu đồ phân tán (scatter):
    #   ax.scatter(y_true, y_pred, alpha=0.6, color='royalblue', edgecolors='k', label='Dự đoán')
    # BƯỚC 4: Vẽ đường chéo lý tưởng y = x (hoàn hảo khi dự đoán trùng khớp thực tế):
    #   min_val = min(y_true.min(), y_pred.min())
    #   max_val = max(y_true.max(), y_pred.max())
    #   ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Đường lý tưởng y = x')
    # BƯỚC 5: Thêm text box chú thích MSE và R² trực tiếp trong biểu đồ nếu có.
    # BƯỚC 6: Thiết lập nhãn trục x ('Actual House Prices (USD)'), trục y ('Predicted House Prices (USD)'),
    #         tiêu đề f'Actual vs. Predicted House Prices ({model_name})', lưới (grid), và chú giải (legend).
    # BƯỚC 7: Nếu có save_path: plt.savefig(save_path, dpi=dpi, bbox_inches='tight').
    # BƯỚC 8: Đóng figure (plt.close(fig)) để giải phóng bộ nhớ.
    """
    pass


def plot_loss_and_mae_curves(
    history: Any,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> None:
    """
    Vẽ đồ thị diễn biến giảm dần của hàm mất mát (Loss) và sai số tuyệt đối (MAE)
    qua 100 epochs cho cả tập Train (80%) và tập Validation (10%).

    Parameters:
    -----------
    history : tf.keras.callbacks.History
        Đối tượng lịch sử huấn luyện trả về từ hàm model.fit().
    save_path : Optional[Union[str, Path]]
        Đường dẫn file lưu biểu đồ (results/figures/loss_curve.png).
    dpi : int
        Độ phân giải ảnh (mặc định 300).

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Trích xuất các chuỗi số từ history.history:
    #   - train_loss = history.history['loss']
    #   - val_loss   = history.history.get('val_loss', [])
    #   - train_mae  = history.history['mae']
    #   - val_mae    = history.history.get('val_mae', [])
    #   - epochs_range = range(1, len(train_loss) + 1)

    # BƯỚC 2: Tạo layout 2 đồ thị con (1 hàng 2 cột):
    #   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # BƯỚC 3: Vẽ đồ thị con 1 (Loss Curve - MSE):
    #   ax1.plot(epochs_range, train_loss, 'b-', label='Train Loss (MSE)')
    #   if val_loss: ax1.plot(epochs_range, val_loss, 'r--', label='Val Loss (MSE)')
    #   ax1.set_title('Training and Validation Loss (MSE) over Epochs')
    #   ax1.set_xlabel('Epochs')
    #   ax1.set_ylabel('Loss (MSE)')
    #   ax1.legend(); ax1.grid(True)

    # BƯỚC 4: Vẽ đồ thị con 2 (MAE Curve):
    #   ax2.plot(epochs_range, train_mae, 'g-', label='Train MAE')
    #   if val_mae: ax2.plot(epochs_range, val_mae, 'orange', linestyle='--', label='Val MAE')
    #   ax2.set_title('Training and Validation MAE over Epochs')
    #   ax2.set_xlabel('Epochs')
    #   ax2.set_ylabel('MAE (USD)')
    #   ax2.legend(); ax2.grid(True)

    # BƯỚC 5: Lưu ảnh ra save_path và đóng figure.
    """
    pass


def plot_model_comparison_bar(
    summary_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> None:
    """
    Vẽ biểu đồ cột đối chiếu trực quan hiệu năng giữa mạng nơ-ron MLP và 3 mô hình Baseline
    (Linear Regression, Decision Tree, Random Forest) trên 2 chỉ số: MSE và R² Score (Task 6).

    Parameters:
    -----------
    summary_df : pd.DataFrame
        Bảng tổng hợp chỉ số chứa các cột ['Model', 'MSE', 'RMSE', 'MAE', 'R2_Score'].
    save_path : Optional[Union[str, Path]]
        Đường dẫn file ảnh lưu kết quả (results/figures/model_comparison_bar.png).
    dpi : int
        Độ phân giải ảnh.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Tạo layout 1 hàng 2 cột (Subplot 1: MSE [thấp hơn là tốt hơn], Subplot 2: R² [cao hơn là tốt hơn]).
    #   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # BƯỚC 2: Vẽ biểu đồ cột MSE:
    #   sns.barplot(data=summary_df, x='Model', y='MSE', ax=ax1, palette='Blues_d')
    #   ax1.set_title('So Sánh MSE giữa Các Mô Hình (Thấp Hơn Là Tốt Hơn)')
    #   ax1.set_ylabel('Mean Squared Error (MSE)')
    #   Hiển thị nhãn giá trị số lên đầu từng cột bar.

    # BƯỚC 3: Vẽ biểu đồ cột R² Score:
    #   sns.barplot(data=summary_df, x='Model', y='R2_Score', ax=ax2, palette='Greens_d')
    #   ax2.set_title('So Sánh R² Score giữa Các Mô Hình (Cao Hơn Là Tốt Hơn)')
    #   ax2.set_ylabel('Hệ Số Xác Định R²')
    #   Hiển thị nhãn giá trị số lên đầu từng cột bar.

    # BƯỚC 4: Tinh chỉnh thẩm mỹ (xoay nhãn x-tick 15 độ, bật grid mờ).
    # BƯỚC 5: Lưu file ảnh chất lượng cao và đóng figure.
    """
    pass


def plot_residuals_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "MLP Keras",
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> None:
    """
    Vẽ biểu đồ phân bố phần dư (Residuals = y_true - y_pred) để kiểm định tính chuẩn của sai số.

    Parameters:
    -----------
    y_true : np.ndarray
        Giá trị thực tế.
    y_pred : np.ndarray
        Giá trị dự đoán.
    model_name : str
        Tên mô hình.
    save_path : Optional[Union[str, Path]]
        Đường dẫn file ảnh.
    dpi : int
        Độ phân giải.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Tính residuals = np.asarray(y_true).flatten() - np.asarray(y_pred).flatten().
    # BƯỚC 2: Vẽ histogram kèm đường cong ước lượng mật độ nhân (KDE) bằng sns.histplot().
    # BƯỚC 3: Vẽ đường thẳng tham chiếu dọc x = 0 màu đỏ đứt nét (kỳ vọng sai số trung bình bằng 0).
    # BƯỚC 4: Đặt tiêu đề f'Residuals Distribution - {model_name}', nhãn trục và lưu file.
    """
    pass