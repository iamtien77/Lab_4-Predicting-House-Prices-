"""
Module Xây Dựng & Huấn Luyện Mạng Nơ-ron Đa Lớp (MLP Model Module)
Phụ trách: Bạn Tường và Bạn Trí (Task 2)
Nhiệm vụ:
- Xây dựng kiến trúc Multilayer Perceptron (MLP) Sequential chuẩn theo đề bài Lab 4:
    Layer 1: Dense(32, activation='relu', input_dim=...)
    Layer 2: Dense(16, activation='relu')
    Output : Dense(1, activation='linear')
- Biên dịch với optimizer='adam', loss='mean_squared_error', metrics=['mae'].
- Huấn luyện mô hình qua 100 epochs với tập validation độc lập (10% từ phép chia 8:1:1).
- Trích xuất lịch sử huấn luyện (history) và thực hiện suy luận dự đoán giá nhà.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def build_mlp_model(
    input_dim: int = 13,
    learning_rate: float = 0.001
) -> Any:
    """
    Khởi tạo và biên dịch mô hình mạng nơ-ron đa lớp (MLP) cho bài toán hồi quy giá nhà.

    Parameters:
    -----------
    input_dim : int
        Số chiều của véc-tơ đặc trưng đầu vào (mặc định 13 với Boston Housing hoặc theo X_train.shape[1]).
    learning_rate : float
        Tốc độ học của thuật toán tối ưu Adam (mặc định 0.001).

    Returns:
    --------
    tf.keras.Model
        Mô hình Keras Sequential đã được compile sẵn sàng cho việc huấn luyện.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Import thư viện Keras:
    #   - from tensorflow.keras.models import Sequential
    #   - from tensorflow.keras.layers import Dense
    #   - from tensorflow.keras.optimizers import Adam

    # BƯỚC 2: Khởi tạo mạng Sequential theo đúng cấu trúc đề tài:
    #   - model = Sequential([
    #         Dense(32, activation='relu', input_dim=input_dim, name='hidden_layer_1'),
    #         Dense(16, activation='relu', name='hidden_layer_2'),
    #         Dense(1, activation='linear', name='output_regression_layer')
    #     ])

    # BƯỚC 3: Thiết lập Optimizer và Biên dịch mô hình (Compile):
    #   - opt = Adam(learning_rate=learning_rate)
    #   - model.compile(
    #         optimizer=opt,
    #         loss='mean_squared_error',   # MSE theo yêu cầu đề bài
    #         metrics=['mae']              # Theo dõi MAE theo yêu cầu đề bài
    #     )

    # BƯỚC 4: In cấu trúc mạng ra màn hình (model.summary()) và trả về đối tượng model.
    """
    pass


def train_mlp_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    verbose: int = 1
) -> Tuple[Any, Any]:
    """
    Huấn luyện mô hình MLP Keras trên tập Train (80%) và giám sát trên tập Validation (10%).

    Parameters:
    -----------
    model : tf.keras.Model
        Mô hình MLP đã được compile.
    X_train : np.ndarray
        Mảng đặc trưng tập huấn luyện đã chuẩn hóa (chiếm 80% dữ liệu).
    y_train : np.ndarray
        Mảng giá trị thực tế tập huấn luyện.
    X_val : np.ndarray
        Mảng đặc trưng tập kiểm định đã chuẩn hóa (chiếm 10% dữ liệu - tách từ 8:1:1).
    y_val : np.ndarray
        Mảng giá trị thực tế tập kiểm định.
    epochs : int
        Số lượt duyệt toàn bộ dữ liệu huấn luyện (mặc định 100 epochs theo đề bài).
    batch_size : int
        Kích thước mini-batch (mặc định 32 theo đề bài).
    verbose : int
        Chế độ in thông tin tiến trình huấn luyện (1: hiển thị progress bar).

    Returns:
    --------
    Tuple[tf.keras.Model, tf.keras.callbacks.History]
        - Model đã được cập nhật trọng số sau huấn luyện.
        - Đối tượng History chứa biến thiên loss, val_loss, mae, val_mae qua từng epoch.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Kiểm tra tính tương thích shape giữa X_train và y_train, X_val và y_val:
    #   - assert X_train.shape[0] == y_train.shape[0]
    #   - assert X_val.shape[0] == y_val.shape[0]

    # BƯỚC 2: Gọi phương thức model.fit():
    #   - history = model.fit(
    #         X_train,
    #         y_train,
    #         validation_data=(X_val, y_val),   # Dùng đúng 10% Validation từ phép chia 8:1:1
    #         epochs=epochs,
    #         batch_size=batch_size,
    #         verbose=verbose
    #     )

    # BƯỚC 3: Trích xuất các mảng loss và metric trong history.history:
    #   - train_loss = history.history['loss']
    #   - val_loss   = history.history['val_loss']
    #   - train_mae  = history.history['mae']
    #   - val_mae    = history.history['val_mae']

    # BƯỚC 4: Trả về cặp (model, history).
    """
    pass


def evaluate_mlp_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Đánh giá mô hình MLP trên tập Test độc lập (10%) sử dụng hàm evaluate tích hợp của Keras.

    Parameters:
    -----------
    model : tf.keras.Model
        Mô hình MLP đã được huấn luyện.
    X_test : np.ndarray
        Mảng đặc trưng tập kiểm thử độc lập đã chuẩn hóa (chiếm 10% dữ liệu).
    y_test : np.ndarray
        Mảng giá trị thực tế tập kiểm thử.

    Returns:
    --------
    Dict[str, float]
        Từ điển chứa giá trị test_loss (MSE) và test_mae.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Gọi model.evaluate(X_test, y_test, verbose=0).
    # BƯỚC 2: Bóc tách loss và mae tương ứng từ kết quả trả về.
    # BƯỚC 3: Đóng gói vào từ điển {'test_mse': float(test_loss), 'test_mae': float(test_mae)}.
    # BƯỚC 4: Trả về từ điển kết quả.
    """
    pass


def predict_mlp(
    model: Any,
    X: np.ndarray
) -> np.ndarray:
    """
    Thực hiện suy luận dự đoán giá nhà từ mô hình MLP đã huấn luyện.

    Parameters:
    -----------
    model : tf.keras.Model
        Mô hình MLP đã huấn luyện.
    X : np.ndarray
        Mảng đặc trưng đầu vào đã chuẩn hóa (shape: [N, D]).

    Returns:
    --------
    np.ndarray
        Véc-tơ 1 chiều chứa các giá trị dự đoán (shape: [N,]).

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Gọi model.predict(X, verbose=0).
    # BƯỚC 2: Làm phẳng mảng kết quả từ shape (N, 1) về (N,) bằng np.squeeze() hoặc .flatten().
    # BƯỚC 3: Trả về mảng y_pred 1 chiều.
    """
    pass