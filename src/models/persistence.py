"""
Module Lưu Trữ & Tải Lại Mô Hình (Model Persistence Module)
Phụ trách: Bạn Tường và Bạn Trí (Task 2)
Nhiệm vụ:
- Lưu trữ mô hình mạng nơ-ron MLP Keras định dạng chuẩn .keras (hoặc .h5).
- Lưu trữ bộ 3 mô hình Baseline Scikit-learn và bộ chuẩn hóa StandardScaler định dạng .joblib.
- Tải lại các artifact phục vụ suy luận dự đoán giá nhà độc lập trong tương lai (Inference pipeline).
"""

from typing import Dict, Any, Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def save_mlp_model(
    model: Any,
    filepath: Union[str, Path]
) -> None:
    """
    Lưu mô hình mạng nơ-ron MLP Keras ra đĩa theo định dạng chuẩn .keras.

    Parameters:
    -----------
    model : tf.keras.Model
        Mô hình MLP đã được huấn luyện hoàn tất.
    filepath : Union[str, Path]
        Đường dẫn file đích (ví dụ: results/saved_models/mlp_house_price_model.keras).

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Chuyển filepath thành đối tượng Path, đảm bảo thư mục cha tồn tại:
    #   - path = Path(filepath)
    #   - path.parent.mkdir(parents=True, exist_ok=True)

    # BƯỚC 2: Gọi phương thức lưu trữ của Keras:
    #   - model.save(str(path))

    # BƯỚC 3: In log thông báo kích thước file và xác nhận lưu thành công.
    """
    pass


def load_mlp_model(
    filepath: Union[str, Path]
) -> Any:
    """
    Tải lại mô hình MLP Keras đã lưu từ đĩa.

    Parameters:
    -----------
    filepath : Union[str, Path]
        Đường dẫn tới file mô hình (.keras).

    Returns:
    --------
    tf.keras.Model
        Mô hình Keras đã phục hồi đầy đủ cấu trúc và trọng số đã học.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Kiểm tra file có tồn tại trên đĩa không: assert Path(filepath).exists().
    # BƯỚC 2: Dùng tensorflow.keras.models.load_model(str(filepath)).
    # BƯỚC 3: Trả về đối tượng mô hình sẵn sàng cho suy luận (predict).
    """
    pass


def save_baseline_models(
    baseline_models: Dict[str, Any],
    filepath: Union[str, Path]
) -> None:
    """
    Lưu trữ toàn bộ 3 mô hình Baseline Scikit-learn thành file .joblib.

    Parameters:
    -----------
    baseline_models : Dict[str, Any]
        Từ điển chứa 3 mô hình đã fit (Linear Regression, Decision Tree, Random Forest).
    filepath : Union[str, Path]
        Đường dẫn file lưu (results/saved_models/baseline_models.joblib).

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Đảm bảo thư mục cha tồn tại: Path(filepath).parent.mkdir(parents=True, exist_ok=True).
    # BƯỚC 2: Import joblib và gọi joblib.dump(baseline_models, filepath).
    # BƯỚC 3: Ghi log xác nhận hoàn thành lưu trữ bộ baseline.
    """
    pass


def load_baseline_models(
    filepath: Union[str, Path]
) -> Dict[str, Any]:
    """
    Tải lại bộ 3 mô hình Baseline từ file .joblib.

    Parameters:
    -----------
    filepath : Union[str, Path]
        Đường dẫn file .joblib đã lưu.

    Returns:
    --------
    Dict[str, Any]
        Từ điển chứa các đối tượng mô hình Baseline đã phục hồi.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: assert Path(filepath).exists().
    # BƯỚC 2: Dùng joblib.load(filepath) để khôi phục dict các mô hình.
    # BƯỚC 3: Trả về từ điển mô hình.
    """
    pass


def save_scaler(
    scaler: StandardScaler,
    filepath: Union[str, Path]
) -> None:
    """
    Lưu trữ đối tượng StandardScaler đã fit trên tập Train (80%).

    Parameters:
    -----------
    scaler : StandardScaler
        Đối tượng chuẩn hóa đã học μ và σ từ tập huấn luyện.
    filepath : Union[str, Path]
        Đường dẫn lưu trữ file .joblib.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Đảm bảo thư mục cha tồn tại.
    # BƯỚC 2: Dùng joblib.dump(scaler, filepath).
    # BƯỚC 3: Ghi log xác nhận lưu trữ scaler.
    """
    pass


def load_scaler(
    filepath: Union[str, Path]
) -> StandardScaler:
    """
    Tải lại StandardScaler từ file .joblib phục vụ chuẩn hóa mẫu dữ liệu mới khi suy luận.

    Parameters:
    -----------
    filepath : Union[str, Path]
        Đường dẫn tới file scaler .joblib.

    Returns:
    --------
    StandardScaler
        Đối tượng scaler đã nạp sẵn các tham số thống kê μ và σ.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: assert Path(filepath).exists().
    # BƯỚC 2: Trả về joblib.load(filepath).
    """
    pass