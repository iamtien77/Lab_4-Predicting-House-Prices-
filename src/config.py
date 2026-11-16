"""
Tập tin cấu hình trung tâm (Centralized Configuration Module)
Quản lý đường dẫn, siêu tham số, tỷ lệ chia dữ liệu 8:1:1 và cài đặt pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class ProjectConfig:
    """
    Class quản lý toàn bộ cấu hình dự án Dự đoán giá nhà (MLP & Baselines).
    California Housing; 8:1:1 là lựa chọn dự án (PDF minh họa train/test 80/20).
    """

    # --------------------------------------------------------------------------
    # 1. ĐƯỜNG DẪN HỆ THỐNG (PATHS)
    # --------------------------------------------------------------------------
    BASE_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    DATA_DIR: Path = field(init=False)
    RAW_DATA_DIR: Path = field(init=False)
    PROCESSED_DATA_DIR: Path = field(init=False)
    RESULTS_DIR: Path = field(init=False)
    FIGURES_DIR: Path = field(init=False)
    METRICS_DIR: Path = field(init=False)
    SAVED_MODELS_DIR: Path = field(init=False)
    LOGS_DIR: Path = field(init=False)

    # Đường dẫn các tệp dữ liệu
    RAW_CSV_PATH: Path = field(init=False)
    PROCESSED_TRAIN_PATH: Path = field(init=False)
    PROCESSED_VAL_PATH: Path = field(init=False)
    PROCESSED_TEST_PATH: Path = field(init=False)
    
    # Đường dẫn tạo phẩm và kết quả đầu ra
    MLP_MODEL_PATH: Path = field(init=False)
    BASELINES_PATH: Path = field(init=False)
    PREPROCESSOR_PATH: Path = field(init=False)
    METRICS_SUMMARY_CSV: Path = field(init=False)
    ACTUAL_VS_PRED_PLOT: Path = field(init=False)
    LOSS_CURVE_PLOT: Path = field(init=False)
    MODEL_COMPARISON_PLOT: Path = field(init=False)
    LOG_FILE: Path = field(init=False)

    # --------------------------------------------------------------------------
    # 2. CẤU HÌNH DỮ LIỆU & TỶ LỆ PHÂN CHIA 8:1:1
    # --------------------------------------------------------------------------
    RANDOM_STATE: int = 42
    
    # Tỷ lệ chia tập 8:1:1 (Train: 80%, Validation: 10%, Test: 10%)
    TRAIN_RATIO: float = 0.80
    VAL_RATIO: float = 0.10
    TEST_RATIO: float = 0.10
    
    # Bước chia 1: Tách Train (80%) và Temp (20%) -> TEST_SIZE_TEMP = 0.20
    TEST_SIZE_TEMP: float = 0.20
    # Bước chia 2: Tách Temp (20%) thành Validation (10%) và Test (10%) -> VAL_SIZE_FROM_TEMP = 0.50
    VAL_SIZE_FROM_TEMP: float = 0.50

    # Phân tầng (Stratified) dựa trên phân vị giá nhà
    USE_STRATIFIED_SPLIT: bool = True
    N_STRATIFY_BINS_TRAIN: int = 10
    N_STRATIFY_BINS_TEMP: int = 5

    # Schema duy nhất: 8 numeric features + 1 categorical feature + target.
    TARGET_COLUMN: str = "median_house_value"
    TARGET_UNIT: str = "USD"
    NUMERIC_FEATURE_NAMES: List[str] = field(default_factory=lambda: [
        "longitude", "latitude", "housing_median_age", "total_rooms",
        "total_bedrooms", "population", "households", "median_income"
    ])
    CATEGORICAL_FEATURE_NAMES: List[str] = field(default_factory=lambda: [
        "ocean_proximity"
    ])

    # --------------------------------------------------------------------------
    # 3. SIÊU THAM SỐ MẠNG NƠ-RON MLP (THEO ĐỀ BÀI LAB 4)
    # --------------------------------------------------------------------------
    # input_dim bắt buộc lấy từ X_train_transformed.shape[1], không cố định.
    MLP_HIDDEN_1_UNITS: int = 32
    MLP_HIDDEN_1_ACTIVATION: str = "relu"
    MLP_HIDDEN_2_UNITS: int = 16
    MLP_HIDDEN_2_ACTIVATION: str = "relu"
    MLP_OUTPUT_UNITS: int = 1
    MLP_OUTPUT_ACTIVATION: str = "linear"

    MLP_OPTIMIZER: str = "adam"
    MLP_LEARNING_RATE: float = 0.001
    MLP_LOSS: str = "mean_squared_error"
    MLP_METRICS: List[str] = field(default_factory=lambda: ["mae"])
    MLP_EPOCHS: int = 100
    MLP_BATCH_SIZE: int = 32
    MLP_VERBOSE: int = 1

    # --------------------------------------------------------------------------
    # 4. SIÊU THAM SỐ CÁC MÔ HÌNH BASELINE REGRESSION (TASK 6)
    # --------------------------------------------------------------------------
    LR_FIT_INTERCEPT: bool = True
    DT_MAX_DEPTH: int = 5
    DT_RANDOM_STATE: int = 42
    RF_N_ESTIMATORS: int = 100
    RF_RANDOM_STATE: int = 42

    # --------------------------------------------------------------------------
    # 5. CẤU HÌNH TRỰC QUAN HÓA (VISUALIZATION)
    # --------------------------------------------------------------------------
    FIGURE_DPI: int = 300
    STYLE: str = "seaborn-v0_8-whitegrid"

    def __post_init__(self):
        """Khởi tạo tất cả các đường dẫn phụ thuộc sau khi BASE_DIR được xác định."""
        self.DATA_DIR = self.BASE_DIR / "data"
        self.RAW_DATA_DIR = self.DATA_DIR / "raw"
        self.PROCESSED_DATA_DIR = self.DATA_DIR / "processed"
        self.RESULTS_DIR = self.BASE_DIR / "results"
        self.FIGURES_DIR = self.RESULTS_DIR / "figures"
        self.METRICS_DIR = self.RESULTS_DIR / "metrics"
        self.SAVED_MODELS_DIR = self.RESULTS_DIR / "saved_models"
        self.LOGS_DIR = self.BASE_DIR / "logs"

        # Đường dẫn tệp
        self.RAW_CSV_PATH = self.RAW_DATA_DIR / "housing.csv"
        self.PROCESSED_TRAIN_PATH = self.PROCESSED_DATA_DIR / "train.csv"
        self.PROCESSED_VAL_PATH = self.PROCESSED_DATA_DIR / "val.csv"
        self.PROCESSED_TEST_PATH = self.PROCESSED_DATA_DIR / "test.csv"

        # Tạo phẩm lưu trữ
        self.MLP_MODEL_PATH = self.SAVED_MODELS_DIR / "mlp_house_price_model.pt"
        self.BASELINES_PATH = self.SAVED_MODELS_DIR / "baseline_models.joblib"
        self.PREPROCESSOR_PATH = self.SAVED_MODELS_DIR / "preprocessor.joblib"
        self.METRICS_SUMMARY_CSV = self.METRICS_DIR / "model_performance_summary.csv"

        # Biểu đồ trực quan
        self.ACTUAL_VS_PRED_PLOT = self.FIGURES_DIR / "actual_vs_predicted.png"
        self.LOSS_CURVE_PLOT = self.FIGURES_DIR / "loss_curve.png"
        self.MODEL_COMPARISON_PLOT = self.FIGURES_DIR / "model_comparison_bar.png"
        self.LOG_FILE = self.LOGS_DIR / "training.log"

    def create_directories(self) -> None:
        """Tự động tạo tất cả các thư mục cần thiết trong dự án nếu chưa tồn tại."""
        dirs = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.RESULTS_DIR,
            self.FIGURES_DIR,
            self.METRICS_DIR,
            self.SAVED_MODELS_DIR,
            self.LOGS_DIR,
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi cấu hình thành dictionary phục vụ in log và kiểm tra."""
        return {
            "RANDOM_STATE": self.RANDOM_STATE,
            "SPLIT_RATIO": f"{int(self.TRAIN_RATIO*10)}:{int(self.VAL_RATIO*10)}:{int(self.TEST_RATIO*10)}",
            "MLP_EPOCHS": self.MLP_EPOCHS,
            "MLP_BATCH_SIZE": self.MLP_BATCH_SIZE,
            "MLP_LEARNING_RATE": self.MLP_LEARNING_RATE,
            "FIGURE_DPI": self.FIGURED_DPI if hasattr(self, "FIGURED_DPI") else self.FIGURE_DPI,
        }


# Đối tượng cấu hình dùng chung toàn dự án
config = ProjectConfig()
# Chỉ gọi create_directories() tại entrypoint khi triển khai; import không ghi đĩa.
