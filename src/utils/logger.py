"""
Module Tiện Ích Ghi Nhật Ký & Đo Thời Gian Thực Thi (Logger & Timing Module)
Phụ trách: Bạn Tiến và Bạn Thịnh (Task 3)
Nhiệm vụ:
- Thiết lập hệ thống logging chuẩn mực ghi đồng thời ra màn hình console và file nhật ký logs/training.log.
- Cung cấp decorator @log_step đo đạc thời gian thực thi của từng công đoạn trong pipeline.
"""

from typing import Optional, Callable, Any
from pathlib import Path
import logging
import time
from functools import wraps


def setup_logger(
    name: str = "HousePriceMLP",
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Cấu hình đối tượng logger ghi thông tin ra cả console và file.

    Parameters:
    -----------
    name : str
        Tên định danh của logger.
    log_file : Optional[Path]
        Đường dẫn file lưu nhật ký (ví dụ: logs/training.log).
    level : int
        Mức độ log (mặc định logging.INFO).

    Returns:
    --------
    logging.Logger
        Đối tượng logger đã được gán các handler sẵn sàng sử dụng.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Khởi tạo logger = logging.getLogger(name), đặt level = level.
    # BƯỚC 2: Kiểm tra nếu logger đã có handler thì return luôn để tránh in lặp log.
    # BƯỚC 3: Tạo formatter chuẩn định dạng:
    #   formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    # BƯỚC 4: Tạo StreamHandler ghi ra console:
    #   console_handler = logging.StreamHandler()
    #   console_handler.setFormatter(formatter)
    #   logger.addHandler(console_handler)
    # BƯỚC 5: Nếu có log_file:
    #   - Tạo thư mục cha: Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    #   - Tạo FileHandler(log_file, encoding='utf-8')
    #   - Gán formatter và thêm vào logger.
    # BƯỚC 6: Trả về đối tượng logger.
    """
    pass


def log_step(step_name: str) -> Callable:
    """
    Decorator dùng để bọc các hàm chức năng, tự động in log khi bắt đầu,
    kết thúc và đo chính xác thời gian thực thi (giây).

    Parameters:
    -----------
    step_name : str
        Tên mô tả công đoạn đang thực hiện (ví dụ: "Nạp Dữ Liệu", "Huấn Luyện MLP").

    Returns:
    --------
    Callable
        Hàm bao bọc (wrapper) đo thời gian.

    Logic chi tiết cần cài đặt:
    ---------------------------
    # BƯỚC 1: Định nghĩa decorator(func):
    #   - Định nghĩa wrapper(*args, **kwargs):
    #       + In log: f"=== [BẮT ĐẦU] {step_name} ==="
    #       + Ghi lại thời điểm bắt đầu: start_time = time.perf_counter()
    #       + Thực thi hàm gốc: result = func(*args, **kwargs)
    #       + Ghi lại thời điểm kết thúc: elapsed = time.perf_counter() - start_time
    #       + In log: f"=== [HOÀN THÀNH] {step_name} (Thời gian: {elapsed:.2f}s) ==="
    #       + Trả về result.
    # BƯỚC 2: Sử dụng @wraps(func) để bảo toàn metadata của hàm gốc.
    # BƯỚC 3: Trả về hàm wrapper.
    """
    pass