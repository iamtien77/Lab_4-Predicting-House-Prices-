"""Hệ thống ghi log ra console/file và đo thời gian thực thi."""

from functools import wraps
import logging
from pathlib import Path
import time


def setup_logger(name="HousePriceMLP", log_file=None, level=logging.INFO):
    """Cấu hình tập hợp handlers ghi log, xử lý an toàn khi thay đổi đường dẫn tệp log."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_step(step_name):
    """Decorator ghi log thời gian thực thi và lan truyền ngoại lệ nếu phát sinh lỗi."""
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("HousePriceMLP")
            logger.info("Bắt đầu: %s", step_name)
            started = time.perf_counter()
            try:
                result = function(*args, **kwargs)
            except Exception:
                logger.exception("Thất bại: %s", step_name)
                raise
            logger.info("Hoàn thành: %s (%.2f giây)", step_name, time.perf_counter() - started)
            return result
        return wrapper
    return decorator
