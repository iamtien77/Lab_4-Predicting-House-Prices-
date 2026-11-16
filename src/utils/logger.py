"""Console/file logging and execution timing."""

from functools import wraps
import logging
from pathlib import Path
import time


def setup_logger(name="HousePriceMLP", log_file=None, level=logging.INFO):
    """Configure one set of handlers, including when the output path changes."""
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
    """Log elapsed time and propagate errors without claiming success."""
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
