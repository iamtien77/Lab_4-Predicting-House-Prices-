"""Đặc tả kiểm thử dữ liệu — Huyền & My. Chưa triển khai assertions."""

import pytest

pytestmark = pytest.mark.skip(reason="Skeleton: chưa triển khai kiểm thử")


def test_data_loader() -> None:
    """Fixture nhỏ cùng schema California: load_housing_data trả đúng df/target.
    Kiểm thiếu file, thiếu/dư cột, dữ liệu rỗng, sai kiểu số, Inf và target thiếu
    đều báo lỗi rõ. Missing feature được giữ nguyên để xử lý sau split.
    Không download/fallback, không sửa file gốc hoặc impute trong loader.
    """
    pass


def test_split_data_8_1_1() -> None:
    """Fixture 100 hàng -> 80/10/10; index đôi một không giao, hợp bằng index gốc.
    X/y cần đồng index; cùng seed tái lập được; sai index X/y phải báo lỗi.
    Kiểm số hàng lẻ được làm tròn đúng. Không có dòng bị mất hoặc xuất hiện hai tập.
    """
    pass


def test_stratified_split_edge_cases() -> None:
    """Kiểm qcut với nhiều target trùng nhau; bins sau duplicates='drop' hợp lệ.
    Nếu target hằng/số mẫu mỗi bin không đủ: báo ValueError; chỉ bỏ stratify khi
    caller truyền stratify=False, không fallback ngầm.
    """
    pass


def test_preprocessing_no_leakage() -> None:
    """Tạo train numeric có missing, val/test có phân phối rất khác.
    fit_preprocessor chỉ trên train. Kiểm median imputer, mean/variance scaler
    khớp tính tay từ train sau imputation; thay val/test không đổi thống kê.
    Kiểm mode và categories cũng chỉ từ train. Target không nằm trong features.
    Numeric train không hằng có mean~0/std~1; cột hằng có std=0 là hợp lệ.
    One-hot giữ 0/1; không kiểm std=1 trên toàn bộ output.
    """
    pass


def test_transform_unknown_category_and_schema() -> None:
    """Category ISLAND chỉ có trong test vẫn transform được, nhánh one-hot toàn 0.
    Output hữu hạn, dense, số chiều bằng train. Đảo thứ tự cột raw không đổi kết
    quả; thiếu/dư cột báo lỗi. Reject cột train thiếu hoàn toàn, transformer chưa fit.
    """
    pass


def test_raw_csv_round_trip(tmp_path) -> None:
    """save_processed_splits ghi train/val/test.csv với feature THÔ + target.
    Đọc lại giữ thứ tự hàng, NaN, categorical, giá USD (cho phép sai số float CSV).
    Không có .npy hoặc dữ liệu đã scale; transform bản đọc lại cho cùng kết quả
    như trước khi lưu bằng chính preprocessor đã fit. Kiểm X/y index trước lưu.
    """
    pass
