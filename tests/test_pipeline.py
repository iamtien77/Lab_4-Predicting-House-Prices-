"""Đặc tả integration test — phối hợp Task 1/2/3. Chưa triển khai.
Dùng fixture nhỏ cùng schema, tmp_path; không train toàn bộ CSV trong test.
"""

import pytest

pytestmark = pytest.mark.skip(reason="Skeleton: chưa triển khai kiểm thử")


def test_pipeline_smoke(tmp_path) -> None:
    """Load -> split -> fit preprocessing train -> transform -> baseline trước MLP
    (1 epoch ở smoke test) -> evaluate cùng test. Kiểm predictions hữu hạn, đúng
    số dòng; metrics cùng đơn vị; đầu vào MLP lấy D động (không cố định 13).
    Xác minh CSV là raw split, bảng so sánh có đủ bốn model, target không rò vào X.
    """
    pass


def test_inference_artifact_round_trip(tmp_path) -> None:
    """Lưu/nạp preprocessor và models từ cùng lần chạy. Với raw sample có NaN,
    category mới và thứ tự cột đổi, predictions trước/sau reload allclose trong
    tolerance phù hợp. Cả model Keras và baseline phải được kiểm. Không fit lại.
    Nạp thiếu artifact hoặc sai schema phải báo lỗi, không trả kết quả giả.
    """
    pass
