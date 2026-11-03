"""Đặc tả feature engineering tùy chọn — Huyền & My. Chưa triển khai."""

import pytest

pytestmark = pytest.mark.skip(reason="Skeleton: chưa triển khai kiểm thử")


def test_feature_engineer_disabled_by_default() -> None:
    """HousingFeatureEngineer mặc định không log/ratios; giữ giá trị và schema."""
    pass


def test_feature_engineer_log_transform() -> None:
    """Bật apply_log trên total_rooms/population; kiểm log1p đúng, dữ liệu không
    bị sửa tại chỗ, giữ index. Reject giá trị ngoài miền; không dùng target.
    """
    pass


def test_feature_engineer_interaction_terms() -> None:
    """Bật create_interactions; kiểm rooms_per_household, bedrooms_ratio,
    people_per_household bằng số biết trước. Reject mẫu số <=0.
    Nếu bật cả log và ratios, ratios phải tính từ giá trị gốc đã impute.
    """
    pass


def test_fit_transform_consistency() -> None:
    """fit_transform(train) khớp fit(train).transform(train); transform val/test
    không thay trạng thái fitted/schema. Không học median từ val/test.
    """
    pass
