"""Tests for optional features and their preprocessing integration."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from src.config import config
from src.features.feature_engineering import HousingFeatureEngineer


def numeric_frame():
    return pd.DataFrame({
        "longitude": [-120.0, -121.0],
        "latitude": [35.0, 36.0],
        "housing_median_age": [10.0, 20.0],
        "total_rooms": [100.0, 200.0],
        "total_bedrooms": [20.0, 50.0],
        "population": [80.0, 150.0],
        "households": [10.0, 25.0],
        "median_income": [2.0, 4.0],
    }, index=[8, 19])[config.NUMERIC_FEATURE_NAMES]


def test_disabled_features_preserve_values_without_mutation():
    frame = numeric_frame()
    transformer = HousingFeatureEngineer()
    result = transformer.fit_transform(frame)
    pd.testing.assert_frame_equal(frame, result)
    assert result is not frame
    np.testing.assert_array_equal(transformer.get_feature_names_out(), frame.columns)
    assert clone(transformer).get_params() == transformer.get_params()


def test_ratios_use_original_values_before_log():
    frame = numeric_frame()
    original = frame.copy(deep=True)
    transformer = HousingFeatureEngineer(apply_log=True, create_interactions=True)
    result = transformer.fit_transform(frame)
    np.testing.assert_allclose(result["rooms_per_household"], [10, 8])
    np.testing.assert_allclose(result["bedrooms_ratio"], [0.2, 0.25])
    np.testing.assert_allclose(result["people_per_household"], [8, 6])
    np.testing.assert_allclose(result["total_rooms"], np.log1p([100, 200]))
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(result, transformer.transform(frame.iloc[:, ::-1]))
    assert result.index.equals(frame.index)
    assert len(transformer.get_feature_names_out()) == 11


def test_fit_transform_consistency_and_target_ignored():
    frame = numeric_frame()
    first = HousingFeatureEngineer(apply_log=True, create_interactions=True)
    second = clone(first)
    pd.testing.assert_frame_equal(
        first.fit_transform(frame, pd.Series([0, np.nan])),
        second.fit(frame, pd.Series([1e20, -1e20])).transform(frame),
    )
    names = first.get_feature_names_out().copy()
    changed = frame * 2
    first.transform(changed)
    np.testing.assert_array_equal(first.get_feature_names_out(), names)


@pytest.mark.parametrize("column,value,kwargs", [
    ("households", 0, {"create_interactions": True}),
    ("total_rooms", -1, {"create_interactions": True}),
    ("population", -1, {"apply_log": True}),
    ("total_bedrooms", np.nan, {}),
    ("latitude", np.inf, {}),
])
def test_invalid_values_fail_explicitly(column, value, kwargs):
    frame = numeric_frame()
    frame.loc[8, column] = value
    with pytest.raises(ValueError):
        HousingFeatureEngineer(**kwargs).fit_transform(frame)


def test_schema_and_fitted_state_are_enforced():
    frame = numeric_frame()
    with pytest.raises(NotFittedError):
        HousingFeatureEngineer().transform(frame)
    with pytest.raises(ValueError):
        HousingFeatureEngineer().fit(frame.assign(median_house_value=1))
    with pytest.raises(ValueError):
        HousingFeatureEngineer(apply_log=True, skewed_cols=["unknown"]).fit(frame)
    transformer = HousingFeatureEngineer().fit(frame)
    with pytest.raises(ValueError):
        transformer.get_feature_names_out(list(reversed(frame.columns)))
