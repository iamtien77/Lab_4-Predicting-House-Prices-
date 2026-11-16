"""Data contracts and leakage regression tests."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from src.config import config
from src.data.loader import load_housing_data, fetch_dataset_info
from src.data.preprocessing import (
    split_data_8_1_1, fit_preprocessor, transform_features, save_processed_splits,
)


def make_data(size=100):
    frame = pd.DataFrame({
        name: np.arange(size, dtype=float) + offset + 1
        for offset, name in enumerate(config.NUMERIC_FEATURE_NAMES)
    })
    frame["ocean_proximity"] = np.where(np.arange(size) % 3 == 0, "NEAR BAY", "INLAND")
    frame.loc[::7, "total_bedrooms"] = np.nan
    target = pd.Series(np.arange(size, dtype=float) * 1000 + 10000,
                       name=config.TARGET_COLUMN)
    return frame, target


def test_loader_and_profile_preserve_raw_values(tmp_path):
    X, y = make_data()
    frame = X.assign(**{config.TARGET_COLUMN: y})
    path = tmp_path / "housing.csv"
    frame.to_csv(path, index=False)
    loaded, target = load_housing_data(path)
    pd.testing.assert_frame_equal(loaded, frame)
    assert target == config.TARGET_COLUMN
    before = loaded.copy(deep=True)
    info = fetch_dataset_info(loaded, target)
    assert info["shape"] == (100, 10)
    assert info["missing"]["total_bedrooms"] == 15
    pd.testing.assert_frame_equal(loaded, before)
    with pytest.raises(FileNotFoundError):
        load_housing_data(tmp_path / "missing.csv")
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_housing_data(path)


@pytest.mark.parametrize("failure", ["missing", "extra", "text", "infinity", "target_missing", "empty"])
def test_loader_rejects_invalid_data(tmp_path, failure):
    X, y = make_data()
    frame = X.assign(**{config.TARGET_COLUMN: y})
    if failure == "missing":
        frame = frame.drop(columns="longitude")
    elif failure == "extra":
        frame["unexpected"] = 1
    elif failure == "text":
        frame["longitude"] = "invalid"
    elif failure == "infinity":
        frame.loc[0, "longitude"] = np.inf
    elif failure == "target_missing":
        frame.loc[0, config.TARGET_COLUMN] = np.nan
    else:
        frame = frame.iloc[:0]
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_housing_data(path)


@pytest.mark.parametrize("size,counts", [(100, [80, 10, 10]), (103, [82, 10, 11])])
def test_split_conserves_rows_and_is_reproducible(size, counts):
    X, y = make_data(size)
    first = split_data_8_1_1(X, y)
    second = split_data_8_1_1(X, y)
    assert [len(frame) for frame in first[:3]] == counts
    all_index = first[0].index.append(first[1].index).append(first[2].index)
    assert all_index.is_unique and set(all_index) == set(X.index)
    for left, right in zip(first, second):
        assert left.equals(right)
    for features, target in zip(first[:3], first[3:]):
        assert features.index.equals(target.index)
        pd.testing.assert_frame_equal(features, X.loc[features.index])
    with pytest.raises(ValueError, match="indexes"):
        split_data_8_1_1(X, y.iloc[::-1])


def test_stratification_rejects_impossible_bins_without_fallback():
    X, y = make_data()
    with pytest.raises(ValueError, match="stratify=False"):
        split_data_8_1_1(X, y * 0)
    with pytest.raises(ValueError, match="stratify=False"):
        split_data_8_1_1(X, y, n_bins_train=100)
    assert len(split_data_8_1_1(X, y * 0, stratify=False)[0]) == 80
    repeated = pd.Series(np.repeat(np.arange(5), 20), dtype=float)
    assert len(split_data_8_1_1(X, repeated)[0]) == 80


def test_preprocessing_statistics_are_train_only():
    X, _ = make_data()
    train = X.iloc[:80].copy()
    train.loc[0, "ocean_proximity"] = np.nan
    processor = fit_preprocessor(train)
    numeric = processor.named_transformers_["numeric"]
    expected = train[config.NUMERIC_FEATURE_NAMES].fillna(train[config.NUMERIC_FEATURE_NAMES].median())
    np.testing.assert_allclose(numeric.named_steps["imputer"].statistics_,
                               train[config.NUMERIC_FEATURE_NAMES].median())
    np.testing.assert_allclose(numeric.named_steps["scaler"].mean_, expected.mean())
    np.testing.assert_allclose(numeric.named_steps["scaler"].var_, expected.var(ddof=0))
    before = numeric.named_steps["scaler"].mean_.copy()
    held_out = X.iloc[80:].copy()
    held_out["longitude"] = 1e8
    held_out["ocean_proximity"] = "ISLAND"
    transformed = transform_features(processor, held_out)
    assert np.isfinite(transformed).all()
    assert (transformed[:, 8:] == 0).all()
    np.testing.assert_array_equal(numeric.named_steps["scaler"].mean_, before)
    np.testing.assert_allclose(transformed, transform_features(processor, held_out.iloc[:, ::-1]))
    with pytest.raises(ValueError):
        transform_features(processor, held_out.drop(columns="longitude"))
    with pytest.raises(ValueError):
        transform_features(processor, held_out.assign(extra=1))
    with pytest.raises(ValueError):
        fit_preprocessor(train.assign(total_bedrooms=np.nan))
    with pytest.raises(NotFittedError):
        transform_features(ColumnTransformer([]), held_out)


def test_csv_and_feature_artifact_round_trip(tmp_path):
    X, y = make_data()
    splits = split_data_8_1_1(X, y)
    processor = fit_preprocessor(splits[0], apply_log=True, create_interactions=True)
    artifact = tmp_path / "preprocessor.joblib"
    joblib.dump(processor, artifact)
    restored = joblib.load(artifact)
    save_processed_splits(*splits, config.TARGET_COLUMN, output_dir=tmp_path)
    for name, original_X, original_y in zip(("train", "val", "test"), splits[:3], splits[3:]):
        frame = pd.read_csv(tmp_path / f"{name}.csv")
        pd.testing.assert_frame_equal(frame.drop(columns=config.TARGET_COLUMN),
                                      original_X.reset_index(drop=True))
        np.testing.assert_allclose(frame[config.TARGET_COLUMN], original_y)
        np.testing.assert_allclose(transform_features(restored, frame.drop(columns=config.TARGET_COLUMN)),
                                   transform_features(processor, original_X))
    assert len(processor.get_feature_names_out()) == 13
    with pytest.raises(ValueError, match="disjoint"):
        save_processed_splits(X, X, X, y, y, y, config.TARGET_COLUMN, tmp_path)
