"""Check regression arithmetic, edge cases, and summary contracts."""

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.evaluation.metrics import RegressionEvaluator


def test_metrics_match_reference():
    actual = np.array([10., 20., 40., 80.])
    predicted = np.array([12., 18., 37., 90.])
    result = RegressionEvaluator.evaluate_predictions(actual, predicted[:, None], "Example")
    assert result["MSE"] == pytest.approx(mean_squared_error(actual, predicted))
    assert result["RMSE"] == pytest.approx(np.sqrt(mean_squared_error(actual, predicted)))
    assert result["MAE"] == pytest.approx(mean_absolute_error(actual, predicted))
    assert result["R2_Score"] == pytest.approx(r2_score(actual, predicted))


@pytest.mark.parametrize("actual,predicted,expected", [([2, 2], [2, 2], 1.), ([2, 2], [1, 1], 0.), ([1, 2, 3], [2, 2, 2], 0.)])
def test_r2_edge_cases(actual, predicted, expected):
    assert RegressionEvaluator.calculate_r2_score(actual, predicted) == expected


@pytest.mark.parametrize("actual,predicted", [([], []), ([1], [1, 2]), ([1, np.nan], [1, 2]), ([[1, 2]], [[1, 2]]), ([1, 2], [np.inf, 2])])
def test_metrics_reject_invalid_vectors(actual, predicted):
    with pytest.raises(ValueError):
        RegressionEvaluator.evaluate_predictions(actual, predicted, "Invalid")


def test_summarize_models():
    results = [RegressionEvaluator.evaluate_predictions([1, 2, 3], [2, 2, 2], "Mean"),
               RegressionEvaluator.evaluate_predictions([1, 2, 3], [1, 2, 3], "Perfect")]
    summary = RegressionEvaluator.summarize_models(results)
    assert summary.columns.tolist() == ["Model", "MSE", "RMSE", "MAE", "R2_Score"]
    assert summary["Model"].tolist() == ["Perfect", "Mean"]
    assert summary.iloc[0]["MSE"] == 0
