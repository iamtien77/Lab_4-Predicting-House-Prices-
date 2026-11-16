"""Compute regression metrics in original target units."""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


class RegressionEvaluator:
    """Validate prediction pairs and produce comparable model summaries."""

    @staticmethod
    def _validate_pair(y_true, y_pred):
        arrays = []
        for values in (y_true, y_pred):
            values = np.asarray(values, dtype=np.float64)
            if values.ndim == 2 and values.shape[1] == 1:
                values = values[:, 0]
            if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
                raise ValueError("Targets and predictions must be nonempty finite vectors.")
            arrays.append(values)
        if arrays[0].shape != arrays[1].shape:
            raise ValueError("Targets and predictions must have matching shapes.")
        return arrays

    @staticmethod
    def calculate_mse(y_true, y_pred):
        y_true, y_pred = RegressionEvaluator._validate_pair(y_true, y_pred)
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def calculate_rmse(y_true, y_pred):
        return float(np.sqrt(RegressionEvaluator.calculate_mse(y_true, y_pred)))

    @staticmethod
    def calculate_mae(y_true, y_pred):
        y_true, y_pred = RegressionEvaluator._validate_pair(y_true, y_pred)
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def calculate_r2_score(y_true, y_pred):
        y_true, y_pred = RegressionEvaluator._validate_pair(y_true, y_pred)
        return float(r2_score(y_true, y_pred, force_finite=True))

    @classmethod
    def evaluate_predictions(cls, y_true, y_pred, model_name):
        """Retain full precision so sorting does not depend on rounding."""
        return {"Model": model_name, "MSE": cls.calculate_mse(y_true, y_pred),
                "RMSE": cls.calculate_rmse(y_true, y_pred),
                "MAE": cls.calculate_mae(y_true, y_pred),
                "R2_Score": cls.calculate_r2_score(y_true, y_pred)}

    @classmethod
    def summarize_models(cls, evaluation_results):
        columns = ["Model", "MSE", "RMSE", "MAE", "R2_Score"]
        result = pd.DataFrame(evaluation_results, columns=columns)
        return result.sort_values("R2_Score", ascending=False, kind="stable").reset_index(drop=True)
