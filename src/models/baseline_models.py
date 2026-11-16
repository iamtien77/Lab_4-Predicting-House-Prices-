"""Train the three reference regression models on a shared feature matrix."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from src.config import config


class BaselineRegressionSuite:
    """Provide ordered training and prediction for the baseline models."""

    def __init__(self, random_state=None, dt_max_depth=None, rf_n_estimators=None, fit_intercept=None):
        random_state = config.RANDOM_STATE if random_state is None else random_state
        dt_max_depth = config.DT_MAX_DEPTH if dt_max_depth is None else dt_max_depth
        rf_n_estimators = config.RF_N_ESTIMATORS if rf_n_estimators is None else rf_n_estimators
        fit_intercept = config.LR_FIT_INTERCEPT if fit_intercept is None else fit_intercept
        self.models = {
            "Linear Regression": LinearRegression(fit_intercept=fit_intercept),
            "Decision Tree": DecisionTreeRegressor(max_depth=dt_max_depth, random_state=random_state),
            "Random Forest": RandomForestRegressor(n_estimators=rf_n_estimators, random_state=random_state, max_features=1.0),
        }
        self.is_fitted = False

    def fit_all(self, X_train, y_train):
        """Fit models in baseline order and return the fitted model mapping."""
        X_train, y_train = check_X_y(X_train, y_train, y_numeric=True)
        self.is_fitted = False
        for name, model in self.models.items():
            print(f"Đang huấn luyện mô hình {name}...")
            model.fit(X_train, y_train)
        self.is_fitted = True
        return self.models

    def predict_all(self, X):
        """Return one finite prediction vector per fitted model."""
        if not self.is_fitted:
            raise ValueError("Baseline models must be fitted before prediction.")
        X = check_array(X)
        predictions = {}
        for name, model in self.models.items():
            check_is_fitted(model)
            prediction = np.asarray(model.predict(X)).reshape(-1)
            if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
                raise ValueError(f"Invalid predictions from {name}.")
            predictions[name] = prediction
        return predictions

    def get_model(self, name):
        """Return a named estimator; unknown names raise KeyError."""
        return self.models[name]
