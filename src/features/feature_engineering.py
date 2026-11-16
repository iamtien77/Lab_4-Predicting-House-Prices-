"""Mô-đun tạo đặc trưng bổ sung cho dữ liệu nhà ở sau khi đã điền khuyết tật trên tập Train."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from src.config import config


class HousingFeatureEngineer(TransformerMixin, BaseEstimator):
    """
    Bộ tạo đặc trưng nhà ở:
    - Tính toán các tỷ số đặc trưng trước khi áp dụng biến đổi log.
    - Hoàn toàn độc lập, không đọc hay sử dụng biến mục tiêu y.
    """

    def __init__(self, apply_log=False, create_interactions=False, skewed_cols=None):
        self.apply_log = apply_log
        self.create_interactions = create_interactions
        self.skewed_cols = skewed_cols

    def _validate(self, X):
        """Kiểm tra tính hợp lệ của DataFrame số trước khi tiến hành tính toán đặc trưng."""
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("Expected a non-empty pandas DataFrame.")
        if not X.columns.is_unique or set(X.columns) != set(config.NUMERIC_FEATURE_NAMES):
            raise ValueError("Expected exactly the configured numeric features.")
        if any(not pd.api.types.is_numeric_dtype(X[c]) for c in X):
            raise ValueError("Feature engineering requires numeric columns.")
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Impute missing values before feature engineering.")

    def fit(self, X, y=None):
        """
        Ghi nhận cấu trúc cột đầu vào và danh sách đặc trưng đầu ra.
        Không sử dụng biến y và không học bất kỳ tham số nào từ tập đánh giá.
        """
        self._validate(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.log_columns_ = list(self.skewed_cols) if self.skewed_cols is not None else [
            "total_rooms", "total_bedrooms", "population", "households"
        ]
        if self.apply_log and not set(self.log_columns_).issubset(X.columns):
            raise ValueError("Log columns must be numeric input features.")
        if len(set(self.log_columns_)) != len(self.log_columns_):
            raise ValueError("Log columns must be unique.")
        self.output_features_ = list(X.columns)
        if self.create_interactions:
            self.output_features_ += ["rooms_per_household", "bedrooms_ratio", "people_per_household"]
        self.transform(X)
        return self

    def transform(self, X):
        """
        Biến đổi và trả về bản sao dữ liệu chứa các đặc trưng mới.
        Bảo toàn tên cột cố định và giữ nguyên chỉ mục (index) gốc của dữ liệu.
        """
        check_is_fitted(self, "output_features_")
        self._validate(X)
        result = X.loc[:, self.feature_names_in_].copy()

        # Tính các tỷ số đặc trưng từ dữ liệu gốc đã được điền khuyết, trước khi áp dụng log
        if self.create_interactions:
            if (result[["households", "total_rooms"]] <= 0).any().any():
                raise ValueError("Ratio denominators households and total_rooms must be positive.")
            result["rooms_per_household"] = result["total_rooms"] / result["households"]
            result["bedrooms_ratio"] = result["total_bedrooms"] / result["total_rooms"]
            result["people_per_household"] = result["population"] / result["households"]

        # Áp dụng biến đổi log1p cho các đặc trưng bị lệch phân phối nếu được kích hoạt
        if self.apply_log:
            if (result[self.log_columns_] <= -1).any().any():
                raise ValueError("log1p inputs must be greater than -1.")
            result[self.log_columns_] = np.log1p(result[self.log_columns_])

        if not np.isfinite(result.to_numpy(dtype=float)).all():
            raise ValueError("Feature engineering produced non-finite values.")

        return result.loc[:, self.output_features_]

    def get_feature_names_out(self, input_features=None):
        """Cung cấp danh sách tên các đặc trưng đầu ra cho sklearn Pipeline và ColumnTransformer."""
        check_is_fitted(self, "output_features_")
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("Input feature names do not match the fitted schema.")
        return np.asarray(self.output_features_, dtype=object)
