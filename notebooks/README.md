# 📓 Sổ Tay Thực Nghiệm (Notebooks) — Lab 4: House Price Prediction

Thư mục này chứa 3 Jupyter Notebooks tương tác phục vụ từng giai đoạn thực nghiệm của đồ án theo quy trình chia dữ liệu chuẩn **8:1:1 (Train 80% - Validation 10% - Test 10%)**:

---

## 📑 Danh mục Notebooks

### 1. `01_eda_and_preprocessing.ipynb`
- **Phụ trách**: Bạn Huyền & Bạn My (Task 1).
- **Mục tiêu**:
  - Nạp dữ liệu an toàn từ `data/raw/housing.csv` (20,640 mẫu).
  - Khám phá dữ liệu (EDA), phân tích phân bố biến mục tiêu `median_house_value`, ma trận tương quan Pearson.
  - Xử lý giá trị khuyết thiếu (NaN) trên cột `total_bedrooms`.
  - Phân chia tập dữ liệu theo tỷ lệ chuẩn **8:1:1** (`Train: 16,512 mẫu (80%)`, `Validation: 2,064 mẫu (10%)`, `Test: 2,064 mẫu (10%)`) có phân tầng `qcut`.
  - Chuẩn hóa đặc trưng bằng `StandardScaler` (chỉ fit trên tập Train để chống rò rỉ dữ liệu).

---

### 2. `02_mlp_model_training.ipynb`
- **Phụ trách**: Bạn Tường & Bạn Trí (Task 2).
- **Mục tiêu**:
  - Xây dựng mô hình mạng nơ-ron đa lớp **Sequential Multilayer Perceptron (MLP)** Keras:
    - Layer 1: `Dense(32, activation='relu', input_dim=...)`
    - Layer 2: `Dense(16, activation='relu')`
    - Output: `Dense(1, activation='linear')`
  - Biên dịch với `optimizer='adam'` và `loss='mean_squared_error'`, metrics `['mae']`.
  - Huấn luyện qua **100 epochs**, batch size 32 với `validation_data=(X_val, y_val)` (10% Val).
  - Trực quan hóa diễn biến học tập (Learning Curves): Loss (MSE) và MAE qua các epoch.
  - Lưu trữ artifact mô hình vào `results/saved_models/mlp_house_price_model.keras`.

---

### 3. `03_baseline_and_comparison.ipynb`
- **Phụ trách**: Bạn Tiến & Bạn Thịnh (Task 3) phối hợp Tường & Trí (Task 2).
- **Mục tiêu**:
  - Huấn luyện 3 mô hình Baseline Scikit-learn (Task 6):
    1. Linear Regression
    2. Decision Tree Regressor (`max_depth=5`)
    3. Random Forest Regressor (`n_estimators=100`)
  - Dự đoán trên tập kiểm thử độc lập **10% Test set** cho cả 4 mô hình.
  - Đánh giá toàn diện 4 chỉ số: **MSE**, **RMSE**, **MAE**, và **$R^2$ Score**.
  - Xuất bảng đối chuẩn tổng hợp `results/metrics/model_performance_summary.csv`.
  - Vẽ biểu đồ chất lượng cao (dpi=300):
    - Actual vs Predicted Scatter Plot có đường chéo lý tưởng $y = x$.
    - Model Comparison Bar Chart so sánh MSE và $R^2$.
    - Phân bố sai số phần dư (Residuals).
  - Demo dự đoán thực tế trên 3 mẫu nhà mới.

---

## 🚀 Hướng Dẫn Khởi Chạy
Bạn có thể mở và chạy các sổ tay này trong môi trường Jupyter Lab / Jupyter Notebook hoặc VS Code:
```bash
jupyter notebook notebooks/01_eda_and_preprocessing.ipynb
```