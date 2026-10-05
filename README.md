# 🏠 Lab 4: Multilayer Perceptron (MLP) Regression — House Price Prediction

## 1. Mô tả bài toán

Dự đoán giá nhà dựa trên các đặc trưng diện tích, số phòng ngủ, vị trí,... sử dụng mạng nơ-ron đa lớp **Multilayer Perceptron (MLP)** và so sánh với các thuật toán hồi quy truyền thống.

- **Dạng bài toán**: Hồi quy (Regression) — Giá trị liên tục.
- **Dataset**: Boston Housing dataset (13 features đầu vào, 1 biến mục tiêu price/MEDV).
- **Thư viện chính**: TensorFlow / Keras, Scikit-learn, Pandas, NumPy, Matplotlib.

---

## 2. Các yêu cầu cốt lõi (Theo đề bài Lab 4)

1. **Tiền xử lý dữ liệu**:
   - Nạp dữ liệu Boston Housing.
   - Tách đặc trưng (X) và biến mục tiêu (y).
   - Chuẩn hóa đặc trưng bằng StandardScaler.
   - Phân chia tập huấn luyện và kiểm thử (80% Train, 20% Test).
2. **Xây dựng mô hình MLP (TensorFlow/Keras)**:
   - Mô hình Sequential:
     - Layer 1: Dense(32, activation='relu', input_dim=13)
     - Layer 2: Dense(16, activation='relu')
     - Output Layer: Dense(1, activation='linear')
   - Biên dịch (Compile): Optimizer='adam', Loss='mean_squared_error' (MSE), Metrics=['mae'].
3. **Huấn luyện mô hình**:
   - epochs=100, atch_size=32, alidation_data=(X_test, y_test).
   - Theo dõi sự suy giảm của hàm mất mát qua từng epoch (Loss curve).
4. **Đánh giá & Trực quan hóa**:
   - Tính toán các chỉ số: **MSE (Mean Squared Error)**, **MAE (Mean Absolute Error)**, **^2$ Score (R-squared)**.
   - Vẽ biểu đồ phân tán so sánh giá trị thực tế vs giá trị dự đoán (Actual vs Predicted Scatter Plot).
5. **Mở rộng so sánh với Baseline Regression Models (Task 6)**:
   - Huấn luyện: **Linear Regression**, **Decision Tree Regressor**, **Random Forest Regressor**.
   - Tính MSE và ^2$ cho từng mô hình và đối chiếu trực tiếp với mô hình MLP.

---

## 3. Cấu trúc thư mục dự án (src/ layout)

`	ext Lab_4/ ├── main.py                         # File chạy chính điều phối toàn bộ workflow ├── Practice exercise 4.pdf         # Đề bài thực hành gốc ├── requirements.txt                # Thư viện yêu cầu (tensorflow, scikit-learn...) ├── README.md                       # Tài liệu hướng dẫn & mô tả kiến trúc ├── data/                           # Thư mục chứa dữ liệu │   ├── raw/                        # Dữ liệu gốc (boston.csv) │   └── processed/                  # Dữ liệu sau chuẩn hóa & chia tập ├── notebooks/                      # Thử nghiệm tương tác & báo cáo trực quan │   ├── 01_eda_and_preprocessing.ipynb   # Khám phá phân bố giá nhà, correlation │   ├── 02_mlp_model_training.ipynb      # Xây dựng MLP, train, loss curves │   ├── 03_baseline_and_comparison.ipynb # Train LR, Decision Tree, Random Forest & So sánh │   └── README.md ├── src/                            # Toàn bộ mã nguồn chính của dự án │   ├── __init__.py │   ├── config.py                   # Cấu hình hyperparameters (epochs, lr, layers, seed) │   ├── data/                       # Module dữ liệu │   │   ├── __init__.py │   │   ├── loader.py               # Tải dữ liệu Boston Housing │   │   └── preprocessing.py        # StandardScaler, split 80/20 │   ├── features/                   # Module kỹ thuật đặc trưng │   │   ├── __init__.py │   │   └── feature_engineering.py  # Tạo biến tương tác, log transform, đa thức (Polynomial) │   ├── models/                     # Module thuật toán │   │   ├── __init__.py │   │   ├── mlp_model.py            # Khởi tạo kiến trúc Sequential MLP Keras │   │   ├── baseline_models.py      # Linear Regression, Decision Tree, Random Forest │   │   └── persistence.py          # Lưu/tải model Keras (.keras) & Scaler (.joblib) │   ├── evaluation/                 # Module đánh giá │   │   ├── __init__.py │   │   ├── metrics.py              # Tính toán MSE, RMSE, MAE, R2-score │   │   └── visualization.py        # Vẽ Actual vs Predicted, Loss vs Epochs, Bar chart │   └── utils/                      # Tiện ích bổ trợ │       ├── __init__.py │       └── logger.py               # Logger theo dõi tiến trình huấn luyện ├── results/                        # Đầu ra sau khi chạy │   ├── figures/                    # Biểu đồ loss_curve.png, scatter_predictions.png │   ├── metrics/                    # Bảng tổng hợp so sánh kết quả (.csv) │   └── saved_models/               # File trọng số model đã huấn luyện (.keras) └── tests/                          # Kiểm thử tự động (Unit tests)     ├── __init__.py     ├── test_data.py                # Kiểm tra nạp và scale dữ liệu     ├── test_features.py            # Kiểm tra tạo đặc trưng mới, biến đổi log     ├── test_models.py              # Kiểm tra khởi tạo MLP và baselines     └── test_evaluation.py          # Kiểm tra tính toán metrics hồi quy `
