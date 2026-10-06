# 🏠 Lab 4: Multilayer Perceptron (MLP) Regression — House Price Prediction

Dự án xây dựng mô hình mạng nơ-ron đa lớp **Multilayer Perceptron (MLP)** sử dụng **TensorFlow / Keras** cho bài toán hồi quy (Regression) dự đoán giá nhà, đồng thời đối chuẩn hiệu năng toàn diện với 3 thuật toán học máy kinh điển (**Linear Regression**, **Decision Tree Regressor**, **Random Forest Regressor**) theo yêu cầu thực hành **Lab 4 (Practice Exercise 4)**.

---

## 📌 1. Tổng Quan & Cải Tiến Kỹ Thuật

- **Dạng bài toán**: Hồi quy (Regression) — Dự đoán biến mục tiêu liên tục (giá nhà).
- **Bộ dữ liệu chính**: `data/raw/housing.csv` (California Housing Dataset gồm **20,640 mẫu**, 10 thuộc tính; hỗ trợ cơ chế nạp dự phòng Boston Housing qua OpenML/StatLib).
- **Phân chia dữ liệu chuẩn 8:1:1 (Train - Validation - Test)**:
  - **Train set (80% — 16,512 mẫu)**: Dùng để cập nhật trọng số mạng nơ-ron và huấn luyện các mô hình Baseline.
  - **Validation set (10% — 2,064 mẫu)**: Dùng làm `validation_data` giám sát sự suy giảm của hàm mất mát (Loss/MAE) qua từng epoch, phát hiện sớm overfitting.
  - **Test set (10% — 2,064 mẫu)**: Tập kiểm thử hoàn toàn độc lập, chỉ đưa vào đánh giá ở bước cuối cùng nhằm triệt tiêu hoàn toàn nguy cơ **rò rỉ dữ liệu (Data Leakage)**.
- **Chuẩn hóa đặc trưng**: Áp dụng `StandardScaler` theo nguyên tắc nghiêm ngặt: **chỉ fit trên tập Train (80%)**, sau đó áp dụng `transform` cho tập Validation và Test.

---

## 🎯 2. Các Yêu Cầu Cốt Lõi (Theo Đề Bài Lab 4)

1. **Tiền xử lý & Phân chia dữ liệu 8:1:1**:
   - Nạp dữ liệu an toàn, xử lý giá trị khuyết thiếu (NaN) trên `total_bedrooms` bằng phương pháp trung vị (median).
   - Tách ma trận đặc trưng ($X$) và biến mục tiêu ($y$).
   - Phân chia tập dữ liệu 2 bước theo phân tầng phân vị (`pd.qcut`): Train (80%) – Validation (10%) – Test (10%).
   - Chuẩn hóa đặc trưng bằng `StandardScaler`.
2. **Xây dựng mô hình MLP (TensorFlow / Keras Sequential)**:
   - **Layer 1**: `Dense(32, activation='relu', input_dim=...)`
   - **Layer 2**: `Dense(16, activation='relu')`
   - **Output Layer**: `Dense(1, activation='linear')`
   - **Biên dịch (Compile)**: `optimizer='adam'` (learning rate = 0.001), `loss='mean_squared_error'` (MSE), `metrics=['mae']`.
3. **Huấn luyện mô hình**:
   - `epochs=100`, `batch_size=32`, `validation_data=(X_val_scaled, y_val)` (10% Validation).
   - Theo dõi sự suy giảm của hàm mất mát và sai số tuyệt đối qua từng epoch (Learning Curves).
4. **Đánh giá & Trực quan hóa**:
   - Đo lường 4 chỉ số hồi quy: **MSE (Mean Squared Error)**, **RMSE (Root MSE)**, **MAE (Mean Absolute Error)**, và **$R^2$ Score (Hệ số xác định)**.
   - Vẽ biểu đồ phân tán giá thực tế vs giá dự đoán (**Actual vs Predicted Scatter Plot**) kèm đường tham chiếu lý tưởng $y = x$ màu đỏ đứt nét (dpi=300).
   - Vẽ biểu đồ diễn biến tiến trình học tập (**Loss & MAE Curves** qua 100 epochs).
5. **Mở rộng đối chuẩn với Baseline Regression Models (Task 6)**:
   - Huấn luyện: **Linear Regression**, **Decision Tree Regressor** (`max_depth=5`), **Random Forest Regressor** (`n_estimators=100`).
   - Xuất bảng đối chuẩn tổng hợp `results/metrics/model_performance_summary.csv`.
   - Vẽ biểu đồ cột đối chiếu MSE và $R^2$ giữa cả 4 mô hình.
   - Demo suy luận dự đoán giá nhà trên 3 mẫu thực tế mới.

---

## 👥 3. Phân Công Trách Nhiệm Thành Viên (3 Tasks)

| Giai đoạn | Trọng tâm công việc | Thành viên phụ trách | Module mã nguồn phụ trách |
| :--- | :--- | :--- | :--- |
| **TASK 1** | Dữ liệu, Khám phá EDA, Tiền xử lý, Chia tập 8:1:1 & Kỹ thuật đặc trưng | **Huyền & My** | `src/config.py`, `src/data/loader.py`, `src/data/preprocessing.py`, `src/features/feature_engineering.py`, `notebooks/01_eda_and_preprocessing.ipynb`, `tests/test_data.py`, `tests/test_features.py` |
| **TASK 2** | Kiến trúc mạng MLP Keras, Huấn luyện 100 Epochs, Baseline Models (Task 6) & Lưu trữ | **Tường & Trí** | `src/models/mlp_model.py`, `src/models/baseline_models.py`, `src/models/persistence.py`, `notebooks/02_mlp_model_training.ipynb`, `tests/test_models.py` |
| **TASK 3** | Đánh giá đa chiều, Trực quan hóa đồ thị (dpi=300), Hệ thống logging & Điều phối `main.py` | **Tiến & Thịnh** | `src/evaluation/metrics.py`, `src/evaluation/visualization.py`, `src/utils/logger.py`, `main.py`, `notebooks/03_baseline_and_comparison.ipynb`, `tests/test_evaluation.py` |

---

## 📁 4. Cấu Trúc Thư Mục Dự Án

```text
Lab_4/
├── main.py                         # File chạy chính điều phối liên hoàn 8 bước của pipeline
├── Practice exercise 4.pdf         # Đề bài thực hành gốc
├── requirements.txt                # Danh sách thư viện yêu cầu (tensorflow, scikit-learn...)
├── README.md                       # Tài liệu hướng dẫn & mô tả kiến trúc hệ thống
├── data/                           # Quản lý dữ liệu
│   ├── raw/                        # Dữ liệu gốc (housing.csv - 20,640 mẫu)
│   └── processed/                  # Dữ liệu sau chia tập 8:1:1 (train.csv, val.csv, test.csv)
├── notebooks/                      # Sổ tay thực nghiệm tương tác Jupyter
│   ├── 01_eda_and_preprocessing.ipynb   # Khám phá EDA, xử lý khuyết thiếu, chia tập 8:1:1
│   ├── 02_mlp_model_training.ipynb      # Xây dựng MLP Sequential Keras, train 100 epochs
│   ├── 03_baseline_and_comparison.ipynb # Huấn luyện LR, Decision Tree, Random Forest & so sánh
│   └── README.md                   # Hướng dẫn sử dụng các sổ tay
├── src/                            # Mã nguồn module hóa chính của dự án
│   ├── __init__.py
│   ├── config.py                   # Cấu hình tập trung (paths, hyperparams, split 8:1:1)
│   ├── data/                       # Module dữ liệu
│   │   ├── __init__.py
│   │   ├── loader.py               # Nạp dữ liệu an toàn đa nguồn
│   │   └── preprocessing.py        # Xử lý khuyết thiếu, chia 8:1:1, StandardScaler
│   ├── features/                   # Module kỹ thuật đặc trưng
│   │   ├── __init__.py
│   │   └── feature_engineering.py  # Biến đổi Log1p, tương tác đặc trưng kinh tế
│   ├── models/                     # Module mô hình học máy & mạng nơ-ron
│   │   ├── __init__.py
│   │   ├── mlp_model.py            # Khởi tạo & huấn luyện Sequential MLP Keras
│   │   ├── baseline_models.py      # Bộ 3 mô hình Baseline Scikit-learn (Task 6)
│   │   └── persistence.py          # Lưu/tải model (.keras, .joblib) & scaler (.joblib)
│   ├── evaluation/                 # Module đánh giá & trực quan hóa
│   │   ├── __init__.py
│   │   ├── metrics.py              # Tính toán 4 chỉ số MSE, RMSE, MAE, R2-score
│   │   └── visualization.py        # Vẽ Actual vs Predicted, Loss curves, Bar chart
│   └── utils/                      # Tiện ích bổ trợ
│       ├── __init__.py
│       └── logger.py               # Logger ghi console/file & decorator đo thời gian @log_step
├── results/                        # Thư mục chứa kết quả xuất bản
│   ├── figures/                    # Đồ thị xuất bản (loss_curve.png, actual_vs_predicted.png...)
│   ├── metrics/                    # Bảng tổng hợp đối chuẩn (model_performance_summary.csv)
│   └── saved_models/               # Trọng số mô hình đã huấn luyện (.keras, .joblib)
└── tests/                          # Kiểm thử tự động (Unit tests)
    ├── __init__.py
    ├── test_data.py                # Kiểm thử nạp dữ liệu & chia tập chuẩn 8:1:1
    ├── test_features.py            # Kiểm thử biến đổi log1p & tạo biến tương tác
    ├── test_models.py              # Kiểm thử kiến trúc MLP, shapes & baseline fit/predict
    └── test_evaluation.py          # Kiểm thử tính đúng đắn toán học của MSE, RMSE, MAE, R2
```

---

## ⚡ 5. Hướng Dẫn Cài Đặt & Thực Thi

### 1. Cài đặt môi trường
Khuyến nghị tạo môi trường ảo Python 3.10+ và cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

### 2. Thực thi toàn bộ pipeline tự động (Một chạm qua `main.py`)
```bash
python main.py
```
*Lệnh trên sẽ tự động thực hiện liên hoàn 8 bước:*
1. Nạp cấu hình hệ thống và khởi tạo seed ngẫu nhiên (`random_state=42`).
2. Nạp dữ liệu và kiểm tra cấu trúc schema.
3. Chia tập theo tỷ lệ **8:1:1** và chuẩn hóa `StandardScaler` (chống data leakage).
4. Huấn luyện mô hình **MLP Keras (100 Epochs)** với tập Validation (10%).
5. Huấn luyện **3 mô hình Baseline** (Linear Regression, Decision Tree, Random Forest).
6. Đánh giá đa chiều trên **tập Test độc lập (10%)** (MSE, RMSE, MAE, $R^2$).
7. Xuất bảng đối chuẩn CSV và vẽ các biểu đồ xuất bản chất lượng cao (dpi=300).
8. Demo suy luận dự đoán giá nhà trên 3 mẫu thực tế mới.

### 3. Khởi chạy sổ tay tương tác (Jupyter Notebooks)
```bash
jupyter notebook notebooks/01_eda_and_preprocessing.ipynb
```

### 4. Chạy kiểm thử tự động (Unit Tests)
```bash
pytest tests/ -v
```

---

## 📊 6. Tiêu Chuẩn Nghiệm Thu Kỹ Thuật (Checklist)

- [x] **Dữ liệu & Phân chia 8:1:1**: Chia chính xác 80% Train, 10% Validation, 10% Test; không rò rỉ dữ liệu qua `StandardScaler`.
- [x] **Kiến trúc MLP chuẩn Keras**: Đủ 3 lớp Sequential (Dense 32 ReLU $\rightarrow$ Dense 16 ReLU $\rightarrow$ Dense 1 Linear), compile `adam` & `mse`.
- [x] **Huấn luyện 100 Epochs**: Theo dõi sát sao biến thiên loss và MAE trên cả 2 tập Train và Validation.
- [x] **Bộ 3 Mô hình Baseline (Task 6)**: Tích hợp đầy đủ Linear Regression, Decision Tree, Random Forest.
- [x] **Đánh giá đa chiều**: Tính chuẩn xác 4 chỉ số (MSE, RMSE, MAE, $R^2$) và xuất file `model_performance_summary.csv`.
- [x] **Trực quan hóa chuẩn mực (dpi=300)**: Scatter plot có đường chéo $y=x$, Loss Learning Curves, Bar chart so sánh.
- [x] **Sổ tay tương tác (Jupyter)**: Cả 3 notebook đều tuân thủ chuẩn JSON v4, phân chia rõ ràng cell mô tả và cell code logic.
