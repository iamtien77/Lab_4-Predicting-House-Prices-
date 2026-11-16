# Hướng dẫn notebook — Lab 4

Ba notebook là chuỗi thực nghiệm riêng, dùng chung `src/`. **Bảng, biểu đồ và nhận xét nằm trong output của chính notebook**, không ghi vào `results/` hoặc `runs/` của `main.py`.

## Thứ tự chạy

| Notebook | Nội dung |
| --- | --- |
| `01_eda_and_preprocessing.ipynb` | CSV, split, EDA train, fit preprocessing |
| `02_mlp_model_training.ipynb` | Baseline trước MLP, nhánh features trên validation, learning curves |
| `03_baseline_and_comparison.ipynb` | Test metrics, residuals, lỗi theo nhóm và inference |

Chọn interpreter `.venv_lab`, dùng **Restart & Run All** lần lượt 01 → 02 → 03, rồi lưu từng `.ipynb`. Ngày 16/11/2026 đã chạy toàn bộ ba notebook và lưu output thành công. Không chỉ chạy cell đánh giá sau khi đổi preprocessing hoặc dữ liệu.

## Dữ liệu và cấu hình

CSV tại `data/raw/housing.csv`; target `median_house_value` theo USD. Split 80/10/10, seed 42; mặc định phân tầng target bằng 10 quantile bins khi tách train/temp rồi 5 bins khi tách validation/test. Không phân tầng theo thu nhập. Ranh giới trùng nhau có thể giảm số bins.

Preprocessing chỉ fit train. Mặc định có 13 features với dữ liệu hiện tại. Notebook 01 có `APPLY_LOG`, `CREATE_INTERACTIONS`; khi bật interactions, số chiều lấy từ fitted preprocessor. Với 5 nhãn one-hot và 3 ratios, số chiều là 16.

Notebook 02 ghi các overrides: cây sâu 5, rừng 100 cây, MLP 100 epochs, batch 32, learning rate 0,001. Nhánh ratios + log chỉ đối chiếu Linear Regression trên validation, không thay cấu hình chuẩn. Đổi default `src/config.py` không đổi các overrides trong cell.

MLP: 32 ReLU → 16 ReLU → 1 linear, Adam/MSE; scaling target chỉ từ train. History báo MSE USD², MAE USD. Chạy đủ epochs rồi khôi phục lowest validation MSE checkpoint, không early stopping. Test chỉ đánh giá sau training.

## Cache bàn giao riêng

Tất cả cache ở `notebooks/artifacts/`:

| File | Vai trò |
| --- | --- |
| `train.csv`, `val.csv`, `test.csv` | Raw splits, giữ missing features và USD |
| `preprocessor.joblib` | Bộ biến đổi fit train |
| `preparation_manifest.json` | Seed, tùy chọn features, raw hash và cache hashes |
| `baseline_models.joblib`, `mlp_house_price_model.pt` | Mô hình đã train |
| `training_history.json` | History bàn giao, không phải báo cáo độc lập |
| `training_manifest.json` | Hash preparation manifest, tham số training, model/history hashes |

Notebook 02/03 kiểm tra hashes trước khi dùng cache. Nếu raw data, preprocessing hoặc model đổi, chạy lại đúng thứ tự. Không chép artifacts CLI vào cache notebook. Nếu chạy bị ngắt, phải xác minh cache trước khi tiếp tục.

## Kết quả và giới hạn

Output gồm validation/test tables, learning curves, actual-vs-predicted, residual histogram, số prediction âm, MAE capped/uncapped và các nhóm `ocean_proximity`. Demo dùng ba dòng test, không phải dữ liệu mới độc lập.

Nhóm `ISLAND` ít mẫu, target có trần và random split không đo generalization theo vùng/thời điểm mới. Không chọn lại model theo test. README chính chỉ trích số liệu CLI kèm run ID; output notebook là nguồn riêng khi trình bày luồng notebook.
