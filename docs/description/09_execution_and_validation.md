# 09 — Thực thi, xác minh và bàn giao

## 1. Mục tiêu và môi trường

Người học cần chạy đúng một luồng thí nghiệm, tìm được outputs và nêu chính xác phần đã kiểm chứng. Input gồm source, CSV và dependencies; output là run hoặc notebook outputs có nguồn, cấu hình và kết quả rõ ràng.

Theo README dự án, dùng Python 3.10 trở lên. Các dependencies nằm trong [requirements.txt](../../requirements.txt); chúng đặt phiên bản tối thiểu, chưa phải lockfile. Manifest ghi phiên bản thực tế của numpy, pandas, torch và scikit-learn cho từng run; không coi đó là phiên bản đang cài nếu chưa kiểm tra.

## 2. Chuẩn bị và chạy CLI

Các lệnh PowerShell sau dành cho người đọc, chạy từ thư mục gốc:

```powershell
python -m venv .venv_lab
.\.venv_lab\Scripts\python.exe -m pip install -r requirements.txt
.\.venv_lab\Scripts\python.exe -X utf8 main.py --help
.\.venv_lab\Scripts\python.exe -X utf8 main.py
```

Nếu môi trường đã tồn tại, dùng môi trường đó thay vì tạo lại. CSV phải có sẵn ở data/raw/housing.csv; loader không tự tải từ Kaggle. Flag -X utf8 hỗ trợ console tiếng Việt.

Smoke run để kiểm tra luồng, không dùng như benchmark 100 epochs:

```powershell
.\.venv_lab\Scripts\python.exe -X utf8 main.py --epochs 5 --skip-plots
```

Thí nghiệm features riêng:

```powershell
.\.venv_lab\Scripts\python.exe -X utf8 main.py --apply-log --create-interactions
```

| Flag | Vai trò |
| --- | --- |
| --epochs, --batch-size, --lr | Ghi đè tham số huấn luyện MLP |
| --random-state | Seed |
| --data-path | CSV đầu vào |
| --output-dir | Root chứa runs và results/latest_run.json |
| --skip-plots | Không xuất PNG trong run này |
| --apply-log | Bật log1p cho bốn cột quy mô |
| --create-interactions | Bật ba ratios |

Không có flag --quick-test. Split 80/10/10, kiến trúc 32–16–1 và Adam/MSE là lựa chọn cố định trong implementation hiện tại. Chạy CLI có thể cập nhật latest pointer; khi viết báo cáo phải chỉ định run tham chiếu, không mặc định latest luôn là benchmark chính.

## 3. Chạy notebook theo thứ tự

1. Chọn kernel .venv_lab.
2. Restart & Run All notebook 01: dữ liệu, split, EDA, preprocessing.
3. Lưu notebook 01; Restart & Run All notebook 02: baseline, nhánh features, MLP.
4. Lưu notebook 02; Restart & Run All notebook 03: test evaluation, nhóm lỗi, inference.
5. Lưu notebook 03 và đọc các output/nhận xét.

Nguồn: [notebooks/README.md](../../notebooks/README.md). Bảng và hình nằm trong output .ipynb; cache bàn giao ở notebooks/artifacts. Preparation/training manifests kiểm tra sự nhất quán cache. Đổi dữ liệu hoặc preprocessing thì chạy lại chuỗi theo thứ tự, không chép model CLI vào cache notebook.

Notebook có overrides trong cell; đổi src/config.py không tự thay các giá trị này. Đặc biệt không trộn số liệu notebook với hình CLI trong báo cáo.

## 4. Kiểm tra outputs và hình

| Giai đoạn | Bằng chứng | Nơi đọc phân tích |
| --- | --- | --- |
| Data/split/EDA | Raw splits, thống kê, 5 EDA PNG | [Chương 02](02_data_understanding.md), [03](03_data_splitting_and_eda.md) |
| Preprocessing | Fitted preprocessor, feature_names, shape | [Chương 04](04_preprocessing_and_features.md) |
| Baseline | Ba predictors và validation metrics | [Chương 05](05_baseline_models.md) |
| MLP training | Checkpoint, history, loss_curve.png | [Chương 06](06_mlp_training.md) |
| Test evaluation | Metrics CSV, 1 comparison + 8 scatter/residual PNG | [Chương 07](07_evaluation_and_error_analysis.md) |
| Save/load/inference | Manifest và sample_predictions.csv | [Chương 08](08_artifacts_and_inference.md) |

Run tham chiếu có 15 PNG: 5 EDA + 1 learning curve + 1 comparison + 4 actual-vs-predicted + 4 residual histograms. Chương không có kết quả ảnh riêng dùng bảng/code để giải thích; không tạo ảnh minh họa rồi trình bày như artifact thực nghiệm.

## 5. Tests phần mềm và ý nghĩa

Lệnh chuẩn:

```powershell
.\.venv_lab\Scripts\python.exe -X utf8 -m pytest -q
```

Nếu cần basetemp riêng, dùng đường dẫn mới dành riêng cho tests:

```powershell
.\.venv_lab\Scripts\python.exe -X utf8 -m pytest -q --basetemp=.test_artifacts/description_validation -o cache_dir=.test_artifacts/cache
```

Pytest có thể dọn nội dung basetemp; không dùng thư mục chứa kết quả cần giữ.

| Nhóm tests | Điều được kiểm tra |
| --- | --- |
| test_data.py | Loader, split, preprocessing và dữ liệu đầu vào |
| test_features.py | Ratios, log và điều kiện hợp lệ |
| test_models.py | Baseline, MLP, scaling, training/prediction |
| test_evaluation.py | Metrics và kiểm tra inputs |
| test_visualization.py | Hàm vẽ và đầu ra hình |
| test_pipeline.py, test_review_regressions.py | Luồng tích hợp, persistence và các hành vi regression |

README ghi 53 tests pass ngày 16/11/2026. **Lần hoàn thiện tài liệu này không chạy lại pytest hoặc training**, nên không báo kết quả lịch sử như kiểm thử mới. Tests kiểm tra hành vi phần mềm; không chứng minh chất lượng ngoài phân phối dữ liệu.

## 6. Bằng chứng kiểm chứng lần hoàn thiện tài liệu

Run: `20261116T113320345877Z_6c8931e5`. Kiểm tra ngày 24/11/2026:

| Kiểm tra | Kết quả | Giới hạn |
| --- | --- | --- |
| Hash artifacts theo manifest | 25/25 khớp qua load_cli_run | Hash so với manifest, không xác thực nguồn bên ngoài |
| Raw CSV fingerprint | Khớp | Không chứng minh bản Kaggle tải mới sẽ giống byte-for-byte |
| Source/dependency hashes | 10/18 khớp | Không khẳng định source hiện tại tái tạo training lịch sử |
| Reload model/preprocessor | Thành công | Chỉ bundle tham chiếu |
| Metrics test bốn models | Khớp CSV trong rtol=1e-5, atol=1e-3 | Không phải training mới hoặc holdout mới |
| Ba sample predictions | Khớp CSV trong cùng tolerance | Chỉ ba mẫu demo test |
| Thống kê dữ liệu và ảnh | Đã đọc CSV và xem trực tiếp các hình được phân tích | Kết luận giới hạn trong run và dữ liệu này |

Tám file source khác hash lưu trong run: main.py, src/evaluation/metrics.py, src/evaluation/visualization.py, src/models/__init__.py, src/models/baseline_models.py, src/models/mlp_model.py, src/models/persistence.py, src/utils/logger.py. Việc chuẩn hóa CRLF sang LF không làm các hash này khớp; chưa kết luận nguyên nhân hoặc ảnh hưởng cụ thể chỉ từ hash.

Có thể chạy kiểm tra artifacts chỉ đọc, không training:

```python
from pathlib import Path
from src.models.persistence import verify_cli_run

run_dir = Path("runs/20261116T113320345877Z_6c8931e5")
manifest = verify_cli_run(run_dir)
print(manifest["run_id"], len(manifest["files"]))
```

verify_cli_run kiểm tra hashes artifacts; không tự so sánh source hiện tại với source_sha256. Đó là kiểm tra bổ sung riêng trong đợt rà soát này.

## 7. Tái lập và các tình huống lỗi

Muốn tái lập, ghi đầy đủ data fingerprint, source version, dependency versions, seed, CLI arguments và run ID. Seed không bảo đảm bit-for-bit giữa các môi trường khác nhau. Muốn so sánh cấu hình mới, giữ quy trình validation riêng; không lặp tối ưu theo test hiện có.

| Tình huống | Việc cần làm |
| --- | --- |
| Không tìm thấy CSV | Kiểm tra cwd/data-path và schema |
| Import error | Kiểm tra đúng interpreter và requirements |
| Hash cache notebook không khớp | Chạy lại các notebook theo thứ tự |
| Thiếu PNG | Kiểm tra skip_plots và trạng thái run |
| Metrics khác báo cáo | Kiểm tra run ID, seed, features, source và môi trường |
| Run lỗi giữa chừng | Đọc log và giữ latest run thành công; không giả định thư mục tồn tại nghĩa là complete |

## 8. Phân tích kết quả đạt được theo 5W1H

| Thành phần | Kết quả |
| --- | --- |
| What | Có hướng dẫn CLI/notebook, mapping outputs, kiểm chứng bundle và giới hạn source mismatch. |
| Why | Người đọc phân biệt chạy được, lưu nhất quán và dự đoán tốt; ba tiêu chí cần bằng chứng khác nhau. |
| Who | Người thực hiện lab, người nhận bàn giao và người kiểm tra báo cáo. |
| When | Artifacts lịch sử 16/11/2026; rà soát tài liệu/bundle ngày 24/11/2026. |
| Where | Code, requirements, tests, notebooks và manifest/run được liên kết. |
| How | Đối chiếu code, CSV, ảnh, hashes và reload predictions; không training lại. |

## 9. Checklist bàn giao

- [x] Bộ tài liệu có đường dẫn code/artifacts và phân biệt CLI/notebook.
- [x] Nêu công thức, quy trình, kết quả và giới hạn theo từng giai đoạn.
- [x] Mọi hình của run tham chiếu có nơi phân tích hoặc liên kết tương ứng.
- [x] Tách kết quả lịch sử khỏi kiểm chứng mới.
- [x] Ghi source mismatch; không tuyên bố tái lập training hoàn toàn.
- [ ] Tái huấn luyện và chạy lại toàn tests trên source hiện tại: chưa thực hiện, ngoài phạm vi hoàn thiện tài liệu.

[Quay lại mục lục](README.md)
