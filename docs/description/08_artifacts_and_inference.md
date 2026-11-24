# 08 — Artifacts, lưu/nạp model và inference

## 1. Mục tiêu và hợp đồng bàn giao

Một mô hình dùng được cần đi cùng đúng fitted preprocessor, schema đầu vào, thống kê target scaling, cấu hình và nguồn kết quả. Chỉ lưu weights mà bỏ preprocessing có thể dẫn đến cùng số chiều nhưng sai ý nghĩa features.

**Input:** fitted preprocessor, ba baseline, MLP checkpoint và outputs của một lần chạy. **Output:** thư mục run độc lập, manifest, con trỏ latest và khả năng reload dự đoán USD.

Chương này mô tả [persistence.py](../../src/models/persistence.py) và [main.py](../../main.py). Kết quả tham chiếu thuộc run `20261116T113320345877Z_6c8931e5`.

## 2. Cấu trúc artifacts

Các đường dẫn sau tương đối với thư mục run:

| Artifact | Nội dung | Cách sử dụng |
| --- | --- | --- |
| data/processed/train.csv, val.csv, test.csv | Raw splits, target USD và missing features còn nguyên | Audit/reload đúng dữ liệu; không phải ma trận đã scale |
| results/saved_models/preprocessor.joblib | Imputers, feature engineer, scaler, category vocabulary | Chỉ transform dữ liệu mới |
| results/saved_models/baseline_models.joblib | Dictionary đủ ba fitted predictors | Dự đoán từ cùng ma trận features |
| results/saved_models/mlp_house_price_model.pt | Metadata kiến trúc, best epoch/loss, state_dict và target buffers | Dựng lại MLP, load weights trên CPU |
| results/metrics/training_history.csv | MSE/MAE train/validation theo epoch | Đọc learning curves |
| results/metrics/validation_summary.csv | Bốn metrics trên validation | Đối chiếu checkpoint/model |
| results/metrics/model_performance_summary.csv | Bốn metrics trên test | Báo cáo đánh giá cuối |
| results/metrics/sample_predictions.csv | Ba dòng test minh họa sau reload | So sánh actual/predicted/absolute error |
| results/figures/ | 15 PNG nếu không skip plots | EDA, training và đánh giá |
| logs/training.log | Log điều phối | Xem diễn biến run |
| manifest.json | Hashes, parameters, versions, feature names, trạng thái | Xác minh tính nhất quán |

Manifest hiện hash các file trong data/processed và results, **không bao gồm training.log**. Không tuyên bố mọi file trong run đều được kiểm tra hash.

## 3. Các bước lưu và công bố run

1. Tạo run ID từ UTC timestamp và chuỗi ngẫu nhiên; tạo thư mục mới, không ghi đè run cũ.
2. Lưu splits, preprocessor và baseline sau fit; lưu MLP sau restore best validation checkpoint.
3. Lưu history, validation/test metrics và figures nếu bật.
4. Nạp lại MLP/preprocessor, dự đoán ba dòng test và so với predictions trước reload.
5. Kiểm tra raw data và danh sách source hashes không đổi trong lúc chạy.
6. Ghi manifest: trạng thái complete, cấu hình có hiệu lực, data/source fingerprints, package versions và artifact hashes.
7. Xác minh manifest/artifacts; chỉ sau đó thay thế nguyên tử results/latest_run.json.

Nếu training lỗi trước publish, latest không được cập nhật; thư mục run chưa hoàn tất có thể vẫn tồn tại. Latest có nghĩa **run hoàn tất mới nhất**, không phải run có RMSE thấp nhất. Một smoke run ít epochs cũng có thể trở thành latest.

## 4. Cơ chế load và những kiểm tra thực hiện

`load_cli_run(output_root)` đọc latest pointer, kiểm tra hash manifest, rồi kiểm tra hashes artifacts trước khi deserialize. Khi truyền `run_dir` cụ thể, hàm dùng run đó và verify artifacts, không dùng latest pointer.

Loader giữ đường dẫn trong output_root/runs; kiểm tra preprocessor fitted, baseline mapping đủ ba tên, checkpoint đúng schema/kiến trúc, weights hữu hạn và target_scale dương. Số features của MLP/baselines phải khớp fitted preprocessor.

MLP dùng `torch.load(..., weights_only=True, map_location="cpu")`; joblib vẫn chỉ nên nạp từ nguồn tin cậy. Hash phát hiện khác biệt nội dung so với manifest, không phải chữ ký xác thực nguồn và không chứng minh model chính xác.

## 5. Ví dụ inference có thể thực hiện

Chạy từ thư mục gốc với môi trường đã cài requirements:

```python
from pathlib import Path
import pandas as pd
from src.models.persistence import load_cli_run
from src.data.preprocessing import transform_features
from src.models.mlp_model import predict_mlp

bundle = load_cli_run(
    Path.cwd(),
    run_dir="runs/20261116T113320345877Z_6c8931e5",
)
raw_test = pd.read_csv(bundle["run_dir"] / "data/processed/test.csv")
sample = raw_test.drop(columns="median_house_value").head(3)
features = transform_features(bundle["preprocessor"], sample)
predicted_usd = predict_mlp(bundle["model"], features)
print(predicted_usd)
```

Mẫu mới thực sự cũng cần 9 features theo schema, đúng kiểu và ý nghĩa đơn vị. Không đưa target vào input, không gọi fit lại; lấy thứ tự biến đổi từ fitted preprocessor. Hàm predict trả vector shape (N,) bằng USD. Nếu dùng output-dir khác, output_root phải là root tương ứng.

## 6. Kết quả minh họa đã đạt được

Nguồn: [sample_predictions.csv](../../runs/20261116T113320345877Z_6c8931e5/results/metrics/sample_predictions.csv).

| Thứ tự trong test | Actual USD | Predicted USD | Absolute error USD |
| --- | ---: | ---: | ---: |
| 1 | 145.600,00 | 213.043,30 | 67.443,30 |
| 2 | 100.000,00 | 105.835,08 | 5.835,08 |
| 3 | 137.600,00 | 129.420,39 | 8.179,61 |

Mẫu 1 bị dự đoán cao; mẫu 3 bị dự đoán thấp. Absolute error luôn không âm, không cho biết hướng sai. Ba mẫu này là ba dòng đầu của test, không phải bộ dữ liệu mới thu thập độc lập.

Ngày 24/11/2026 đã reload bundle và đối chiếu prediction với CSV bằng rtol=1e-5, atol=1e-3: đạt. Đây là parity của artifacts đã lưu với prediction được nạp lại; không tự chứng minh generalization. Chương này dùng bảng làm đầu ra phù hợp; pipeline không xuất hình inference riêng. Các scatter toàn test ở [chương 07](07_evaluation_and_error_analysis.md) không được gắn nhãn là hình của riêng ba mẫu.

## 7. Failure modes và xử lý

| Dấu hiệu | Ý nghĩa | Cách xử lý |
| --- | --- | --- |
| Artifact mismatch | File thiếu hoặc khác hash manifest | Xác định đúng run còn nguyên; không sửa hash chỉ để vượt kiểm tra |
| Feature dimension mismatch | Model/preprocessor không cùng biểu diễn | Nạp toàn bundle cùng run |
| Invalid checkpoint/schema | Sai định dạng hoặc metadata | Dùng checkpoint do pipeline tương ứng lưu |
| Sai tên/kiểu input | Dữ liệu mới không đúng hợp đồng | Kiểm tra 9 features trước transform |
| Latest trỏ smoke run | Chọn nhầm kết quả mới nhất | Truyền run_dir cụ thể khi báo cáo |
| Load thành công nhưng dự đoán kém | Tính toàn vẹn khác chất lượng thống kê | Đọc metrics và giới hạn dữ liệu |

Runs giữ riêng làm tăng dung lượng theo thời gian. Quy trình hiện không tự dọn; việc quản lý dung lượng cần chọn run có chủ đích.

## 8. Phân tích kết quả đạt được theo 5W1H

| Thành phần | Kết quả |
| --- | --- |
| What | Bundle nạp được, đủ 25 artifacts được hash và ba sample predictions khớp CSV sau reload. |
| Why | Cho phép bàn giao model cùng preprocessing và phát hiện bộ kết quả bị trộn hoặc đổi nội dung. |
| Who | Người tiếp nhận model, người chạy inference và người kiểm tra báo cáo. |
| When | Run lưu ngày 16/11/2026; reload kiểm chứng ngày 24/11/2026. |
| Where | Run tham chiếu, manifest, saved_models và sample_predictions.csv. |
| How | Verify hashes, load schema/state, transform raw test, predict USD, assert_allclose. |

## 9. Checklist và bàn giao

- [x] Chỉ định đúng run và đủ artifacts.
- [x] Đối chiếu 25 artifact hashes; raw CSV khớp fingerprint.
- [x] Reload sample parity và metrics test đã kiểm chứng.
- [x] Nêu rõ giới hạn: chỉ 10/18 source hashes hiện tại khớp run lịch sử.
- [x] Có hướng dẫn inference và cách xử lý lỗi; không cần ảnh mới cho kết quả dạng bảng.

Xem [chương 09](09_execution_and_validation.md) cho lệnh thực thi và phạm vi tái lập.

[Quay lại mục lục](README.md)
