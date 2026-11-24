# 01 — Xác định bài toán và metrics

## 1. Mục tiêu, người sử dụng và I/O

Bài lab phục vụ người học thực hành **supervised regression** và người đánh giá báo cáo: từ 9 features của khu dân cư, dự đoán `median_house_value` theo USD. Mục tiêu là pipeline đúng, có baseline và so sánh công bằng với bằng chứng truy được về run. Dự án chưa thống nhất ngưỡng RMSE chấp nhận cho nghiệp vụ định giá thực tế.

| Thành phần | Contract |
| --- | --- |
| Input thô | DataFrame với 8 numeric features và `ocean_proximity` |
| Nhãn training | Vector target số hữu hạn, không missing, cùng thứ tự X |
| Input model | Ma trận số từ preprocessor đã fit trên train |
| Output | Một giá trị dự đoán USD cho mỗi dòng |
| Phạm vi | Trung vị khu dân cư trong dataset, không phải giá giao dịch hiện tại |

Với dữ liệu có nhãn `(x_i, y_i)`, học hàm `ŷ_i = fθ(x_i)`. Training điều chỉnh θ trên train. Validation theo dõi hoặc đối chiếu cấu hình/checkpoint. Test đánh giá cuối sau khi cấu hình được cố định. Inference dùng model/preprocessor đã lưu và không nhất thiết có nhãn y; không fit lại trên mẫu mới.

## 2. Thống nhất tiêu chí trước lựa chọn model

1. Chốt target, đơn vị và đơn vị quan sát từ [chương 00](00_data_overview.md).
2. Quy định train/validation/test và chống leakage trước khi fit thống kê.
3. Chọn metrics, hướng tốt/xấu và tiêu chí checkpoint.
4. Huấn luyện Linear Regression làm mốc tuyến tính, thêm Tree/Forest để đối chứng phi tuyến.
5. Huấn luyện MLP trên cùng splits/features; không mặc định MLP phải thắng.
6. Báo kết quả theo cấu hình/run, không đổi tiêu chí hoặc tuning theo test.

Đây là contract hiện hành trong [README](../../README.md) và [main.py](../../main.py), không phải đề xuất đổi thuật toán.

## 3. Công thức và cách đọc metrics

Với N mẫu, sai số `e_i = y_i − ŷ_i`, và `ȳ` là mean target của tập đang đánh giá:

$$
MSE = \frac{1}{N}\sum_i e_i^2, \qquad RMSE = \sqrt{MSE}
$$

$$
MAE = \frac{1}{N}\sum_i |e_i|, \qquad
R^2 = 1 - \frac{\sum_i e_i^2}{\sum_i (y_i-\bar y)^2}
$$

| Metric | Đơn vị | Chiều tốt | Ý nghĩa | Hạn chế |
| --- | --- | --- | --- | --- |
| MSE | USD² | Thấp | Bình phương sai số, phạt lỗi lớn | Khó diễn giải trực tiếp thành tiền |
| RMSE | USD | Thấp | Căn MSE, mốc đối chiếu chính theo README | Nhạy với lỗi lớn; không phải lỗi của mọi mẫu |
| MAE | USD | Thấp | Độ lệch tuyệt đối trung bình | Không cho biết riêng phần đuôi lỗi hoặc nhóm |
| R² | Không đơn vị | Cao | So với dự đoán mean của tập đánh giá | Có thể âm; không phải accuracy |

Trên cùng tập có target không hằng, MSE/RMSE/R² cho thứ hạng tương đương. `summarize_models()` thực tế sắp R² giảm dần nên phù hợp thứ hạng RMSE trên cùng tập. Không dùng điều này để so R² giữa các tập có độ biến thiên target khác nhau.

Ví dụ minh họa công thức, **không phải kết quả run**: `y = [100000, 200000]`, `ŷ = [110000, 180000]`. Sai số là −10.000 và 20.000 USD: MAE = 15.000 USD, MSE = 250.000.000 USD², RMSE ≈ 15.811,39 USD, R² = 0,9. R² 0,9 không có nghĩa 90% nhà được dự đoán đúng.

### Trường hợp biên và kiểm tra input

[RegressionEvaluator](../../src/evaluation/metrics.py) chuyển input thành float64, nhận vector `(N,)` hoặc cột `(N, 1)`; từ chối rỗng, NaN/Inf, shape không phù hợp và hai vector khác độ dài. Không tự căn chỉnh index; người gọi phải bảo toàn thứ tự y và prediction.

R² dùng `r2_score(force_finite=True)`: target hằng số với ít nhất hai mẫu nhận 1 khi prediction hoàn hảo, 0 nếu không hoàn hảo. Một mẫu có R² không xác định. MAE/RMSE của một mẫu chỉ mô tả sai lệch mẫu đó, không đại diện chất lượng tổng thể.

## 4. Objective và checkpoint MLP

Trong [mlp_model.py](../../src/models/mlp_model.py), target được chuẩn hóa bằng thống kê train:

$$
z_i = \frac{y_i-\mu_{train}}{s_{train}}, \qquad
\hat y_i = \hat z_i s_{train}+\mu_{train}
$$

Mean và population standard deviation chỉ tính từ train; target hằng dùng scale = 1. Adam tối ưu MSE trên z. Prediction sau đó inverse-transform về USD; với scale cố định, `MSE_USD = scale² × MSE_z` và `MAE_USD = scale × MAE_z`. Chuẩn hóa không biến mọi phân phối thành phân phối chuẩn.

History báo MSE USD² và MAE USD trên snapshot cuối mỗi epoch. Chọn checkpoint có **validation MSE thấp nhất**, tương đương validation RMSE thấp nhất. Training chạy đủ epochs rồi khôi phục checkpoint; không early stopping, không chọn epoch theo test. Xem [chương 06](06_mlp_training.md).

## 5. Thành công và giới hạn kết luận

Correctness yêu cầu X/y căn chỉnh, preprocessing chỉ fit train, model dùng cùng split/features, prediction hữu hạn, metrics đúng đơn vị và reload parity trong tolerance. Các điều kiện này khác với generalization tốt.

Chất lượng cần đối chiếu baseline, RMSE/MAE và failure modes. Không tuyên bố “đạt chất lượng nghiệp vụ” khi chưa có threshold, chi phí sai số hoặc dữ liệu triển khai. Prediction âm từ Linear/MLP phải được ghi nhận, không clipping để làm đẹp metrics. Target capping, nhóm hiếm và random split giới hạn kết luận. Phân tích ở [chương 07](07_evaluation_and_error_analysis.md).

## 6. Phân tích kết quả đạt được theo 5W1H

| Thành phần | Kết quả có bằng chứng |
| --- | --- |
| **What** | Xác định regression, 9 features, target USD và bốn metrics tương ứng code. |
| **Why** | RMSE/MAE diễn giải sai số tiền; MSE liên hệ objective; R² cung cấp mốc mean. |
| **Who** | Người training theo validation criterion; người đọc báo cáo đánh giá trên cùng tập và đơn vị. |
| **When** | Tiêu chí được đặt trước các chương training; đối chiếu code/README ngày 24/11/2026, không training mới. |
| **Where** | README, `metrics.py`, `mlp_model.py`, `main.py`. |
| **How** | Đọc công thức, validation input, target scaling và điều kiện cập nhật checkpoint. |

**Kết luận:** đạt mục tiêu giải thích bài toán và tiêu chí phần mềm hiện có. Chưa đủ cơ sở kết luận đạt chất lượng ứng dụng thực tế. Bàn giao target, bộ metrics và checkpoint criterion cho các giai đoạn dữ liệu/model.

### Checklist nghiệm thu

- [x] Xác định I/O, target và đơn vị quan sát.
- [x] Đủ công thức, đơn vị, giới hạn và trường hợp biên.
- [x] Phân biệt objective chuẩn hóa, metrics USD và validation checkpoint.
- [x] Không mặc định MLP thắng, không chọn cấu hình bằng test.
- [x] Không tự đặt ngưỡng chưa được thống nhất.

[Tiếp: Kiểm tra dữ liệu](02_data_understanding.md) · [Mục lục](README.md)
