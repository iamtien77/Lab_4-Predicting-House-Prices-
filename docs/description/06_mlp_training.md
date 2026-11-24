# 06 — Huấn luyện MLP

Giai đoạn này học quan hệ phi tuyến bằng PyTorch MLP và so sánh với các baseline đã có. Mục tiêu là tạo checkpoint được chọn bằng validation, prediction theo USD và history có thể diễn giải; không mặc định neural network phải thắng Random Forest.

## 1. Input, output và kiến trúc

Input gồm train/validation features hữu hạn, cùng thứ tự với target USD. Số chiều input lấy từ `train_features.shape[1]`. [HousePriceMLP](../../src/models/mlp_model.py) có kiến trúc cố định:

```text
(B, D) → Linear(D, 32) → ReLU → Linear(32, 16) → ReLU
       → Linear(16, 1) → squeeze(-1) → (B,)
```

Với run tham chiếu, `D=13`. Mạng có `13×32+32 + 32×16+16 + 16×1+1 = 993` trainable parameters, suy ra từ kiến trúc. Các tầng ẩn kết hợp features qua trọng số và ReLU; tầng cuối linear không giới hạn khoảng prediction. Giá dự đoán âm vẫn có thể xảy ra về mặt cấu trúc.

Output của `forward()` là target chuẩn hóa. Hàm công khai `predict_mlp()` chuyển về USD và kiểm tra vector `(N,)`, hữu hạn. Output training gồm model đã restore best checkpoint và dictionary history. Implementation chạy CPU; không có dropout, batch normalization, learning-rate scheduler hoặc early stopping.

## 2. Target scaling: học từ train và hoàn nguyên

Features đã được xử lý ở giai đoạn 04; target có độ lớn hàng trăm nghìn USD được chuẩn hóa riêng:

$$z_i=\frac{y_i-\mu_{train}}{\sigma_{train}},\qquad
\hat y_i=\hat z_i\sigma_{train}+\mu_{train}.$$

`train_mlp_model()` tính mean và population standard deviation (`unbiased=False`) từ `y_train`. Nếu target hằng, scale dùng 1 để tránh chia 0. Validation không tham gia tính statistics. Hai giá trị nằm trong registered buffers `target_mean` và `target_scale` kiểu float64, được lưu cùng weights; không có class độc lập tên `TargetScaler` trong implementation.

Target scaling đưa objective về thang đo thuận tiện hơn cho tối ưu; nó không biến mọi phân phối thành Gaussian, không xóa target capping và không bảo đảm hội tụ. Nếu bỏ inverse-transform mà tính metrics với USD, kết quả sẽ sai đơn vị.

## 3. Training loop từng bước

Cấu hình run tham chiếu từ [manifest](../../runs/20261116T113320345877Z_6c8931e5/manifest.json): 100 epochs, batch size 32, learning rate 0,001, seed 42. `main.py` thiết lập seed cho Python/NumPy/PyTorch và bật deterministic algorithms. DataLoader shuffle train bằng generator có seed; `num_workers=0`.

1. Kiểm tra X/y, số chiều features, epochs và batch size dương. Chuyển features thành float32 tensors; giữ target USD float64 cho đánh giá.
2. Tính target statistics train; tạo normalized target float32.
3. Tạo `TensorDataset`, DataLoader và Adam. Với 16.512 mẫu train và batch 32, một epoch có 516 mini-batches.
4. Đầu epoch gọi `model.train()`; mỗi batch gọi `optimizer.zero_grad(set_to_none=True)` để không cộng dồn gradient từ batch trước.
5. Forward tạo prediction chuẩn hóa và tính `nn.MSELoss()`:

   $$L=\frac{1}{B}\sum_{i=1}^{B}(z_i-\hat z_i)^2.$$

6. Kiểm tra loss hữu hạn; gọi `loss.backward()` tính gradients theo chain rule rồi `optimizer.step()` cập nhật weights bằng Adam.
7. Cuối epoch đánh giá snapshot hiện tại trên toàn bộ train và validation; ghi history và kiểm tra checkpoint tốt nhất.
8. Sau đủ epochs, restore best state, chuyển eval mode và đánh dấu fitted.

`train_mlp_model()` gọi lại trên một model cũ sẽ reset Adam và tính lại target statistics nhưng kế thừa weights hiện có. Muốn thực nghiệm độc lập phải build model mới và đặt seed phù hợp; không coi việc gọi train lần hai là run khởi tạo mới tương đương.

## 4. Objective, history và checkpoint khác nhau thế nào?

| Giá trị | Cách tính | Đơn vị/vai trò |
| --- | --- | --- |
| Batch objective | MSE giữa normalized target và output mạng trong training | Không đơn vị; phục vụ backward/Adam |
| `loss` | MSE train sau inverse-transform ở snapshot cuối epoch | USD²; learning curve |
| `val_loss` | MSE validation tại cùng snapshot | USD²; chọn checkpoint |
| `mae`, `val_mae` | Mean absolute error train/validation tại snapshot | USD; diễn giải độ lệch |

`_epoch_metrics()` dùng `model.eval()` và `torch.inference_mode()`, cộng tổng errors qua các batch rồi chia tổng số mẫu. Vì vậy mỗi mẫu có trọng số như nhau, không lấy trung bình không trọng số của các batch nếu batch cuối ngắn hơn. History cũng không phải trung bình các loss trong khi weights đang thay đổi giữa mini-batches.

Khi `val_mse` nhỏ hơn best đã có, code clone `state_dict()` và ghi epoch. Nếu bằng nhau thì giữ checkpoint trước. Chạy đủ 100 epochs rồi mới restore; **không phải early stopping**. Tiêu chí là validation MSE thấp nhất, tương đương RMSE thấp nhất trên cùng validation vì căn bậc hai là hàm đơn điệu. Test không tham gia chọn epoch.

## 5. Learning curves của run tham chiếu

![Learning curves train và validation: MSE USD² và MAE USD theo epoch](../../runs/20261116T113320345877Z_6c8931e5/results/figures/loss_curve.png)

Hình lấy từ run `20261116T113320345877Z_6c8931e5`; đã xem ảnh và đối chiếu [training_history.csv](../../runs/20261116T113320345877Z_6c8931e5/results/metrics/training_history.csv). Trục trái của MSE dùng hệ số `1e9`; không đọc 3,0 thành 3 USD².

| Epoch | Train MSE (USD²) | Validation MSE (USD²) | Train MAE (USD) | Validation MAE (USD) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3.879.874.695,74 | 4.293.457.373,29 | 43.556,31 | 45.211,11 |
| 25 | 2.815.802.848,13 | 3.382.943.913,12 | 36.443,27 | 39.079,24 |
| 50 | 2.578.068.542,36 | 3.172.054.990,55 | 34.357,77 | 37.144,63 |
| 75 | 2.526.982.001,15 | 3.130.322.806,29 | 33.822,81 | 36.551,65 |
| 100 | 2.459.671.608,07 | 3.056.805.714,16 | 33.491,61 | 36.551,04 |

**Quan sát:** MSE/MAE giảm mạnh giai đoạn đầu, rồi giảm chậm và dao động. Validation ở phía trên train; tại epoch 100, MAE gap khoảng 3.059,43 USD. Manifest ghi best epoch 100, trùng cuối run này.

**Ý nghĩa:** training đã giảm lỗi trên cả train và validation, không chỉ ghi nhớ train mà validation xấu dần xuyên suốt. Tuy vậy generalization gap còn tồn tại. Một số đỉnh ở cả hai đường cho thấy snapshot giữa epochs dao động; đây không tự chứng minh data leakage hay lỗi chương trình.

**Hạn chế:** best epoch nằm cuối không chứng minh đã hội tụ hoặc chạy thêm chắc chắn tốt hơn. Một đường cong với một seed không định lượng độ ổn định, không đủ kết luận nguyên nhân gap là duy nhất do overfitting. Các thay đổi features, capacity, learning rate, số epochs cần được đánh giá riêng bằng validation, không suy từ hình rằng phải thêm tầng hoặc dropout.

**Hành động:** giữ checkpoint epoch 100 cho báo cáo run hiện tại. Nếu mở rộng thực nghiệm, định nghĩa trước ngân sách epochs và tiêu chí validation, lưu run mới, so sánh nhiều seeds khi được triển khai; không tiếp tục chọn cấu hình theo test hiện tại đã quan sát.

## 6. Kết quả validation và so với baseline

[validation_summary.csv](../../runs/20261116T113320345877Z_6c8931e5/results/metrics/validation_summary.csv) ghi MLP có MSE **3.056.805.741,45 USD²**, RMSE **55.288,39 USD**, MAE **36.551,04 USD**, R² **0,7689** trên 2.064 mẫu.

MSE CSV summary chênh khoảng 27,29 USD² so với `val_loss` epoch 100 của history, tương đối khoảng `9×10^-9`. Code tính history theo batch còn summary gọi prediction toàn tập; sai khác rất nhỏ phù hợp với phép tính floating point theo batch, không nên ép hai số bằng nhau tuyệt đối hoặc gọi chúng là hai checkpoints khác nhau. Đây là giải thích dựa trên code, không phải một phép đo nguyên nhân số học độc lập.

MLP giảm RMSE khoảng 18.763,20 USD so với Linear Regression và 16.534,26 USD so với Decision Tree, nhưng cao hơn Random Forest khoảng 4.813,56 USD. Kết quả này cho thấy MLP là mô hình phi tuyến hữu ích trong cấu hình đã chạy, chưa vượt baseline forest. Không dùng tên gọi “deep learning” để bỏ qua đối chứng này.

Các hình test và diễn giải residuals nằm ở [giai đoạn 07](07_evaluation_and_error_analysis.md); giữ tách biệt learning curves dùng cho checkpoint với test dùng cho báo cáo cuối.

## 7. Failure modes và cách xử lý

| Dấu hiệu | Giải thích cần kiểm tra | Hành động trong phạm vi thực nghiệm |
| --- | --- | --- |
| NaN/infinite objective | Input/target không hợp lệ, tối ưu không ổn định | Kiểm tra dữ liệu, scaling và learning rate; code dừng thay vì lưu model lỗi |
| Prediction sai thang USD | Quên inverse-transform hoặc dùng statistics khác | Gọi `predict_mlp()`, nạp đúng checkpoint cùng preprocessor |
| Dimension mismatch | Vocabulary/features khác model đã train | Dùng đúng fitted preprocessor và feature names của run |
| Train giảm, validation xấu kéo dài | Có thể overfitting hoặc phân phối khác | Phân tích split và validation; dùng checkpoint tốt nhất, không chọn theo test |
| Cả train/validation cao | Có thể thiếu capacity/features hoặc tối ưu chưa đủ | So baseline, kiểm tra learning rate/history trước khi tăng độ phức tạp |
| Prediction âm | Output linear không ràng buộc miền | Ghi nhận error; không clipping sau khi xem test để cải thiện số liệu |

Code hiện tại không có automatic tuning hoặc cơ chế tự thêm regularization. Các hành động thử cấu hình là hướng mở rộng có kiểm soát, không phải tính năng đã chạy trong run tham chiếu.

## 8. Kết quả theo 5W1H và bàn giao

| Thành phần | Kết quả thực tế |
| --- | --- |
| **What** | Hoàn tất 100 epochs; best epoch 100; validation RMSE 55.288,39 USD; history có đủ train/validation MSE/MAE. |
| **Why** | Target scaling hỗ trợ tối ưu; checkpoint chọn theo validation để tránh phụ thuộc epoch cuối tùy ý. Run này best trùng epoch cuối nhưng quy tắc vẫn là best validation. |
| **Who** | Người huấn luyện đọc history; người đánh giá và inference dùng weights đã cố định. |
| **When** | Run ngày 16/11/2026 sau baseline, trước test; seed 42, batch 32, learning rate 0,001. |
| **Where** | Manifest, CSV và learning curves liên kết ở trên; [MLP checkpoint](../../runs/20261116T113320345877Z_6c8931e5/results/saved_models/mlp_house_price_model.pt). |
| **How** | Đối chiếu ảnh/history/summary/manifest. [test_models.py](../../tests/test_models.py) mô tả kiểm tra train-only scaling, best checkpoint và round-trip; không chạy training lại để viết chương này. |

**Kết luận:** đã có checkpoint và bằng chứng giảm lỗi qua training; chưa có bằng chứng MLP vượt Random Forest hoặc đạt chất lượng ứng dụng thực tế. Bàn giao checkpoint chứa weights/target statistics, fitted preprocessor và history sang đánh giá cuối. Các thông tin code hiện tại giải thích cơ chế; không khẳng định mọi source hash hiện tại khớp run lịch sử, xem [xác minh tái lập](09_execution_and_validation.md).

[Tiếp: đánh giá và phân tích sai số](07_evaluation_and_error_analysis.md) · [Quay lại mục lục](README.md)
