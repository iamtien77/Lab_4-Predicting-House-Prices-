# Hướng dẫn phân tích bài lab Predicting House Prices

Bộ tài liệu mô tả dữ liệu, phương pháp và kết quả theo từng giai đoạn, dành cho người học thực hiện và giải thích bài lab. Nội dung bám source hiện tại và phân biệt rõ kết quả của run lịch sử với kiểm chứng mới.

## 1. Thứ tự đọc

| Chương | Nội dung chính |
| --- | --- |
| [00_data_overview.md](00_data_overview.md) | Dữ liệu đầu vào |
| [01_problem_and_metrics.md](01_problem_and_metrics.md) | Xác định bài toán và metrics |
| [02_data_understanding.md](02_data_understanding.md) | Kiểm tra và phân tích chất lượng dữ liệu |
| [03_data_splitting_and_eda.md](03_data_splitting_and_eda.md) | Chia tập và EDA |
| [04_preprocessing_and_features.md](04_preprocessing_and_features.md) | Tiền xử lý và feature engineering |
| [05_baseline_models.md](05_baseline_models.md) | Xây dựng các baseline |
| [06_mlp_training.md](06_mlp_training.md) | Huấn luyện MLP |
| [07_evaluation_and_error_analysis.md](07_evaluation_and_error_analysis.md) | Đánh giá và phân tích sai số |
| [08_artifacts_and_inference.md](08_artifacts_and_inference.md) | Lưu artifacts và inference |
| [09_execution_and_validation.md](09_execution_and_validation.md) | Thực thi, xác minh và bàn giao |

Luồng học tập: dataset → bài toán/metrics → audit dữ liệu → split/EDA → preprocessing → baseline → MLP → đánh giá → lưu/nạp/inference → thực thi và bàn giao.

## 2. Quy ước bằng chứng

Run tham chiếu chung: **20261116T113320345877Z_6c8931e5**, ngày 16/11/2026, seed 42, 13 features, MLP 100 epochs, batch size 32, learning rate 0,001, không bật log/ratios. Nguồn: [manifest](../../runs/20261116T113320345877Z_6c8931e5/manifest.json).

Ngày 24/11/2026 đã đọc ảnh, kiểm tra CSV, xác minh 25 artifact hashes và reload model để đối chiếu metrics/sample predictions. Raw CSV khớp fingerprint. **Chỉ 10/18 source hashes hiện tại khớp run cũ**; không khẳng định training lại source hiện tại tái tạo toàn bộ kết quả. Chi tiết ở [chương 09](09_execution_and_validation.md).

Nguồn dataset: Dữ liệu Tổng điều tra Dân số Hoa Kỳ 1990 (1990 U.S. Census), công bố học thuật gốc bởi Pace & Barry (1997) trong bài báo *"Sparse Spatial Autoregressions"*, được Aurélien Géron làm sạch và chia sẻ lại trên Kaggle [California Housing Prices — Kaggle / camnugent](https://www.kaggle.com/datasets/camnugent/california-housing-prices). Chi tiết xuất xứ học thuật và ý nghĩa từng đặc trưng được trình bày đầy đủ tại [chương 00](00_data_overview.md).

- Mỗi chương có mục tiêu, quy trình, kết quả 5W1H, giới hạn và bàn giao.
- Giá được báo theo USD; MSE có đơn vị USD², R² không đơn vị.
- Mọi bảng/ảnh định lượng phải gắn split và nguồn; không trộn CLI với notebook.
- Số liệu do tính lại trên bundle lịch sử không được gọi là benchmark training mới.
- Kết quả 53 tests pass trong README gốc là ghi nhận lịch sử, không phải tests vừa chạy.
- Đề xuất mở rộng được phân biệt với tính năng đã triển khai.

## 3. Bản đồ kết quả dạng ảnh

| Nhóm artifacts | Số PNG | Chương phân tích | Nội dung |
| --- | ---: | --- | --- |
| EDA | 5 | [03](03_data_splitting_and_eda.md) | Histograms, địa lý, heatmap, target, categorical |
| Learning curves | 1 | [06](06_mlp_training.md) | MSE và MAE train/validation theo epoch |
| Model comparison | 1 | [07](07_evaluation_and_error_analysis.md) | MSE/RMSE/MAE/R² test |
| Actual vs predicted | 4 | [07](07_evaluation_and_error_analysis.md) | Giá thực tế so với dự đoán từng model |
| Residual histograms | 4 | [07](07_evaluation_and_error_analysis.md) | Hướng và phân bố sai số từng model |

Tổng cộng **15 PNG** của cùng run. Mỗi hình đã được xem và phân tích qua cách đọc, quan sát, ý nghĩa, hạn chế và hành động đề xuất. Các chương dữ liệu, metrics, preprocessing, persistence và thực thi sử dụng bảng/code hoặc dẫn sang hình liên quan; không có PNG riêng cho mọi giai đoạn.

## 4. Phạm vi các kết luận

Baseline được huấn luyện trước MLP; preprocessing chỉ fit train. MLP dùng PyTorch và khôi phục checkpoint tốt nhất trên validation sau khi chạy đủ epochs, không early stopping. Test dùng cho đánh giá cuối.

Trong run tham chiếu, Random Forest có test RMSE 47.929,97 USD; MLP là 51.403,48 USD. Đây là kết quả một cấu hình/split, không chứng minh thứ hạng ở dữ liệu mới. Dữ liệu lịch sử, đơn vị khu dân cư, capping và nhóm hiếm đều giới hạn cách ứng dụng.

## 5. Đối chiếu tài liệu thiết kế cũ

Một số phần trong docs/algorithims/README.md còn nhắc Keras/đường dẫn cũ, trong khi code hiện dùng PyTorch và runs/<run_id>. docs/design.md có ghi chú cập nhật nhưng thân còn cấu trúc cũ. Bộ hướng dẫn này mô tả hành vi code đã đọc và nêu khác biệt; không sửa các tài liệu cũ ngoài phạm vi.

Latest pointer hiện có thể trỏ một run khác run tham chiếu. Khi mở bảng/hình theo tài liệu, dùng liên kết run cụ thể, không thay bằng latest. Chi tiết nguồn và phạm vi xác minh xem chương 09.
