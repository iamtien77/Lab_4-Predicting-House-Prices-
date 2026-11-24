# 02 — Kiểm tra và phân tích chất lượng dữ liệu

## 1. Mục tiêu và input/output

Input là [CSV thô](../../data/raw/housing.csv) đã nhận diện ở chương 00. Output gồm DataFrame hợp lệ, tên target và báo cáo audit: shape, kiểu dữ liệu, missing, duplicates, phân bố nhãn phân loại và giới hạn target. Giai đoạn này không tự sửa dữ liệu hoặc huấn luyện model.

Hai câu hỏi khác nhau cần được trả lời: **file có đáp ứng contract phần mềm không** và **dữ liệu có những hạn chế gì khi diễn giải kết quả**. Loader trở thành công cụ giải đáp câu hỏi thứ nhất trong phạm vi quy tắc đã cài đặt; không chứng minh nhãn đúng với thực tế hay mẫu đại diện dân số.

## 2. Thực hiện audit cụ thể tái lập

Từ thư mục gốc, dùng môi trường sẵn có và chạy đoạn sau trong Python:

```python
import hashlib
from pathlib import Path
import numpy as np
from src.data.loader import load_housing_data, fetch_dataset_info

path = Path("data/raw/housing.csv")
df, target = load_housing_data(path)
info = fetch_dataset_info(df, target)
print(info)
print("sha256:", hashlib.sha256(path.read_bytes()).hexdigest())
print("target tại 500001:", int(df[target].eq(500001).sum()))
print("numeric infinity:", int(np.isinf(
    df.select_dtypes(include="number").to_numpy()
).sum()))
```

`fetch_dataset_info()` tính thống kê mà không thay đổi hoặc loại dòng. Hàm kiểm tra target nhưng không thay thế toàn bộ feature validation của loader; gọi loader trước như ví dụ. `target_max_count` của helper đếm số dòng bằng **max đang có**, không mặc định mọi CSV đều có trần 500.001; nên đối chiếu cả max và phép đếm bằng giá trị cụ thể.

## 3. Contract schema và quy tắc từ chối

Nguồn trực tiếp là [src/data/loader.py](../../src/data/loader.py):

| Kiểm tra | Hành vi thực tế | Ý nghĩa |
| --- | --- | --- |
| Đường dẫn | Phải là file cục bộ; không có thì `FileNotFoundError` | Không tự tải dữ liệu thay thế |
| CSV | Bắt `EmptyDataError`/`ParserError` thành `ValueError` | Không bỏ qua lỗi parse |
| Target | Phải có cột `median_house_value`, Series số không rỗng, không boolean, mọi giá trị hữu hạn | Không chấp nhận missing/Inf/NaN và không điền target |
| Features | DataFrame không rỗng, tên cột duy nhất, tập cột đúng 9 cột cấu hình | Cột thiếu hoặc thừa đều bị từ chối |
| Numeric features | Dtype numeric, không boolean, không infinity | NaN features được phép để imputer xử lý sau |
| Categorical features | Mọi giá trị không missing phải là chuỗi | Không ép mã số thành category ngầm |
| Cột train hoàn toàn missing | Bị từ chối khi validation chạy với `require_complete_columns=True` trong fit preprocessing | Loader mặc định chưa áp dụng điều kiện này |

Numeric text đọc thành chuỗi, ví dụ một cột có thêm chữ, có thể bị từ chối thay vì được tự ép kiểu. Không thêm cột index CSV vào input model: schema strict sẽ coi đó là cột thừa. Kiểm tra tên cột ở đây áp dụng DataFrame sau `read_csv`; không coi nó là một công cụ kiểm định mọi đặc điểm của header CSV thô.

Loader không có bộ quy tắc đầy đủ cho địa lý, tính hợp lý kinh tế hoặc miền giá trị. Một số điều kiện mẫu số/log thuộc feature engineering, được kiểm tra khi bật biến đổi tương ứng; xem [chương 04](04_preprocessing_and_features.md). Vì thế “đọc được” không đồng nghĩa “mọi giá trị đều hợp lý về domain”.

## 4. Kết quả audit CSV ngày 24/11/2026

Phạm vi bảng này là **toàn bộ 20.640 dòng trước split**, hash `b214a65099c1dcc2ea7797935aff103fa0445220a366fb74beacb59486e548ff`.

| Hạng mục | Kết quả | Diễn giải |
| --- | ---: | --- |
| Dòng × cột | 20.640 × 10 | 9 features và 1 target |
| Missing `total_bedrooms` | 207 / 20.640 ≈ 1,003% | Giữ dòng, xử lý bằng imputer fit train |
| Missing các cột khác | 0 | Không suy ra mọi bản CSV tương lai cũng như vậy |
| Dòng trùng hoàn toàn | 0 | Không chứng minh không có quan sát gần giống hoặc phụ thuộc không gian |
| Numeric infinity | 0 | Kiểm đếm các cột số |
| Target missing/không hữu hạn | 0 | Loader đã xác thực thành công |
| Target nhỏ nhất | 14.999 USD | Giá trị quan sát trong file |
| Target lớn nhất | 500.001 USD | 965 dòng cùng đạt mức này |
| Target bằng 500.001 | 965 / 20.640 ≈ 4,675% | Cần chú ý khi đánh giá nhóm nhãn tại trần |

Các thống kê target từ pandas `describe()`:

| Count | Mean | Sample std | Q1 | Median | Q3 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20.640 | 206.855,82 | 115.395,62 | 119.600 | 179.700 | 264.725 |

Các giá trị tiền đều theo USD. Sample std của bảng dùng mẫu số N-1, không phải population std dùng cho target scaling. Không đưa mean/std của toàn CSV này vào quá trình fit MLP.

### Missing features khác missing target

Thiếu feature vẫn cho phép sử dụng target thật của dòng khi training; median imputation là một lựa chọn để giữ mẫu. Nhưng điền target bằng median rồi huấn luyện sẽ tạo nhãn giả và thay đổi bài toán, nên loader từ chối target thiếu. Tỷ lệ missing khoảng 1% không chứng minh missing là ngẫu nhiên; audit hiện tại chưa phân tích cơ chế missing.

Không gọi `dropna()` trên toàn bảng cho tiện hoặc dùng median của toàn CSV. Sau split, imputer chỉ học median từ train. Với run tham chiếu, chương 03 kiểm tra 167 missing `total_bedrooms` trong train; đó là phạm vi khác với 207 trên toàn CSV.

### Nhãn tại trần và nhóm hiếm

Theo README, target có hiện tượng capping tại 500.001 USD. Audit xác nhận **965 dòng bằng đúng mức max này**, nhưng phép đếm không tự chứng minh cơ chế thu thập hoặc giá trị thật phía trên trần. Model được đánh giá so với nhãn có sẵn; prediction gần 500.001 không chứng minh dự đoán đúng giá chưa bị chặn.

| `ocean_proximity` | Số dòng | Tỷ lệ trên 20.640 |
| --- | ---: | --- |
| `<1H OCEAN` | 9.136 | 44,264% |
| `INLAND` | 6.551 | 31,739% |
| `NEAR OCEAN` | 2.658 | 12,878% |
| `NEAR BAY` | 2.290 | 11,095% |
| `ISLAND` | 5 | 0,024% |

`ISLAND` rất ít mẫu nên một vài dự đoán có thể làm metric của nhóm thay đổi mạnh. Random split phân tầng target không bảo đảm mọi nhãn category xuất hiện với số lượng đủ trong cả ba tập. One-hot cần xử lý category chưa thấy, nhưng khả năng chạy không lỗi khác với khả năng dự đoán tốt cho nhóm đó. Không gộp/xóa nhóm chỉ vì ít mẫu mà không có quyết định và đánh giá riêng.

## 5. Phân biệt audit với EDA và fit

| Hoạt động | Phạm vi | Có học tham số để biến đổi không? |
| --- | --- | --- |
| Audit schema, missing, duplicates, đếm nhóm | Toàn CSV để mô tả và kiểm tra hợp lệ | Không |
| EDA chi tiết để hiểu phân bố/quan hệ và đề xuất features | Train sau split | Không fit model, nhưng có thể tác động lựa chọn thiết kế |
| Imputation, scaling, học danh sách one-hot | Chỉ train | Có |
| Áp dụng cho validation/test/inference | Dữ liệu tương ứng | Chỉ transform, không fit lại |

Các ảnh EDA và diễn giải đã có ở [chương 03 — Chia tập và phân tích EDA](03_data_splitting_and_eda.md). Chương này không tạo ảnh mới hoặc trộn thống kê toàn CSV vào các nhận xét ghi là train-only. Phân bố hoặc correlation không đủ chứng minh nhân quả, và histogram residual không phải bằng chứng duy nhất để kết luận phân phối chuẩn.

## 6. Điều kiện dừng và bàn giao

Dừng xử lý nếu file thiếu/hỏng, schema không đúng, target không hữu hạn hoặc kiểu cột không hợp lý. Trước fit preprocessing phải dừng nếu có cột train hoàn toàn missing. Các trường hợp này cần xác định nguyên nhân, không âm thầm thay file hoặc điền giá trị để vượt validation.

Missing feature rời rạc, nhóm hiếm và target capping là những hạn chế đã nhận diện, không tự động là lý do loại dòng. Bàn giao DataFrame thô nguyên số dòng và tên target để [chia tập ở chương 03](03_data_splitting_and_eda.md), cùng danh sách lưu ý cho preprocessing/đánh giá.

## 7. Phân tích kết quả đạt được theo 5W1H

| Thành phần | Kết quả có bằng chứng |
| --- | --- |
| **What** | Loader đọc thành công 20.640 dòng; xác nhận 207 missing feature, 0 duplicate đầy đủ, 0 numeric Inf, 965 nhãn tại 500.001 và 5 mẫu ISLAND. |
| **Why** | Missing dẫn tới imputation train-only; capped target và nhóm hiếm giới hạn cách diễn giải metrics. |
| **Who** | Người chuẩn bị dữ liệu giữ raw CSV và contract; người đánh giá cần báo riêng phạm vi/nhóm khi nhận xét chất lượng. |
| **When** | Audit trực tiếp ngày 24/11/2026, trước mọi thay đổi dữ liệu trong nhiệm vụ tài liệu này. |
| **Where** | `data/raw/housing.csv`, `src/data/loader.py`; các hành vi kiểm thử được mô tả tại `tests/test_data.py`. |
| **How** | Chạy loader/helper, đếm missing/duplicates/category/max, kiểm infinity và hash bằng Python; không chạy lại training hoặc toàn test suite. |

**Kết luận:** Đạt mục tiêu kiểm tra hợp lệ và thống kê chất lượng bản CSV hiện tại. Chưa chứng minh tính đúng của nhãn ngoài thực tế, cơ chế missing, tính đại diện hoặc nguồn gốc thu thập. Bàn giao dữ liệu thô hợp lệ cho split và các giới hạn cho các chương tiếp theo.

### Checklist nghiệm thu

- [x] File/schema/target được loader xác nhận.
- [x] Audit có ngày, phạm vi và hash; tỷ lệ dùng mẫu số rõ ràng.
- [x] Không sửa raw CSV, tự loại outlier hoặc điền target.
- [x] Phân biệt audit toàn bộ với EDA/fit chỉ trên train.
- [x] Ghi rõ kiểm tra miền/domain và provenance chưa đầy đủ.

Nguồn implementation: [loader.py](../../src/data/loader.py), [preprocessing.py](../../src/data/preprocessing.py), [test_data.py](../../tests/test_data.py). Các liên kết test là nguồn tham khảo hành vi, không phải tuyên bố tests vừa được chạy.

[Tiếp: Chia tập và EDA](03_data_splitting_and_eda.md) • [Mục lục](README.md)
