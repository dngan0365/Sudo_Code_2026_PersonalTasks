# Vietnamese Review Sentiment Classification

Pipeline hướng đối tượng cho notebook `sudocode-week3-ml.ipynb`: đọc dữ liệu review tiếng Việt, tiền xử lý văn bản, trích xuất đặc trưng TF-IDF, train model ML cổ điển và lưu kết quả bằng một lệnh.

Các lớp chính:

- `ReviewDataLoader`: đọc CSV, JSON, JSON Lines hoặc Parquet và chuẩn hóa cột `label`, `review`.
- `VietnameseReviewPreprocessor`: chuẩn hóa Unicode, lowercase, xóa URL/HTML/ký tự nhiễu và tùy chọn tokenize bằng `underthesea`.
- `FeatureBuilder`: tạo TF-IDF unigram/bigram, có thể thêm `review_length` và `sentence_count`.
- `ModelFactory`: tạo Logistic Regression, SVM hoặc Naive Bayes giống các thí nghiệm trong notebook.
- `ModelEvaluator`: tính accuracy, precision, recall, F1 và classification report.
- `SentimentTrainingPipeline`: điều phối toàn bộ quá trình train, đánh giá và lưu model.
- `SentimentPredictor`: load model đã train để dự đoán câu mới.

## Cài đặt

Yêu cầu Python 3.10 trở lên. Từ thư mục `week3/ML`, chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Trên macOS/Linux, kích hoạt môi trường bằng `source .venv/bin/activate`.

## Chuẩn bị dữ liệu

Notebook dùng bộ dữ liệu [Vietnamese Text Classification Dataset](https://www.kaggle.com/datasets/tuannguyenvananh/vietnamese-text-classification-dataset). Tải file `train.csv` và đặt tại:

```text
week3/ML/data/train.csv
```

Mặc định file CSV không có header và có 2 cột:

```text
label,review
```

Nhãn được hiểu theo mapping:

```text
0 = Negative
1 = Neutral
2 = Positive
```

Nếu file của bạn có header, thêm flag `--has-header`.

## Chạy bằng một lệnh

Train model mặc định Logistic Regression:

```powershell
python src/main.py
```

So sánh toàn bộ model trong notebook và tự lưu model tốt nhất theo `macro_f1`:

```powershell
python src/main.py --model all
```

Chạy giống hướng thí nghiệm notebook hơn, thêm đặc trưng độ dài review và số câu:

```powershell
python src/main.py --model all --include-numeric
```

Tokenize bằng `underthesea` nếu muốn thử lại nhánh thí nghiệm trong notebook:

```powershell
python src/main.py --model svm-rbf --tokenize
```

Kết quả được lưu trong `output/`:

- `metrics.csv`: bảng so sánh model.
- `classification_reports.json`: classification report chi tiết cho từng model.
- `classification_report.txt`: report dạng text khi train một model.
- `test_predictions.csv`: prediction trên tập test khi train một model.
- `models/*.joblib`: bundle gồm model, preprocessor và feature builder.
- `models/best_model.joblib`: model tốt nhất theo `macro_f1`.

## Dự đoán câu mới

Sau khi train xong:

```powershell
python src/predict.py --text "Sản phẩm rất tốt, giao hàng nhanh"
python src/predict.py --model output/models/best_model.joblib --text "Dịch vụ quá tệ"
```

Output có dạng:

```text
2    Positive    Sản phẩm rất tốt, giao hàng nhanh
0    Negative    Dịch vụ quá tệ
```

## Cấu trúc

```text
ML/
├── data/
│   └── train.csv              # tự tải từ Kaggle
├── notebooks/
│   └── sudocode-week3-ml.ipynb
├── src/
│   ├── main.py
│   ├── predict.py
│   └── utils/
│       └── sentiment_ml/
│           ├── data_loader.py
│           ├── evaluator.py
│           ├── features.py
│           ├── models.py
│           ├── pipeline.py
│           ├── predictor.py
│           └── preprocessor.py
├── output/                    # tự tạo khi chạy
├── README.md
└── requirements.txt
```

Notebook cũ vẫn được giữ để xem lại EDA và kết quả thí nghiệm; pipeline chính nằm trong `src/` để chạy lại ổn định bằng CLI.
