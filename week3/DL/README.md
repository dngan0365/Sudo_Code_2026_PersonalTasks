# Vietnamese News Text Classification With Deep Learning

Refactor từ ý tưởng notebook thành pipeline hướng đối tượng để train một neural network đơn giản cho bài toán phân loại tin tức tiếng Việt 10 lớp.

Dataset trong thư mục này có dạng:

```text
data/
├── Train_Full/
│   ├── Chinh tri Xa hoi/
│   ├── Doi song/
│   └── ...
├── Test_Full/
│   ├── Chinh tri Xa hoi/
│   ├── Doi song/
│   └── ...
└── Stats.txt
```

## Cài đặt

Từ thư mục `week3/DL`, chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Trên macOS/Linux, kích hoạt môi trường bằng `source .venv/bin/activate`.

## Chạy training

Chạy đầy đủ theo cấu hình mặc định:

```powershell
python src/main.py
```

Chạy nhanh để kiểm tra pipeline:

```powershell
python src/main.py --epochs 1 --limit-per-class 200
```

Dùng `underthesea` để tách từ tiếng Việt:

```powershell
python src/main.py --use-underthesea
```

Các tham số hardcode nằm tại `config/default.yaml`, gồm đường dẫn dữ liệu, vocab size, batch size, epoch, learning rate, embedding dimension và device.

## Mô hình

Pipeline dùng mô hình PyTorch đơn giản:

```text
tokens -> vocabulary ids -> EmbeddingBag(mean) -> Linear -> ReLU -> Dropout -> Linear -> 10 classes
```

`EmbeddingBag` giúp xử lý văn bản có độ dài khác nhau mà không cần padding dài. Đây là baseline neural network gọn, nhanh, dễ giải thích trước khi thử CNN, LSTM hoặc Transformer.

## Kết quả

Sau khi chạy, thư mục `output/` sẽ có:

- `dataset_report.csv`: số file theo split và class.
- `history.csv`: loss, accuracy, macro-F1 và weighted-F1 theo epoch.
- `metrics.json`: cấu hình, vocab size, best epoch và classification report.
- `classification_report.txt`: báo cáo precision/recall/F1 dạng text.
- `best_model.pt`: model bundle để predict lại.
- `best_model_state.pt`: state dict tốt nhất theo macro-F1.
- `metadata.joblib`: vocabulary và label names.

## Predict

Sau khi train:

```powershell
python src/predict.py --text "Google đang ra mắt dịch vụ tìm kiếm mới tại châu Á"
```

Output:

```text
9    Vi tinh    0.8421    Google đang ra mắt dịch vụ tìm kiếm mới tại châu Á
```

## Cấu trúc code

```text
DL/
├── config/default.yaml
├── data/
├── notebooks/text_classification_dl_draft.ipynb
├── src/
│   ├── main.py
│   ├── predict.py
│   └── utils/news_dl/
│       ├── config.py
│       ├── data_loader.py
│       ├── dataset.py
│       ├── evaluator.py
│       ├── model.py
│       ├── pipeline.py
│       ├── predictor.py
│       ├── preprocessor.py
│       ├── trainer.py
│       └── vocabulary.py
├── README.md
└── requirements.txt
```

Notebook trong `notebooks/` dùng để phân tích dữ liệu và phác thảo ý tưởng DL; pipeline chính nằm trong `src/` để chạy lại bằng CLI.
