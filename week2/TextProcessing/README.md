# Vietnamese Text Processing

Pipeline hướng đối tượng để làm sạch và tokenize bộ dữ liệu **Vietnamese Online News**.

- `DataLoader`: đọc/ghi JSON, JSON Lines, CSV và Parquet.
- `VietnameseTextPreprocessor`: chuẩn hóa Unicode, chữ thường, loại số/dấu câu,
  xử lý dữ liệu thiếu, stopword và tokenize tiếng Việt bằng underthesea.
- `TextProcessingPipeline`: điều phối toàn bộ quá trình và lưu kết quả.

## Cài đặt

Yêu cầu Python 3.10 trở lên. Từ thư mục `week2/TextProcessing`, chạy:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Trên macOS/Linux, kích hoạt bằng `source .venv/bin/activate`.

## Chuẩn bị dữ liệu

Tải [Vietnamese Online News Dataset](https://www.kaggle.com/datasets/haitranquangofficial/vietnamese-online-news-dataset),
đổi tên thành `news_dataset.json` và đặt tại `data/news_dataset.json`. Dataset tối
thiểu phải có cột `content`; mặc định pipeline cũng xử lý `title`.

## Chạy bằng một lệnh

```powershell
python src/main.py
```

Kết quả mặc định là `output/news_processed.csv`. Mỗi trường văn bản sinh thêm
`*_clean`, `*_tokens`; `content_len` là độ dài nội dung gốc. Hàng thiếu `content`
bị loại. Metadata thiếu được điền `Unknown`; `source` được thử suy ra từ URL trước.

Ví dụ tùy chỉnh:

```powershell
python src/main.py --input data/news_dataset.json --output output/news.jsonl
python src/main.py --text-columns content --no-tokens
python src/main.py --keep-digits
python src/main.py --help
```

Pipeline dùng `underthesea.word_tokenize(..., format="text")`, nên các từ ghép
tiếng Việt được nối bằng dấu gạch dưới và khớp tốt với file stopwords dạng dash.

Hỗ trợ `.json`, `.jsonl`, `.csv`, `.parquet`. Notebook cũ vẫn được giữ để tham
khảo EDA; pipeline chính trong `src/` không cần chạy từng cell.

## Cấu trúc

```text
TextProcessing/
├── data/
├── notebooks/TextProcessing.ipynb
├── src/
│   ├── main.py
│   └── utils/text_processing/
│       ├── data_loader.py
│       ├── pipeline.py
│       └── preprocessor.py
├── output/                 # tự tạo khi chạy
├── README.md
└── requirements.txt
```
