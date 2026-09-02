# Vietnamese Text Feature Extraction

Pipeline hướng đối tượng trích xuất đặc trưng BoW hoặc TF-IDF từ dữ liệu tin tức
tiếng Việt. Notebook được giữ để tham khảo; code chạy chính nằm trong `src/`.

## Cài đặt và dữ liệu

```powershell
cd week2/TextFeatureExtraction
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Đặt `news_dataset.json` vào `data/`. Dữ liệu cần có `content` và, với cấu hình
mặc định, `title`. Hỗ trợ input JSON, JSONL, CSV và Parquet.

## Chạy một lệnh

```powershell
python src/main.py
```

Mặc định pipeline chuẩn hóa và tách từ bằng PyVi, loại stopword, nối `title` với
`content`, rồi sinh TF-IDF unigram + bigram tối đa 5.000 features. Output:

- `output/tfidf_features.npz`: sparse matrix, tránh chuyển dataset lớn thành dense.
- `output/tfidf_vectorizer.joblib`: fitted vectorizer để transform dữ liệu mới.
- `output/tfidf_vocabulary.txt`: danh sách feature theo đúng thứ tự cột.
- `output/metadata.csv`: metadata tương ứng với từng hàng của matrix.

Ví dụ khác:

```powershell
python src/main.py --method bow --ngram-min 1 --ngram-max 1
python src/main.py --method tfidf --max-features 10000 --min-df 2
python src/main.py --input data/news.jsonl --output-dir artifacts
python src/main.py --help
```

## Cấu trúc class

```text
src/
├── main.py
└── utils/text_feature_extraction/
    ├── data_loader.py
    ├── preprocessor.py
    ├── feature_extractor.py
    └── pipeline.py
```
