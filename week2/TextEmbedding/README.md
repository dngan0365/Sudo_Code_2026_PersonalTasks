# Vietnamese Text Embedding

Pipeline hướng đối tượng để chuẩn bị corpus Wikipedia tiếng Việt và huấn luyện
Word2Vec CBOW/Skip-gram bằng gensim. Corpus được đọc theo chunk để không nạp toàn
bộ file vào RAM.

## Cài đặt

```powershell
cd week2/TextEmbedding
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Chạy một lệnh

Repository đã có `data/viwik18.txt`, vì vậy có thể huấn luyện cả hai model bằng:

```powershell
python src/main.py
```

Model được lưu thành `models/viwik18_cbow.model` và
`models/viwik18_skipgram.model`. Cấu hình mặc định: vector 100 chiều, window 5,
`min_count=5`, negative sampling 10 và 5 epochs.

Chỉ huấn luyện một kiến trúc hoặc chạy cấu hình nhanh:

```powershell
python src/main.py --architecture skipgram --vector-size 50 --epochs 2
python src/main.py --architecture cbow --workers 4
```

Có thể gộp các phần corpus và huấn luyện trong cùng một lệnh:

```powershell
python src/main.py --merge-dir data/viwik18-master/dataset
python src/main.py --merge-dir data/viwik18-master/dataset --max-files 2 --epochs 1
```

`--max-files` phù hợp để smoke test; bỏ tùy chọn này để dùng toàn bộ corpus. Xem
tất cả tham số bằng `python src/main.py --help`.

## Cấu trúc class

- `CorpusMerger`: gộp các file corpus theo stream.
- `VietnameseWikiCorpus`: iterator có thể duyệt lại, đọc theo chunk.
- `VietnameseTextPreprocessor`: chuẩn hóa và tách từ bằng underthesea.
- `Word2VecTrainer`: train/save/load/query model.
- `EmbeddingPipeline`: điều phối merge, train và lưu model.

```text
src/
├── main.py
└── utils/text_embedding/
    ├── corpus.py
    ├── preprocessor.py
    ├── trainer.py
    └── pipeline.py
```
