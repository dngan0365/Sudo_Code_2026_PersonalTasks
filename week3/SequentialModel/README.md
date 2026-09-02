# LSTM Text Generation

Week 6 implementation: train mot LSTM don gian de sinh van ban tu corpus sach/truyen trong `data/output`.

Pipeline huong doi tuong:

- `TextCorpusLoader`: doc nhieu file `.txt`, chuan hoa Unicode va whitespace.
- `CharVocabulary`: build character vocabulary va encode/decode text.
- `CharSequenceDataset`: tao sequence input/target cho next-character prediction.
- `CharLSTM`: Embedding + LSTM + Linear classifier du doan ky tu tiep theo.
- `LSTMTrainer`: train model, report loss/perplexity va sinh sample sau moi epoch.
- `TextGenerator`: load checkpoint va sinh van ban moi tu prompt.
- `TextGenerationPipeline`: dieu phoi toan bo qua trinh.

## Cai dat

Tu thu muc `week3/SequentialModel`, chay:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Tren macOS/Linux, kich hoat moi truong bang `source .venv/bin/activate`.

## Chay training

Train theo config mac dinh:

```powershell
python src/main.py
```

Chay nhanh de kiem tra pipeline:

```powershell
python src/main.py --epochs 1 --max-files 2 --max-chars 50000 --batch-size 32 --device cpu
```

Toan bo tham so hardcode nam trong `config/default.yaml`: corpus path, so file, so ky tu toi da, sequence length, batch size, epoch, learning rate, kien truc LSTM, prompt sinh mau, temperature va top-k.

## Report ket qua

Sau khi train, `output/` se co:

- `history.csv`: training loss va perplexity theo epoch.
- `generated_samples.csv`: text samples sinh ra sau moi epoch.
- `generated_samples.md`: samples dang Markdown de doc.
- `training_report.json`: corpus size, vocab size, so sequence, final metrics va config.
- `best_model.pt`: checkpoint tot nhat theo training loss.

## Generate sau khi train

```powershell
python src/generate.py --prompt "Ngay xua" --length 500 --temperature 0.8 --top-k 40
```

`temperature` thap hon se sinh text chac tay hon nhung de lap; cao hon se da dang hon nhung de nhieu.

## Cau truc

```text
SequentialModel/
├── config/default.yaml
├── data/
│   ├── archive.zip
│   └── output/*.txt
├── notebooks/lstm_text_generation_draft.ipynb
├── src/
│   ├── main.py
│   ├── generate.py
│   └── utils/text_generation/
│       ├── config.py
│       ├── corpus.py
│       ├── dataset.py
│       ├── generator.py
│       ├── model.py
│       ├── pipeline.py
│       ├── trainer.py
│       └── vocabulary.py
├── README.md
└── requirements.txt
```
