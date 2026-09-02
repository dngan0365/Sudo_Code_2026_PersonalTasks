# Attention Text Summarization

Week 4 implementation: train an encoder-decoder LSTM with additive attention for Vietnamese text summarization.

Dataset files in `data/` are parquet files with two columns:

- `Content`: source article/content.
- `Summary`: target summary.

## Setup

From `week4/Attention`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Train

Run the default experiment:

```powershell
python src/main.py
```

Run a small smoke test:

```powershell
python src/main.py --epochs 1 --train-limit 200 --valid-limit 50 --test-limit 20 --batch-size 8 --device cpu
```

All default parameters are in `config/default.yaml`: data paths, text columns, max input/summary tokens, vocab sizes, model dimensions, training settings and generation sample count.

## Model

The summarizer is a classic attention baseline:

```text
Content tokens -> source vocab ids -> BiLSTM encoder
Summary tokens -> decoder input -> LSTMCell decoder + Bahdanau attention -> next summary token
```

The attention layer scores each encoder state against the decoder hidden state and builds a context vector at every decoding step.

## Results

After training, `output/` contains:

- `history.csv`: train loss and validation loss by epoch.
- `best_model.pt`: checkpoint with model state, vocabularies and preprocessor.
- `generated_summaries.csv`: generated summaries for test samples.
- `generated_summaries.md`: generated summaries in a readable report format.
- `evaluation_report.json`: row counts, vocab sizes, best validation loss and sample ROUGE-L.

Generate a summary after training:

```powershell
python src/summarize.py --text "Noi dung bai viet can tom tat o day"
```

## Structure

```text
Attention/
├── config/default.yaml
├── data/
│   ├── train-00000-of-00001.parquet
│   ├── valid-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
├── notebooks/attention_summarization_draft.ipynb
├── src/
│   ├── main.py
│   ├── summarize.py
│   └── utils/attention_summarization/
│       ├── config.py
│       ├── data_loader.py
│       ├── dataset.py
│       ├── generator.py
│       ├── metrics.py
│       ├── model.py
│       ├── pipeline.py
│       ├── preprocessor.py
│       ├── trainer.py
│       └── vocabulary.py
├── README.md
└── requirements.txt
```
