# Transformer Translation

Week 4 implementation: build a Transformer sequence-to-sequence model for English-to-Vietnamese translation and report translation quality.

Dataset is EVBCorpus EVBNews v2.0 in SGML format. Each file contains aligned sentence pairs:

```xml
<s id='en1'>What is a Fenqing ?</s>
<s id='vn1'>Fenqing la gi ?</s>
```

## Setup

From `week4/Transformer`:

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
python src/main.py --epochs 1 --max-pairs 1000 --batch-size 16 --device cpu
```

Default hyperparameters live in `config/default.yaml`: corpus path, source/target language ids, vocab sizes, max sequence lengths, Transformer dimensions, training settings and sample count.

## Model

The model uses PyTorch `nn.Transformer`:

```text
English tokens -> source ids -> source embedding + positional encoding -> Transformer encoder
Vietnamese tokens -> target ids -> target embedding + positional encoding -> causal Transformer decoder
decoder output -> linear layer -> next Vietnamese token
```

## Translation Quality Report

After training, `output/` contains:

- `history.csv`: train and validation loss by epoch.
- `best_model.pt`: checkpoint with model state, vocabularies and preprocessor.
- `translation_samples.csv`: source, reference and predicted translations.
- `translation_samples.md`: readable sample report.
- `evaluation_report.json`: split sizes, vocab sizes, best validation loss and sample BLEU.

Translate text after training:

```powershell
python src/translate.py --text "What is a Fenqing ?"
```

## Structure

```text
Transformer/
├── config/default.yaml
├── data/EVBCorpus_EVBNews_v2.0/*.sgml
├── notebooks/transformer_translation_draft.ipynb
├── src/
│   ├── main.py
│   ├── translate.py
│   └── utils/transformer_translation/
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
