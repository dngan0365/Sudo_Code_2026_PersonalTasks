from __future__ import annotations

import math
from collections import Counter


def corpus_bleu(references: list[str], predictions: list[str], max_n: int = 4) -> float:
    matches = [0] * max_n
    totals = [0] * max_n
    ref_len = 0
    pred_len = 0
    for reference, prediction in zip(references, predictions):
        ref_tokens = reference.split()
        pred_tokens = prediction.split()
        ref_len += len(ref_tokens)
        pred_len += len(pred_tokens)
        for n in range(1, max_n + 1):
            ref_ngrams = _ngrams(ref_tokens, n)
            pred_ngrams = _ngrams(pred_tokens, n)
            totals[n - 1] += sum(pred_ngrams.values())
            overlap = pred_ngrams & ref_ngrams
            matches[n - 1] += sum(overlap.values())
    if pred_len == 0:
        return 0.0
    precisions = [(matches[i] + 1) / (totals[i] + 1) for i in range(max_n)]
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(1, pred_len))
    return brevity_penalty * math.exp(sum(math.log(value) for value in precisions) / max_n)


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + n]) for index in range(0, max(0, len(tokens) - n + 1)))
