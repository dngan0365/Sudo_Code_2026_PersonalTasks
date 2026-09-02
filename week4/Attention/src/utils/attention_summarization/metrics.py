from __future__ import annotations


def rouge_l_score(reference: str, prediction: str) -> float:
    reference_tokens = reference.split()
    prediction_tokens = prediction.split()
    if not reference_tokens or not prediction_tokens:
        return 0.0
    lcs = _lcs_length(reference_tokens, prediction_tokens)
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def average_rouge_l(references: list[str], predictions: list[str]) -> float:
    if not references:
        return 0.0
    return sum(rouge_l_score(reference, prediction) for reference, prediction in zip(references, predictions)) / len(references)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]
