"""Metrics: accuracy, macro-F1, Recall@k, MRR, EM, token F1, ECE, Brier."""

from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD normalisation: lower-case, strip punctuation and articles, squash spaces."""
    text = text.lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def exact_match(pred: str, golds: list[str]) -> bool:
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in golds)


def token_f1(pred: str, golds: list[str]) -> float:
    p = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        gt = normalize_answer(g).split()
        common = sum((Counter(p) & Counter(gt)).values())
        if common:
            pr, rc = common / len(p), common / len(gt)
            best = max(best, 2 * pr * rc / (pr + rc))
    return best


def qa_outcome(pred: str, golds: list[str], conflict: str) -> str:
    """Contradiction outcome for QA: 'true', 'conflicting' or 'other'."""
    if not normalize_answer(pred):
        return "other"  # no answer given
    if exact_match(pred, golds):
        return "true"
    if exact_match(pred, [conflict]):
        return "conflicting"
    t, c = token_f1(pred, golds), token_f1(pred, [conflict])  # partial overlap decides
    if t > c:
        return "true"
    if c > t:
        return "conflicting"
    return "other"


def first_gold_rank(ranking: list[str], gold_ids: set[str]) -> int | None:
    """1-based rank of the first gold paragraph in a ranked id list."""
    for i, pid in enumerate(ranking, 1):
        if pid in gold_ids:
            return i
    return None


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> float:
    f1s = []
    for k in range(n_classes):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    if conf.size == 0:
        return float("nan")
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(conf[m].mean() - correct[m].mean())
    return float(total / conf.size)


def brier_multiclass(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between the probability vector and the one-hot label."""
    probs = np.asarray(probs, float)
    onehot = np.eye(probs.shape[1])[np.asarray(y, int)]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
