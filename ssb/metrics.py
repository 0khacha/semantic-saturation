"""Per-answer scoring and calibration metrics."""

from __future__ import annotations

import math
import re
import string
from collections import Counter

import numpy as np

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)
_PREFIX = re.compile(r"^\s*(final answer|answer)\s*[:\-]\s*", re.I)


def clean_answer(text: str) -> str:
    """First non-empty line, without an 'Answer:' prefix."""
    for line in text.strip().splitlines():
        line = _PREFIX.sub("", line).strip()
        if line:
            return line
    return ""


def normalize(text: str) -> str:
    """SQuAD-style: lower-case, drop punctuation and articles, collapse whitespace."""
    text = clean_answer(text).lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def is_abstention(pred: str) -> bool:
    return normalize(pred) == "unanswerable"


def exact_match(pred: str, aliases: list[str]) -> bool:
    p = normalize(pred)
    return any(p == normalize(a) for a in aliases)


def mentions(pred: str, phrase: str) -> bool:
    """Whole-word mention of a phrase in the normalised prediction."""
    p, a = normalize(pred), normalize(phrase)
    return bool(a) and re.search(rf"(^|\s){re.escape(a)}($|\s)", p) is not None


def token_f1(pred: str, aliases: list[str]) -> float:
    p = normalize(pred).split()
    best = 0.0
    for a in aliases:
        g = normalize(a).split()
        common = sum((Counter(p) & Counter(g)).values())
        if common:
            prec, rec = common / len(p), common / len(g)
            best = max(best, 2 * prec * rec / (prec + rec))
    return best


def confidence(token_logprobs: list[float]) -> float | None:
    """Mean probability of the generated answer tokens."""
    if not token_logprobs:
        return None
    return float(sum(math.exp(lp) for lp in token_logprobs) / len(token_logprobs))


def contradiction_outcome(pred: str, gold: str, conflict: str) -> str:
    """'gold', 'conflict' or 'other' (abstained, mixed, or something else)."""
    g, c = mentions(pred, gold), mentions(pred, conflict)
    if g and not c:
        return "gold"
    if c and not g:
        return "conflict"
    return "other"


def score(sample: dict, generation: dict) -> dict:
    """One row of per-answer metrics."""
    pred = generation["text"]
    answerable = sample["task_type"] != "unanswerable"
    abstained = is_abstention(pred)
    row = {
        "prediction": clean_answer(pred),
        "abstained": abstained,
        "confidence": confidence(generation.get("answer_logprobs") or []),
        "prompt_tokens": generation.get("prompt_tokens"),
    }
    if answerable:
        row["correct"] = exact_match(pred, sample["answer_aliases"])
        row["f1"] = token_f1(pred, sample["answer_aliases"])
        row["lenient"] = any(mentions(pred, a) for a in sample["answer_aliases"])
        row["hallucinated"] = None
        row["false_abstention"] = abstained
    else:
        row["correct"] = abstained  # the right behaviour is to abstain
        row["f1"] = float(abstained)
        row["lenient"] = abstained
        row["hallucinated"] = not abstained
        row["false_abstention"] = None
    row["outcome"] = (contradiction_outcome(pred, sample["gold_answer"],
                                            sample["conflict_answer"])
                      if sample["axis"] == "contradiction" else None)
    return row


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
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


def brier(conf: np.ndarray, correct: np.ndarray) -> float:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return float(np.mean((conf - correct) ** 2)) if conf.size else float("nan")
