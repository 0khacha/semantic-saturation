"""Loaders for AG News, SQuAD 2.0 and the Wikipedia filler pool.

Datasets come from the Hugging Face Hub and are cached under `data.raw_dir`.
Random filler for every task is SQuAD 2.0 *train* paragraphs: Wikipedia text whose
442 articles share no title with the 35 dev articles the questions come from.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass
class NewsItem:
    id: str
    text: str
    label: int


@dataclass
class SquadQuestion:
    id: str
    title: str
    context: str
    question: str
    answers: list[str]  # distinct gold answer strings; empty when unanswerable
    answer_start: int | None = None  # of the first answer
    meta: dict = field(default_factory=dict)

    @property
    def answerable(self) -> bool:
        return bool(self.answers)


def _load(name: str, raw_dir: str):
    from datasets import load_dataset

    return load_dataset(name, cache_dir=raw_dir)


def load_agnews(raw_dir: str) -> tuple[list[NewsItem], list[NewsItem]]:
    ds = _load("fancyzhx/ag_news", raw_dir)
    train = [NewsItem(f"agtr{i}", r["text"], r["label"]) for i, r in enumerate(ds["train"])]
    test = [NewsItem(f"agte{i}", r["text"], r["label"]) for i, r in enumerate(ds["test"])]
    return train, test


def load_squad(raw_dir: str) -> tuple[list[SquadQuestion], list[dict]]:
    """Dev questions, and train paragraphs as [{'title', 'context'}] (deduplicated)."""
    ds = _load("rajpurkar/squad_v2", raw_dir)
    dev = []
    for r in ds["validation"]:
        texts = list(dict.fromkeys(t.strip() for t in r["answers"]["text"] if t.strip()))
        start = r["answers"]["answer_start"][0] if r["answers"]["answer_start"] else None
        dev.append(SquadQuestion(r["id"], r["title"], r["context"], r["question"], texts, start))
    seen, train = set(), []
    for r in ds["train"]:
        if r["context"] not in seen:
            seen.add(r["context"])
            train.append({"title": r["title"], "context": r["context"]})
    return dev, train


def stratified_sample(items: list, key, n: int, rng: random.Random) -> list:
    """n items, as equal as possible across strata, deterministic given rng."""
    groups: dict = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups, key=str)
    out = []
    while len(out) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    if len(out) < n:
        raise ValueError(f"only {len(out)} items available, wanted {n}")
    return out
