"""Conflicting signal: NER entity swap (SQuAD) and cross-class insertion (AG News).

The swapped copy is the gold paragraph with every occurrence of the answer replaced by
another entity of the same spaCy type that appears nowhere in the paragraph. It reads
almost exactly like the gold but asserts a different answer.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

# entity types whose swap yields a fluent, type-consistent sentence
SWAPPABLE = {"PERSON", "ORG", "GPE", "LOC", "NORP", "DATE", "CARDINAL", "EVENT", "FAC",
             "WORK_OF_ART", "LAW", "LANGUAGE", "PRODUCT", "QUANTITY", "MONEY", "PERCENT",
             "ORDINAL", "TIME"}


def _pattern(answer: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(answer)}(?!\w)")


class EntitySwapper:
    def __init__(self, model: str = "en_core_web_sm"):
        import spacy

        self.nlp = spacy.load(model, disable=["parser", "lemmatizer"])
        self.bank: dict[str, list[str]] = defaultdict(list)

    def fit(self, texts: list[str]) -> None:
        """Collect replacement candidates per entity type from a corpus."""
        seen: dict[str, set] = defaultdict(set)
        for doc in self.nlp.pipe(texts, batch_size=64):
            for e in doc.ents:
                t = e.text.strip()
                if e.label_ in SWAPPABLE and len(t) > 1 and t.lower() not in seen[e.label_]:
                    seen[e.label_].add(t.lower())
                    self.bank[e.label_].append(t)
        for v in self.bank.values():
            v.sort()

    def answer_types(self, items: list[tuple[str, str, int | None]]) -> list[str | None]:
        """spaCy type of each (context, answer, start) answer span, if it is essentially one
        entity: the entity and the answer span overlap by at least 60% of the longer of the
        two (so "the Denver Broncos" vs entity "Denver Broncos" still counts)."""
        out = []
        contexts = [c for c, _, _ in items]
        for doc, (_, answer, start) in zip(self.nlp.pipe(contexts, batch_size=64), items,
                                           strict=True):
            label = None
            a0 = start
            a1 = None if start is None else start + len(answer)
            for e in doc.ents:
                if e.label_ not in SWAPPABLE:
                    continue
                if a0 is None:
                    ok = e.text.strip() == answer.strip()
                else:
                    overlap = min(a1, e.end_char) - max(a0, e.start_char)
                    ok = overlap >= 0.6 * max(len(answer), e.end_char - e.start_char)
                if ok:
                    label = e.label_
                    break
            out.append(label)
        return out

    def swap(self, context: str, answers: list[str], label: str,
             rng: random.Random) -> tuple[str, str] | None:
        """(corrupted context, replacement) or None when no safe replacement exists."""
        lowered = context.lower()
        candidates = [c for c in self.bank.get(label, [])
                      if c.lower() not in lowered
                      and not any(a.lower() in c.lower() or c.lower() in a.lower()
                                  for a in answers)]
        if not candidates:
            return None
        repl = rng.choice(candidates)
        new = context
        for a in sorted(answers, key=len, reverse=True):  # every gold alias goes
            new = _pattern(a).sub(repl, new)
        if new == context or any(_pattern(a).search(new) for a in answers):
            return None
        return new, repl


def cross_class(label: int, pool_by_label: dict[int, list], rng: random.Random):
    """A news item from a different class than `label` (class chosen uniformly)."""
    other = rng.choice(sorted(k for k in pool_by_label if k != label))
    return rng.choice(pool_by_label[other])
