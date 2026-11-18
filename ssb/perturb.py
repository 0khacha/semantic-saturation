"""Perturbations: which paragraphs surround the gold evidence, and where everything goes.

All functions are pure given their inputs (randomness comes in as a seeded Random).
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter

from .world import Paragraph, contains_phrase

N_TEMPLATES = 6  # fact templates per relation; see world.RELATIONS


# --------------------------------------------------------------------------- placement


def random_slots(n: int, k: int, rng: random.Random) -> list[int]:
    """k distinct positions in [0, n), sorted."""
    if k > n:
        raise ValueError(f"cannot place {k} paragraphs in a context of {n}")
    return sorted(rng.sample(range(n), k))


def depth_block_start(n: int, block: int, depth: float) -> int:
    """Start index of a contiguous block of `block` paragraphs at relative depth `depth`.

    depth 0 puts the block first, 1.0 puts it last, 0.5 centres it.
    """
    if block > n:
        raise ValueError(f"block of {block} does not fit in {n}")
    return round(depth * (n - block))


def contradiction_slots(n: int, position: str) -> tuple[int, int]:
    """(gold index, conflict index) at the one-third and two-thirds marks of the context.

    Both claims sit the same distance from the edges, so `before`/`after` changes only
    which claim the model reads last, not how deep either one is.
    """
    first, second = round((n - 1) / 3), round(2 * (n - 1) / 3)
    if first == second:
        raise ValueError(f"context of {n} is too short to separate gold and conflict")
    if position == "before":  # conflict comes before gold
        return second, first
    if position == "after":
        return first, second
    raise ValueError(f"unknown contradiction position {position!r}")


def assemble(n: int, placed: dict[int, Paragraph], fillers: list[Paragraph]) -> list[Paragraph]:
    """Context of n paragraphs: `placed` at fixed indices, fillers in order elsewhere."""
    need = n - len(placed)
    if any(not 0 <= i < n for i in placed):
        raise ValueError("placed index out of range")
    if len(fillers) < need:
        raise ValueError(f"need {need} fillers, have {len(fillers)}")
    it = iter(fillers[:need])
    return [placed[i] if i in placed else next(it) for i in range(n)]


# --------------------------------------------------------------------------- redundancy


def paraphrase_variants(base_variant: int, k: int) -> list[int]:
    """Template variants for k renderings of one fact: the base first, then paraphrases."""
    if k > N_TEMPLATES - 1:  # one template is reserved for the conflicting claim
        raise ValueError(f"k={k} exceeds the {N_TEMPLATES - 1} available paraphrases")
    return [(base_variant + j) % N_TEMPLATES for j in range(k)]


def conflict_variant(base_variant: int) -> int:
    """Template for the conflicting claim: never used by the gold or its paraphrases."""
    return (base_variant + N_TEMPLATES - 1) % N_TEMPLATES


# --------------------------------------------------------------------------- noise


def filter_candidates(pool: list[Paragraph], exclude_ids: set[str],
                      banned_phrases: list[str]) -> list[Paragraph]:
    """Pool paragraphs that are not gold and mention none of the banned phrases."""
    return [p for p in pool if p.id not in exclude_ids
            and not any(contains_phrase(p.text, b) for b in banned_phrases)]


class SimilarityScorer:
    """Cosine similarity between a question and paragraphs.

    `embedding` uses a sentence-transformers model; `tfidf` is a dependency-free
    fallback (used by the unit tests, never by the published dataset).
    """

    def __init__(self, kind: str, model_name: str | None = None):
        self.kind = kind
        if kind == "embedding":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name, device="cpu")
        elif kind != "tfidf":
            raise ValueError(f"unknown scorer {kind!r}")

    def fit(self, texts: list[str]) -> None:
        """Encode the paragraph collection once."""
        self._texts = texts
        if self.kind == "embedding":
            self._emb = self._model.encode(texts, batch_size=64, normalize_embeddings=True,
                                           convert_to_numpy=True, show_progress_bar=False)
        else:
            docs = [_words(t) for t in texts]
            df = Counter(w for d in docs for w in set(d))
            self._idf = {w: math.log((1 + len(docs)) / (1 + c)) + 1 for w, c in df.items()}
            self._vecs = [self._tfidf(d) for d in docs]

    def scores(self, query: str) -> list[float]:
        if self.kind == "embedding":
            q = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True,
                                   show_progress_bar=False)[0]
            return [float(x) for x in self._emb @ q]
        qv = self._tfidf(_words(query))
        return [sum(qv.get(w, 0.0) * v for w, v in d.items()) for d in self._vecs]

    def _tfidf(self, words: list[str]) -> dict[str, float]:
        tf = Counter(words)
        vec = {w: c * self._idf.get(w, 0.0) for w, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {w: v / norm for w, v in vec.items()}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rank_hard_negatives(candidates: list[Paragraph], sims: dict[str, float]) -> list[Paragraph]:
    """Most similar to the question first; ties broken by id for determinism."""
    return sorted(candidates, key=lambda p: (-sims[p.id], p.id))
