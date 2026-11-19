"""Perturbations: where the signal, its copies, the conflict and the filler go.

Placement functions are pure (randomness arrives as a seeded Random). The hard-negative
selector embeds with an encoder that is *not* one of the evaluated models
(all-mpnet-base-v2 by default), so selection does not favour any retriever.
"""

from __future__ import annotations

import random

import numpy as np

from .paragraphs import Paragraph


def n_paragraphs(level: int, n_evidence: int) -> int:
    """Context length for size level N: never shorter than its own evidence.

    N = 1 with five redundant copies is five paragraphs; with a conflict, two. So every
    example takes part at every level and curves always compare the same examples.
    """
    return max(level, n_evidence)


def random_slots(n: int, k: int, rng: random.Random) -> list[int]:
    if k > n:
        raise ValueError(f"cannot place {k} paragraphs in a context of {n}")
    return sorted(rng.sample(range(n), k))


def depth_index(n: int, depth: float) -> int:
    """Index of a single paragraph at relative depth (0 = first, 1 = last)."""
    return round(depth * (n - 1))


def contradiction_slots(position: str) -> tuple[int, int]:
    """(signal index, conflict index): adjacent, at the start of the context.

    Only the order differs between `before` and `after`, and both paragraphs stay
    inside a 512-token truncation window, so truncation cannot decide the outcome.
    """
    if position == "before":  # the conflict is read first
        return 1, 0
    if position == "after":
        return 0, 1
    raise ValueError(f"unknown contradiction position {position!r}")


def assemble(n: int, placed: dict[int, Paragraph], fillers: list[Paragraph]) -> list[Paragraph]:
    """Context of n paragraphs: `placed` at fixed indices, fillers in order elsewhere."""
    if any(not 0 <= i < n for i in placed):
        raise ValueError("placed index out of range")
    need = n - len(placed)
    if len(fillers) < need:
        raise ValueError(f"need {need} fillers, have {len(fillers)}")
    it = iter(fillers[:need])
    return [placed[i] if i in placed else next(it) for i in range(n)]


class Selector:
    """Cosine similarity with a sentence encoder; embeddings cached by paragraph id."""

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, ids: list[str], texts: list[str], batch_size: int = 64) -> np.ndarray:
        todo = [(i, t) for i, t in zip(ids, texts, strict=True) if i not in self._cache]
        if todo:
            emb = self.model.encode([t for _, t in todo], batch_size=batch_size,
                                    normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False)
            for (i, _), e in zip(todo, emb, strict=True):
                self._cache[i] = e
        return np.stack([self._cache[i] for i in ids])

    def query(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True,
                                 show_progress_bar=False)[0]


def rank_by_similarity(query_vec: np.ndarray, pool: list[Paragraph],
                       pool_emb: np.ndarray, limit: int) -> list[tuple[Paragraph, float]]:
    """Top `limit` pool paragraphs by cosine similarity (ties broken by id)."""
    sims = pool_emb @ query_vec
    order = sorted(range(len(pool)), key=lambda i: (-sims[i], pool[i].id))[:limit]
    return [(pool[i], float(sims[i])) for i in order]
