"""Task B: passage retrieval. The pool is the sample's context paragraphs; gold is any
copy of the signal.

TF-IDF and BM25 use corpus statistics from every paragraph in the dataset (BM25 scores
only the candidate subset with `get_batch_scores`), so a score never depends on which
other paragraphs happen to share the pool. Exact score ties, common between a gold
paragraph and its NER-swapped copy because the swapped entity is rarely in the question,
are broken by a seeded random key, never by position, and reported as their own outcome.
"""

from __future__ import annotations

import random
import re

import numpy as np

from ..config import ModelConfig
from ..metrics import first_gold_rank

_WORD = re.compile(r"\w+")


def _tok(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class _Ranker:
    def scores(self, question: str, pids: list[str]) -> np.ndarray:
        raise NotImplementedError

    def predict(self, samples: list[dict]) -> list[dict]:
        out = []
        for s in samples:
            pids = s["paragraphs"]
            # round so float noise (e.g. slightly different vector norms) cannot break a
            # real tie; ranking and the tie outcome then agree
            sc = np.round(np.asarray(self.scores(s["question"], pids), dtype=float), 9)
            rng = random.Random(s["seed"])
            keys = [rng.random() for _ in pids]
            order = sorted(range(len(pids)), key=lambda i: (-sc[i], keys[i]))
            ranking = [pids[i] for i in order]
            gold = set(s["signal_ids"])
            r = first_gold_rank(ranking, gold)
            res = {"rank": r, "r1": float(r == 1), "r5": float(r is not None and r <= 5),
                   "mrr": 1.0 / r if r else 0.0,
                   "top_score": float(sc[order[0]]), "n_candidates": len(pids)}
            if s["conflict_id"]:
                g = max(sc[i] for i, p in enumerate(pids) if p in gold)
                c = sc[pids.index(s["conflict_id"])]
                res["outcome"] = ("tie" if g == c else
                                  "true" if g > c else "conflicting")
            out.append(res)
        return out


class TfidfRanker(_Ranker):
    def __init__(self, m: ModelConfig, corpus: dict[str, str]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.ids = list(corpus)
        self.index = {p: i for i, p in enumerate(self.ids)}
        self.vec = TfidfVectorizer(sublinear_tf=True)
        self.X = self.vec.fit_transform([corpus[p] for p in self.ids])  # rows L2-normalised

    def scores(self, question, pids):
        q = self.vec.transform([question])
        rows = self.X[[self.index[p] for p in pids]]
        return np.asarray((rows @ q.T).todense()).ravel()


class BM25Ranker(_Ranker):
    def __init__(self, m: ModelConfig, corpus: dict[str, str]):
        from rank_bm25 import BM25Okapi

        self.ids = list(corpus)
        self.index = {p: i for i, p in enumerate(self.ids)}
        self.bm25 = BM25Okapi([_tok(corpus[p]) for p in self.ids])

    def scores(self, question, pids):
        return np.asarray(self.bm25.get_batch_scores(_tok(question),
                                                     [self.index[p] for p in pids]))


class DenseRanker(_Ranker):
    def __init__(self, m: ModelConfig, corpus: dict[str, str], device: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(m.model_id, device=device)
        self.ids = list(corpus)
        self.index = {p: i for i, p in enumerate(self.ids)}
        self.emb = self.model.encode([corpus[p] for p in self.ids], batch_size=m.batch_size,
                                     normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=False)
        self._q: dict[str, np.ndarray] = {}

    def scores(self, question, pids):
        if question not in self._q:
            self._q[question] = self.model.encode([question], normalize_embeddings=True,
                                                  convert_to_numpy=True,
                                                  show_progress_bar=False)[0]
        return self.emb[[self.index[p] for p in pids]] @ self._q[question]
