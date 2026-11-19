"""Shared fixtures: a small but real build (AG News + SQuAD 2.0 from the local cache, real
spaCy NER) with light stand-ins for the three heavy models: a TF-IDF selector instead of
mpnet, a rule-based paraphraser instead of MarianMT, and a lookup judge."""

import dataclasses

import numpy as np
import pytest

from ssb.build import Builder
from ssb.config import load_config
from ssb.data import load_agnews, load_squad
from ssb.paragraphs import TokenCounter, split_sentences

PREFIXES = ["Notably, ", "In short, ", "Reportedly, ", "Indeed, ", "Overall, ", "Today, "]


class FakeTranslator:
    """Deterministic 'paraphrases': lead-in words and sentence rotations. Keeps answers."""

    def __init__(self):
        self.source_of: dict[str, str] = {}

    def paraphrases(self, texts):
        out = {}
        for t in texts:
            sents = split_sentences(t)
            rot = [" ".join(sents[i:] + sents[:i]) for i in range(1, len(sents))]
            cands = [p + t[0].lower() + t[1:] for p in PREFIXES] + rot
            cands = [c for c in dict.fromkeys(cands) if c != t]
            for c in cands:
                self.source_of[c] = t
            out[t] = cands
        return out


class FakeJudge:
    def __init__(self, translator, label_of_text):
        self.tr, self.label_of = translator, label_of_text
        self.perm, self.calibration_accuracy = [0, 1, 2, 3], 1.0

    def predict(self, texts):
        return [self.label_of.get(self.tr.source_of.get(t, t), -1) for t in texts]


class FakeSelector:
    """TF-IDF cosine instead of a sentence encoder."""

    def __init__(self):
        self.vec = None
        self._cache = {}

    def embed(self, ids, texts, batch_size=64):
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self.vec is None:
            self.vec = TfidfVectorizer(stop_words="english").fit(texts)
        todo = [(i, t) for i, t in zip(ids, texts, strict=True) if i not in self._cache]
        if todo:
            m = self.vec.transform([t for _, t in todo]).toarray()
            m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
            for (i, _), row in zip(todo, m, strict=True):
                self._cache[i] = row
        return np.stack([self._cache[i] for i in ids])

    def query(self, text):
        v = self.vec.transform([text]).toarray()[0]
        return v / (np.linalg.norm(v) + 1e-12)


@pytest.fixture(scope="session")
def counter():
    return TokenCounter("bert-base-uncased")


@pytest.fixture(scope="session")
def raw():
    cfg = load_config("configs/smoke.yaml")
    return load_agnews(cfg.data.raw_dir), load_squad(cfg.data.raw_dir)


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory):
    cfg = load_config("configs/smoke.yaml")
    d = dataclasses.replace(cfg.data, examples={"classification": 8, "retrieval": 8, "qa": 8},
                            random_pool=200, hard_pool=200,
                            path=str(tmp_path_factory.mktemp("data") / "ssb_test.jsonl"))
    return dataclasses.replace(cfg, data=d)


def make_builder(cfg, counter, raw):
    (ag_train, ag_test), (dev, train_paras) = raw
    tr = FakeTranslator()
    judge = FakeJudge(tr, {x.text: x.label for x in ag_test})
    return Builder(cfg, counter=counter, selector=FakeSelector(), translator=tr, judge=judge,
                   loaders={"agnews": lambda: (ag_train, ag_test),
                            "squad": lambda: (dev, train_paras)},
                   log=lambda *_: None)


@pytest.fixture(scope="session")
def built(small_cfg, counter, raw):
    b = make_builder(small_cfg, counter, raw)
    samples, paras = b.build()
    return b, samples, paras
