import dataclasses

import pytest

from ssb.build import Builder
from ssb.config import load_config
from ssb.perturb import SimilarityScorer
from ssb.world import TokenCounter


@pytest.fixture(scope="session")
def counter():
    return TokenCounter("Qwen/Qwen2.5-1.5B-Instruct")


@pytest.fixture(scope="session")
def small_cfg():
    cfg = load_config("configs/main.yaml")
    ds = dataclasses.replace(cfg.dataset, questions={"single_hop": 8, "multi_hop": 8,
                                                     "unanswerable": 6},
                             hard_negative_scorer="tfidf", path="unused.jsonl")
    return dataclasses.replace(cfg, dataset=ds)


@pytest.fixture(scope="session")
def built(small_cfg, counter):
    """A full build on 22 questions (TF-IDF scorer keeps the tests fast and offline)."""
    b = Builder(small_cfg, counter=counter, scorer=SimilarityScorer("tfidf"))
    samples, paras = b.build()
    return b, samples, paras
