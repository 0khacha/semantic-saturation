"""Metrics, lexical rankers, analysis engine and runner resume (no large models)."""

import json

import numpy as np
import pandas as pd
import pytest

from ssb import metrics
from ssb.analysis import Boot, ModelAnalysis, Stat, holm
from ssb.config import ModelConfig, load_config
from ssb.tasks.retrieval import BM25Ranker, TfidfRanker

# ---- metrics


@pytest.mark.parametrize("pred,ok", [("Denver Broncos", True), ("the denver broncos.", True),
                                     ("Broncos", False), ("", False)])
def test_exact_match(pred, ok):
    assert metrics.exact_match(pred, ["Denver Broncos"]) is ok


def test_token_f1_and_qa_outcome():
    assert metrics.token_f1("Denver", ["Denver Broncos"]) == pytest.approx(2 / 3)
    assert metrics.qa_outcome("Denver Broncos", ["Denver Broncos"], "Carolina Panthers") == "true"
    assert metrics.qa_outcome("Carolina Panthers", ["Denver Broncos"],
                              "Carolina Panthers") == "conflicting"
    assert metrics.qa_outcome("in 1990", ["1990"], "1987") == "true"  # partial overlap
    assert metrics.qa_outcome("", ["1990"], "1987") == "other"
    assert metrics.qa_outcome("the stadium", ["1990"], "1987") == "other"


def test_rank_ece_brier_macro_f1():
    assert metrics.first_gold_rank(["a", "b", "c"], {"c", "z"}) == 3
    assert metrics.first_gold_rank(["a"], {"z"}) is None
    assert metrics.ece(np.array([0.95, 0.95, 0.15, 0.15]), np.array([1, 1, 0, 0])) == \
        pytest.approx(0.1)
    assert metrics.brier_multiclass(np.array([[1, 0, 0, 0]]), np.array([0])) == 0.0
    y = np.array([0, 1, 2, 3])
    assert metrics.macro_f1(y, y) == 1.0


# ---- retrieval rankers


CORPUS = {"g": "The Broncos won Super Bowl 50 in Santa Clara in February 2016.",
          "c": "The Panthers won Super Bowl 50 in Santa Clara in February 2016.",
          "f1": "Bread dough rises faster in a warm kitchen.",
          "f2": "Tides rise and fall with the pull of the moon."}
# BM25Okapi gives zero IDF to terms in half the documents, so a realistic corpus needs
# more than a handful of unrelated paragraphs
CORPUS.update({f"x{i}": f"Unrelated note {i} about gardening, weather and old maps."
               for i in range(12)})


def _sample(paras, signal="g", conflict=None, seed=0):
    return {"question": "Where was Super Bowl 50 played?", "paragraphs": paras,
            "signal_ids": [signal], "conflict_id": conflict, "seed": seed}


@pytest.mark.parametrize("cls", [TfidfRanker, BM25Ranker])
def test_rankers_find_gold_and_report_ties(cls):
    r = cls(ModelConfig(name="x", task="retrieval", kind="bm25"), CORPUS)
    (res,) = r.predict([_sample(["f1", "g", "f2"])])
    assert res["rank"] == 1 and res["r1"] == 1.0 and res["mrr"] == 1.0
    # swapped copy differs only in the answer entity, which the question never mentions
    (res,) = r.predict([_sample(["f1", "c", "g"], conflict="c")])
    assert res["outcome"] == "tie"
    ranks = {r.predict([_sample(["c", "g"], conflict="c", seed=s)])[0]["rank"]
             for s in range(30)}
    assert ranks == {1, 2}  # ties broken by a seeded coin, not by position


def test_rankers_are_position_invariant():
    r = TfidfRanker(ModelConfig(name="x", task="retrieval", kind="tfidf"), CORPUS)
    a = r.predict([_sample(["g", "f1", "f2"])])[0]["rank"]
    b = r.predict([_sample(["f1", "f2", "g"])])[0]["rank"]
    assert a == b == 1


# ---- analysis


def test_holm_and_stat():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05) == {"a"}
    s = Stat(0.3, np.linspace(0.1, 0.5, 1001))
    assert s.p_greater_than_zero() == 0.0
    assert Stat(0, np.linspace(-1, 1, 1001)).p_two_sided() == pytest.approx(1.0, abs=0.01)


def test_boot_weighted_mean_matches_resampling():
    ids = np.array(["a", "a", "b", "c"])
    boot = Boot(ids, 100, 0)
    v = np.array([1.0, 0.0, 1.0, 0.0])
    point, b = boot.mean(np.arange(4), v)
    rows = np.repeat(np.arange(4), boot.W[5][boot.codes].astype(int))
    assert point == 0.5 and b[5] == pytest.approx(v[rows].mean())


def _frame(acc_by_n, n_base=200):
    rows = []
    for i in range(n_base):
        for n, acc in acc_by_n.items():
            rows.append({"model": "m", "task": "classification", "kind": "tfidf_lr",
                         "base_id": f"b{i}", "axis": "dilution", "N": n, "depth": 0.0,
                         "noise_type": "random", "redundancy_mode": None,
                         "contradiction_pos": None, "answerable": True, "label": 0,
                         "pred": 0 if i < acc * n_base else 1,
                         "correct": float(i < acc * n_base), "confidence": 0.9})
    return pd.DataFrame(rows)


def test_curve_quantities():
    cfg = load_config("configs/main.yaml")
    ma = ModelAnalysis(_frame({1: 0.9, 5: 0.9, 20: 0.6, 100: 0.2}), cfg)
    assert ma.score("dilution", 20).point == pytest.approx(0.6)
    assert ma.saturation_point("dilution") == 5
    onset, peak = ma.degradation_onset("dilution")
    assert peak == 1 and onset == 20
    x = np.log10([1, 5, 20, 100])
    assert ma.sss("dilution").point == pytest.approx(np.trapezoid([0.9, 0.9, 0.6, 0.2], x) / 2)
    assert ma.stats[("ece", "dilution", 100)].point == pytest.approx(0.7)
    assert ma.stats[("overconfidence", "dilution", 100)].point == pytest.approx(0.7)


# ---- runner resume


def test_runner_resumes_and_skips_finished(tmp_path, monkeypatch, built):
    import dataclasses

    from ssb.build import write_dataset
    from ssb.run import Runner

    b, samples, paras = built
    cfg = load_config("configs/smoke.yaml")
    data = dataclasses.replace(cfg.data, path=str(tmp_path / "d.jsonl"))
    run = dataclasses.replace(cfg.run, output_dir=str(tmp_path / "out"), device="cpu")
    m = ModelConfig(name="bm25", task="retrieval", kind="bm25", batch_size=4)
    cfg = dataclasses.replace(cfg, data=data, run=run, models=[m])
    write_dataset(cfg, b, samples, paras)
    calls = []
    real = BM25Ranker.predict

    def counting(self, batch):
        calls.append(len(batch))
        if len(calls) == 3:
            raise RuntimeError("simulated crash")
        return real(self, batch)

    monkeypatch.setattr(BM25Ranker, "predict", counting)
    with pytest.raises(RuntimeError):
        Runner(cfg, log=lambda *_: None).run()
    path = tmp_path / "out" / "predictions" / "bm25.jsonl"
    done = len(path.read_text().splitlines())
    assert done == 8  # two finished batches of 4 survived
    calls.clear()
    monkeypatch.setattr(BM25Ranker, "predict", lambda self, batch: calls.append(len(batch))
                        or real(self, batch))
    Runner(cfg, log=lambda *_: None).run()
    n_ret = sum(s["task"] == "retrieval" for s in samples)
    assert sum(calls) == n_ret - done
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert len({x["sample_id"] for x in lines}) == n_ret == len(lines)
