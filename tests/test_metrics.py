import math

import numpy as np
import pytest

from ssb import metrics


@pytest.mark.parametrize("pred,ok", [
    ("Kusorin", True), ("kusorin.", True), ("Answer: Kusorin", True), ("The Kusorin", True),
    ("Kusorin\nbecause the context says so", True), ("Kusorin town", False), ("", False),
])
def test_exact_match(pred, ok):
    assert metrics.exact_match(pred, ["Kusorin"]) is ok


def test_abstention_is_strict():
    assert metrics.is_abstention("UNANSWERABLE")
    assert metrics.is_abstention("unanswerable.")
    assert not metrics.is_abstention("I think it is unanswerable")
    assert not metrics.is_abstention("1702")


def test_token_f1():
    assert metrics.token_f1("Zeldel Lehun", ["Zeldel Lehun"]) == 1.0
    assert metrics.token_f1("Zeldel", ["Zeldel Lehun"]) == pytest.approx(2 / 3)
    assert metrics.token_f1("nothing", ["Zeldel Lehun"]) == 0.0


def test_contradiction_outcome():
    assert metrics.contradiction_outcome("1692", "1692", "1686") == "gold"
    assert metrics.contradiction_outcome("It was 1686.", "1692", "1686") == "conflict"
    assert metrics.contradiction_outcome("1692 or 1686", "1692", "1686") == "other"
    assert metrics.contradiction_outcome("UNANSWERABLE", "1692", "1686") == "other"
    assert metrics.contradiction_outcome("16920", "1692", "1686") == "other"


def test_confidence_is_mean_token_probability():
    assert metrics.confidence([math.log(0.5), math.log(1.0)]) == pytest.approx(0.75)
    assert metrics.confidence([]) is None


def test_ece_and_brier():
    conf = np.array([0.95, 0.95, 0.15, 0.15])
    correct = np.array([1, 1, 0, 0])
    assert metrics.ece(conf, correct) == pytest.approx(0.1)
    assert metrics.brier(conf, correct) == pytest.approx((0.05**2 * 2 + 0.15**2 * 2) / 4)
    # perfectly overconfident: always 1.0, half right
    assert metrics.ece(np.ones(4), np.array([1, 0, 1, 0])) == pytest.approx(0.5)


def test_score_unanswerable_and_answerable():
    s_un = {"task_type": "unanswerable", "axis": "noise", "answer_aliases": ["UNANSWERABLE"],
            "gold_answer": "UNANSWERABLE", "conflict_answer": None}
    r = metrics.score(s_un, {"text": "1777", "answer_logprobs": [-0.1]})
    assert r["correct"] is False and r["hallucinated"] is True
    r = metrics.score(s_un, {"text": "UNANSWERABLE", "answer_logprobs": [-0.1]})
    assert r["correct"] is True and r["hallucinated"] is False
    s_a = {"task_type": "single_hop", "axis": "contradiction", "answer_aliases": ["1692"],
           "gold_answer": "1692", "conflict_answer": "1686"}
    r = metrics.score(s_a, {"text": "UNANSWERABLE", "answer_logprobs": []})
    assert r["false_abstention"] is True and r["correct"] is False and r["outcome"] == "other"
