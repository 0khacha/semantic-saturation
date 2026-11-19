import numpy as np
import pandas as pd
import pytest

from ssb.analysis import Boot, ModelAnalysis, Stat, holm
from ssb.config import load_config


def test_holm():
    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05) == {"a"}
    assert holm({"a": 0.01, "b": 0.02, "c": 0.03}, 0.05) == {"a", "b", "c"}
    assert holm({"a": 0.2}, 0.05) == set()


def test_boot_weighted_mean_matches_resampling():
    q = np.array(["a", "a", "b", "c"])
    boot = Boot(q, 200, seed=0)
    v = np.array([1.0, 0.0, 1.0, 0.0])
    point, b = boot.mean(np.arange(4), v)
    assert point == pytest.approx(0.5)
    # recompute one resample by hand
    w = boot.W[7]
    rows = np.repeat(np.arange(4), w[boot.codes].astype(int))
    assert b[7] == pytest.approx(v[rows].mean())


def test_stat_ci_and_p():
    s = Stat(0.3, np.linspace(0.1, 0.5, 1001))
    lo, hi = s.ci()
    assert lo == pytest.approx(0.11, abs=1e-3) and hi == pytest.approx(0.49, abs=1e-3)
    assert s.p_greater_than_zero() == 0.0
    assert Stat(0.0, np.linspace(-1, 1, 1001)).p_greater_than_zero() == pytest.approx(0.5, 0.01)


def _synthetic(acc_by_n: dict[int, float], n_q: int = 200) -> pd.DataFrame:
    """Every question answered right at N with probability acc_by_n[N] (deterministic)."""
    rows = []
    for i in range(n_q):
        for n, acc in acc_by_n.items():
            rows.append({"model": "m", "model_id": "org/m", "question_id": f"q{i}",
                         "task_type": "single_hop", "axis": "noise", "N": n, "depth": None,
                         "contradiction_pos": None, "correct": i < acc * n_q, "f1": 0.0,
                         "lenient": False, "false_abstention": False, "hallucinated": None,
                         "confidence": 0.9, "outcome": None, "is_gold": False})
        rows.append({**rows[-1], "axis": "closed_book", "N": 0, "correct": False})
    return pd.DataFrame(rows)


def test_saturation_onset_and_sss():
    cfg = load_config("configs/main.yaml")
    ma = ModelAnalysis(_synthetic({1: 0.9, 5: 0.9, 20: 0.6, 100: 0.2}), cfg)
    assert ma.acc("noise", 20).point == pytest.approx(0.6)
    assert ma.saturation_point("noise") == 5          # gain 1 -> 5 is 0 < 1 point
    onset, info = ma.degradation_onset("noise")
    assert info["peak"] == 1 and onset == 20          # N=5 ties the peak; N=20 is far below
    x = np.log10([1, 5, 20, 100])
    expected = np.trapezoid([0.9, 0.9, 0.6, 0.2], x) / 2
    assert ma.sss("noise").point == pytest.approx(expected)
    lo, hi = ma.sss("noise").ci()
    assert lo <= expected <= hi
    # confidence 0.9 throughout: ECE at N=100 is |0.9 - 0.2|
    assert ma.stats[("ece", "noise", 100)].point == pytest.approx(0.7)


def test_closed_book_correct_questions_are_excluded():
    cfg = load_config("configs/main.yaml")
    df = _synthetic({1: 1.0, 5: 1.0})
    df.loc[(df.axis == "closed_book") & (df.question_id == "q3"), "correct"] = True
    ma = ModelAnalysis(df, cfg)
    assert ma.closed_book_correct == ["q3"]
    assert "q3" not in set(ma.df.question_id)
    assert "q3" in set(ModelAnalysis(df, cfg, exclude_closed_book=False).df.question_id)
