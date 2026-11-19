import numpy as np

from ssb.tasks.qa import best_span


def test_best_span_picks_context_span_and_reports_null():
    # tokens: [CLS] q q [SEP] c c c c [SEP]
    mask = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0])
    start = np.array([2.0, 9.0, 9.0, 0.0, 0.0, 5.0, 0.0, 0.0, 9.0])
    end = np.array([2.0, 9.0, 9.0, 0.0, 0.0, 0.0, 5.0, 0.0, 9.0])
    i, j, score, null = best_span(start, end, mask, cls_index=0)
    assert (i, j) == (5, 6)          # question and SEP tokens are never chosen
    assert 0 < null < score < 1


def test_best_span_respects_order_and_max_length():
    mask = np.array([0] + [1] * 40)
    start = np.zeros(41)
    end = np.zeros(41)
    start[30], end[2] = 10.0, 10.0   # end before start: invalid
    i, j, _, _ = best_span(start, end, mask, cls_index=0)
    assert i <= j < i + 30


def test_null_wins_when_cls_dominates():
    mask = np.array([0, 0, 1, 1, 1])
    start = np.array([10.0, 0, 0, 0, 0])
    end = np.array([10.0, 0, 0, 0, 0])
    _, _, score, null = best_span(start, end, mask, cls_index=0)
    assert null > score
