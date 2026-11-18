"""Dataset invariants, checked independently of Builder's own check_sample."""

from collections import Counter, defaultdict
from pathlib import Path

import pytest

from ssb.build import (
    UNANSWERABLE,
    Builder,
    check_sample,
    load_dataset,
    load_paragraphs,
    render_prompt,
)
from ssb.config import load_config
from ssb.perturb import SimilarityScorer
from ssb.world import contains_phrase


def _ctx(s, paras):
    return [paras[pid] for pid in s["paragraphs"]]


def _answerable_with_context(s):
    return s["task_type"] != "unanswerable" and s["axis"] != "closed_book"


def test_builder_checks_pass(built, counter, small_cfg):
    _, samples, paras = built
    for s in samples:
        assert check_sample(s, paras, counter, small_cfg.token_range) == [], s["id"]


def test_question_balance(built):
    b, _, _ = built
    assert Counter(q.task_type for q in b.questions) == {"single_hop": 8, "multi_hop": 8,
                                                          "unanswerable": 6}
    assert len({q.text for q in b.questions}) == len(b.questions)


def test_paragraph_and_token_counts_match_condition(built, counter):
    _, samples, paras = built
    for s in samples:
        ctx = _ctx(s, paras)
        assert len(ctx) == s["N"]
        assert all(108 <= p.tokens <= 132 for p in ctx)
        prompt = render_prompt(s["question"], [p.text for p in ctx])
        assert s["token_count"] == counter.count(prompt)


def test_gold_present_and_unmodified(built, counter):
    b, samples, paras = built
    w, (lo, hi) = b.world, b.cfg.token_range
    for s in filter(_answerable_with_context, samples):
        for fid, pid in zip(s["gold_fact_ids"], s["base_gold_ids"], strict=True):
            f = w.facts[fid]
            v = int(pid.rsplit(":v", 1)[1])
            fresh = w.render_fact_paragraph(pid, f.subject, f.relation, f.value, v, counter,
                                            lo, hi)
            assert pid in s["paragraphs"], s["id"]
            assert paras[pid].text == fresh.text  # identical to an independent render


def test_answer_only_inside_gold(built):
    _, samples, paras = built
    for s in filter(_answerable_with_context, samples):
        for p in _ctx(s, paras):
            if p.id not in s["gold_paragraph_ids"]:
                assert not contains_phrase(p.text, s["gold_answer"]), (s["id"], p.id)


def test_gold_absent_in_unanswerable(built):
    _, samples, paras = built
    for s in samples:
        if s["task_type"] != "unanswerable":
            continue
        assert s["gold_paragraph_ids"] == [] and s["gold_answer"] == UNANSWERABLE
        subj, rel = s["target_fact_id"].split(".")
        for p in _ctx(s, paras):
            assert not contains_phrase(p.text, s["hidden_value"])
            assert (p.subject, p.relation) != (subj, rel)
        assert s["axis"] in ("closed_book", "dilution", "noise")


def test_multi_hop_facts_in_different_paragraphs(built):
    _, samples, _ = built
    for s in filter(_answerable_with_context, samples):
        if s["task_type"] == "multi_hop":
            a, b = s["base_gold_ids"]
            assert a != b and a in s["paragraphs"] and b in s["paragraphs"]


def test_contradiction_really_contradicts(built):
    b, samples, paras = built
    seen = Counter()
    for s in samples:
        if s["axis"] != "contradiction":
            continue
        c = paras[s["conflict_paragraph_id"]]
        target = b.world.facts[s["target_fact_id"]]
        assert (c.subject, c.relation) == (target.subject, target.relation)
        assert c.value == s["conflict_answer"] != s["gold_answer"]
        assert contains_phrase(c.text, s["conflict_answer"])
        assert not contains_phrase(c.text, s["gold_answer"])
        others = [p for p in _ctx(s, paras) if p.id != c.id]
        assert not any(contains_phrase(p.text, s["conflict_answer"]) for p in others)
        g = s["paragraphs"].index(s["base_gold_ids"][-1])
        ci = s["paragraphs"].index(c.id)
        assert (ci < g) == (s["contradiction_pos"] == "before")
        seen[s["contradiction_pos"]] += 1
    assert seen["before"] == seen["after"] > 0


def test_depth_placement(built):
    _, samples, _ = built
    for s in samples:
        if s["axis"] != "position":
            continue
        n, hops = s["N"], len(s["gold_fact_ids"])
        start = s["paragraphs"].index(s["base_gold_ids"][0])
        if s["depth"] == 0.0:
            assert start == 0
        elif s["depth"] == 1.0:
            assert start + hops == n  # gold block ends the context
        else:
            assert start == round(s["depth"] * (n - hops))
        assert s["gold_positions"] == list(range(start, start + hops))


def test_redundancy_paraphrases(built):
    _, samples, paras = built
    for s in samples:
        if s["axis"] == "redundancy":
            k = s["redundancy_k"]
            gold = [paras[p] for p in s["gold_paragraph_ids"]]
            assert len(gold) == k * len(s["gold_fact_ids"])
            assert len({p.text for p in gold}) == len(gold)
            assert set(Counter((p.subject, p.relation) for p in gold).values()) == {k}


def test_filler_types(built):
    _, samples, paras = built
    for s in samples:
        if s["axis"] == "closed_book":
            assert s["paragraphs"] == []
            continue
        fill = [paras[p] for p in s["paragraphs"]
                if p not in s["gold_paragraph_ids"] and p != s["conflict_paragraph_id"]]
        kinds = {p.kind for p in fill}
        assert kinds <= ({"offtopic"} if s["noise_type"] == "random" else {"fact"})


def test_hard_negatives_are_more_similar_than_random(built):
    _, samples, _ = built
    by = defaultdict(dict)
    for s in samples:
        if s["axis"] in ("dilution", "noise") and s["N"] == 100:
            by[s["question_id"]][s["noise_type"]] = s["filler_similarity"]
    assert by and all(v["hard_negative"] > v["random"] for v in by.values())


def test_fillers_are_nested_across_sizes(built):
    _, samples, _ = built
    sets = defaultdict(dict)
    for s in samples:
        if s["axis"] in ("dilution", "noise"):
            fill = set(s["paragraphs"]) - set(s["gold_paragraph_ids"])
            sets[(s["question_id"], s["noise_type"])][s["N"]] = fill
    for by_n in sets.values():
        ns = sorted(by_n)
        for a, b in zip(ns, ns[1:], strict=False):
            assert by_n[a] <= by_n[b]


def test_every_question_has_a_closed_book_sample(built):
    b, samples, _ = built
    assert {s["question_id"] for s in samples if s["axis"] == "closed_book"} == \
        {q.id for q in b.questions}


def test_build_is_deterministic(built, small_cfg, counter):
    _, samples, paras = built
    again, paras2 = Builder(small_cfg, counter=counter, scorer=SimilarityScorer("tfidf")).build()
    assert [s["id"] for s in again] == [s["id"] for s in samples]
    assert [s["paragraphs"] for s in again] == [s["paragraphs"] for s in samples]
    assert {k: p.text for k, p in paras2.items()} == {k: p.text for k, p in paras.items()}


MAIN = load_config("configs/main.yaml")


@pytest.mark.skipif(not Path(MAIN.dataset.path).exists(), reason="dataset not built yet")
def test_published_dataset_passes_all_checks(counter):
    paras = load_paragraphs(MAIN.dataset.path)
    samples = load_dataset(MAIN.dataset.path)
    assert len({s["question_id"] for s in samples}) == sum(MAIN.dataset.questions.values())
    for s in samples:
        assert check_sample(s, paras, counter, MAIN.token_range) == [], s["id"]
        assert s["prompt"] == render_prompt(s["question"], s["context"])
