"""Dataset invariants, checked independently of Builder's own check_sample."""

from collections import Counter, defaultdict

from ssb.build import check_sample, context_text
from ssb.paragraphs import contains_answer

from .conftest import make_builder


def test_builder_checks_pass(built, counter, small_cfg):
    _, samples, paras = built
    for s in samples:
        assert check_sample(s, paras, counter, small_cfg.token_range) == [], s["id"]


def test_base_counts_and_balance(built):
    b, _, _ = built
    by_task = Counter(x.task for x in b.bases)
    assert by_task == {"classification": 8, "retrieval": 8, "qa": 8}
    assert Counter(x.label for x in b.bases if x.task == "classification") == {0: 2, 1: 2, 2: 2,
                                                                               3: 2}
    qa = [x for x in b.bases if x.task == "qa"]
    assert sum(not x.answerable for x in qa) == 4
    # QA's answerable questions are retrieval questions (paraphrases and swaps are shared)
    ret_q = {x.question for x in b.bases if x.task == "retrieval"}
    assert {x.question for x in qa if x.answerable} <= ret_q


def test_signal_present_and_unmodified(built, raw):
    (_, ag_test), (dev, _) = raw
    originals = {x.text for x in ag_test} | {q.context for q in dev}
    _, samples, paras = built
    for s in samples:
        assert s["signal_id"] in s["paragraphs"], s["id"]
        assert paras[s["signal_id"]].text in originals  # byte-identical to the source data


def test_paragraph_count_tokens_and_depth(built, counter):
    _, samples, paras = built
    for s in samples:
        ctx = [paras[p] for p in s["paragraphs"]]
        evidence = (s["redundancy_k"] if s["axis"] == "redundancy" else 1) + bool(s["conflict_id"])
        assert len(ctx) == s["n_paragraphs"] == max(s["N"], evidence)
        assert s["token_count"] == counter.count(context_text([p.text for p in ctx]))
        for p in ctx:
            assert p.tokens == counter.count(p.text)
            news = p.label is not None  # AG articles and their paraphrases keep their length
            if not news and not (p.id == s["signal_id"]):
                assert 108 <= p.tokens <= 132, (s["id"], p.id, p.tokens)
        i = s["paragraphs"].index(s["signal_id"])
        if s["axis"] == "position":
            n = s["n_paragraphs"]
            assert i == {0.0: 0, 1.0: n - 1}.get(s["depth"], round(s["depth"] * (n - 1)))
        assert s["depth"] == (round(i / (s["n_paragraphs"] - 1), 4)
                              if s["axis"] != "position" and s["n_paragraphs"] > 1
                              else s["depth"])


def test_ner_swap_changes_the_answer(built):
    _, samples, paras = built
    seen = 0
    for s in samples:
        if s["axis"] == "contradiction" and s["task"] in ("retrieval", "qa"):
            c, g = paras[s["conflict_id"]], paras[s["signal_id"]]
            assert c.text != g.text
            assert not any(contains_answer(c.text, a) for a in s["answers"])
            assert contains_answer(c.text, s["conflict_value"])
            assert any(contains_answer(g.text, a) for a in s["answers"])
            seen += 1
    assert seen


def test_cross_class_paragraph_has_a_different_label(built):
    _, samples, paras = built
    for s in samples:
        if s["axis"] == "contradiction" and s["task"] == "classification":
            assert paras[s["conflict_id"]].label not in (None, s["label"])
            si, ci = s["paragraphs"].index(s["signal_id"]), s["paragraphs"].index(s["conflict_id"])
            assert abs(si - ci) == 1 and (ci < si) == (s["contradiction_pos"] == "before")


def test_paraphrases_keep_answer_or_label(built):
    b, samples, paras = built
    for s in samples:
        if s["axis"] != "redundancy" or s["redundancy_mode"] != "paraphrase":
            continue
        copies = [paras[i] for i in s["signal_ids"] if i != s["signal_id"]]
        assert len(copies) == s["redundancy_k"] - 1
        for p in copies:
            assert p.source == "paraphrase"
            if s["task"] == "classification":
                assert p.label == s["label"]
            else:
                assert any(contains_answer(p.text, a) for a in s["answers"])


def test_exact_copies_repeat_the_signal(built):
    _, samples, _ = built
    for s in samples:
        if s["axis"] == "redundancy" and s["redundancy_mode"] == "copy":
            assert s["paragraphs"].count(s["signal_id"]) == s["redundancy_k"]


def test_every_base_is_present_at_every_level(built, small_cfg):
    b, samples, _ = built
    sizes = set(small_cfg.conditions.sizes)
    levels = defaultdict(set)
    for s in samples:
        levels[(s["base_id"], s["axis"])].add(s["N"])
    for x in b.bases:
        axes = ["dilution", "noise"]
        if x.answerable:
            axes += ["redundancy", "contradiction"]
        for axis in axes:
            assert levels[(x.id, axis)] == sizes, (x.id, axis)
        if x.answerable:
            assert levels[(x.id, "position")] == sizes - {1}
        else:
            assert (x.id, "position") not in levels


def test_fillers_are_nested_across_sizes(built):
    _, samples, _ = built
    sets = defaultdict(dict)
    for s in samples:
        if s["axis"] in ("dilution", "noise"):
            sets[(s["base_id"], s["noise_type"])][s["N"]] = set(s["paragraphs"]) - {s["signal_id"]}
    for by_n in sets.values():
        ns = sorted(by_n)
        for a, b in zip(ns, ns[1:], strict=False):
            assert by_n[a] <= by_n[b]


def test_random_filler_is_unrelated_wikipedia(built, raw):
    (_, _), (dev, _) = raw
    dev_titles = {q.title for q in dev}
    _, samples, paras = built
    for s in samples:
        if s["noise_type"] == "random":
            for pid in s["paragraphs"]:
                if pid not in s["signal_ids"] and pid != s["conflict_id"]:
                    assert paras[pid].source == "squad_train"
                    assert paras[pid].title not in dev_titles


def test_build_is_deterministic(built, small_cfg, counter, raw):
    _, samples, paras = built
    again, paras2 = make_builder(small_cfg, counter, raw).build()
    assert [s["id"] for s in again] == [s["id"] for s in samples]
    assert [s["paragraphs"] for s in again] == [s["paragraphs"] for s in samples]
    assert {k: p.text for k, p in paras2.items()} == {k: p.text for k, p in paras.items()}
