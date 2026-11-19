"""Unit tests: config, normalisation, placement, entity swap."""

import random

import pytest
import yaml

from ssb import perturb
from ssb.config import ConfigError, load_config
from ssb.contradict import EntitySwapper, cross_class
from ssb.paragraphs import Paragraph, contains_answer, normalize, split_sentences

# ---- config


@pytest.mark.parametrize("path", ["configs/main.yaml", "configs/smoke.yaml"])
def test_shipped_configs_load(path):
    cfg = load_config(path)
    assert {m.task for m in cfg.models} == {"classification", "retrieval", "qa"}
    assert cfg.token_range == (108, 132)


def _write(tmp_path, data):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, {"name": "x", "seed": 1, "data": {"exampels": {}}}))


@pytest.mark.parametrize("bad", [
    {"conditions": {"sizes": [5, 1]}},
    {"conditions": {"depths": [1.5]}},
    {"conditions": {"redundancy_modes": ["mirror"]}},
    {"models": [{"name": "m", "task": "qa", "kind": "bm25"}]},
    {"models": [{"name": "m", "task": "classification", "kind": "transformer_cls",
                 "model_id": "x"}]},
])
def test_invalid_settings_are_rejected(tmp_path, bad):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, {"name": "x", "seed": 1, **bad}))


# ---- paragraphs


def test_split_sentences_keeps_abbreviations():
    assert split_sentences("The U.S. Army came in 1945. Dr. Smith left. Then it rained.") == [
        "The U.S. Army came in 1945.", "Dr. Smith left.", "Then it rained."]


def test_normalize_keeps_in_range_text_and_cuts_long_text(counter):
    para = " ".join(["The committee met in the old hall and discussed the budget at length."] * 6)
    n = counter.count(para)
    kept = normalize(para, counter, n - 5, n + 5)
    assert kept == (para, n)  # already inside the window: unchanged
    long = " ".join(f"Sentence number {i} talks about a different topic entirely."
                    for i in range(60))
    t, k = normalize(long, counter, 108, 132)
    assert 108 <= k <= 132 and k == counter.count(t)
    assert t in long  # whole consecutive sentences, no edited words
    assert normalize("Too short.", counter, 108, 132) is None


def test_contains_answer_respects_word_boundaries():
    assert contains_answer("It opened in 1990.", "1990")
    assert not contains_answer("It opened in 19900.", "1990")
    assert contains_answer("the Denver Broncos won", "denver broncos")


# ---- placement


def test_size_level_never_drops_evidence():
    assert perturb.n_paragraphs(1, 5) == 5
    assert perturb.n_paragraphs(20, 5) == 20
    assert perturb.n_paragraphs(1, 2) == 2


def test_depth_and_contradiction_slots():
    assert [perturb.depth_index(100, d) for d in (0, 0.5, 1)] == [0, 50, 99]
    assert perturb.contradiction_slots("before") == (1, 0)
    assert perturb.contradiction_slots("after") == (0, 1)
    with pytest.raises(ValueError):
        perturb.contradiction_slots("middle")


def test_assemble_allows_repeated_signal_and_fills_in_order():
    sig = Paragraph("s", "signal", 50, "agnews")
    fill = [Paragraph(f"f{i}", "x", 120, "squad_train") for i in range(5)]
    ctx = perturb.assemble(5, {0: sig, 3: sig}, fill)
    assert [p.id for p in ctx] == ["s", "f0", "f1", "s", "f2"]
    with pytest.raises(ValueError):
        perturb.assemble(5, {0: sig}, fill[:2])


# ---- contradiction


@pytest.fixture(scope="module")
def swapper():
    s = EntitySwapper("en_core_web_sm")
    s.fit(["Paris is the capital of France.", "Berlin hosted the summit in 1990.",
           "Madrid and Rome signed the deal.", "Barack Obama met Angela Merkel in 2009."])
    return s


def test_entity_swap_replaces_every_occurrence(swapper):
    ctx = "In 1987 the bridge opened. By 1987 it carried trains."
    (label,) = swapper.answer_types([(ctx, "1987", 3)])
    assert label == "DATE"  # whatever spaCy says, the swap stays within that type
    new, repl = swapper.swap(ctx, ["1987"], label, random.Random(0))
    assert "1987" not in new and new.count(repl) == 2 and repl in swapper.bank["DATE"]
    assert swapper.answer_types([(ctx, "bridge", 9)]) == [None]  # not an entity: no swap


def test_entity_swap_ignores_case_like_the_dataset_check(swapper):
    ctx = "Oxygen was named by Lavoisier. Liquid oxygen is pale blue."
    new, repl = swapper.swap(ctx, ["Oxygen"], "GPE", random.Random(0))
    assert not contains_answer(new, "Oxygen") and new.count(repl) == 2


def test_cross_class_picks_another_label():
    pool = {0: ["a"], 1: ["b"], 2: ["c"], 3: ["d"]}
    for seed in range(20):
        assert cross_class(1, pool, random.Random(seed)) != "b"
