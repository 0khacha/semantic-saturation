import re

import pytest

from ssb.world import (
    _NEUTRAL,
    _OFFTOPIC,
    _OFFTOPIC_SLOTS,
    _SLOTS,
    FACT_WORDS,
    HIDEABLE,
    MULTI_HOP,
    RELATIONS,
    World,
    _fill_slots,
    contains_phrase,
    rng_for,
)


@pytest.fixture(scope="module")
def world():
    return World(123, 80, 40, 40, 40)


def test_world_is_deterministic(world):
    again = World(123, 80, 40, 40, 40)
    assert again.facts == world.facts and again.entities == world.entities
    assert World(124, 80, 40, 40, 40).facts != world.facts


def test_names_never_overlap_as_whole_words(world):
    names = sorted(world.all_names())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not contains_phrase(a, b) and not contains_phrase(b, a), (a, b)


def test_unique_values_are_unique(world):
    for rel in HIDEABLE:
        values = [f.value for f in world.facts.values() if f.relation == rel]
        assert len(values) == len(set(values)), rel
    years = [f.value for f in world.facts.values() if RELATIONS[f.relation].value_kind == "year"]
    assert len(years) == len(set(years))


def test_relation_templates_have_six_distinct_paraphrases():
    for rel in RELATIONS.values():
        assert len(set(rel.templates)) == 6, rel.name
        for t in rel.templates:
            assert "{s}" in t and "{v}" in t, t
    for (r1, r2) in MULTI_HOP:
        assert RELATIONS[r1].value_kind == RELATIONS[r2].subject_type


def test_neutral_sentences_state_no_facts():
    for etype, templates in _NEUTRAL.items():
        for t in templates:
            bare = re.sub(r"\{\w+\}", "", t)
            assert not FACT_WORDS.search(bare), (etype, t)
    for slot, values in _SLOTS.items():
        for v in values:
            assert not FACT_WORDS.search(v), (slot, v)


def test_offtopic_sentences_have_no_digits():
    rng = rng_for(0)
    for templates in _OFFTOPIC.values():
        for t in templates:
            assert not re.search(r"\d", _fill_slots(t, _OFFTOPIC_SLOTS, rng, {}))


def test_paragraphs_hit_the_token_window(world, counter):
    for f in list(world.facts.values())[::7]:
        p = world.render_fact_paragraph(f"f:{f.id}:v0", f.subject, f.relation, f.value, 0,
                                        counter, 108, 132)
        assert 108 <= p.tokens <= 132
        assert p.tokens == counter.count(p.text)
        assert contains_phrase(p.text, f.value)
    for p in world.offtopic_pool(40, counter, 108, 132):
        assert 108 <= p.tokens <= 132
        assert not any(contains_phrase(p.text, n) for n in world.all_names())


def test_paragraph_text_depends_only_on_its_id(world, counter):
    f = next(iter(world.facts.values()))
    args = (f.subject, f.relation, f.value, 0, counter, 108, 132)
    a = world.render_fact_paragraph("f:x:v0", *args)
    b = world.render_fact_paragraph("f:x:v0", *args)
    c = world.render_fact_paragraph("f:y:v0", *args)
    assert a.text == b.text != c.text
