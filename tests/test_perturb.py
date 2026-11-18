import random

import pytest

from ssb import perturb
from ssb.world import Paragraph


def test_depth_block_start():
    assert [perturb.depth_block_start(100, 1, d) for d in (0, 0.5, 1)] == [0, 50, 99]
    assert [perturb.depth_block_start(20, 2, d) for d in (0, 0.5, 1)] == [0, 9, 18]
    with pytest.raises(ValueError):
        perturb.depth_block_start(1, 2, 0)


@pytest.mark.parametrize("n", [5, 20, 100])
def test_contradiction_slots_are_mirror_images(n):
    g_b, c_b = perturb.contradiction_slots(n, "before")
    g_a, c_a = perturb.contradiction_slots(n, "after")
    assert c_b < g_b and g_a < c_a
    assert (g_b, c_b) == (c_a, g_a)  # same two slots, order swapped
    assert min(g_a, n - 1 - g_a) == min(c_a, n - 1 - c_a)  # equally far from the edges


def test_contradiction_needs_room():
    with pytest.raises(ValueError):
        perturb.contradiction_slots(3, "before")


def _p(i):
    return Paragraph(f"p{i}", f"text {i}", "filler", 120, "offtopic")


def test_assemble_places_and_fills_in_order():
    ctx = perturb.assemble(4, {2: _p("g")}, [_p(0), _p(1), _p(2), _p(3)])
    assert [p.id for p in ctx] == ["p0", "p1", "pg", "p2"]
    with pytest.raises(ValueError):
        perturb.assemble(4, {2: _p("g")}, [_p(0)])


def test_paraphrase_variants_never_use_the_conflict_template():
    for base in range(perturb.N_TEMPLATES):
        vs = perturb.paraphrase_variants(base, 5)
        assert vs[0] == base and len(set(vs)) == 5
        assert perturb.conflict_variant(base) not in vs
    with pytest.raises(ValueError):
        perturb.paraphrase_variants(0, 6)


def test_random_slots():
    s = perturb.random_slots(10, 3, random.Random(0))
    assert s == sorted(s) and len(set(s)) == 3
    with pytest.raises(ValueError):
        perturb.random_slots(2, 3, random.Random(0))


def test_tfidf_ranks_related_text_first():
    sc = perturb.SimilarityScorer("tfidf")
    sc.fit(["Bread dough rises in a warm kitchen.", "Kusorin was founded in 1702."])
    s = sc.scores("When was Kusorin founded?")
    assert s[1] > s[0]
