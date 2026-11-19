"""Assemble the dataset: questions x conditions -> samples, then verify every sample.

Storage is normalised: `<name>.jsonl` holds one sample per line with `paragraphs` as an
ordered list of paragraph ids; `<name>.paragraphs.jsonl` holds each distinct paragraph
once. `load_dataset` joins them back. (Inlining the text would make the v0.1 file
~50 MB, too large to version comfortably.)
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import perturb
from .config import Config, config_to_dict
from .world import (
    HIDEABLE,
    MULTI_HOP,
    RELATIONS,
    Fact,
    Paragraph,
    TokenCounter,
    World,
    contains_phrase,
    derive_seed,
    rng_for,
)

UNANSWERABLE = "UNANSWERABLE"

PROMPT_CONTEXT = (
    "Answer the question using only the context below. Give a short answer of a few "
    "words. If the context does not contain the answer, reply exactly UNANSWERABLE.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)
PROMPT_CLOSED_BOOK = (
    "Answer the question with a short answer of a few words. If you do not know the "
    "answer, reply exactly UNANSWERABLE.\n\nQuestion: {question}\nAnswer:"
)


def render_prompt(question: str, texts: list[str]) -> str:
    """The exact user message sent to every model (before its chat template)."""
    if not texts:
        return PROMPT_CLOSED_BOOK.format(question=question)
    return PROMPT_CONTEXT.format(context="\n\n".join(texts), question=question)


# --------------------------------------------------------------------------- questions


@dataclass
class Question:
    id: str
    task_type: str
    text: str
    answer: str  # gold answer; UNANSWERABLE for unanswerable questions
    aliases: list[str]
    facts: list[str]  # gold fact ids in hop order (empty when unanswerable)
    target: str  # fact id of the final hop, or of the hidden fact
    hidden_value: str | None = None
    meta: dict = field(default_factory=dict)


def _aliases(world: World, fact: Fact) -> list[str]:
    v = fact.value
    kind = RELATIONS[fact.relation].value_kind
    if kind == "river":
        return [v, f"the {v}", f"river {v}", f"the river {v}", f"{v} river"]
    if kind == "town":
        return [v, f"the town of {v}"]
    return [v]


def _round_robin(groups: dict[str, list], n: int) -> list:
    """Take n items cycling over sorted group keys (keeps relation types balanced)."""
    keys = sorted(groups)
    queues = {k: list(groups[k]) for k in keys}
    out = []
    while len(out) < n:
        progressed = False
        for k in keys:
            if queues[k] and len(out) < n:
                out.append(queues[k].pop(0))
                progressed = True
        if not progressed:
            raise ValueError(f"not enough candidates: wanted {n}, found {len(out)}")
    return out


def select_questions(world: World, counts: dict[str, int], seed: int) -> list[Question]:
    rng = rng_for(seed, "questions")
    facts = sorted(world.facts.values(), key=lambda f: f.id)
    questions: list[Question] = []

    # 1. Unanswerable: hide a uniquely-valued fact so its value appears nowhere.
    by_rel: dict[str, list[Fact]] = defaultdict(list)
    for f in facts:
        if f.relation in HIDEABLE:
            by_rel[f.relation].append(f)
    for v in by_rel.values():
        rng.shuffle(v)
    hidden = _round_robin(by_rel, counts.get("unanswerable", 0))
    hidden_ids = {f.id for f in hidden}
    for f in hidden:
        questions.append(Question("", "unanswerable", world.question_text(f, rng.randrange(3)),
                                  UNANSWERABLE, [UNANSWERABLE], [], f.id, f.value,
                                  {"relation": f.relation}))

    # 2. Single hop: one visible fact.
    by_rel = defaultdict(list)
    for f in facts:
        if f.id not in hidden_ids:
            by_rel[f.relation].append(f)
    for v in by_rel.values():
        rng.shuffle(v)
    for f in _round_robin(by_rel, counts.get("single_hop", 0)):
        questions.append(Question("", "single_hop", world.question_text(f, rng.randrange(3)),
                                  f.value, _aliases(world, f), [f.id], f.id,
                                  meta={"relation": f.relation}))

    # 3. Multi hop: two visible facts chained through a bridge entity.
    by_subject: dict[str, list[Fact]] = defaultdict(list)
    for f in facts:
        by_subject[f.subject].append(f)
    chains: dict[str, list[tuple[Fact, Fact]]] = defaultdict(list)
    for f1 in facts:
        if f1.id in hidden_ids or f1.object_id is None:
            continue
        for f2 in by_subject[f1.object_id]:
            key = (f1.relation, f2.relation)
            if key in MULTI_HOP and f2.id not in hidden_ids:
                chains["+".join(key)].append((f1, f2))
    for v in chains.values():
        rng.shuffle(v)
    for f1, f2 in _round_robin(chains, counts.get("multi_hop", 0)):
        subject = world.name(f1.subject)
        text = MULTI_HOP[(f1.relation, f2.relation)].format(s=subject)
        questions.append(Question("", "multi_hop", text, f2.value, _aliases(world, f2),
                                  [f1.id, f2.id], f2.id,
                                  meta={"relation": f"{f1.relation}+{f2.relation}",
                                        "bridge": f1.value}))

    for i, q in enumerate(questions):
        q.id = f"q{i:03d}"
    return questions


# --------------------------------------------------------------------------- conditions


def n_paragraphs(n_level: int, n_gold: int, n_conflict: int = 0) -> int:
    """Actual context length for size level N: never fewer than the evidence itself."""
    return max(n_level, n_gold + n_conflict) if n_level else 0


def conditions_for(q: Question, cfg: Config) -> list[dict]:
    """Every condition a question takes part in. One axis varies at a time.

    The base context (dilution axis) is gold + `base_filler`; the noise axis adds the
    other filler type at the same N, so the random-vs-hard comparison reuses the
    dilution samples instead of duplicating them.
    """
    c = cfg.conditions
    hops = len(q.facts)
    answerable = q.task_type != "unanswerable"
    other_filler = "hard_negative" if c.base_filler == "random" else "random"
    out: list[dict] = []
    if c.closed_book:
        out.append({"axis": "closed_book", "N": 0})
    for n in c.sizes:
        # Dilution, noise and redundancy exist at every size level: when the level is
        # smaller than the evidence (multi-hop at N=1, k=5 copies at N=1 or 5) the
        # context is the evidence alone (see n_paragraphs), so every question is
        # present at every level and curves compare the same questions.
        if "dilution" in c.axes:
            out.append({"axis": "dilution", "N": n, "noise_type": c.base_filler})
        if "noise" in c.axes:
            out.append({"axis": "noise", "N": n, "noise_type": other_filler})
        if not answerable:
            continue
        if "redundancy" in c.axes:
            for k in c.redundancy_k:
                if k > 1:
                    out.append({"axis": "redundancy", "N": n, "noise_type": c.base_filler,
                                "redundancy_k": k})
        if "contradiction" in c.axes and n >= hops + 1:
            for pos in c.contradiction_positions:
                try:
                    perturb.contradiction_slots(n, pos)
                except ValueError:
                    continue
                out.append({"axis": "contradiction", "N": n, "noise_type": c.base_filler,
                            "contradiction_pos": pos})
        if "position" in c.axes and n >= 2:
            seen: set[int] = set()
            for d in c.depths:
                start = perturb.depth_block_start(n, hops, d)
                if start in seen:  # depths that land on the same slot at small N
                    continue
                seen.add(start)
                out.append({"axis": "position", "N": n, "noise_type": c.base_filler,
                            "depth": d})
    return out


# --------------------------------------------------------------------------- builder


class Builder:
    def __init__(self, cfg: Config, counter: TokenCounter | None = None,
                 scorer: perturb.SimilarityScorer | None = None):
        self.cfg = cfg
        d = cfg.dataset
        self.lo, self.hi = cfg.token_range
        self.counter = counter or TokenCounter(d.tokenizer)
        self.world = World(cfg.seed, d.world.people, d.world.towns, d.world.companies,
                           d.world.inventions)
        self.questions = select_questions(self.world, d.questions, cfg.seed)
        hidden = {q.target for q in self.questions if q.task_type == "unanswerable"}

        # Every visible fact rendered once with its base template: the hard-negative pool,
        # and the unmodified gold paragraph whenever that fact is gold.
        self.base_variant: dict[str, int] = {}
        self.fact_para: dict[str, Paragraph] = {}
        for f in sorted(self.world.facts.values(), key=lambda f: f.id):
            if f.id in hidden:
                continue
            v = derive_seed(cfg.seed, "variant", f.id) % perturb.N_TEMPLATES
            self.base_variant[f.id] = v
            self.fact_para[f.id] = self._fact_paragraph(f, v)
        self.offtopic = self.world.offtopic_pool(d.offtopic_pool, self.counter, self.lo,
                                                 self.hi)
        self.paragraphs: dict[str, Paragraph] = {p.id: p for p in self.fact_para.values()}
        self.paragraphs.update({p.id: p for p in self.offtopic})

        self.scorer = scorer or perturb.SimilarityScorer(d.hard_negative_scorer,
                                                         d.embedding_model)
        pool = list(self.fact_para.values()) + self.offtopic
        self._pool_ids = [p.id for p in pool]
        self.scorer.fit([p.text for p in pool])

        # Conflicting claims, created in a fixed order so fresh values are reproducible.
        self.conflicts: dict[str, Paragraph] = {}
        for q in self.questions:
            if q.task_type != "unanswerable":
                self.conflicts[q.id] = self._conflict(q)

    # ---- paragraph factories
    def _fact_paragraph(self, f: Fact, variant: int) -> Paragraph:
        return self.world.render_fact_paragraph(f"f:{f.id}:v{variant}", f.subject, f.relation,
                                                f.value, variant, self.counter, self.lo,
                                                self.hi)

    def _paraphrases(self, fact_id: str, k: int) -> list[Paragraph]:
        f = self.world.facts[fact_id]
        out = [self.fact_para[fact_id]]
        for v in perturb.paraphrase_variants(self.base_variant[fact_id], k)[1:]:
            p = self._fact_paragraph(f, v)
            self.paragraphs[p.id] = p
            out.append(p)
        return out

    def _conflict(self, q: Question) -> Paragraph:
        f = self.world.facts[q.target]
        kind = RELATIONS[f.relation].value_kind
        value = self.world.fresh_value(kind, rng_for(self.cfg.seed, "conflict", q.id))
        v = perturb.conflict_variant(self.base_variant[f.id])
        p = self.world.render_fact_paragraph(f"c:{q.id}:{f.id}", f.subject, f.relation, value,
                                             v, self.counter, self.lo, self.hi)
        self.paragraphs[p.id] = p
        return p

    # ---- fillers
    def _fillers(self, q: Question, kind: str, sims: dict[str, float]) -> list[Paragraph]:
        banned = [q.hidden_value] if q.task_type == "unanswerable" else [q.answer]
        if kind == "random":
            cands = perturb.filter_candidates(self.offtopic, set(), banned)
            rng_for(self.cfg.seed, "random-order", q.id).shuffle(cands)
            return cands
        gold_ids = {self.fact_para[f].id for f in q.facts}
        cands = perturb.filter_candidates(list(self.fact_para.values()), gold_ids, banned)
        return perturb.rank_hard_negatives(cands, sims)

    # ---- samples
    def build_samples(self, q: Question) -> list[dict]:
        scores = self.scorer.scores(q.text)
        sims = dict(zip(self._pool_ids, scores, strict=True))
        fillers = {k: self._fillers(q, k, sims) for k in ("random", "hard_negative")}
        return [self._sample(q, cond, fillers, sims) for cond in conditions_for(q, self.cfg)]

    def _sample(self, q: Question, cond: dict, fillers: dict, sims: dict) -> dict:
        level, axis = cond["N"], cond["axis"]
        n = 0
        seed = derive_seed(self.cfg.seed, q.id, *sorted(cond.items()))
        rng = random.Random(seed)
        hops = len(q.facts)
        placed: dict[int, Paragraph] = {}
        gold: list[Paragraph] = []
        conflict: Paragraph | None = None
        depth: float | None = cond.get("depth")

        if axis == "closed_book":
            context: list[Paragraph] = []
        else:
            if axis == "redundancy":
                gold = [p for fid in q.facts for p in self._paraphrases(fid, cond["redundancy_k"])]
            else:
                gold = [self.fact_para[fid] for fid in q.facts]
            n = n_paragraphs(level, len(gold), 1 if axis == "contradiction" else 0)

            if axis == "position":
                start = perturb.depth_block_start(n, hops, cond["depth"])
                placed = {start + i: p for i, p in enumerate(gold)}
            elif axis == "contradiction":
                conflict = self.conflicts[q.id]
                g_idx, c_idx = perturb.contradiction_slots(n, cond["contradiction_pos"])
                placed = {g_idx: self.fact_para[q.target], c_idx: conflict}
                for fid in q.facts[:-1]:  # earlier hops go anywhere else
                    free = [i for i in range(n) if i not in placed]
                    placed[rng.choice(free)] = self.fact_para[fid]
            else:
                order = gold[:]
                rng.shuffle(order)
                placed = dict(zip(perturb.random_slots(n, len(order), rng), order, strict=True))

            pool = fillers[cond["noise_type"]]
            need = n - len(placed)
            chosen = pool[:need]
            if cond["noise_type"] == "hard_negative":
                chosen = chosen[:]
                rng.shuffle(chosen)  # nested sets across N, but similarity never sets order
            context = perturb.assemble(n, placed, chosen)

        ids = [p.id for p in context]
        gold_set = {p.id for p in gold}
        gold_ids = [pid for pid in ids if pid in gold_set]
        target_pid = self.fact_para[q.target].id if q.target in self.fact_para else None
        if depth is None and target_pid in ids:
            depth = round(ids.index(target_pid) / (n - 1), 4) if n > 1 else 0.0
        texts = [p.text for p in context]
        conflict_id = conflict.id if conflict else None
        filler_sims = [sims[pid] for pid in ids if pid not in gold_set and pid != conflict_id]
        return {
            "id": "|".join([q.id, axis, f"N{level}"] + [f"{k}={v}" for k, v in sorted(cond.items())
                                                     if k not in ("axis", "N")]),
            "question_id": q.id,
            "task_type": q.task_type,
            "question": q.text,
            "gold_answer": q.answer,
            "answer_aliases": q.aliases,
            "gold_fact_ids": q.facts,
            "target_fact_id": q.target,
            "gold_paragraph_ids": gold_ids,
            "base_gold_ids": [self.fact_para[f].id for f in q.facts] if axis != "closed_book"
            else [],
            "paragraphs": ids,
            "axis": axis,
            "N": level,
            "n_paragraphs": n,
            "depth": depth,
            "gold_positions": [i for i, pid in enumerate(ids) if pid in gold_set],
            "noise_type": cond.get("noise_type"),
            "redundancy_k": cond.get("redundancy_k", 1 if gold else 0),
            "contradiction_pos": cond.get("contradiction_pos"),
            "conflict_answer": conflict.value if conflict else None,
            "conflict_paragraph_id": conflict.id if conflict else None,
            "hidden_value": q.hidden_value,
            "relation": q.meta.get("relation"),
            "filler_similarity": round(sum(filler_sims) / len(filler_sims), 4)
            if filler_sims else None,
            "token_count": self.counter.count(render_prompt(q.text, texts)),
            "seed": seed,
        }

    # ---- whole dataset
    def build(self) -> tuple[list[dict], dict[str, Paragraph]]:
        samples = [s for q in self.questions for s in self.build_samples(q)]
        ids = [s["id"] for s in samples]
        if len(ids) != len(set(ids)):
            raise RuntimeError("duplicate sample ids")
        used = {pid for s in samples for pid in s["paragraphs"]}
        paras = {pid: self.paragraphs[pid] for pid in sorted(used)}
        problems = [f"{s['id']}: {e}" for s in samples
                    for e in check_sample(s, paras, self.counter, (self.lo, self.hi))]
        if problems:
            raise RuntimeError("dataset failed its checks:\n" + "\n".join(problems[:20]))
        return samples, paras


# --------------------------------------------------------------------------- checks


def check_sample(s: dict, paras: dict, counter: TokenCounter | None,
                 token_range: tuple[int, int]) -> list[str]:
    """Return every invariant the sample violates (empty list = valid)."""
    errs: list[str] = []
    ctx = [paras[pid] for pid in s["paragraphs"]]
    texts = [p.text for p in ctx]
    n = s["n_paragraphs"]
    lo, hi = token_range

    def text_of(pid: str) -> str:
        return paras[pid].text

    n_evidence = len(s["gold_paragraph_ids"]) + (1 if s["conflict_paragraph_id"] else 0)
    if n != n_paragraphs(s["N"], n_evidence):
        errs.append(f"n_paragraphs={n} but level N={s['N']} with {n_evidence} evidence "
                    "paragraphs requires max(N, evidence)")
    if len(ctx) != n:
        errs.append(f"{len(ctx)} paragraphs, expected {n}")
    if len(set(s["paragraphs"])) != len(s["paragraphs"]):
        errs.append("a paragraph appears twice")
    for p in ctx:
        if not lo <= p.tokens <= hi:
            errs.append(f"paragraph {p.id} has {p.tokens} tokens, outside [{lo}, {hi}]")
    if counter is not None:
        real = counter.count(render_prompt(s["question"], texts))
        if real != s["token_count"]:
            errs.append(f"token_count {s['token_count']} != recount {real}")

    non_gold = [pid for pid in s["paragraphs"] if pid not in s["gold_paragraph_ids"]]
    if s["task_type"] == "unanswerable":
        if s["gold_paragraph_ids"] or s["base_gold_ids"]:
            errs.append("unanswerable sample has gold paragraphs")
        if any(contains_phrase(t, s["hidden_value"]) for t in texts):
            errs.append("hidden answer appears in the context")
        subj, rel = s["target_fact_id"].split(".")
        if any(p.kind == "fact" and p.subject == subj and p.relation == rel for p in ctx):
            errs.append("a paragraph states the hidden fact")
    elif s["axis"] != "closed_book":
        missing = [b for b in s["base_gold_ids"] if b not in s["paragraphs"]]
        if missing:
            errs.append(f"base gold paragraph(s) missing: {missing}")
        target = [pid for pid in s["gold_paragraph_ids"]
                  if paras[pid].relation == s["target_fact_id"].split(".")[1]
                  and paras[pid].subject == s["target_fact_id"].split(".")[0]]
        if not target:
            errs.append("no gold paragraph states the target fact")
        for pid in target:
            if not contains_phrase(text_of(pid), s["gold_answer"]):
                errs.append(f"gold paragraph {pid} lacks the answer")
        leaks = [pid for pid in non_gold if contains_phrase(text_of(pid), s["gold_answer"])]
        if leaks:
            errs.append(f"answer appears outside gold paragraphs: {leaks}")

    if s["axis"] == "closed_book" and ctx:
        errs.append("closed-book sample has context")

    if s["axis"] == "redundancy":
        k = s["redundancy_k"]
        if len(s["gold_paragraph_ids"]) != k * len(s["gold_fact_ids"]):
            errs.append(f"expected {k} renderings per gold fact")
        if len({text_of(p) for p in s["gold_paragraph_ids"]}) != len(s["gold_paragraph_ids"]):
            errs.append("redundant gold copies are not distinct paraphrases")

    if s["axis"] == "contradiction":
        cid = s["conflict_paragraph_id"]
        if cid not in s["paragraphs"]:
            errs.append("conflict paragraph missing")
        else:
            c = paras[cid]
            subj, rel = s["target_fact_id"].split(".")
            if (c.subject, c.relation) != (subj, rel):
                errs.append("conflict paragraph is about a different fact")
            if c.value != s["conflict_answer"] or c.value == s["gold_answer"]:
                errs.append("conflict does not assert a different answer")
            if not contains_phrase(c.text, c.value) or contains_phrase(c.text, s["gold_answer"]):
                errs.append("conflict text does not carry exactly the conflicting value")
            gold_t = next(i for i, pid in enumerate(s["paragraphs"])
                          if pid == s["base_gold_ids"][-1])
            conf_t = s["paragraphs"].index(cid)
            if (conf_t < gold_t) != (s["contradiction_pos"] == "before"):
                errs.append("conflict is on the wrong side of the gold paragraph")

    if s["axis"] == "position":
        hops = len(s["gold_fact_ids"])
        start = round(s["depth"] * (n - hops))
        if s["gold_positions"] != list(range(start, start + hops)):
            errs.append(f"gold at {s['gold_positions']}, expected block at {start}")

    if s["axis"] in ("dilution", "noise", "redundancy", "contradiction", "position"):
        want = "offtopic" if s["noise_type"] == "random" else "fact"
        fill = [paras[p] for p in non_gold if p != s["conflict_paragraph_id"]]
        if any(p.kind != want for p in fill):
            errs.append(f"filler kind does not match noise_type={s['noise_type']}")
    return errs


# --------------------------------------------------------------------------- io


def paragraphs_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".paragraphs.jsonl")


def manifest_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".manifest.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dataset(cfg: Config, samples: list[dict], paras: dict[str, Paragraph]) -> dict:
    out = Path(cfg.dataset.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    ppath = paragraphs_path(out)
    with ppath.open("w", encoding="utf-8", newline="\n") as fh:
        for p in paras.values():
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    manifest = {
        "version": cfg.dataset.version,
        "samples": len(samples),
        "questions": len({s["question_id"] for s in samples}),
        "paragraphs": len(paras),
        "by_axis": dict(sorted(Counter(s["axis"] for s in samples).items())),
        "by_task": dict(sorted(Counter(s["task_type"] for s in samples).items())),
        "by_N": {str(k): v for k, v in sorted(Counter(s["N"] for s in samples).items())},
        "prompt_tokens_total": sum(s["token_count"] for s in samples),
        "prompt_tokens_max": max(s["token_count"] for s in samples),
        "sha256": {out.name: _sha256(out), ppath.name: _sha256(ppath)},
        "config": config_to_dict(cfg),
    }
    manifest_path(out).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
                                  newline="\n")
    return manifest


def load_dataset(path: str | Path) -> list[dict]:
    """Samples with `context` (list of paragraph texts) and `prompt` filled in."""
    path = Path(path)
    paras = {}
    with paragraphs_path(path).open(encoding="utf-8") as fh:
        for line in fh:
            p = json.loads(line)
            paras[p["id"]] = p
    samples = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            s["context"] = [paras[pid]["text"] for pid in s["paragraphs"]]
            s["prompt"] = render_prompt(s["question"], s["context"])
            samples.append(s)
    return samples


def load_paragraphs(path: str | Path) -> dict[str, Paragraph]:
    out = {}
    with paragraphs_path(path).open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out[d["id"]] = Paragraph(d["id"], d["text"], "filler", d["tokens"], d["kind"],
                                     d.get("subject"), d.get("relation"), d.get("value"))
    return out
