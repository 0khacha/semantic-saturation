"""Assemble every condition for all three tasks into one versioned dataset.

Storage is normalised: `<name>.jsonl` has one sample per line, with `paragraphs` the
ordered list of paragraph ids in the context; `<name>.paragraphs.jsonl` holds each
distinct paragraph once. `load_dataset` joins them. Every sample is checked by
`check_sample` before anything is written.

Signals are never edited: AG News articles keep their natural length (median ~48
tokens), and SQuAD signals are dev paragraphs that already fall inside the 108-132
token window. Filler, paraphrases and NER-swapped copies are held to that window.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import perturb
from .config import Config, config_to_dict
from .contradict import EntitySwapper, cross_class
from .data import NewsItem, SquadQuestion, load_agnews, load_squad, stratified_sample
from .paragraphs import Paragraph, TokenCounter, contains_answer, normalize, text_id


def derive_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def rng_for(*parts: object) -> random.Random:
    return random.Random(derive_seed(*parts))


def context_text(texts: list[str]) -> str:
    """How every model sees a context: paragraphs separated by a blank line."""
    return "\n\n".join(texts)


# --------------------------------------------------------------------------- bases


@dataclass
class Base:
    """One base example: the fixed signal plus everything its conditions need."""
    id: str
    task: str
    signal: Paragraph
    paraphrases: list[Paragraph] = field(default_factory=list)
    conflict: Paragraph | None = None
    label: int | None = None  # classification
    question: str | None = None  # retrieval / qa
    answers: list[str] = field(default_factory=list)  # empty = unanswerable
    conflict_value: str | int | None = None  # swapped entity, or the other class
    meta: dict = field(default_factory=dict)

    @property
    def answerable(self) -> bool:
        return self.task == "classification" or bool(self.answers)


class LabelJudge:
    """An AG News classifier that is *not* evaluated, used to check paraphrases keep
    their label. Its label order is learned on AG News train data, not assumed."""

    def __init__(self, model_id: str, device: str, calibration: list[NewsItem]):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        self.model.eval()
        raw = self._raw([x.text for x in calibration])
        gold = [x.label for x in calibration]
        best = max(itertools.permutations(range(4)),
                   key=lambda p: sum(p[r] == g for r, g in zip(raw, gold, strict=True)))
        self.perm = list(best)
        self.calibration_accuracy = sum(self.perm[r] == g for r, g in zip(raw, gold,
                                                                         strict=True)) / len(gold)

    def _raw(self, texts: list[str]) -> list[int]:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), 32):
                enc = self.tok(texts[i:i + 32], truncation=True, max_length=256, padding=True,
                               return_tensors="pt").to(self.device)
                out += self.model(**enc).logits.argmax(-1).tolist()
        return out

    def predict(self, texts: list[str]) -> list[int]:
        return [self.perm[r] for r in self._raw(texts)]


def _device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Builder:
    """Builds the dataset. Heavy components can be injected (the tests use light fakes)."""

    def __init__(self, cfg: Config, counter=None, selector=None, translator=None,
                 swapper=None, judge=None, loaders=None, log=print):
        self.cfg = cfg
        self.log = log
        self.lo, self.hi = cfg.token_range
        d = cfg.data
        self.device = _device(d.device)
        self.counter = counter or TokenCounter(d.tokenizer)
        self._selector, self._translator = selector, translator
        self._swapper, self._judge = swapper, judge
        self.loaders = loaders or {"agnews": lambda: load_agnews(d.raw_dir),
                                   "squad": lambda: load_squad(d.raw_dir)}
        self.paragraphs: dict[str, Paragraph] = {}
        self.info: dict = {}

    # ---- lazily built heavy components
    @property
    def selector(self):
        if self._selector is None:
            self._selector = perturb.Selector(self.cfg.data.selector_model, self.device)
        return self._selector

    @property
    def translator(self):
        if self._translator is None:
            from .paraphrase import BackTranslator

            self._translator = BackTranslator(
                self.cfg.data.paraphrase_models,
                Path(self.cfg.data.raw_dir) / "paraphrases.jsonl", self.device)
        return self._translator

    @property
    def swapper(self):
        if self._swapper is None:
            self._swapper = EntitySwapper(self.cfg.data.ner_model)
        return self._swapper

    def _add(self, p: Paragraph) -> Paragraph:
        self.paragraphs.setdefault(p.id, p)
        return self.paragraphs[p.id]

    def _norm_para(self, text: str, source: str, title: str | None = None,
                   label: int | None = None) -> Paragraph | None:
        r = normalize(text, self.counter, self.lo, self.hi)
        if r is None:
            return None
        t, n = r
        return Paragraph(text_id("w" if source.startswith("squad") else "n", t), t, n, source,
                         label=label, title=title)

    # ---- pools
    def _wiki_pools(self, train_paras: list[dict], dev: list[SquadQuestion]):
        d = self.cfg.data
        rng = rng_for(self.cfg.seed, "wiki-pool")
        order = list(range(len(train_paras)))
        rng.shuffle(order)
        need = d.random_pool + d.hard_pool
        pool: list[Paragraph] = []
        seen: set[str] = set()
        for i in order:
            if len(pool) >= need:
                break
            r = train_paras[i]
            p = self._norm_para(r["context"], "squad_train", r["title"])
            if p is not None and p.id not in seen:
                seen.add(p.id)
                pool.append(p)
        if len(pool) < need:
            raise RuntimeError(f"only {len(pool)} normalisable train paragraphs, need {need}")
        self.random_pool = pool[:d.random_pool]
        dev_paras = []
        for ctx, title in dict.fromkeys((q.context, q.title) for q in dev):
            p = self._norm_para(ctx, "squad_dev", title)
            if p is not None:
                p.origin = text_id("ctx", ctx)
                dev_paras.append(p)
        self.hard_pool = pool[d.random_pool:] + dev_paras
        self.info["pools"] = {"random": len(self.random_pool), "hard": len(self.hard_pool),
                              "dev_paragraphs_normalised": len(dev_paras)}

    # ---- classification bases
    def _classification_bases(self, train: list[NewsItem], test: list[NewsItem]) -> list[Base]:
        d, c = self.cfg.data, self.cfg.conditions
        n = d.examples["classification"]
        rng = rng_for(self.cfg.seed, "agnews-bases")
        need_para = "redundancy" in c.axes and "paraphrase" in c.redundancy_modes
        cands = stratified_sample(test, lambda x: x.label, len(test), rng)  # full, balanced order
        judge_ok: dict[str, list[str]] = {}
        if need_para:
            if self._judge is None:
                cal = stratified_sample(train, lambda x: x.label, 400,
                                        rng_for(self.cfg.seed, "cal"))
                self._judge = LabelJudge(d.label_judge, self.device, cal)
                self.info["label_judge"] = {
                    "model": d.label_judge, "label_order": self._judge.perm,
                    "train_accuracy": round(self._judge.calibration_accuracy, 4)}
            # translate in label-balanced order until each class has enough valid bases
            per_label = n // 4 + (1 if n % 4 else 0)
            chosen_by_label: dict[int, list] = defaultdict(list)
            chunk = min(64, max(8, n))
            for i in range(0, len(cands), chunk):
                todo = [x for x in cands[i:i + chunk]
                        if len(chosen_by_label[x.label]) < per_label]
                if not todo:
                    continue
                para = self.translator.paraphrases([x.text for x in todo])
                flat = [(x, p) for x in todo for p in para[x.text]]
                labels = self._judge.predict([p for _, p in flat]) if flat else []
                for (x, p), lab in zip(flat, labels, strict=True):
                    if lab == x.label:
                        judge_ok.setdefault(x.id, []).append(p)
                for x in todo:
                    if len(judge_ok.get(x.id, [])) >= c.redundancy_k - 1:
                        chosen_by_label[x.label].append(x)
                if all(len(chosen_by_label[k]) >= per_label for k in range(4)):
                    break
            tried = min(len(cands), i + chunk)
            self.info["classification_funnel"] = {
                "candidates_tried": tried,
                "with_k-1_label_keeping_paraphrases": sum(len(v) for v in chosen_by_label.values())}
            chosen = stratified_sample([x for v in chosen_by_label.values() for x in v],
                                       lambda x: x.label, n, rng)
        else:
            chosen = cands[:n]
        base_ids = {x.id for x in chosen}
        conflict_pool: dict[int, list[NewsItem]] = defaultdict(list)
        for x in test:
            if x.id not in base_ids:
                conflict_pool[x.label].append(x)
        bases = []
        for x in sorted(chosen, key=lambda x: x.id):
            sig = self._add(Paragraph(f"ag:{x.id}", x.text, self.counter.count(x.text),
                                      "agnews", label=x.label))
            paras = [self._add(Paragraph(text_id("pp", p), p, self.counter.count(p),
                                         "paraphrase", label=x.label))
                     for p in judge_ok.get(x.id, [])[:c.redundancy_k - 1]]
            other = cross_class(x.label, conflict_pool, rng_for(self.cfg.seed, "conflict", x.id))
            conf = self._add(Paragraph(f"ag:{other.id}", other.text,
                                       self.counter.count(other.text), "agnews",
                                       label=other.label))
            bases.append(Base(f"cls-{x.id}", "classification", sig, paras, conf, label=x.label,
                              conflict_value=other.label))
        return bases

    # ---- SQuAD bases (retrieval and QA share their answerable questions)
    def _squad_bases(self, dev: list[SquadQuestion]) -> tuple[list[Base], list[Base]]:
        d, c = self.cfg.data, self.cfg.conditions
        in_range = [q for q in dev if self.lo <= self.counter.count(q.context) <= self.hi]
        answerable = [q for q in in_range if q.answerable]
        unanswerable = [q for q in in_range if not q.answerable]
        n_ret = d.examples["retrieval"]
        n_qa_un = round(d.examples["qa"] * d.qa_unanswerable_share)
        n_qa_ans = d.examples["qa"] - n_qa_un
        n_ans = max(n_ret, n_qa_ans)
        self.swapper.fit(sorted({q.context for q in dev}))
        rng = rng_for(self.cfg.seed, "squad-bases")
        cands = stratified_sample(answerable, lambda q: q.title, len(answerable), rng)
        funnel = Counter()
        need_para = "redundancy" in c.axes and "paraphrase" in c.redundancy_modes
        chosen: list[tuple[SquadQuestion, str, Paragraph, list[str]]] = []
        chunk = min(64, max(8, n_ans))
        for i in range(0, len(cands), chunk):
            part = cands[i:i + chunk]
            types = self.swapper.answer_types([(q.context, q.answers[0], q.answer_start)
                                               for q in part])
            funnel["tried"] += len(part)
            batch = [(q, t) for q, t in zip(part, types, strict=True) if t]
            funnel["answer_is_entity"] += len(batch)
            para = (self.translator.paraphrases([q.context for q, _ in batch])
                    if need_para and batch else {})
            for q, t in batch:
                sw = None
                srng = rng_for(self.cfg.seed, "swap", q.id)
                for _ in range(8):  # a few draws until the copy stays in the token window
                    r = self.swapper.swap(q.context, q.answers, t, srng)
                    if r and self.lo <= self.counter.count(r[0]) <= self.hi:
                        sw = r
                        break
                if sw is None:
                    continue
                funnel["swap_ok"] += 1
                good = [p for p in para.get(q.context, [])
                        if any(contains_answer(p, a) for a in q.answers)
                        and self.lo <= self.counter.count(p) <= self.hi]
                if need_para and len(good) < c.redundancy_k - 1:
                    continue
                funnel["paraphrases_ok"] += 1
                conf = Paragraph(text_id("cf", sw[0]), sw[0], self.counter.count(sw[0]),
                                 "conflict", title=q.title)
                chosen.append((q, t, conf, good[:c.redundancy_k - 1]))
                q.meta["swap"] = sw[1]
            if len(chosen) >= n_ans:
                break
        self.info["squad_funnel"] = dict(funnel)
        if len(chosen) < n_ans:
            raise RuntimeError(f"only {len(chosen)} usable answerable questions, need {n_ans}")
        picked = chosen[:n_ans]  # candidates were already in title-stratified order
        self.info["squad_answer_types"] = dict(Counter(t for _, t, _, _ in picked))

        def make(q: SquadQuestion, task: str, conf=None, paras=()) -> Base:
            sig = self._add(Paragraph(f"sq:{text_id('ctx', q.context)[4:]}", q.context,
                                      self.counter.count(q.context), "squad_dev", title=q.title,
                                      origin=text_id("ctx", q.context)))
            pp = [self._add(Paragraph(text_id("pp", p), p, self.counter.count(p), "paraphrase",
                                      title=q.title)) for p in paras]
            return Base(f"{task[:3]}-{q.id}", task, sig, pp, self._add(conf) if conf else None,
                        question=q.question, answers=q.answers,
                        conflict_value=q.meta.get("swap"), meta={"title": q.title})

        retrieval = [make(q, "retrieval", conf, paras) for q, _, conf, paras in picked[:n_ret]]
        qa_ans = [make(q, "qa", conf, paras) for q, _, conf, paras in picked[:n_qa_ans]]
        un = stratified_sample(unanswerable, lambda q: q.title, n_qa_un,
                               rng_for(self.cfg.seed, "unanswerable")) if n_qa_un else []
        qa = qa_ans + [make(q, "qa") for q in un]
        return retrieval, qa

    # ---- fillers
    def _fillers(self, b: Base) -> dict[str, list[Paragraph]]:
        sizes = self.cfg.conditions.sizes
        need = max(sizes) + 2
        banned = b.answers

        def ok(p: Paragraph) -> bool:
            if p.id == b.signal.id or (p.origin is not None and p.origin == b.signal.origin):
                return False
            return not any(contains_answer(p.text, a) for a in banned)

        rand = [p for p in self.random_pool if ok(p)]
        rng_for(self.cfg.seed, "random-order", b.id).shuffle(rand)
        cands = [p for p in self.hard_pool if ok(p)]
        query = b.signal.text if b.task == "classification" else b.question
        emb = self.selector.embed([p.id for p in cands], [p.text for p in cands])
        ranked = perturb.rank_by_similarity(self.selector.query(query), cands, emb, need)
        b.meta["hard_similarity"] = [round(s, 4) for _, s in ranked[:max(sizes)]]
        return {"random": [self._add(p) for p in rand[:need]],
                "hard_negative": [self._add(p) for p, _ in ranked]}

    # ---- conditions
    def conditions_for(self, b: Base) -> list[dict]:
        c = self.cfg.conditions
        other = "hard_negative" if c.base_filler == "random" else "random"
        out = []
        for n in c.sizes:
            if "dilution" in c.axes:
                out.append({"axis": "dilution", "N": n, "noise_type": c.base_filler})
            if "noise" in c.axes:
                out.append({"axis": "noise", "N": n, "noise_type": other})
            if not b.answerable:  # unanswerable QA: only how much and what kind of text
                continue
            if "redundancy" in c.axes:
                for mode in c.redundancy_modes:
                    out.append({"axis": "redundancy", "N": n, "noise_type": c.base_filler,
                                "redundancy_k": c.redundancy_k, "redundancy_mode": mode})
            if "contradiction" in c.axes and b.conflict is not None:
                for pos in c.contradiction_positions:
                    out.append({"axis": "contradiction", "N": n, "noise_type": c.base_filler,
                                "contradiction_pos": pos})
            if "position" in c.axes and n >= 2:
                seen = set()
                for depth in c.depths:
                    i = perturb.depth_index(n, depth)
                    if i not in seen:
                        seen.add(i)
                        out.append({"axis": "position", "N": n, "noise_type": c.base_filler,
                                    "depth": depth})
        return out

    def _sample(self, b: Base, cond: dict, fillers: dict) -> dict:
        level, axis = cond["N"], cond["axis"]
        seed = derive_seed(self.cfg.seed, b.id, *sorted(cond.items()))
        rng = random.Random(seed)
        k = cond.get("redundancy_k", 1)
        if axis == "redundancy":
            signals = ([b.signal] * k if cond["redundancy_mode"] == "copy"
                       else [b.signal] + b.paraphrases[:k - 1])
        else:
            signals = [b.signal]
        conflict = b.conflict if axis == "contradiction" else None
        n = perturb.n_paragraphs(level, len(signals) + (1 if conflict else 0))
        if axis == "position":
            placed = {perturb.depth_index(n, cond["depth"]): b.signal}
        elif axis == "contradiction":
            s_i, c_i = perturb.contradiction_slots(cond["contradiction_pos"])
            placed = {s_i: b.signal, c_i: conflict}
        else:
            order = signals[:]
            rng.shuffle(order)
            placed = dict(zip(perturb.random_slots(n, len(order), rng), order, strict=True))
        pool = fillers[cond["noise_type"]][:n - len(placed)]
        if cond["noise_type"] == "hard_negative":
            pool = pool[:]
            rng.shuffle(pool)  # nested sets across N, but similarity never sets the order
        ctx = perturb.assemble(n, placed, pool)
        ids = [p.id for p in ctx]
        signal_ids = {p.id for p in signals}
        sig_pos = [i for i, pid in enumerate(ids) if pid in signal_ids]
        first = ids.index(b.signal.id)
        depth = cond.get("depth", round(first / (n - 1), 4) if n > 1 else 0.0)
        s = {
            "id": "|".join([b.id, axis, f"N{level}"] + [f"{key}={v}" for key, v in
                                                       sorted(cond.items())
                                                       if key not in ("axis", "N")]),
            "task": b.task, "base_id": b.id,
            "label": b.label, "question": b.question, "answers": b.answers,
            "answerable": b.answerable,
            "signal_id": b.signal.id,
            "signal_ids": sorted(signal_ids),
            "paragraphs": ids,
            "axis": axis, "N": level, "n_paragraphs": n, "depth": depth,
            "signal_positions": sig_pos,
            "noise_type": cond.get("noise_type"),
            "redundancy_k": k if axis == "redundancy" else 1,
            "redundancy_mode": cond.get("redundancy_mode"),
            "contradiction_pos": cond.get("contradiction_pos"),
            "conflict_id": conflict.id if conflict else None,
            "conflict_value": b.conflict_value if conflict else None,
            "token_count": self.counter.count(context_text([p.text for p in ctx])),
            "seed": seed,
        }
        return s

    # ---- whole dataset
    def build(self) -> tuple[list[dict], dict[str, Paragraph]]:
        ag_train, ag_test = self.loaders["agnews"]()
        dev, train_paras = self.loaders["squad"]()
        self.log(f"loaded AG News ({len(ag_train)} train / {len(ag_test)} test), SQuAD 2.0 "
                 f"({len(dev)} dev questions, {len(train_paras)} train paragraphs)")
        self._wiki_pools(train_paras, dev)
        self.log(f"pools: {self.info['pools']}")
        bases = []
        if self.cfg.data.examples.get("classification"):
            bases += self._classification_bases(ag_train, ag_test)
        if self.cfg.data.examples.get("retrieval") or self.cfg.data.examples.get("qa"):
            ret, qa = self._squad_bases(dev)
            bases += ret + qa
        self.log(f"bases: {dict(Counter(b.task for b in bases))}")
        samples = []
        for b in bases:
            fillers = self._fillers(b)
            samples += [self._sample(b, cond, fillers) for cond in self.conditions_for(b)]
        self.bases = bases
        used = {pid for s in samples for pid in s["paragraphs"]}
        used |= {b.conflict.id for b in bases if b.conflict}
        paras = {pid: self.paragraphs[pid] for pid in sorted(used)}
        problems = [f"{s['id']}: {e}" for s in samples
                    for e in check_sample(s, paras, self.counter, (self.lo, self.hi))]
        if problems:
            raise RuntimeError("dataset failed its checks:\n" + "\n".join(problems[:20]))
        return samples, paras


# --------------------------------------------------------------------------- checks


def check_sample(s: dict, paras: dict[str, Paragraph], counter: TokenCounter | None,
                 token_range: tuple[int, int]) -> list[str]:
    """Every invariant the sample violates (empty list = valid)."""
    errs: list[str] = []
    lo, hi = token_range
    ctx = [paras[pid] for pid in s["paragraphs"]]
    n = s["n_paragraphs"]
    signal = paras[s["signal_id"]]
    n_evidence = (s["redundancy_k"] if s["axis"] == "redundancy" else 1) + bool(s["conflict_id"])
    if n != max(s["N"], n_evidence):
        errs.append(f"n_paragraphs={n}, expected max(N={s['N']}, evidence={n_evidence})")
    if len(ctx) != n:
        errs.append(f"{len(ctx)} paragraphs, expected {n}")
    if s["signal_id"] not in s["paragraphs"]:
        errs.append("signal paragraph missing")
    copy_mode = s["redundancy_mode"] == "copy"
    if not copy_mode and len(set(s["paragraphs"])) != len(s["paragraphs"]):
        errs.append("a paragraph appears twice")
    if copy_mode and s["paragraphs"].count(s["signal_id"]) != s["redundancy_k"]:
        errs.append("exact-copy redundancy does not repeat the signal k times")
    for p in ctx:
        if p.label is not None or (p.id in s["signal_ids"] and p.source == "squad_dev"):
            continue  # news articles (and their paraphrases) and dev signals keep their length
        if not lo <= p.tokens <= hi:
            errs.append(f"paragraph {p.id} has {p.tokens} tokens, outside [{lo}, {hi}]")
    if counter is not None:
        real = counter.count(context_text([p.text for p in ctx]))
        if real != s["token_count"]:
            errs.append(f"token_count {s['token_count']} != recount {real}")
    # the answer appears only where it should
    fill = [p for p in ctx if p.id not in s["signal_ids"] and p.id != s["conflict_id"]]
    if s["task"] in ("retrieval", "qa") and s["answerable"]:
        for p in [signal] + [paras[i] for i in s["signal_ids"]]:
            if not any(contains_answer(p.text, a) for a in s["answers"]):
                errs.append(f"signal copy {p.id} lacks the answer")
        if any(contains_answer(p.text, a) for p in fill for a in s["answers"]):
            errs.append("answer appears in a filler paragraph")
    if s["task"] == "classification":
        if any(p.source == "agnews" for p in fill):
            errs.append("a news article is used as filler")
        for i in s["signal_ids"]:
            if paras[i].label != s["label"]:
                errs.append(f"signal copy {i} has a different label")
    # filler type
    want = "squad_train" if s["noise_type"] == "random" else None
    if want and any(p.source != want for p in fill):
        errs.append("random filler must be unrelated Wikipedia (SQuAD train) paragraphs")
    if s["noise_type"] == "hard_negative" and any(p.source not in ("squad_train", "squad_dev")
                                                  for p in fill):
        errs.append("hard negatives must be Wikipedia paragraphs")
    # axis-specific
    if s["axis"] == "contradiction":
        c = paras[s["conflict_id"]]
        si, ci = s["paragraphs"].index(s["signal_id"]), s["paragraphs"].index(s["conflict_id"])
        if abs(si - ci) != 1 or (ci < si) != (s["contradiction_pos"] == "before"):
            errs.append("conflict is not adjacent on the requested side")
        if s["task"] == "classification":
            if c.label is None or c.label == s["label"] or c.label != s["conflict_value"]:
                errs.append("cross-class paragraph does not carry a different label")
        else:
            if any(contains_answer(c.text, a) for a in s["answers"]):
                errs.append("NER-swapped copy still contains the answer")
            if not contains_answer(c.text, s["conflict_value"]):
                errs.append("NER-swapped copy lacks the replacement entity")
            if c.text == signal.text:
                errs.append("NER-swapped copy equals the signal")
    if s["axis"] == "position":
        want_i = round(s["depth"] * (n - 1))
        if s["paragraphs"].index(s["signal_id"]) != want_i:
            errs.append(f"signal at {s['paragraphs'].index(s['signal_id'])}, expected {want_i}")
    if s["axis"] == "redundancy" and not copy_mode:
        texts = [paras[i].text for i in s["signal_ids"]]
        if len(set(texts)) != s["redundancy_k"]:
            errs.append("paraphrase redundancy needs k distinct texts")
    return errs


# --------------------------------------------------------------------------- io


def paragraphs_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".paragraphs.jsonl")


def manifest_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".manifest.json")


def write_dataset(cfg: Config, builder: Builder, samples: list[dict],
                  paras: dict[str, Paragraph]) -> dict:
    out = Path(cfg.data.path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    with paragraphs_path(out).open("w", encoding="utf-8", newline="\n") as fh:
        for p in paras.values():
            fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    sha = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
           for f in (out, paragraphs_path(out))}
    manifest = {
        "version": cfg.data.version,
        "samples": len(samples),
        "bases": dict(Counter(b.task for b in builder.bases)),
        "paragraphs": len(paras),
        "by_task_axis": {f"{t}/{a}": v for (t, a), v in
                         sorted(Counter((s["task"], s["axis"]) for s in samples).items())},
        "context_tokens_total": sum(s["token_count"] for s in samples),
        "context_tokens_max": max(s["token_count"] for s in samples),
        "build_info": builder.info,
        "sha256": sha,
        "config": config_to_dict(cfg),
    }
    manifest_path(out).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
                                  newline="\n")
    return manifest


def load_paragraphs(path: str | Path) -> dict[str, Paragraph]:
    out = {}
    with paragraphs_path(path).open(encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            out[d["id"]] = Paragraph.from_dict(d)
    return out


def load_dataset(path: str | Path) -> list[dict]:
    """Samples with `context` (list of paragraph texts) filled in."""
    paras = load_paragraphs(path)
    samples = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            s = json.loads(line)
            s["context"] = [paras[pid].text for pid in s["paragraphs"]]
            samples.append(s)
    return samples
