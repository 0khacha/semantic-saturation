"""Scoring, bootstrap statistics, N*, N†, SSS and the pre-registered hypothesis tests.

One bootstrap over base examples drives everything: a resample is a vector of example
multiplicities, and every statistic is a weighted mean (or weighted ECE) under it. All
statistics of a model share the same resamples, so differences are paired.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .build import load_dataset
from .config import Config
from .metrics import ece as ece_fn
from .run import select_examples

PRIMARY = {"classification": "correct", "retrieval": "r1", "qa": "em"}
SCORE_NAME = {"classification": "accuracy", "retrieval": "Recall@1", "qa": "exact match"}

# curve -> (label, pandas query)
CURVES: dict[str, tuple[str, str]] = {
    "dilution": ("Dilution (random filler)", "axis == 'dilution'"),
    "noise": ("Hard-negative noise", "axis == 'noise'"),
    "redundancy_copy": ("Redundancy k=5, copies", "axis == 'redundancy' and "
                        "redundancy_mode == 'copy'"),
    "redundancy_paraphrase": ("Redundancy k=5, paraphrases", "axis == 'redundancy' and "
                              "redundancy_mode == 'paraphrase'"),
    "contradiction": ("Contradiction", "axis == 'contradiction'"),
    "position": ("Position d=0.5", "axis == 'position' and depth == 0.5"),
    "position_d0": ("Position d=0", "axis == 'position' and depth == 0.0"),
    "position_d1": ("Position d=1", "axis == 'position' and depth == 1.0"),
}
MAIN_CURVES = ["dilution", "noise", "redundancy_copy", "redundancy_paraphrase",
               "contradiction", "position"]


# --------------------------------------------------------------------------- scoring


def score_run(cfg: Config) -> pd.DataFrame:
    samples = {s["id"]: s for s in select_examples(load_dataset(cfg.data.path),
                                                    cfg.run.examples)}
    rows = []
    for m in cfg.models:
        path = Path(cfg.run.output_dir) / "predictions" / f"{m.name}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = samples.get(p["sample_id"])
            if s is None:
                continue
            r = {"model": m.name, "task": m.task, "kind": m.kind, "sample_id": s["id"],
                 "base_id": s["base_id"], "axis": s["axis"], "N": s["N"],
                 "n_paragraphs": s["n_paragraphs"], "depth": s["depth"],
                 "noise_type": s["noise_type"], "redundancy_mode": s["redundancy_mode"],
                 "contradiction_pos": s["contradiction_pos"], "answerable": s["answerable"],
                 "token_count": s["token_count"], "label": s["label"]}
            if m.task == "classification":
                r.update(pred=p["pred"], confidence=p["confidence"],
                         correct=float(p["pred"] == s["label"]),
                         n_chunks=p.get("n_chunks"))
                if s["axis"] == "contradiction":
                    r["outcome"] = ("true" if p["pred"] == s["label"] else "conflicting"
                                    if p["pred"] == s["conflict_value"] else "other")
                r["probs"] = p["probs"]
            elif m.task == "retrieval":
                r.update({k: p[k] for k in ("rank", "r1", "r5", "mrr")})
                r["outcome"] = p.get("outcome")
            else:
                r.update(answer=p["answer"], confidence=p["confidence"], windows=p["windows"],
                         abstained=p["abstained"], em=p.get("em"), f1=p.get("f1"),
                         noans_correct=p.get("noans_correct"), outcome=p.get("outcome"))
            rows.append(r)
    if not rows:
        raise FileNotFoundError(f"no predictions under {cfg.run.output_dir}; run `ssb run`")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- bootstrap


class Boot:
    def __init__(self, ids: np.ndarray, n_boot: int, seed: int):
        self.keys, self.codes = np.unique(ids, return_inverse=True)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(self.keys), size=(n_boot, len(self.keys)))
        self.W = np.stack([np.bincount(d, minlength=len(self.keys)) for d in draws]).astype(float)

    def mean(self, idx: np.ndarray, v: np.ndarray) -> tuple[float, np.ndarray]:
        x = v[idx].astype(float)
        if x.size == 0:
            return math.nan, np.full(len(self.W), math.nan)
        w = self.W[:, self.codes[idx]]
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(x.mean()), (w * x).sum(1) / w.sum(1)

    def ece(self, idx, conf, correct, bins):
        c, y = conf[idx].astype(float), correct[idx].astype(float)
        if c.size == 0:
            return math.nan, np.full(len(self.W), math.nan)
        onehot = np.eye(bins)[np.minimum((c * bins).astype(int), bins - 1)]
        w = self.W[:, self.codes[idx]]
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = np.abs(w @ (onehot * (c - y)[:, None])).sum(1) / w.sum(1)
        return ece_fn(c, y, bins), boot


@dataclass
class Stat:
    point: float
    boot: np.ndarray = field(repr=False)

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        b = self.boot[~np.isnan(self.boot)]
        if b.size == 0:
            return math.nan, math.nan
        a = (1 - level) / 2
        return float(np.quantile(b, a)), float(np.quantile(b, 1 - a))

    def p_greater_than_zero(self) -> float:
        b = self.boot[~np.isnan(self.boot)]
        return float((b <= 0).mean()) if b.size else math.nan

    def p_two_sided(self) -> float:
        b = self.boot[~np.isnan(self.boot)]
        if not b.size:
            return math.nan
        return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))

    def __sub__(self, o: Stat) -> Stat:
        return Stat(self.point - o.point, self.boot - o.boot)


def holm(pvalues: dict, alpha: float) -> set:
    order = sorted(pvalues, key=lambda k: pvalues[k])
    rejected = set()
    for i, k in enumerate(order):
        if pvalues[k] <= alpha / (len(order) - i):
            rejected.add(k)
        else:
            break
    return rejected


# --------------------------------------------------------------------------- per model


class ModelAnalysis:
    """Every statistic for one model, under one shared bootstrap."""

    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.model, self.task = df.model.iloc[0], df.task.iloc[0]
        self.boot = Boot(self.df.base_id.to_numpy(), cfg.eval.bootstrap, cfg.seed)
        self.stats: dict[tuple, Stat] = {}
        self._compute()

    def _rows(self, query: str, answerable: bool | None = True) -> np.ndarray:
        d = self.df.query(query) if query else self.df
        if answerable is not None:
            d = d[d.answerable == answerable]
        return d.index.to_numpy()

    def _mean(self, idx, col) -> Stat:
        v = pd.to_numeric(self.df[col], errors="coerce").to_numpy(dtype=float)
        ok = idx[~np.isnan(v[idx])]
        return Stat(*self.boot.mean(ok, v))

    def ns(self, curve: str) -> list[int]:
        d = self.df.query(CURVES[curve][1])
        d = d[d.answerable]
        return sorted(d.N.unique().tolist())

    def _compute(self) -> None:
        primary = PRIMARY[self.task]
        has_conf = "confidence" in self.df and self.df.confidence.notna().any()
        conf = (pd.to_numeric(self.df.confidence, errors="coerce").to_numpy(float)
                if has_conf else None)
        for curve, (_, q) in CURVES.items():
            for n in self.ns(curve):
                idx = self._rows(f"({q}) and N == {n}")
                self.stats[("score", curve, n)] = self._mean(idx, primary)
                for col in ("r5", "mrr", "f1"):
                    if col in self.df:
                        self.stats[(col, curve, n)] = self._mean(idx, col)
                if has_conf:
                    corr = pd.to_numeric(self.df[primary], errors="coerce").to_numpy(float)
                    cidx = idx[~np.isnan(conf[idx]) & ~np.isnan(corr[idx])]
                    if cidx.size:
                        c, y = Stat(*self.boot.mean(cidx, conf)), Stat(*self.boot.mean(cidx, corr))
                        self.stats[("confidence", curve, n)] = c
                        self.stats[("overconfidence", curve, n)] = c - y
                        self.stats[("ece", curve, n)] = Stat(*self.boot.ece(cidx, conf, corr,
                                                                            self.cfg.eval.ece_bins))
        if self.task == "classification":
            for n in self.ns("dilution"):
                for curve in ("dilution", "noise"):
                    d = self.df.query(f"({CURVES[curve][1]}) and N == {n}")
                    if len(d):
                        self.stats[("macro_f1", curve, n)] = Stat(
                            _macro_f1(d), np.full(len(self.boot.W), math.nan))
        if self.task == "qa":
            for axis in ("dilution", "noise"):
                for n in sorted(self.df[self.df.axis == axis].N.unique()):
                    idx = self._rows(f"axis == '{axis}' and N == {n}", answerable=False)
                    if idx.size:
                        self.stats[("noans", axis, n)] = self._mean(idx, "noans_correct")
        if "outcome" in self.df:
            for pos in ("before", "after"):
                for n in self.ns("contradiction"):
                    idx = self._rows(f"axis == 'contradiction' and contradiction_pos == '{pos}'"
                                     f" and N == {n}")
                    for oc in ("true", "conflicting", "other", "tie"):
                        v = (self.df.outcome == oc).astype(float).to_numpy()
                        self.stats[(f"outcome_{oc}", pos, n)] = Stat(*self.boot.mean(idx, v))
            decided = self.df.outcome.isin(["true", "conflicting"]).to_numpy()
            conflicting = (self.df.outcome == "conflicting").astype(float).to_numpy()
            for pos in ("all", "before", "after"):
                q = "axis == 'contradiction'" + (f" and contradiction_pos == '{pos}'"
                                                 if pos != "all" else "")
                idx = np.intersect1d(self._rows(q), np.flatnonzero(decided))
                self.stats[("conflict_share", pos, 0)] = Stat(*self.boot.mean(idx, conflicting))

    # ---- curve summaries
    def score(self, curve, n) -> Stat:
        return self.stats[("score", curve, n)]

    def sss(self, curve) -> Stat | None:
        ns = self.ns(curve)
        if len(ns) < 2:
            return None
        x = np.log10(ns)
        pts = [self.score(curve, n) for n in ns]
        span = x[-1] - x[0]
        return Stat(float(np.trapezoid([p.point for p in pts], x) / span),
                    np.trapezoid(np.stack([p.boot for p in pts], 1), x, axis=1) / span)

    def saturation_point(self, curve) -> int | None:
        ns = self.ns(curve)
        for prev, n in zip(ns, ns[1:], strict=False):
            if self.score(curve, n).point - self.score(curve, prev).point < \
                    self.cfg.eval.saturation_epsilon:
                return n
        return None

    def degradation_onset(self, curve) -> tuple[int | None, int | None]:
        ns = self.ns(curve)
        if not ns:
            return None, None
        peak = max(ns, key=lambda n: (self.score(curve, n).point, -n))
        pv = {n: (self.score(curve, peak) - self.score(curve, n)).p_greater_than_zero()
              for n in ns if n > peak}
        rej = holm(pv, self.cfg.eval.alpha)
        return (min(rej) if rej else None), peak


def _macro_f1(d: pd.DataFrame) -> float:
    from .metrics import macro_f1

    return macro_f1(d.label.to_numpy(int), d.pred.to_numpy(int))


# --------------------------------------------------------------------------- hypotheses


def _f(s: Stat, k: float = 100) -> str:
    lo, hi = s.ci()
    return f"{s.point * k:+.1f} [{lo * k:+.1f}, {hi * k:+.1f}]"


def _inside(s: Stat, bound: float) -> bool:
    lo, hi = s.ci()
    return -bound <= lo and hi <= bound


def test_hypotheses(an: dict[str, ModelAnalysis], cfg: Config) -> list[dict]:
    a = cfg.eval.alpha
    out = []

    def add(hid, verdict, evidence):
        out.append({"id": hid, "verdict": verdict, "evidence": evidence})

    def has(m, *keys):
        return m in an and all(k in an[m].stats for k in keys)

    big = max(cfg.conditions.sizes)
    # H1
    p_dil, p_noise, parts = {}, {}, []
    for name, ma in an.items():
        if has(name, ("score", "dilution", 1), ("score", "dilution", big),
               ("score", "noise", big)):
            dd = ma.score("dilution", 1) - ma.score("dilution", big)
            dn = ma.score("dilution", big) - ma.score("noise", big)
            p_dil[name], p_noise[name] = dd.p_greater_than_zero(), dn.p_greater_than_zero()
            parts.append(f"{name}: N=1→{big} {_f(dd)}, random−hard {_f(dn)}")
    if p_dil:
        r1, r2 = holm(p_dil, a), holm(p_noise, a)
        n = len(p_dil)
        verdict = ("supported" if len(r1) == n and len(r2) == n else
                   "partly supported" if len(r1) > n / 2 and len(r2) > n / 2 else
                   "not supported")
        add("H1", verdict, f"degrades: {len(r1)}/{n}, hard worse: {len(r2)}/{n}. "
            + "; ".join(parts))
    # H2
    bag = [m for m in ("tfidf-lr", "tfidf", "bm25") if has(m, ("score", "position_d0", big))]
    ok_a, parts = True, []
    for m in bag:
        ma = an[m]
        d1 = ma.score("position_d0", big) - ma.score("position", big)
        d2 = ma.score("position_d0", big) - ma.score("position_d1", big)
        ok_a &= _inside(d1, 0.05) and _inside(d2, 0.05)
        parts.append(f"{m}: d0−d0.5 {_f(d1)}, d0−d1 {_f(d2)}")
    ok_b = None
    if has("bert-truncate", ("score", "position_d0", big)):
        ma = an["bert-truncate"]
        mid, end, start = (ma.score("position", big), ma.score("position_d1", big),
                           ma.score("position_d0", big))
        ok_b = (mid.point < 0.5 and end.point < 0.5
                and (start - mid).p_greater_than_zero() < a
                and (start - end).p_greater_than_zero() < a)
        parts.append(f"bert-truncate N={big}: d0 {start.point:.1%}, d0.5 {mid.point:.1%}, "
                     f"d1 {end.point:.1%}")
    if bag or ok_b is not None:
        verdict = ("supported" if ok_a and ok_b else
                   "partly supported" if ok_a or ok_b else "not supported")
        add("H2", verdict, "; ".join(parts))
    # H3
    if has("bert-chunk-mean", ("score", "position_d0", big), ("score", "dilution", 1)):
        ma = an["bert-chunk-mean"]
        d1 = ma.score("position_d0", big) - ma.score("position", big)
        d2 = ma.score("position_d0", big) - ma.score("position_d1", big)
        fix = _inside(d1, 0.05) and _inside(d2, 0.05)
        deg = ma.score("dilution", 1) - ma.score("dilution", big)
        dg = deg.p_greater_than_zero() < a
        add("H3", "supported" if fix and dg else "partly supported" if fix or dg
            else "not supported",
            f"position d0−d0.5 {_f(d1)}, d0−d1 {_f(d2)}; dilution N=1→{big} {_f(deg)}")
    # H4
    if has("minilm", ("score", "noise", big)):
        comps = {}
        for lex in ("bm25", "tfidf"):
            if has(lex, ("score", "noise", big)):
                comps[f"{lex}>minilm hard"] = an[lex].score("noise", big) - \
                    an["minilm"].score("noise", big)
                comps[f"minilm>{lex} random"] = an["minilm"].score("dilution", big) - \
                    an[lex].score("dilution", big)
        rej = holm({k: v.p_greater_than_zero() for k, v in comps.items()}, a)
        hard_ok = all(k in rej for k in comps if "hard" in k)
        rand_ok = all(k in rej for k in comps if "random" in k)
        add("H4", "supported" if hard_ok and rand_ok else "partly supported"
            if hard_ok or rand_ok else "not supported",
            "; ".join(f"{k}: {_f(v)}{' *' if k in rej else ''}" for k, v in comps.items()))
    # H5
    gains, parts = {}, []
    for name, ma in an.items():
        if has(name, ("score", "redundancy_copy", big), ("score", "redundancy_paraphrase", big)):
            base = ma.score("dilution", big)
            gains[name] = {mode: ma.score(f"redundancy_{mode}", big) - base
                           for mode in ("copy", "paraphrase")}
    if gains:
        slows = sum(max(g.values(), key=lambda s: s.point).p_greater_than_zero() < a
                    for g in gains.values())
        lex = [m for m in ("tfidf-lr", "tfidf") if m in gains]
        neural = [m for m in ("bert-truncate", "bert-chunk-mean", "minilm", "distilbert-squad",
                              "roberta-squad2") if m in gains]
        b = {m: gains[m]["copy"] - gains[m]["paraphrase"] for m in lex}
        c = {m: gains[m]["paraphrase"] - gains[m]["copy"] for m in neural}
        rb = holm({m: v.p_greater_than_zero() for m, v in b.items()}, a)
        rc = holm({m: v.p_greater_than_zero() for m, v in c.items()}, a)
        ok = [slows > len(gains) / 2, len(rb) == len(b), len(rc) == len(c)]
        parts = [f"slows degradation for {slows}/{len(gains)} models"]
        parts += [f"{m} copy−paraphrase gain {_f(v)}{' *' if m in rb else ''}"
                  for m, v in b.items()]
        parts += [f"{m} paraphrase−copy gain {_f(v)}{' *' if m in rc else ''}"
                  for m, v in c.items()]
        add("H5", "supported" if all(ok) else "partly supported" if any(ok)
            else "not supported", "; ".join(parts))
    # H6
    for m in ("distilbert-squad", "roberta-squad2"):
        if has(m, ("conflict_share", "all", 0)):
            ma = an[m]
            share = ma.stats[("conflict_share", "all", 0)]
            diff = ma.stats[("conflict_share", "before", 0)] - \
                ma.stats[("conflict_share", "after", 0)]
            half = 0.35 <= share.point <= 0.65
            pos = diff.p_two_sided() < a
            add("H6", ("supported" if half and pos else "partly supported" if half or pos
                       else "not supported") + f" ({m})",
                f"conflict share {_f(share)}%; before − after {_f(diff)} pts "
                f"(p={diff.p_two_sided():.3f})")
    # H7
    parts, oks = [], []
    for name, ma in an.items():
        if ma.task == "retrieval":
            continue
        if has(name, ("overconfidence", "dilution", 1), ("overconfidence", "dilution", big)):
            d = ma.stats[("overconfidence", "dilution", big)] - \
                ma.stats[("overconfidence", "dilution", 1)]
            oks.append(d.ci()[0] > 0)
            parts.append(f"{name}: overconfidence change {_f(d)} pts")
    if parts:
        add("H7", "supported" if all(oks) else "partly supported" if any(oks)
            else "not supported", "; ".join(parts))
    # H8
    if has("roberta-squad2", ("noans", "dilution", 1), ("noans", "dilution", big),
           ("noans", "noise", 1), ("noans", "noise", big)):
        ma = an["roberta-squad2"]
        dd = ma.stats[("noans", "dilution", 1)] - ma.stats[("noans", "dilution", big)]
        dn = ma.stats[("noans", "noise", 1)] - ma.stats[("noans", "noise", big)]
        ok = [dd.p_greater_than_zero() < a, dn.p_greater_than_zero() < a]
        add("H8", "supported" if all(ok) else "partly supported" if any(ok)
            else "not supported",
            f"no-answer accuracy drop N=1→{big}: dilution {_f(dd)} pts, noise {_f(dn)} pts")
    return out


# --------------------------------------------------------------------------- tables


def metrics_table(an: dict[str, ModelAnalysis], level: float) -> pd.DataFrame:
    rows = []
    for name, ma in an.items():
        for (metric, curve, n), s in ma.stats.items():
            lo, hi = s.ci(level)
            rows.append({"model": name, "task": ma.task, "metric": metric, "curve": curve,
                         "N": n, "value": s.point, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def leaderboard(an: dict[str, ModelAnalysis], level: float) -> pd.DataFrame:
    rows = []
    for name, ma in an.items():
        for c in MAIN_CURVES:
            s = ma.sss(c)
            if s is None:
                continue
            lo, hi = s.ci(level)
            onset, peak = ma.degradation_onset(c)
            rows.append({"task": ma.task, "model": name, "axis": c, "SSS": s.point,
                         "SSS_low": lo, "SSS_high": hi,
                         "N*": ma.saturation_point(c) or "not reached",
                         "N†": onset or f">{max(ma.ns(c))}", "peak N": peak})
    return pd.DataFrame(rows)


def analyse(cfg: Config, df: pd.DataFrame) -> dict:
    an = {m.name: ModelAnalysis(df[df.model == m.name], cfg) for m in cfg.models
          if (df.model == m.name).any()}
    return {"analyses": an, "hypotheses": test_hypotheses(an, cfg),
            "metrics": metrics_table(an, cfg.eval.ci),
            "leaderboard": leaderboard(an, cfg.eval.ci)}
