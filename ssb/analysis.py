"""Scoring, bootstrap statistics, N*, N†, SSS and the pre-registered hypothesis tests.

One question-level bootstrap drives everything: each resample is a vector of question
multiplicities, and every statistic is a weighted mean (or weighted ECE) under it. All
statistics of a model therefore share the same resamples, so differences between
conditions are paired automatically.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .build import load_dataset
from .config import Config
from .run import select_subset

# name -> (label, filter on the scores frame, value column)
CURVES: dict[str, tuple[str, str, str]] = {
    "dilution": ("Dilution (random filler)", "axis == 'dilution'", "correct"),
    "noise": ("Hard-negative noise", "axis == 'noise'", "correct"),
    "redundancy": ("Redundancy (k=5)", "axis == 'redundancy'", "correct"),
    "contradiction": ("Contradiction", "axis == 'contradiction'", "correct"),
    "position": ("Position (d=0.5)", "axis == 'position' and depth == 0.5", "correct"),
    "position_d0": ("Position d=0", "axis == 'position' and depth == 0.0", "correct"),
    "position_d1": ("Position d=1", "axis == 'position' and depth == 1.0", "correct"),
    "contradiction_before": ("Conflict before gold", "axis == 'contradiction' and "
                             "contradiction_pos == 'before'", "correct"),
    "contradiction_after": ("Conflict after gold", "axis == 'contradiction' and "
                            "contradiction_pos == 'after'", "correct"),
    "contradiction_gold": ("Contradiction gold rate", "axis == 'contradiction'", "is_gold"),
}
MAIN_CURVES = ["dilution", "noise", "redundancy", "contradiction", "position"]
H7_CURVES = ["dilution", "noise", "redundancy", "position", "contradiction_gold"]


# --------------------------------------------------------------------------- scoring


def score_run(cfg: Config) -> pd.DataFrame:
    """Join every model's generations with the dataset and score each answer."""
    samples = {s["id"]: s for s in select_subset(load_dataset(cfg.dataset.path),
                                                  cfg.run.questions)}
    rows = []
    gen_dir = Path(cfg.run.output_dir) / "generations"
    for path in sorted(gen_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                g = json.loads(line)
                s = samples.get(g["sample_id"])
                if s is None:
                    continue
                r = metrics.score(s, g)
                rows.append({
                    "model": g["model"], "model_id": g["model_id"], "backend": g["backend"],
                    "sample_id": s["id"], "question_id": s["question_id"],
                    "task_type": s["task_type"], "axis": s["axis"], "N": s["N"],
                    "depth": s["depth"], "noise_type": s["noise_type"],
                    "redundancy_k": s["redundancy_k"],
                    "contradiction_pos": s["contradiction_pos"],
                    "relation": s["relation"], "token_count": s["token_count"],
                    "gold_answer": s["gold_answer"], "conflict_answer": s["conflict_answer"],
                    "text": g["text"], **r,
                })
    if not rows:
        raise FileNotFoundError(f"no generations under {gen_dir}; run `ssb run` first")
    df = pd.DataFrame(rows)
    df["is_gold"] = df["outcome"].eq("gold")
    return df


# --------------------------------------------------------------------------- bootstrap


class Boot:
    def __init__(self, question_ids: np.ndarray, n_boot: int, seed: int):
        self.qs, self.codes = np.unique(question_ids, return_inverse=True)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(self.qs), size=(n_boot, len(self.qs)))
        self.W = np.stack([np.bincount(d, minlength=len(self.qs)) for d in draws]).astype(float)

    def _w(self, idx: np.ndarray) -> np.ndarray:
        return self.W[:, self.codes[idx]]

    def mean(self, idx: np.ndarray, v: np.ndarray) -> tuple[float, np.ndarray]:
        x = v[idx].astype(float)
        if x.size == 0:
            return math.nan, np.full(len(self.W), math.nan)
        w = self._w(idx)
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(x.mean()), (w * x).sum(1) / w.sum(1)

    def ece(self, idx: np.ndarray, conf: np.ndarray, correct: np.ndarray,
            bins: int) -> tuple[float, np.ndarray]:
        c, y = conf[idx].astype(float), correct[idx].astype(float)
        if c.size == 0:
            return math.nan, np.full(len(self.W), math.nan)
        onehot = np.eye(bins)[np.minimum((c * bins).astype(int), bins - 1)]  # r x bins
        gap = onehot * (c - y)[:, None]
        w = self._w(idx)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = np.abs(w @ gap).sum(1) / w.sum(1)
        return metrics.ece(c, y, bins), boot


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
        """One-sided bootstrap p-value for H0: value <= 0."""
        b = self.boot[~np.isnan(self.boot)]
        return float((b <= 0).mean()) if b.size else math.nan

    def __sub__(self, other: Stat) -> Stat:
        return Stat(self.point - other.point, self.boot - other.boot)

    def __add__(self, other: Stat) -> Stat:
        return Stat(self.point + other.point, self.boot + other.boot)

    def scale(self, k: float) -> Stat:
        return Stat(self.point * k, self.boot * k)


def holm(pvalues: dict, alpha: float) -> set:
    """Keys whose null is rejected under Holm's step-down correction."""
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
    def __init__(self, df: pd.DataFrame, cfg: Config, exclude_closed_book: bool = True):
        self.cfg = cfg
        self.model = df["model"].iloc[0]
        self.model_id = df["model_id"].iloc[0]
        cb = df[(df.axis == "closed_book") & (df.task_type != "unanswerable")]
        self.closed_book_correct = sorted(cb.loc[cb.correct.astype(bool), "question_id"])
        self.all = df.reset_index(drop=True)
        if exclude_closed_book:
            df = df[~df.question_id.isin(self.closed_book_correct)]
        self.df = df.reset_index(drop=True)
        e = cfg.eval
        self.boot = Boot(self.df.question_id.to_numpy(), e.bootstrap, cfg.seed)
        self.stats: dict[tuple, Stat] = {}
        self._compute()

    # ---- primitives
    def _idx(self, query: str, answerable: bool | None = True) -> np.ndarray:
        d = self.df.query(query) if query else self.df
        if answerable is True:
            d = d[d.task_type != "unanswerable"]
        elif answerable is False:
            d = d[d.task_type == "unanswerable"]
        return d.index.to_numpy()

    def _mean(self, idx: np.ndarray, col: str) -> Stat:
        v = self.df[col].astype(float).to_numpy()
        ok = idx[~np.isnan(v[idx])]
        return Stat(*self.boot.mean(ok, v))

    def curve_ns(self, curve: str) -> list[int]:
        q = CURVES[curve][1]
        return sorted(self.df.query(q).N.unique().tolist())

    def _compute(self) -> None:
        d = self.df
        bins = self.cfg.eval.ece_bins
        conf = d.confidence.astype(float).to_numpy()
        correct = d.correct.astype(float).to_numpy()
        for curve, (_, q, col) in CURVES.items():
            for n in self.curve_ns(curve):
                idx = self._idx(f"({q}) and N == {n}")
                self.stats[("acc", curve, n)] = self._mean(idx, col)
                self.stats[("f1", curve, n)] = self._mean(idx, "f1")
                self.stats[("lenient", curve, n)] = self._mean(idx, "lenient")
                self.stats[("false_abstention", curve, n)] = self._mean(idx, "false_abstention")
                cidx = idx[~np.isnan(conf[idx])]
                if cidx.size:
                    self.stats[("confidence", curve, n)] = Stat(*self.boot.mean(cidx, conf))
                    acc_c = Stat(*self.boot.mean(cidx, correct))
                    self.stats[("overconfidence", curve, n)] = (
                        self.stats[("confidence", curve, n)] - acc_c)
                    self.stats[("ece", curve, n)] = Stat(*self.boot.ece(cidx, conf, correct,
                                                                        bins))
                    sq = (conf - correct) ** 2
                    self.stats[("brier", curve, n)] = Stat(*self.boot.mean(cidx, sq))
        for axis in ("dilution", "noise"):
            for n in sorted(d[d.axis == axis].N.unique()):
                idx = self._idx(f"axis == '{axis}' and N == {n}", answerable=False)
                if idx.size:
                    self.stats[("hallucination", axis, n)] = self._mean(idx, "hallucinated")
        cb_un = self._idx("axis == 'closed_book'", answerable=False)
        if cb_un.size:
            self.stats[("hallucination", "closed_book", 0)] = self._mean(cb_un, "hallucinated")
        # contradiction outcomes and the "later claim" share
        for pos in ("before", "after"):
            for n in self.curve_ns("contradiction"):
                idx = self._idx(f"axis == 'contradiction' and contradiction_pos == '{pos}' "
                                f"and N == {n}")
                for oc in ("gold", "conflict", "other"):
                    v = (d.outcome == oc).astype(float).to_numpy()
                    self.stats[(f"outcome_{oc}", f"contradiction_{pos}", n)] = Stat(
                        *self.boot.mean(idx, v))
        later = (((d.contradiction_pos == "before") & (d.outcome == "gold"))
                 | ((d.contradiction_pos == "after") & (d.outcome == "conflict")))
        d_later = later.astype(float).to_numpy()
        decided = d.outcome.isin(["gold", "conflict"])
        for n in [None] + self.curve_ns("contradiction"):
            q = "axis == 'contradiction'" + (f" and N == {n}" if n else "")
            idx = np.intersect1d(self._idx(q), np.flatnonzero(decided))
            self.stats[("later_share", "contradiction", n or "all")] = Stat(
                *self.boot.mean(idx, d_later))

    # ---- curve summaries
    def acc(self, curve: str, n: int) -> Stat:
        return self.stats[("acc", curve, n)]

    def sss(self, curve: str) -> Stat | None:
        ns = self.curve_ns(curve)
        if len(ns) < 2:
            return None
        x = np.log10(ns)
        pts = [self.acc(curve, n) for n in ns]
        span = x[-1] - x[0]
        point = float(np.trapezoid([p.point for p in pts], x) / span)
        boot = np.trapezoid(np.stack([p.boot for p in pts], 1), x, axis=1) / span
        return Stat(point, boot)

    def saturation_point(self, curve: str) -> int | None:
        """N*: smallest N whose gain over the previous N is below epsilon."""
        ns = self.curve_ns(curve)
        eps = self.cfg.eval.saturation_epsilon
        for prev, n in zip(ns, ns[1:], strict=False):
            if self.acc(curve, n).point - self.acc(curve, prev).point < eps:
                return n
        return None

    def degradation_onset(self, curve: str) -> tuple[int | None, dict]:
        """N†: smallest N after the peak significantly below it (Holm-corrected)."""
        ns = self.curve_ns(curve)
        if not ns:
            return None, {}
        peak = max(ns, key=lambda n: (self.acc(curve, n).point, -n))
        later = [n for n in ns if n > peak]
        pvals = {n: (self.acc(curve, peak) - self.acc(curve, n)).p_greater_than_zero()
                 for n in later}
        rejected = holm(pvals, self.cfg.eval.alpha)
        onset = min(rejected) if rejected else None
        return onset, {"peak": peak, "p": pvals}


# --------------------------------------------------------------------------- hypotheses


def _fmt(s: Stat, pct: bool = True, level: float = 0.95) -> str:
    k = 100 if pct else 1
    lo, hi = s.ci(level)
    return f"{s.point * k:+.1f} [{lo * k:+.1f}, {hi * k:+.1f}]" if pct else \
        f"{s.point:.3f} [{lo:.3f}, {hi:.3f}]"


def _has(ma: ModelAnalysis, *keys) -> bool:
    return all(k in ma.stats for k in keys)


def test_hypotheses(ma: ModelAnalysis) -> list[dict]:
    a = ma.cfg.eval.alpha
    out = []

    def add(hid, verdict, evidence):
        out.append({"id": hid, "model": ma.model, "verdict": verdict, "evidence": evidence})

    # H1
    if _has(ma, ("acc", "noise", 1), ("acc", "noise", 5), ("acc", "dilution", 100)):
        decline_1_5 = ma.acc("noise", 1) - ma.acc("noise", 5)
        plateau = decline_1_5.p_greater_than_zero() >= a
        onset, info = ma.degradation_onset("noise")
        ns = ma.curve_ns("noise")
        drop = {c: ma.acc(c, ma.degradation_onset(c)[1]["peak"]) - ma.acc(c, ns[-1])
                for c in ("noise", "dilution")}
        steeper = drop["noise"] - drop["dilution"]
        larger = steeper.ci()[0] > 0
        verdict = ("supported" if plateau and onset and larger
                   else "partly supported" if onset else "not supported")
        add("H1", verdict,
            f"hard-negative N†={onset or '>100'}; drop N=1→5 {_fmt(decline_1_5)} pts "
            f"(plateau {'yes' if plateau else 'no'}); peak→N={ns[-1]} drop "
            f"{_fmt(drop['noise'])} vs dilution {_fmt(drop['dilution'])} pts")
    # H2
    diffs = {n: ma.acc("dilution", n) - ma.acc("noise", n) for n in (5, 20, 100)
             if _has(ma, ("acc", "dilution", n), ("acc", "noise", n))}
    if diffs:
        rej = holm({n: d.p_greater_than_zero() for n, d in diffs.items()}, a)
        ok = 100 in rej and all(diffs[n].point > 0 for n in (20, 100) if n in diffs)
        add("H2", "supported" if ok else "not supported",
            "; ".join(f"N={n}: random − hard {_fmt(d)} pts"
                      f"{' *' if n in rej else ''}" for n, d in diffs.items()))
    # H3
    gaps = {}
    for n in (5, 20, 100):
        if _has(ma, ("acc", "position_d0", n), ("acc", "position_d1", n), ("acc", "position", n)):
            gaps[n] = (ma.acc("position_d0", n) + ma.acc("position_d1", n)).scale(0.5) - \
                ma.acc("position", n)
    if 100 in gaps and 20 in gaps:
        first = gaps[100].p_greater_than_zero() < a
        widen = (gaps[100] - gaps[20]).ci()[0] > 0
        add("H3", "supported" if first and widen else "partly supported" if first
            else "not supported",
            "; ".join(f"N={n}: edges − middle {_fmt(g)} pts" for n, g in gaps.items())
            + f"; widening N=20→100 {_fmt(gaps[100] - gaps[20])} pts")
    # H4
    if _has(ma, ("acc", "redundancy", 100), ("acc", "dilution", 100)):
        d100 = ma.acc("redundancy", 100) - ma.acc("dilution", 100)
        ev = f"N=100: k=5 − k=1 {_fmt(d100)} pts"
        if _has(ma, ("acc", "redundancy", 20)):
            ev += f"; N=20: {_fmt(ma.acc('redundancy', 20) - ma.acc('dilution', 20))} pts"
        add("H4", "supported" if d100.p_greater_than_zero() < a else "not supported", ev)
    # H5
    key = ("later_share", "contradiction", "all")
    if key in ma.stats and not math.isnan(ma.stats[key].point):
        s = ma.stats[key]
        lo, hi = s.ci()
        verdict = ("supported" if lo > 0.5 else
                   "not supported (favours the earlier claim)" if hi < 0.5 else "not supported")
        add("H5", verdict, f"share choosing the later claim {_fmt(s, pct=False)}")
    # H6
    parts, oks = [], []
    for c in ("dilution", "noise"):
        if _has(ma, ("overconfidence", c, 1), ("overconfidence", c, 100)):
            d = ma.stats[("overconfidence", c, 100)] - ma.stats[("overconfidence", c, 1)]
            de = ma.stats[("ece", c, 100)] - ma.stats[("ece", c, 1)]
            oks.append(d.ci()[0] > 0)
            parts.append(f"{c}: overconfidence change N=1→100 {_fmt(d)} pts, "
                         f"ECE change {_fmt(de)} pts")
    if parts:
        add("H6", "supported" if all(oks) and len(oks) == 2 else "not supported",
            "; ".join(parts))
    return out


def test_h7(analyses: list[ModelAnalysis]) -> dict | None:
    if len(analyses) < 2:
        return None
    small, large = analyses[0], analyses[-1]

    def rank(x):
        return math.inf if x is None else x

    later = earlier = 0
    parts = []
    for c in H7_CURVES:
        a, b = small.degradation_onset(c)[0], large.degradation_onset(c)[0]
        later += rank(b) > rank(a)
        earlier += rank(b) < rank(a)
        parts.append(f"{c}: {a or '>100'} vs {b or '>100'}")
    ok = later > len(H7_CURVES) / 2 and earlier == 0
    return {"id": "H7", "model": f"{small.model} vs {large.model}",
            "verdict": "supported" if ok else "not supported",
            "evidence": f"N† small vs large — {'; '.join(parts)}"}


# --------------------------------------------------------------------------- tables


def metrics_table(analyses: list[ModelAnalysis], level: float) -> pd.DataFrame:
    rows = []
    for ma in analyses:
        for (metric, curve, n), s in ma.stats.items():
            lo, hi = s.ci(level)
            rows.append({"model": ma.model, "model_id": ma.model_id, "metric": metric,
                         "curve": curve, "N": str(n), "value": s.point, "ci_low": lo,
                         "ci_high": hi})
    return pd.DataFrame(rows)


def leaderboard(analyses: list[ModelAnalysis], sensitivity: list[ModelAnalysis],
                level: float) -> pd.DataFrame:
    rows = []
    for ma, sens in zip(analyses, sensitivity, strict=True):
        for c in MAIN_CURVES:
            s = ma.sss(c)
            if s is None:
                continue
            lo, hi = s.ci(level)
            onset, _ = ma.degradation_onset(c)
            s_all = sens.sss(c)
            rows.append({"model": ma.model, "axis": c, "SSS": s.point, "SSS_low": lo,
                         "SSS_high": hi, "N*": ma.saturation_point(c) or "not reached",
                         "N†": onset or ">100",
                         "SSS_incl_closed_book": s_all.point if s_all else math.nan})
    return pd.DataFrame(rows)


def analyse(cfg: Config, df: pd.DataFrame) -> dict:
    # config order (small model first); a fallback model takes its primary's place
    from .run import fallback_config

    wanted = []
    for m in cfg.models:
        wanted.append(m.name)
        if m.fallback_model_id:
            wanted.append(fallback_config(m).name)
    present = list(dict.fromkeys(df.model))
    models = [m for m in wanted if m in present] + [m for m in present if m not in wanted]
    analyses = [ModelAnalysis(df[df.model == m], cfg) for m in models]
    sensitivity = [ModelAnalysis(df[df.model == m], cfg, exclude_closed_book=False)
                   for m in models]
    hyps = [h for ma in analyses for h in test_hypotheses(ma)]
    h7 = test_h7(analyses)
    if h7:
        hyps.append(h7)
    return {"analyses": analyses, "sensitivity": sensitivity, "hypotheses": hyps,
            "metrics": metrics_table(analyses, cfg.eval.ci),
            "leaderboard": leaderboard(analyses, sensitivity, cfg.eval.ci)}
