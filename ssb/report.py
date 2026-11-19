"""`ssb report`: score, analyse, plot, and write REPORT.md plus the README results."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

from .analysis import CURVES, SCORE_NAME, analyse, score_run
from .config import Config

HYPOTHESES = {
    "H1": "All models degrade as N grows; hard negatives hurt more than random noise",
    "H2": "Bag-of-words models show no position bias; truncating BERT collapses past 512 tokens",
    "H3": "Chunk-mean fixes position bias but still degrades from dilution",
    "H4": "BM25/TF-IDF beat the dense encoder on hard negatives; dense wins on random noise",
    "H5": "Redundancy slows degradation; copies help TF-IDF, paraphrases help transformers",
    "H6": "QA models pick the conflicting answer about half the time; position matters",
    "H7": "Confidence stays high while accuracy drops",
    "H8": "Unanswerable questions get more false answers as N grows",
}
START, END = "<!-- results:start -->", "<!-- results:end -->"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for _, r in df.iterrows()]
    return "\n".join(lines)


def _answer(res: dict, cfg: Config) -> str:
    """One-line answer from the measured dilution and redundancy effects."""
    big = max(cfg.conditions.sizes)
    drops, helps = [], []
    for name, ma in res["analyses"].items():
        if ("score", "dilution", 1) in ma.stats and ("score", "dilution", big) in ma.stats:
            d = ma.score("dilution", 1) - ma.score("dilution", big)
            if d.p_greater_than_zero() < cfg.eval.alpha:
                drops.append((name, d.point))
        for mode in ("copy", "paraphrase"):
            key = ("score", f"redundancy_{mode}", big)
            if key in ma.stats and ("score", "dilution", big) in ma.stats:
                g = ma.stats[key] - ma.score("dilution", big)
                if g.p_greater_than_zero() < cfg.eval.alpha:
                    helps.append(name)
    n = len(res["analyses"])
    worst = max(drops, key=lambda x: x[1]) if drops else None
    head = (f"More context did not help: {len(drops)} of {n} models got significantly worse "
            f"from N=1 to N={big}" if drops else
            f"No model got significantly worse from N=1 to N={big}")
    if worst:
        head += f" (largest drop: {worst[0]}, {worst[1] * 100:.1f} points)"
    tail = (f". Repeating the signal 5 times significantly softened the drop for "
            f"{len(set(helps))} of {n} models." if helps else
            ". Repeating the signal did not significantly soften the drop.")
    return head + tail


def results_markdown(cfg: Config, res: dict, fig: str) -> str:
    out = Path(cfg.run.output_dir)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8")) \
        if (out / "run_manifest.json").exists() else {}
    board = res["leaderboard"].copy()
    board["SSS [95% CI]"] = board.apply(lambda r: f"{r.SSS:.3f} [{r.SSS_low:.3f}, "
                                        f"{r.SSS_high:.3f}]", axis=1)
    board["axis"] = board["axis"].map(lambda c: CURVES[c][0])
    board = board[["task", "model", "axis", "SSS [95% CI]", "N*", "N†"]]
    hyp = pd.DataFrame([{"": h["id"], "hypothesis": HYPOTHESES[h["id"]],
                         "verdict": f"**{h['verdict']}**", "evidence": h["evidence"]}
                        for h in res["hypotheses"]])
    pts = []
    for name, ma in res["analyses"].items():
        for c in ("dilution", "noise"):
            for n in ma.ns(c):
                s = ma.score(c, n)
                lo, hi = s.ci()
                e = ma.stats.get(("ece", c, n))
                pts.append({"model": name, "metric": SCORE_NAME[ma.task], "filler":
                            "random" if c == "dilution" else "hard", "N": n,
                            "score [95% CI]": f"{_pct(s.point)} [{_pct(lo)}, {_pct(hi)}]",
                            "ECE": f"{e.point:.3f}" if e else "n/a"})
    secs = {k: v.get("seconds") for k, v in manifest.get("models", {}).items()}
    env = manifest.get("environment", {})
    total_h = sum(v or 0 for v in secs.values()) / 3600
    lines = [
        f"## Results ({cfg.name} run)", "",
        f"**Answer:** {_answer(res, cfg)}", "",
        f"![Score vs N]({fig}/fig1_main.png)", "",
        "### Leaderboard", "",
        "SSS is the normalised area under score vs log N (1 = perfect at every N); N* is "
        "where the gain from the previous N drops below 1 point; N† is the first N "
        "significantly below the curve's peak (bootstrap, Holm, p < 0.05).", "",
        _table(board), "",
        "### Pre-registered hypotheses", "",
        "Verdicts follow the tests fixed in [HYPOTHESES.md](HYPOTHESES.md). Differences in "
        "points with 95% bootstrap CIs over base examples; * = Holm-significant.", "",
        _table(hyp) if len(hyp) else "n/a", "",
        "### Figures", "",
        *[f"- [{t}]({fig}/{f}.png)" for t, f in [
            ("Noise types", "fig2_noise_types"), ("Position heatmap", "fig3_position_heatmap"),
            ("Redundancy", "fig4_redundancy"), ("Contradiction outcomes", "fig5_contradiction"),
            ("Calibration", "fig6_calibration"), ("Leaderboard", "fig7_leaderboard")]], "",
        "<details><summary>All dilution and noise points</summary>", "",
        _table(pd.DataFrame(pts)) if pts else "n/a", "", "</details>", "",
        "### Run details", "",
        f"- Device: {env.get('gpu') or manifest.get('device', 'n/a')}; torch "
        f"{env.get('torch')}, transformers {env.get('transformers')}.",
        f"- Runtime: {total_h:.2f} h in total; per model (s): {secs}.",
        f"- Raw predictions: `{out.as_posix()}/predictions/`; per-answer scores: "
        f"`{out.as_posix()}/scores.parquet`; every metric with CIs: "
        f"`{out.as_posix()}/metrics.parquet`.",
    ]
    return "\n".join(lines)


def write_report(cfg: Config, update_readme: bool) -> dict:
    from .plots import make_all

    out = Path(cfg.run.output_dir)
    df = score_run(cfg)
    df.drop(columns=["probs"], errors="ignore").to_parquet(out / "scores.parquet", index=False)
    res = analyse(cfg, df)
    res["metrics"].to_parquet(out / "metrics.parquet", index=False)
    res["leaderboard"].to_csv(out / "leaderboard.csv", index=False)
    (out / "hypotheses.json").write_text(json.dumps(res["hypotheses"], indent=2) + "\n",
                                         encoding="utf-8", newline="\n")
    fig_dir = Path(cfg.figures_dir)
    make_all(res["analyses"], res["leaderboard"], [m.name for m in cfg.models], fig_dir)
    rel = os.path.relpath(fig_dir, out).replace("\\", "/")
    (out / "REPORT.md").write_text(results_markdown(cfg, res, rel) + "\n", encoding="utf-8",
                                   newline="\n")
    if update_readme:
        readme = Path("README.md")
        body = results_markdown(cfg, res, fig_dir.as_posix())
        text = re.sub(rf"{re.escape(START)}.*?{re.escape(END)}",
                      lambda _: f"{START}\n{body}\n{END}",
                      readme.read_text(encoding="utf-8"), flags=re.S)
        readme.write_text(text, encoding="utf-8", newline="\n")
    return res
