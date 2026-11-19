"""`ssb eval` and `ssb report`: score, analyse, plot, and write the results section."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

from .analysis import CURVES, MAIN_CURVES, analyse, score_run
from .config import Config

HYPOTHESES = {
    "H1": "Accuracy plateaus, then declines as N grows, most with hard-negative noise",
    "H2": "Hard negatives hurt more than random noise at equal N",
    "H3": "Gold at depth 0.5 underperforms depths 0 and 1.0, and the gap widens with N",
    "H4": "Redundancy (k=5) protects against dilution at large N",
    "H5": "Under contradiction, models favour the claim that appears last",
    "H6": "Calibration degrades faster than accuracy",
    "H7": "The larger model has a later degradation onset N†",
}
START, END = "<!-- results:start -->", "<!-- results:end -->"


def evaluate(cfg: Config) -> dict:
    out = Path(cfg.run.output_dir)
    df = score_run(cfg)
    df.to_parquet(out / "scores.parquet", index=False)
    res = analyse(cfg, df)
    res["metrics"].to_parquet(out / "metrics.parquet", index=False)
    res["leaderboard"].to_csv(out / "leaderboard.csv", index=False)
    (out / "hypotheses.json").write_text(json.dumps(res["hypotheses"], indent=2) + "\n",
                                         encoding="utf-8", newline="\n")
    res["scores"] = df
    return res


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _one_line_answer(res: dict) -> str:
    """Summary sentence whose wording follows the measured effects and their significance."""
    ma = res["analyses"][-1]  # the largest model
    s = ma.stats
    alpha = ma.cfg.eval.alpha
    model = ma.model_id.split("/")[-1]

    def change(a, b):  # (point change b - a, significant decline?, significant gain?)
        d = s[b] - s[a]
        return d.point, (s[a] - s[b]).p_greater_than_zero() < alpha, d.p_greater_than_zero() < alpha

    need = [("acc", c, n) for c in ("dilution", "noise") for n in (1, 100)]
    if not all(k in s for k in need):
        return "Not enough conditions in this run for a summary."
    d_rand, hurt_rand, _ = change(("acc", "dilution", 1), ("acc", "dilution", 100))
    d_hard, hurt_hard, _ = change(("acc", "noise", 1), ("acc", "noise", 100))
    if hurt_rand or hurt_hard:
        head = "Yes, past a point more context makes the model less reliable"
    else:
        head = "No significant degradation was measured up to N=100"
    text = (f"{head}: for {model}, going from 1 to 100 paragraphs changed accuracy by "
            f"{d_rand * 100:+.1f} points with random filler and {d_hard * 100:+.1f} with hard "
            "negatives")
    if ("acc", "redundancy", 100) in s:
        d_red, _, helped = change(("acc", "dilution", 100), ("acc", "redundancy", 100))
        text += (f"; repeating the evidence 5 times {'significantly ' if helped else ''}"
                 f"changed accuracy at N=100 by {d_red * 100:+.1f} points")
    return text + "."


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _curve_table(res: dict) -> str:
    rows = []
    for ma in res["analyses"]:
        for c in MAIN_CURVES:
            for n in ma.curve_ns(c):
                st = ma.stats[("acc", c, n)]
                lo, hi = st.ci()
                ece = ma.stats.get(("ece", c, n))
                len_ = ma.stats.get(("lenient", c, n))
                rows.append({"model": ma.model, "axis": CURVES[c][0], "N": n,
                             "accuracy [95% CI]": f"{_pct(st.point)} [{_pct(lo)}, {_pct(hi)}]",
                             "lenient (answer mentioned)": _pct(len_.point) if len_ else "n/a",
                             "ECE": f"{ece.point:.3f}" if ece else "n/a"})
    return _table(pd.DataFrame(rows))


def results_markdown(cfg: Config, res: dict, fig_rel: str) -> str:
    out = Path(cfg.run.output_dir)
    manifest = json.loads((out / "run_manifest.json").read_text()) \
        if (out / "run_manifest.json").exists() else {}
    env = manifest.get("environment", {})
    board = res["leaderboard"].copy()
    board["SSS [95% CI]"] = board.apply(
        lambda r: f"{r.SSS:.3f} [{r.SSS_low:.3f}, {r.SSS_high:.3f}]", axis=1)
    board["axis"] = board["axis"].map(lambda c: CURVES[c][0])
    board = board[["model", "axis", "SSS [95% CI]", "N*", "N†"]]

    hyp_rows = []
    for h in res["hypotheses"]:
        hyp_rows.append({"": h["id"], "hypothesis": HYPOTHESES[h["id"]], "model": h["model"],
                         "verdict": f"**{h['verdict']}**", "evidence": h["evidence"]})
    excluded = {ma.model: len(ma.closed_book_correct) for ma in res["analyses"]}
    hall = []
    for ma in res["analyses"]:
        for axis in ("closed_book", "dilution", "noise"):
            for (m, a, n), st in ma.stats.items():
                if m == "hallucination" and a == axis:
                    lo, hi = st.ci()
                    hall.append({"model": ma.model, "condition": axis, "N": n,
                                 "hallucination rate": f"{_pct(st.point)} "
                                                       f"[{_pct(lo)}, {_pct(hi)}]"})
    runtime = ", ".join(f"{k}: {v.get('gpu_hours', 'n/a')} GPU-h ({v.get('backend')}, "
                        f"{v.get('model_id')})" for k, v in manifest.get("models", {}).items())
    md = [
        f"## Results ({cfg.name} run)",
        "",
        f"**Answer:** {_one_line_answer(res)}",
        "",
        f"![Accuracy vs N]({fig_rel}/fig1_main.png)",
        "",
        "### Leaderboard",
        "",
        "SSS is the normalised area under accuracy against log N (1 = perfect at every N). "
        "N* is where the gain from the previous N drops below 1 point. N† is the first N "
        "significantly below the curve's peak (bootstrap, Holm-corrected, p < 0.05).",
        "",
        _table(board),
        "",
        "### Pre-registered hypotheses",
        "",
        "Verdicts follow the tests fixed in [HYPOTHESES.md](HYPOTHESES.md). Differences are in "
        "accuracy points, with 95% bootstrap CIs over questions; * marks a Holm-significant "
        "difference.",
        "",
        _table(pd.DataFrame(hyp_rows)),
        "",
        "### Abstention",
        "",
        _table(pd.DataFrame(hall)) if hall else "n/a",
        "",
        "### Figures",
        "",
        f"- [Position heatmap]({fig_rel}/fig2_position_heatmap.png)",
        f"- [Noise types]({fig_rel}/fig3_noise_types.png)",
        f"- [Contradiction outcomes]({fig_rel}/fig4_contradiction.png)",
        f"- [Accuracy vs ECE]({fig_rel}/fig5_accuracy_vs_ece.png)",
        f"- [Leaderboard]({fig_rel}/fig6_leaderboard.png)",
        "",
        "<details><summary>All accuracy points</summary>",
        "",
        _curve_table(res),
        "",
        "</details>",
        "",
        "### Run details",
        "",
        f"- Questions: {manifest.get('questions', 'n/a')}; samples per model: "
        f"{manifest.get('samples', 'n/a')}.",
        f"- Questions answered correctly closed-book (dropped from the primary analysis): "
        f"{excluded}. The leaderboard CSV also lists SSS with them included.",
        f"- Hardware: {env.get('gpu') or 'n/a'}; torch {env.get('torch')}, transformers "
        f"{env.get('transformers')}, vLLM {env.get('vllm')}.",
        f"- Runtime: {runtime or 'n/a'}.",
        f"- Raw generations: `{out.as_posix()}/generations/`; per-answer scores: "
        f"`{out.as_posix()}/scores.parquet`; all metrics with CIs: "
        f"`{out.as_posix()}/metrics.parquet`.",
    ]
    return "\n".join(md)


def write_report(cfg: Config, update_readme: bool) -> dict:
    from .plots import make_all

    res = evaluate(cfg)
    fig_dir = Path(cfg.figures_dir)
    make_all(res["analyses"], res["leaderboard"], fig_dir)
    out = Path(cfg.run.output_dir)
    rel = os.path.relpath(fig_dir, out).replace("\\", "/")
    (out / "REPORT.md").write_text(results_markdown(cfg, res, rel) + "\n", encoding="utf-8",
                                   newline="\n")
    if update_readme:
        readme = Path("README.md")
        text = readme.read_text(encoding="utf-8")
        body = results_markdown(cfg, res, fig_dir.as_posix())
        new = re.sub(rf"{re.escape(START)}.*?{re.escape(END)}",
                     lambda _: f"{START}\n{body}\n{END}", text, flags=re.S)
        readme.write_text(new, encoding="utf-8", newline="\n")
    return res
