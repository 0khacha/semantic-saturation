"""The six report figures, exported as PNG and PDF.

Colours are a validated colourblind-safe categorical order, fixed per axis (never cycled).
Every series also has its own marker shape, so identity never rests on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import CURVES, MAIN_CURVES, ModelAnalysis  # noqa: E402

SURFACE = "#fcfcfb"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"dilution": "#2a78d6", "noise": "#eb6834", "redundancy": "#1baf7a",
          "contradiction": "#eda100", "position": "#e87ba4"}
MARKERS = {"dilution": "o", "noise": "s", "redundancy": "^", "contradiction": "D",
           "position": "v"}
OUTCOME = {"gold": "#2a78d6", "conflict": "#eb6834", "other": "#b5b3ad"}
LABEL = {c: CURVES[c][0] for c in CURVES}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
        "axes.titleweight": "bold", "font.size": 9.5, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.markersize": 5.5, "font.family": "DejaVu Sans",
    })


def _save(fig, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _log_x(ax, ns) -> None:
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("Context size N (paragraphs, log scale)")


def _pct_y(ax, label="Accuracy (exact match)") -> None:
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel(label)


def _line(ax, ma: ModelAnalysis, curve: str, key: str, color: str, marker: str, label: str):
    ns = ma.curve_ns(curve)
    pts = [ma.stats[("acc", curve, n)] for n in ns]
    y = np.array([p.point for p in pts])
    lo = np.array([p.ci()[0] for p in pts])
    hi = np.array([p.ci()[1] for p in pts])
    ax.fill_between(ns, lo, hi, color=color, alpha=0.14, linewidth=0)
    ax.plot(ns, y, color=color, marker=marker, label=label, markeredgecolor=SURFACE,
            markeredgewidth=1.2)
    return ns


def _panels(n: int, width: float = 4.6, height: float = 3.6):
    fig, axes = plt.subplots(1, n, figsize=(width * n, height), squeeze=False, sharey=True)
    return fig, axes[0]


def fig_main(analyses, out):
    fig, axes = _panels(len(analyses))
    for ax, ma in zip(axes, analyses, strict=True):
        ns_all = set()
        for c in MAIN_CURVES:
            if ma.curve_ns(c):
                ns_all |= set(_line(ax, ma, c, "acc", SERIES[c], MARKERS[c], LABEL[c]))
        _log_x(ax, sorted(ns_all))
        _pct_y(ax)
        ax.set_title(ma.model_id.split("/")[-1])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Accuracy vs context size, one line per axis (95% bootstrap CI)", x=0.01,
                 ha="left", fontsize=12, fontweight="bold")
    return _save(fig, out, "fig1_main")


def fig_position(analyses, out):
    fig, axes = plt.subplots(1, len(analyses), figsize=(4.4 * len(analyses) + 1, 3.0),
                             squeeze=False, layout="constrained")
    rows = [("position_d0", "0 (start)"), ("position", "0.5 (middle)"), ("position_d1", "1 (end)")]
    for k, (ax, ma) in enumerate(zip(axes[0], analyses, strict=True)):
        ns = ma.curve_ns("position")
        grid = np.array([[ma.stats[("acc", c, n)].point if ("acc", c, n) in ma.stats else np.nan
                          for n in ns] for c, _ in rows])
        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.0%}", ha="center", va="center", fontsize=9,
                            color="white" if grid[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(ns)), [str(n) for n in ns])
        ax.set_yticks(range(len(rows)), [r[1] for r in rows] if k == 0 else [])
        ax.set_xlabel("Context size N")
        if k == 0:
            ax.set_ylabel("Gold depth d")
        ax.grid(False)
        ax.set_title(ma.model_id.split("/")[-1])
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label="Accuracy")
    fig.suptitle("Accuracy by gold depth and context size", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    return _save(fig, out, "fig2_position_heatmap")


def fig_noise(analyses, out):
    fig, axes = _panels(len(analyses))
    for ax, ma in zip(axes, analyses, strict=True):
        ns = _line(ax, ma, "dilution", "acc", SERIES["dilution"], MARKERS["dilution"],
                   "Random off-topic filler")
        _line(ax, ma, "noise", "acc", SERIES["noise"], MARKERS["noise"], "Hard negatives")
        for c, lab in (("dilution", "random"), ("noise", "hard")):
            last = ma.curve_ns(c)[-1]
            ax.annotate(lab, (last, ma.stats[("acc", c, last)].point), xytext=(6, 0),
                        textcoords="offset points", va="center", color=INK2, fontsize=9)
        _log_x(ax, ns)
        _pct_y(ax)
        ax.set_title(ma.model_id.split("/")[-1])
    axes[0].legend(loc="lower left")
    fig.suptitle("Noise type at equal N", x=0.01, ha="left", fontsize=12, fontweight="bold")
    return _save(fig, out, "fig3_noise_types")


def fig_contradiction(analyses, out):
    fig, axes = _panels(len(analyses), width=4.8)
    for ax, ma in zip(axes, analyses, strict=True):
        ns = ma.curve_ns("contradiction")
        labels, x = [], []
        bottoms = {}
        for i, n in enumerate(ns):
            for j, pos in enumerate(("before", "after")):
                labels.append(f"{pos}\nN={n}")
                x.append(i * 2.6 + j)
        x = np.array(x, float)
        bottom = np.zeros(len(x))
        for oc in ("gold", "conflict", "other"):
            vals = np.array([ma.stats[(f"outcome_{oc}", f"contradiction_{pos}", n)].point
                             for n in ns for pos in ("before", "after")])
            ax.bar(x, vals, bottom=bottom, width=0.85, color=OUTCOME[oc], label=oc.capitalize()
                   if oc != "other" else "Other / abstain", edgecolor=SURFACE, linewidth=1.5)
            bottom += vals
            bottoms[oc] = vals
        ax.set_xticks(x, labels, fontsize=8)
        _pct_y(ax, "Share of answers")
        ax.grid(axis="x", visible=False)
        ax.set_title(ma.model_id.split("/")[-1])
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Which claim wins: conflict placed before vs after the gold paragraph",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    return _save(fig, out, "fig4_contradiction")


def fig_calibration(analyses, out):
    fig, axes = _panels(len(analyses))
    for ax, ma in zip(axes, analyses, strict=True):
        for c in ("dilution", "noise"):
            ns = [n for n in ma.curve_ns(c) if ("ece", c, n) in ma.stats]
            if not ns:
                continue
            xs = [ma.stats[("ece", c, n)].point for n in ns]
            ys = [ma.stats[("acc", c, n)].point for n in ns]
            ax.plot(xs, ys, color=SERIES[c], marker=MARKERS[c], label=LABEL[c],
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            for xv, yv, n in zip(xs, ys, ns, strict=True):
                ax.annotate(f"N={n}", (xv, yv), xytext=(5, 4), textcoords="offset points",
                            fontsize=8, color=INK2)
        ax.set_xlabel("Expected calibration error (10 bins)")
        _pct_y(ax)
        ax.set_xlim(left=0)
        ax.set_title(ma.model_id.split("/")[-1])
    axes[0].legend(loc="lower left")
    fig.suptitle("Accuracy vs calibration error as N grows", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    return _save(fig, out, "fig5_accuracy_vs_ece")


def fig_leaderboard(board, out):
    cols = ["model", "axis", "SSS", "N*", "N†"]
    cells = [[r["model"], LABEL[r["axis"]],
              f"{r['SSS']:.3f} [{r['SSS_low']:.3f}, {r['SSS_high']:.3f}]", str(r["N*"]),
              str(r["N†"])] for _, r in board.iterrows()]
    fig, ax = plt.subplots(figsize=(8.6, 0.32 * (len(cells) + 2)))
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="left", colLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.25)
    for (r, _), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK)
    ax.set_title("Leaderboard: SSS (normalised area under accuracy vs log N), N*, N†",
                 loc="left", fontsize=11, fontweight="bold")
    return _save(fig, out, "fig6_leaderboard")


def make_all(analyses, board, out: Path) -> list[Path]:
    _style()
    paths = []
    for f in (fig_main, fig_position, fig_noise, fig_contradiction, fig_calibration):
        paths += f(analyses, out)
    paths += fig_leaderboard(board, out)
    return paths
