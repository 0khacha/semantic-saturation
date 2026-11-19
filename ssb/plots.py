"""The seven report figures, as PNG and PDF.

Each model has a fixed colour from a validated colour-blind-safe categorical order
(assigned by model, never cycled) and its own marker, so identity never rests on colour
alone. Sequential data (the position heatmap) uses a single-hue ramp.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .analysis import CURVES, SCORE_NAME  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
           "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
OUTCOME = {"true": "#2a78d6", "conflicting": "#eb6834", "tie": "#eda100", "other": "#b5b3ad"}
TASK_TITLE = {"classification": "A. Topic classification", "retrieval": "B. Passage retrieval",
              "qa": "C. Extractive QA"}

# Every visible string, by language. English is the default; `make_all(..., lang="fr")`
# draws the same figures with French labels.
TEXT = {
    "en": {},
    "fr": {
        "A. Topic classification": "A. Classification thématique",
        "B. Passage retrieval": "B. Recherche de passage",
        "C. Extractive QA": "C. QA extractive",
        "accuracy": "précision",
        "Context size N (paragraphs, log scale)":
            "Taille du contexte N (paragraphes, échelle log)",
        "Score vs context size (random filler), 95% bootstrap CI":
            "Score selon la taille du contexte (remplissage aléatoire), IC bootstrap 95 %",
        "random": "aléatoire", "hard": "difficile",
        "Noise type: random (solid) vs hard negatives (dashed)":
            "Type de bruit : aléatoire (trait plein) vs négatifs difficiles (tirets)",
        "Signal depth": "Profondeur du signal",
        "Position: score by signal depth and N":
            "Position : score selon la profondeur du signal et N",
        "Redundancy: does repeating the signal slow the decline?":
            "Redondance : répéter le signal ralentit-il la baisse ?",
        "true": "vraie réponse", "conflicting": "réponse contradictoire", "tie": "égalité",
        "other": "autre", "Share": "Part", "B": "Av", "A": "Ap",
        "Contradiction: conflict placed Before or After the signal, by N":
            "Contradiction : conflit placé avant (Av) ou après (Ap) le signal, selon N",
        "perfect calibration": "calibration parfaite", "Mean confidence": "Confiance moyenne",
        "Calibration as N grows (random filler)":
            "Calibration quand N augmente (remplissage aléatoire)",
        "task": "tâche", "model": "modèle", "axis": "axe", "SSS [95% CI]": "SSS [IC 95 %]",
        "Leaderboard: SSS, saturation point N*, degradation onset N†":
            "Classement : SSS, point de saturation N*, début de dégradation N†",
        "Dilution (random filler)": "Dilution (remplissage aléatoire)",
        "Hard-negative noise": "Bruit (négatifs difficiles)",
        "Redundancy k=5, copies": "Redondance k=5, copies",
        "Redundancy k=5, paraphrases": "Redondance k=5, paraphrases",
        "Position d=0.5": "Position d=0,5",
        "retrieval": "recherche", "qa": "QA",
    },
}
LANG = "en"


def _t(s: str) -> str:
    return TEXT[LANG].get(s, s)


def _pct_label(v: float) -> str:
    return f"{v:.0%}" if LANG == "en" else f"{v * 100:.0f} %"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "font.size": 9, "legend.frameon": False, "lines.linewidth": 1.6,
        "lines.markersize": 5.5, "font.family": "DejaVu Sans"})


def _save(fig, out: Path, name: str) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.png", out / f"{name}.pdf"]
    fig.savefig(paths[0], dpi=200, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt.close(fig)
    return paths


def _title(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="bold")


def _logx(ax, ns):
    ax.set_xscale("log")
    ax.set_xticks(ns, [str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel(_t("Context size N (paragraphs, log scale)"))


def _pct(ax, label):
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], [_pct_label(v) for v in (0, 0.25, 0.5, 0.75, 1)])
    ax.set_ylabel(_t(label))


def _line(ax, ma, curve, color, marker, label, ls="-"):
    ns = ma.ns(curve)
    if not ns:
        return []
    pts = [ma.score(curve, n) for n in ns]
    y = np.array([p.point for p in pts])
    lo, hi = np.array([p.ci()[0] for p in pts]), np.array([p.ci()[1] for p in pts])
    ax.fill_between(ns, lo, hi, color=color, alpha=0.13, linewidth=0)
    ax.plot(ns, y, color=color, marker=marker, ls=ls, label=label, markeredgecolor=SURFACE,
            markeredgewidth=1.1)
    return ns


class Figures:
    def __init__(self, analyses: dict, model_order: list[str], out: Path):
        self.an = analyses
        self.order = [m for m in model_order if m in analyses]
        self.color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(model_order)}
        self.marker = {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(model_order)}
        self.out = out
        self.tasks = [t for t in TASK_TITLE if any(a.task == t for a in analyses.values())]

    def _models(self, task):
        return [m for m in self.order if self.an[m].task == task]

    def main(self):
        fig, axes = plt.subplots(1, len(self.tasks), figsize=(4.4 * len(self.tasks), 3.6),
                                 squeeze=False)
        for ax, task in zip(axes[0], self.tasks, strict=True):
            ns = []
            for m in self._models(task):
                ns = _line(ax, self.an[m], "dilution", self.color[m], self.marker[m], m) or ns
            _logx(ax, ns)
            _pct(ax, SCORE_NAME[task])
            ax.set_title(_t(TASK_TITLE[task]))
            ax.legend(loc="lower left", fontsize=8)
        _title(fig, _t("Score vs context size (random filler), 95% bootstrap CI"))
        return _save(fig, self.out, "fig1_main")

    def noise(self):
        fig, axes = plt.subplots(1, len(self.tasks), figsize=(4.4 * len(self.tasks), 3.6),
                                 squeeze=False)
        for ax, task in zip(axes[0], self.tasks, strict=True):
            ns = []
            for m in self._models(task):
                ns = _line(ax, self.an[m], "dilution", self.color[m], self.marker[m],
                           f"{m} {_t('random')}") or ns
                _line(ax, self.an[m], "noise", self.color[m], self.marker[m], f"{m} {_t('hard')}",
                      ls="--")
            _logx(ax, ns)
            _pct(ax, SCORE_NAME[task])
            ax.set_title(_t(TASK_TITLE[task]))
            ax.legend(loc="lower left", fontsize=7)
        _title(fig, _t("Noise type: random (solid) vs hard negatives (dashed)"))
        return _save(fig, self.out, "fig2_noise_types")

    def position(self):
        models = self.order
        cols = min(4, len(models))
        rows = int(np.ceil(len(models) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.6 * rows), squeeze=False,
                                 layout="constrained")
        depth_rows = [("position_d0", "0"), ("position", "0.5"), ("position_d1", "1")]
        im = None
        for k, ax in enumerate(axes.flat):
            if k >= len(models):
                ax.axis("off")
                continue
            ma = self.an[models[k]]
            ns = ma.ns("position")
            grid = np.array([[ma.stats[("score", c, n)].point if ("score", c, n) in ma.stats
                              else np.nan for n in ns] for c, _ in depth_rows])
            im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if not np.isnan(grid[i, j]):
                        ax.text(j, i, _pct_label(grid[i, j]), ha="center", va="center", fontsize=8,
                                color="white" if grid[i, j] > 0.6 else INK)
            ax.set_xticks(range(len(ns)), [str(n) for n in ns])
            ax.set_yticks(range(3), [r[1] for r in depth_rows] if k % cols == 0 else [])
            ax.grid(False)
            ax.set_title(models[k], fontsize=9.5)
            if k % cols == 0:
                ax.set_ylabel(_t("Signal depth"))
            if k // cols == rows - 1:
                ax.set_xlabel("N")
        if im is not None:
            cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label=_t("Score"))
            cb.set_ticks([0, 0.25, 0.5, 0.75, 1],
                         labels=[_pct_label(v) for v in (0, 0.25, 0.5, 0.75, 1)])
        _title(fig, _t("Position: score by signal depth and N"))
        return _save(fig, self.out, "fig3_position_heatmap")

    def redundancy(self):
        models = self.order
        cols = min(4, len(models))
        rows = int(np.ceil(len(models) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.7 * rows), squeeze=False,
                                 sharey=True)
        series = [("dilution", "k=1", PALETTE[0], "o"),
                  ("redundancy_copy", "k=5 copies", PALETTE[1], "s"),
                  ("redundancy_paraphrase", "k=5 paraphrases", PALETTE[2], "^")]
        for k, ax in enumerate(axes.flat):
            if k >= len(models):
                ax.axis("off")
                continue
            ma = self.an[models[k]]
            ns = []
            for curve, lab, col, mk in series:
                ns = _line(ax, ma, curve, col, mk, _t(lab)) or ns
            _logx(ax, ns)
            ax.set_xlabel("N")
            _pct(ax, SCORE_NAME[ma.task] if k % cols == 0 else "")
            ax.set_title(models[k], fontsize=9.5)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
        _title(fig, _t("Redundancy: does repeating the signal slow the decline?"))
        fig.tight_layout()
        return _save(fig, self.out, "fig4_redundancy")

    def contradiction(self):
        models = [m for m in self.order if any(k[0].startswith("outcome_")
                                               for k in self.an[m].stats)]
        if not models:
            return []
        cols = min(4, len(models))
        rows = int(np.ceil(len(models) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows), squeeze=False,
                                 sharey=True)
        for k, ax in enumerate(axes.flat):
            if k >= len(models):
                ax.axis("off")
                continue
            ma = self.an[models[k]]
            ns = ma.ns("contradiction")
            x, labels = [], []
            for i, n in enumerate(ns):
                for j, pos in enumerate(("before", "after")):
                    x.append(i * 2.6 + j)
                    labels.append(f"{_t(pos[0].upper())}\n{n}")
            x = np.array(x, float)
            bottom = np.zeros(len(x))
            for oc in ("true", "conflicting", "tie", "other"):
                vals = np.array([ma.stats.get((f"outcome_{oc}", pos, n), None).point
                                 if (f"outcome_{oc}", pos, n) in ma.stats else 0.0
                                 for n in ns for pos in ("before", "after")])
                if not vals.any():
                    continue
                ax.bar(x, vals, bottom=bottom, width=0.85, color=OUTCOME[oc], label=_t(oc),
                       edgecolor=SURFACE, linewidth=1.5)
                bottom += vals
            ax.set_xticks(x, labels, fontsize=7)
            _pct(ax, _t("Share") if k % cols == 0 else "")
            ax.grid(axis="x", visible=False)
            ax.set_title(models[k], fontsize=9.5)
        handles, labs = [], []
        for a in axes.flat:
            for h, lab in zip(*a.get_legend_handles_labels(), strict=True):
                if lab not in labs:
                    handles.append(h)
                    labs.append(lab)
        fig.legend(handles, labs, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
        _title(fig, _t("Contradiction: conflict placed Before or After the signal, by N"))
        fig.tight_layout()
        return _save(fig, self.out, "fig5_contradiction")

    def calibration(self):
        models = [m for m in self.order if ("confidence", "dilution", 1) in self.an[m].stats]
        if not models:
            return []
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls=":", label=_t("perfect calibration"))
        ends: list[tuple[float, float]] = []
        for m in models:
            ma = self.an[m]
            ns = [n for n in ma.ns("dilution") if ("confidence", "dilution", n) in ma.stats]
            xs = [ma.stats[("confidence", "dilution", n)].point for n in ns]
            ys = [ma.score("dilution", n).point for n in ns]
            ax.plot(xs, ys, color=self.color[m], marker=self.marker[m], label=m,
                    markeredgecolor=SURFACE)
            # endpoints that land close together get their labels stacked, not overlapped
            near = sum(abs(xs[-1] - x) < 0.06 and abs(ys[-1] - y) < 0.06 for x, y in ends)
            ends.append((xs[-1], ys[-1]))
            ax.annotate(f"N={ns[-1]}", (xs[-1], ys[-1]), xytext=(-6, -10 - 10 * near),
                        textcoords="offset points", fontsize=7, color=self.color[m],
                        ha="right")
        ax.set_xlabel(_t("Mean confidence"))
        ax.set_xlim(0, 1.02)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], [_pct_label(v) for v in (0, 0.25, 0.5, 0.75, 1)])
        _pct(ax, "Score")
        ax.legend(loc="upper left", fontsize=8)
        _title(fig, _t("Calibration as N grows (random filler)"))
        return _save(fig, self.out, "fig6_calibration")

    def leaderboard(self, board):
        cells = [[_t(r["task"]), r["model"], _t(CURVES[r["axis"]][0]),
                  f"{r['SSS']:.3f} [{r['SSS_low']:.3f}, {r['SSS_high']:.3f}]", str(r["N*"]),
                  str(r["N†"])] for _, r in board.iterrows()]
        fig, ax = plt.subplots(figsize=(10, 0.27 * (len(cells) + 2)))
        ax.axis("off")
        t = ax.table(cellText=cells, colLabels=[_t(c) for c in ("task", "model", "axis",
                                                           "SSS [95% CI]", "N*", "N†")], loc="center", cellLoc="left",
                     colLoc="left")
        t.auto_set_font_size(False)
        t.set_fontsize(7.5)
        t.scale(1, 1.2)
        for (r, _), cell in t.get_celld().items():
            cell.set_edgecolor(GRID)
            cell.set_facecolor(SURFACE)
            if r == 0:
                cell.set_text_props(weight="bold")
        ax.set_title(_t("Leaderboard: SSS, saturation point N*, degradation onset N†"),
                 loc="left",
                     fontsize=11, fontweight="bold")
        return _save(fig, self.out, "fig7_leaderboard")


def make_all(analyses, board, model_order, out: Path, lang: str = "en") -> list[Path]:
    global LANG
    LANG = lang
    _style()
    f = Figures(analyses, model_order, out)
    paths = []
    for fn in (f.main, f.noise, f.position, f.redundancy, f.contradiction, f.calibration):
        paths += fn()
    paths += f.leaderboard(board)
    LANG = "en"
    return paths
