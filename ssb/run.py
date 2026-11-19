"""Resumable runner: every model scores every sample of its task.

Predictions are appended to results/<run>/predictions/<model>.jsonl after each batch and
fsynced, so a crash loses at most the batch in flight; a rerun skips finished samples.
Before the bulk of the work the runner times one batch per N for each model and prints
the extrapolated runtime (`--estimate` stops there).
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

from .build import load_dataset, load_paragraphs
from .config import Config, ModelConfig
from .data import load_agnews


def device_for(name: str) -> str:
    if name != "auto":
        return name
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def select_examples(samples: list[dict], n: int | None) -> list[dict]:
    """All samples of a balanced subset of n base examples per task."""
    if n is None:
        return samples
    keep: set[str] = set()
    by_task: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for s in samples:
        stratum = s["label"] if s["task"] == "classification" else s["answerable"]
        if s["base_id"] not in by_task[s["task"]][stratum]:
            by_task[s["task"]][stratum].append(s["base_id"])
    for strata in by_task.values():
        chosen: list[str] = []
        keys = sorted(strata, key=str)
        while len(chosen) < n and any(strata[k] for k in keys):
            for k in keys:
                if strata[k] and len(chosen) < n:
                    chosen.append(strata[k].pop(0))
        keep |= set(chosen)
    return [s for s in samples if s["base_id"] in keep]


def environment() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "sentence_transformers", "sklearn", "rank_bm25",
                "spacy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    try:
        import torch

        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        out["gpu"] = None
    return out


class Runner:
    def __init__(self, cfg: Config, log=print):
        self.cfg, self.log = cfg, log
        self.out = Path(cfg.run.output_dir)
        self.device = device_for(cfg.run.device)
        self.samples = select_examples(load_dataset(cfg.data.path), cfg.run.examples)
        self.manifest_path = self.out / "run_manifest.json"
        self.manifest = (json.loads(self.manifest_path.read_text(encoding="utf-8"))
                         if self.manifest_path.exists() else {"models": {}})

    def _save(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest.update({"config": self.cfg.name, "dataset": self.cfg.data.path,
                              "device": self.device, "environment": environment()})
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2) + "\n",
                                      encoding="utf-8", newline="\n")

    def _load(self, m: ModelConfig):
        from .tasks import classification, qa, retrieval

        if m.kind == "tfidf_lr":
            train, _ = load_agnews(self.cfg.data.raw_dir)
            return classification.TfidfLR(m, self.cfg.data.raw_dir, [x.text for x in train],
                                          [x.label for x in train], self.cfg.seed)
        if m.kind == "transformer_cls":
            return classification.TransformerClassifier(m, self.device)
        if m.kind in ("tfidf", "bm25", "dense"):
            paras = load_paragraphs(self.cfg.data.path)
            corpus = {p.id: p.text for p in paras.values()}
            if m.kind == "tfidf":
                return retrieval.TfidfRanker(m, corpus)
            if m.kind == "bm25":
                return retrieval.BM25Ranker(m, corpus)
            return retrieval.DenseRanker(m, corpus, self.device)
        return qa.QAModel(m, self.device, self.cfg.run.max_seq_len, self.cfg.run.doc_stride)

    def _todo(self, m: ModelConfig) -> tuple[list[dict], Path]:
        path = self.out / "predictions" / f"{m.name}.jsonl"
        done = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    done.add(json.loads(line)["sample_id"])
                except (json.JSONDecodeError, KeyError):  # a torn last line after a crash
                    continue
        mine = [s for s in self.samples if s["task"] == m.task
                and (s["answerable"] or m.handles_unanswerable)]
        return [s for s in mine if s["id"] not in done], path

    @staticmethod
    def _append(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def run(self, only: list[str] | None = None, estimate_only: bool = False) -> dict:
        estimates = {}
        for m in self.cfg.models:
            if only and m.name not in only:
                continue
            todo, path = self._todo(m)
            self.log(f"[{m.name}] {m.task}: {len(todo)} samples to go")
            if not todo:
                continue
            t0 = time.time()
            model = self._load(m)
            self.log(f"[{m.name}] loaded in {time.time() - t0:.0f}s on {self.device}")
            by_n: dict[int, list[dict]] = defaultdict(list)
            for s in todo:
                by_n[s["N"]].append(s)
            bs = max(1, m.batch_size)
            # probe: one batch per N, then extrapolate
            per_sample: dict[int, float] = {}
            elapsed = 0.0
            for n in sorted(by_n):
                batch, by_n[n] = by_n[n][:bs], by_n[n][bs:]
                t = time.time()
                self._write(m, path, batch, model.predict(batch))
                dt = time.time() - t
                elapsed += dt
                per_sample[n] = dt / len(batch)
            est = elapsed + sum(per_sample[n] * len(v) for n, v in by_n.items())
            estimates[m.name] = est
            entry = self.manifest["models"].setdefault(m.name, {})
            entry.update({"task": m.task, "kind": m.kind, "model_id": m.model_id,
                          "estimated_seconds": round(est, 1)})
            self._save()
            self.log(f"[{m.name}] measured s/sample: "
                     + ", ".join(f"N={n}: {v:.3f}" for n, v in sorted(per_sample.items()))
                     + f" -> estimated {est / 60:.1f} min")
            if estimate_only:
                continue
            last = time.time()
            for n in sorted(by_n):
                rest = by_n[n]
                for i in range(0, len(rest), bs):
                    batch = rest[i:i + bs]
                    t = time.time()
                    self._write(m, path, batch, model.predict(batch))
                    elapsed += time.time() - t
                    if time.time() - last > 60:
                        last = time.time()
                        self.log(f"[{m.name}] N={n}: {i + len(batch)}/{len(rest)}")
            entry["seconds"] = round(elapsed, 1)
            self._save()
            self.log(f"[{m.name}] done in {elapsed / 60:.1f} min")
            del model
        if estimates:
            total = sum(estimates.values())
            self.log("estimated runtime: " + ", ".join(f"{k} {v / 60:.1f} min"
                                                      for k, v in estimates.items())
                     + f"; total {total / 3600:.2f} h")
        return estimates

    def _write(self, m: ModelConfig, path: Path, batch: list[dict], preds: list[dict]) -> None:
        self._append(path, [{"sample_id": s["id"], "model": m.name, **p}
                            for s, p in zip(batch, preds, strict=True)])
