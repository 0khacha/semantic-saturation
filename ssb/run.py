"""Resumable runner.

Every finished batch goes straight to an append-only cache (fsynced), so a crash or a
timeout never loses finished work: re-running the same command skips what is cached.

Before the bulk of the work, the runner times one batch from each context size N (real
work, cached like the rest) and extrapolates the model's remaining GPU time. If that
exceeds the budget, it switches to the configured fallback model; if there is none, it
stops and says how many questions would fit (cut questions, never axes).
"""

from __future__ import annotations

import dataclasses
import json
import math
import platform
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from .build import load_dataset
from .config import Config, ModelConfig
from .models import Backend, GenerationCache, load_backend, model_id_for, resolve_backend


class BudgetExceeded(RuntimeError):
    pass


class TimeLimitReached(RuntimeError):
    pass


def select_subset(samples: list[dict], n_questions: int | None) -> list[dict]:
    """All samples of a balanced subset of questions (round-robin over task types)."""
    if n_questions is None:
        return samples
    by_type: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        q = s["question_id"]
        if q not in by_type[s["task_type"]]:
            by_type[s["task_type"]].append(q)
    order = ["single_hop", "multi_hop", "unanswerable"]
    chosen: list[str] = []
    while len(chosen) < n_questions and any(by_type[t] for t in order):
        for t in order:
            if by_type[t] and len(chosen) < n_questions:
                chosen.append(by_type[t].pop(0))
    keep = set(chosen)
    return [s for s in samples if s["question_id"] in keep]


def make_batches(samples: list[dict], m: ModelConfig) -> list[list[dict]]:
    """Group by N (similar lengths), then cut by batch size and padded-token budget."""
    by_n: dict[int, list[dict]] = defaultdict(list)
    for s in samples:
        by_n[s["N"]].append(s)
    batches = []
    for n in sorted(by_n, reverse=True):  # longest first: OOM shows up at once
        group = sorted(by_n[n], key=lambda s: -s["token_count"])
        cur: list[dict] = []
        for s in group:
            longest = max([x["token_count"] for x in cur] + [s["token_count"]])
            if cur and (len(cur) + 1 > m.max_batch_size
                        or longest * (len(cur) + 1) > m.max_batch_tokens):
                batches.append(cur)
                cur = []
            cur.append(s)
        if cur:
            batches.append(cur)
    return batches


def library_versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "accelerate", "bitsandbytes", "vllm", "tokenizers"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - optional libraries
            out[mod] = None
    try:
        import torch

        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        out["gpu"] = None
    return out


class Runner:
    def __init__(self, cfg: Config, time_limit_hours: float | None = None,
                 force: bool = False, log=print):
        self.cfg = cfg
        self.out = Path(cfg.run.output_dir)
        self.deadline = time.time() + time_limit_hours * 3600 if time_limit_hours else None
        self.force = force
        self.log = log
        samples = load_dataset(cfg.dataset.path)
        self.samples = select_subset(samples, cfg.run.questions)
        self.n_questions = len({s["question_id"] for s in self.samples})
        self.manifest_path = self.out / "run_manifest.json"
        self.manifest = (json.loads(self.manifest_path.read_text())
                         if self.manifest_path.exists() else {"models": {}})

    # ---- helpers
    def _key(self, m: ModelConfig, backend: str, model_id: str, prompt: str) -> str:
        return GenerationCache.key(model_id, backend, m.quantization,
                                   self.cfg.generation.max_new_tokens, prompt)

    def _save_manifest(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest.update({"config": self.cfg.name, "dataset": self.cfg.dataset.path,
                              "questions": self.n_questions, "samples": len(self.samples),
                              "max_gpu_hours": self.cfg.run.max_gpu_hours,
                              "environment": library_versions()})
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2) + "\n")

    def _run_batch(self, backend: Backend, batch: list[dict], cache: GenerationCache,
                   keys: list[str]) -> float:
        t0 = time.time()
        gens = backend.generate([s["prompt"] for s in batch])
        dt = time.time() - t0
        cache.add_many([{"key": k, "sample_id": s["id"], **g, "seconds": dt / len(batch)}
                        for k, s, g in zip(keys, batch, gens, strict=True)])
        return dt

    # ---- main loop
    def run(self, only: list[str] | None = None) -> dict:
        spent_hours = 0.0
        status = {}
        for m in self.cfg.models:
            if only and m.name not in only:
                continue
            remaining = self.cfg.run.max_gpu_hours - spent_hours
            hours = self._run_model(m, remaining)
            spent_hours += hours
            status[m.name] = "done"
        return status

    def _run_model(self, m: ModelConfig, budget_hours: float) -> float:
        backend_name = resolve_backend(m)
        model_id = model_id_for(m, backend_name)
        cache = GenerationCache(Path(self.cfg.run.cache_dir) / f"{m.name}.jsonl")
        key = {s["id"]: self._key(m, backend_name, model_id, s["prompt"])
               for s in self.samples}
        todo = [s for s in self.samples if key[s["id"]] not in cache]
        self.log(f"[{m.name}] {model_id} via {backend_name}: {len(self.samples) - len(todo)}"
                 f"/{len(self.samples)} cached, {len(todo)} to go")
        elapsed = 0.0
        if todo:
            t_load = time.time()
            backend = load_backend(m, self.cfg.generation.max_new_tokens)
            if backend.name != backend_name:  # vLLM fell back to HF: keys change
                backend_name = backend.name
                key = {s["id"]: self._key(m, backend_name, backend.model_id, s["prompt"])
                       for s in self.samples}
                todo = [s for s in self.samples if key[s["id"]] not in cache]
            model_id = backend.model_id
            self.log(f"[{m.name}] loaded in {time.time() - t_load:.0f}s")
            batches = make_batches(todo, m)
            try:
                elapsed = self._probe_and_run(m, backend, batches, cache, key, budget_hours)
            except TimeLimitReached:
                self._materialise(m, cache, key, backend_name, model_id)  # keep partial output
                raise
            if elapsed < 0:  # over budget: run the fallback model instead
                return self._run_model(fallback_config(m), budget_hours)
            del backend
            _free_gpu()
        self._materialise(m, cache, key, backend_name, model_id)
        return elapsed / 3600

    def _probe_and_run(self, m, backend, batches, cache, key, budget_hours):
        by_n: dict[int, list[int]] = defaultdict(list)
        for i, b in enumerate(batches):
            by_n[b[0]["N"]].append(i)
        probe = [idxs[0] for idxs in by_n.values()]
        timings: dict[int, float] = {}
        elapsed = 0.0
        for i in probe:
            b = batches[i]
            dt = self._run_batch(backend, b, cache, [key[s["id"]] for s in b])
            timings[b[0]["N"]] = dt
            elapsed += dt
        est = sum(timings[n] * (len(idxs) - 1) for n, idxs in by_n.items()) + elapsed
        est_h = est / 3600
        detail = ", ".join(f"N={n}: {timings[n]:.1f}s x {len(idxs)} batches"
                           for n, idxs in sorted(by_n.items()))
        self.log(f"[{m.name}] measured: {detail}")
        self.log(f"[{m.name}] estimated GPU time for this model: {est_h:.2f} h "
                 f"(budget left: {budget_hours:.2f} h)")
        entry = self.manifest["models"].setdefault(m.name, {})
        entry.update({"model_id": backend.model_id, "backend": backend.name,
                      "info": backend.info(), "estimated_hours": round(est_h, 3)})
        self._save_manifest()
        if est_h > budget_hours and not self.force:
            if m.fallback_model_id:
                self.log(f"[{m.name}] over budget: switching to {m.fallback_model_id}")
                entry["fallback_reason"] = f"estimate {est_h:.2f} h > budget {budget_hours:.2f} h"
                self._save_manifest()
                del backend
                _free_gpu()
                return -1.0
            fit = max(1, math.floor(self.n_questions * budget_hours / est_h))
            raise BudgetExceeded(
                f"{m.name}: estimated {est_h:.2f} h exceeds the {budget_hours:.2f} h left. "
                f"Set run.questions to about {fit} (cut questions, not axes), or pass --force.")
        t_start = last_log = time.time()
        rest = [i for i in range(len(batches)) if i not in set(probe)]
        for j, i in enumerate(rest, 1):
            if self.deadline and time.time() > self.deadline:
                raise TimeLimitReached(f"{m.name}: time limit reached after {j - 1} batches; "
                                       "rerun the same command to resume")
            b = batches[i]
            elapsed += self._run_batch(backend, b, cache, [key[s["id"]] for s in b])
            if time.time() - last_log > 60 or j == len(rest):
                last_log = time.time()
                rate = (time.time() - t_start) / j
                self.log(f"[{m.name}] {j}/{len(rest)} batches, "
                         f"~{rate * (len(rest) - j) / 60:.0f} min left")
        entry["gpu_hours"] = round(elapsed / 3600 + entry.get("gpu_hours", 0.0), 3)
        entry["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
        self._save_manifest()
        return elapsed

    def _materialise(self, m, cache, key, backend_name, model_id) -> None:
        """results/<run>/generations/<model>.jsonl: one line per sample, in dataset order."""
        path = self.out / "generations" / f"{m.name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        missing = 0
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for s in self.samples:
                rec = cache.get(key[s["id"]])
                if rec is None:
                    missing += 1
                    continue
                fh.write(json.dumps({
                    "sample_id": s["id"], "model": m.name, "model_id": model_id,
                    "backend": backend_name, "text": rec["text"],
                    "answer_logprobs": rec["answer_logprobs"],
                    "prompt_tokens": rec["prompt_tokens"], "gen_tokens": rec["gen_tokens"],
                }, ensure_ascii=False) + "\n")
        self.log(f"[{m.name}] wrote {path} ({len(self.samples) - missing} generations"
                 + (f", {missing} missing" if missing else "") + ")")
        entry = self.manifest["models"].setdefault(m.name, {})
        entry.update({"model_id": model_id, "backend": backend_name,
                      "complete": missing == 0})
        self._save_manifest()


def fallback_config(m: ModelConfig) -> ModelConfig:
    """The fallback runs as its own model: full fp16 (a 3B model fits a T4), own name."""
    name = m.fallback_model_id.split("/")[-1].lower().replace("-instruct", "")
    return dataclasses.replace(m, name=name, model_id=m.fallback_model_id, quantization=None,
                               vllm_model_id=None, fallback_model_id=None)


def _free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def estimate(cfg: Config) -> list[dict]:
    """Static estimate from prior throughputs (the runner measures the real one)."""
    samples = select_subset(load_dataset(cfg.dataset.path), cfg.run.questions)
    tokens = sum(s["token_count"] for s in samples)
    rows = []
    for m in cfg.models:
        rows.append({"model": m.name, "model_id": m.model_id, "prompts": len(samples),
                     "prompt_tokens": tokens,
                     "gpu_hours": tokens / m.prior_tokens_per_s / 3600})
    return rows
