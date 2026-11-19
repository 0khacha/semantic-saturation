"""Runner: caching, crash-resume, time limit, budget and fallback (fake backend, no GPU)."""

import dataclasses
import json
import time
from pathlib import Path

import pytest

from ssb import models
from ssb.config import ModelConfig, load_config
from ssb.run import BudgetExceeded, Runner, TimeLimitReached, make_batches, select_subset

DATA = Path("data/ssb_v0.1.jsonl")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="dataset not built yet")


def _cfg(tmp_path, questions=4, **model_kw):
    cfg = load_config("configs/smoke.yaml")
    run = dataclasses.replace(cfg.run, output_dir=str(tmp_path / "out"),
                              cache_dir=str(tmp_path / "cache"), questions=questions)
    m = ModelConfig(name="fake", model_id="fake/model", backend="fake", **model_kw)
    return dataclasses.replace(cfg, run=run, models=[m])


def _quiet(*_):
    pass


def test_subset_is_balanced():
    from ssb.build import load_dataset

    s = select_subset(load_dataset(DATA), 6)
    tasks = {}
    for x in s:
        tasks.setdefault(x["task_type"], set()).add(x["question_id"])
    assert {k: len(v) for k, v in tasks.items()} == {"single_hop": 2, "multi_hop": 2,
                                                       "unanswerable": 2}


def test_batches_respect_limits(tmp_path):
    cfg = _cfg(tmp_path, max_batch_size=3, max_batch_tokens=30000)
    r = Runner(cfg, log=_quiet)
    batches = make_batches(r.samples, cfg.models[0])
    assert sum(len(b) for b in batches) == len(r.samples)
    for b in batches:
        assert len(b) <= 3 and len({s["N"] for s in b}) == 1
        assert max(s["token_count"] for s in b) * len(b) <= 30000 or len(b) == 1


def test_run_writes_every_generation_and_resumes_from_cache(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    calls = {"n": 0}
    real = models.FakeBackend.generate

    def crashing(self, prompts):
        calls["n"] += 1
        if calls["n"] == 6:
            raise RuntimeError("simulated crash")
        return real(self, prompts)

    monkeypatch.setattr(models.FakeBackend, "generate", crashing)
    with pytest.raises(RuntimeError, match="simulated crash"):
        Runner(cfg, log=_quiet).run()
    cache_file = Path(cfg.run.cache_dir) / "fake.jsonl"
    done_before = len(cache_file.read_text().splitlines())
    assert done_before > 0  # finished batches survived the crash

    generated = []
    monkeypatch.setattr(models.FakeBackend, "generate",
                        lambda self, p: generated.extend(p) or real(self, p))
    r = Runner(cfg, log=_quiet)
    r.run()
    assert len(generated) == len(r.samples) - done_before  # nothing regenerated
    out = Path(cfg.run.output_dir) / "generations" / "fake.jsonl"
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert [x["sample_id"] for x in lines] == [s["id"] for s in r.samples]
    manifest = json.loads((Path(cfg.run.output_dir) / "run_manifest.json").read_text())
    assert manifest["models"]["fake"]["complete"] is True


def test_time_limit_pauses_cleanly(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    real = models.FakeBackend.generate
    monkeypatch.setattr(models.FakeBackend, "generate",
                        lambda self, p: time.sleep(0.02) or real(self, p))
    with pytest.raises(TimeLimitReached):
        Runner(cfg, time_limit_hours=0.03 / 3600, log=_quiet).run()
    out = Path(cfg.run.output_dir) / "generations" / "fake.jsonl"
    assert out.exists()  # partial output written
    Runner(cfg, log=_quiet).run()  # and the rerun completes


def test_budget_stops_and_recommends_fewer_questions(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, questions=6)
    cfg = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, max_gpu_hours=1e-7))
    real = models.FakeBackend.generate
    monkeypatch.setattr(models.FakeBackend, "generate",
                        lambda self, p: time.sleep(0.01) or real(self, p))
    with pytest.raises(BudgetExceeded, match="run.questions"):
        Runner(cfg, log=_quiet).run()


def test_over_budget_switches_to_fallback(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, quantization=None, fallback_model_id="fake/Small-3B-Instruct")
    cfg = dataclasses.replace(cfg, run=dataclasses.replace(cfg.run, max_gpu_hours=1e-7))
    real = models.FakeBackend.generate

    def slow_unless_fallback(self, p):
        if self.model_id == "fake/model":
            time.sleep(0.01)
        return real(self, p)

    monkeypatch.setattr(models.FakeBackend, "generate", slow_unless_fallback)
    # the fallback would also exceed a ~zero budget, so give it room once switched
    seen = []
    orig = Runner._run_model

    def tracking(self, m, budget):
        seen.append(m.name)
        return orig(self, m, budget if m.name == "fake" else 10.0)

    monkeypatch.setattr(Runner, "_run_model", tracking)
    Runner(cfg, log=_quiet).run()
    assert seen == ["fake", "small-3b"]
    manifest = json.loads((Path(cfg.run.output_dir) / "run_manifest.json").read_text())
    assert "fallback_reason" in manifest["models"]["fake"]
    assert (Path(cfg.run.output_dir) / "generations" / "small-3b.jsonl").exists()
