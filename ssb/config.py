"""YAML config schema. Unknown keys are errors, so a typo never silently does nothing."""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TASK_TYPES = ("single_hop", "multi_hop", "unanswerable")
AXES = ("dilution", "noise", "redundancy", "contradiction", "position")


@dataclass
class WorldConfig:
    people: int = 80
    towns: int = 40
    companies: int = 40
    inventions: int = 40


@dataclass
class DatasetConfig:
    path: str = "data/ssb_v0.1.jsonl"
    version: str = "0.1"
    questions: dict[str, int] = field(
        default_factory=lambda: {"single_hop": 34, "multi_hop": 33, "unanswerable": 33})
    paragraph_tokens: int = 120
    tolerance: float = 0.10
    tokenizer: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hard_negative_scorer: str = "embedding"  # "embedding" | "tfidf"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    offtopic_pool: int = 400
    world: WorldConfig = field(default_factory=WorldConfig)


@dataclass
class ConditionsConfig:
    sizes: list[int] = field(default_factory=lambda: [1, 5, 20, 100])
    axes: list[str] = field(default_factory=lambda: list(AXES))
    # Filler used by every axis except `noise`, which compares filler types.
    base_filler: str = "random"  # "random" | "hard_negative"
    redundancy_k: list[int] = field(default_factory=lambda: [1, 5])
    contradiction_positions: list[str] = field(default_factory=lambda: ["before", "after"])
    depths: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])
    closed_book: bool = True


@dataclass
class ModelConfig:
    name: str
    model_id: str
    quantization: str | None = None  # None | "4bit"
    backend: str = "auto"  # "auto" (vLLM if installed with a GPU, else HF) | "vllm" | "hf" | "fake"
    vllm_model_id: str | None = None  # e.g. an AWQ checkpoint when vLLM serves the 4-bit model
    max_model_len: int = 16384
    max_batch_size: int = 16
    max_batch_tokens: int = 24576  # padded prompt tokens per batch (HF backend)
    fallback_model_id: str | None = None  # used when the runtime estimate is too high
    prior_tokens_per_s: float = 2000.0  # T4 prompt throughput guess for `ssb estimate`


@dataclass
class GenerationConfig:
    max_new_tokens: int = 32
    temperature: float = 0.0


@dataclass
class RunConfig:
    output_dir: str = "results/main"
    cache_dir: str = "results/cache"  # kept under results/ so Kaggle saves it with the output
    questions: int | None = None  # take a balanced subset of the dataset (smoke runs)
    max_gpu_hours: float = 6.0


@dataclass
class EvalConfig:
    bootstrap: int = 1000
    ci: float = 0.95
    saturation_epsilon: float = 0.01  # N*: marginal gain below 1 accuracy point
    alpha: float = 0.05  # N†: significance level
    ece_bins: int = 10


@dataclass
class Config:
    name: str
    seed: int
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    conditions: ConditionsConfig = field(default_factory=ConditionsConfig)
    models: list[ModelConfig] = field(default_factory=list)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    run: RunConfig = field(default_factory=RunConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    figures_dir: str = "figures"

    @property
    def token_range(self) -> tuple[int, int]:
        t, tol = self.dataset.paragraph_tokens, self.dataset.tolerance
        return round(t * (1 - tol)), round(t * (1 + tol))


class ConfigError(ValueError):
    pass


def _build(cls, data, where: str):
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(names)}")
    kwargs = {}
    for key, value in data.items():
        hint = hints[key]
        origin = typing.get_origin(hint)
        if dataclasses.is_dataclass(hint):
            value = _build(hint, value, f"{where}.{key}")
        elif origin is list and dataclasses.is_dataclass(typing.get_args(hint)[0]):
            item = typing.get_args(hint)[0]
            value = [_build(item, v, f"{where}.{key}[{i}]") for i, v in enumerate(value)]
        kwargs[key] = value
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigError(f"{where}: {e}") from None


def validate(cfg: Config) -> None:
    d, c = cfg.dataset, cfg.conditions
    if set(d.questions) - set(TASK_TYPES):
        raise ConfigError(f"dataset.questions: task types must be in {TASK_TYPES}")
    if any(n < 0 for n in d.questions.values()) or sum(d.questions.values()) == 0:
        raise ConfigError("dataset.questions: counts must be >= 0 and not all zero")
    if d.hard_negative_scorer not in ("embedding", "tfidf"):
        raise ConfigError("dataset.hard_negative_scorer must be 'embedding' or 'tfidf'")
    if not 0 <= d.tolerance < 1:
        raise ConfigError("dataset.tolerance must be in [0, 1)")
    if c.sizes != sorted(set(c.sizes)) or c.sizes[0] < 1:
        raise ConfigError("conditions.sizes must be strictly increasing positive integers")
    if set(c.axes) - set(AXES):
        raise ConfigError(f"conditions.axes must be a subset of {AXES}")
    if c.base_filler not in ("random", "hard_negative"):
        raise ConfigError("conditions.base_filler must be 'random' or 'hard_negative'")
    if any(not 0 <= x <= 1 for x in c.depths):
        raise ConfigError("conditions.depths must be in [0, 1]")
    if set(c.contradiction_positions) - {"before", "after"}:
        raise ConfigError("conditions.contradiction_positions must be 'before'/'after'")
    if any(not 1 <= k <= 5 for k in c.redundancy_k):
        raise ConfigError("conditions.redundancy_k must be in [1, 5] (6 templates per fact, "
                          "one reserved for the conflicting claim)")
    names = [m.name for m in cfg.models]
    if len(names) != len(set(names)):
        raise ConfigError("models: names must be unique")
    for m in cfg.models:
        if m.quantization not in (None, "4bit"):
            raise ConfigError(f"models.{m.name}: quantization must be null or '4bit'")
        if m.backend not in ("auto", "vllm", "hf", "fake"):
            raise ConfigError(f"models.{m.name}: backend must be auto, vllm, hf or fake")
        if m.max_batch_size < 1 or m.max_batch_tokens < 1:
            raise ConfigError(f"models.{m.name}: batch limits must be positive")
    if cfg.generation.temperature != 0:
        raise ConfigError("generation.temperature must be 0 (greedy) in v0.1")
    if cfg.run.questions is not None and cfg.run.questions < 1:
        raise ConfigError("run.questions must be a positive integer or null")


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = _build(Config, data, path.name)
    validate(cfg)
    return cfg


def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)
