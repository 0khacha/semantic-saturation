"""YAML config schema. Unknown keys are errors, so a typo never silently does nothing."""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TASKS = ("classification", "retrieval", "qa")
AXES = ("dilution", "noise", "redundancy", "contradiction", "position")


@dataclass
class DataConfig:
    path: str = "data/ssb_v0.1.jsonl"
    version: str = "0.1"
    raw_dir: str = "data/raw"  # Hugging Face dataset cache, model-made caches (gitignored)
    tokenizer: str = "bert-base-uncased"  # reference tokenizer for every token count
    paragraph_tokens: int = 120
    tolerance: float = 0.10
    # base examples per task; QA mixes answerable and unanswerable questions
    examples: dict[str, int] = field(
        default_factory=lambda: {"classification": 300, "retrieval": 300, "qa": 300})
    qa_unanswerable_share: float = 0.5
    random_pool: int = 3000  # normalised SQuAD-train paragraphs used as random filler
    hard_pool: int = 4000  # extra normalised paragraphs searched for hard negatives
    selector_model: str = "sentence-transformers/all-mpnet-base-v2"  # picks hard negatives
    ner_model: str = "en_core_web_sm"
    paraphrase_models: list[str] = field(default_factory=lambda: [
        "Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en"])
    # judge that checks a classification paraphrase keeps its label (not an evaluated model)
    label_judge: str = "fabriceyhc/bert-base-uncased-ag_news"
    device: str = "auto"  # "auto" | "cpu" | "cuda" for the build-time models


@dataclass
class ConditionsConfig:
    sizes: list[int] = field(default_factory=lambda: [1, 5, 20, 100])
    axes: list[str] = field(default_factory=lambda: list(AXES))
    base_filler: str = "random"  # filler for every axis except noise
    redundancy_k: int = 5
    redundancy_modes: list[str] = field(default_factory=lambda: ["copy", "paraphrase"])
    contradiction_positions: list[str] = field(default_factory=lambda: ["before", "after"])
    depths: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.0])


@dataclass
class ModelConfig:
    name: str
    task: str
    kind: str  # tfidf_lr | transformer_cls | tfidf | bm25 | dense | qa
    model_id: str | None = None
    long_input: str | None = None  # transformer_cls: "truncate" | "chunk_mean"
    handles_unanswerable: bool = True  # qa: False -> evaluated on answerable only
    batch_size: int = 32


@dataclass
class RunConfig:
    output_dir: str = "results/main"
    device: str = "auto"
    max_seq_len: int = 384  # QA sliding window
    doc_stride: int = 128
    examples: int | None = None  # balanced subset of base examples per task


@dataclass
class EvalConfig:
    bootstrap: int = 1000
    ci: float = 0.95
    saturation_epsilon: float = 0.01
    alpha: float = 0.05
    ece_bins: int = 10


@dataclass
class Config:
    name: str
    seed: int
    data: DataConfig = field(default_factory=DataConfig)
    conditions: ConditionsConfig = field(default_factory=ConditionsConfig)
    models: list[ModelConfig] = field(default_factory=list)
    run: RunConfig = field(default_factory=RunConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    figures_dir: str = "figures"

    @property
    def token_range(self) -> tuple[int, int]:
        t, tol = self.data.paragraph_tokens, self.data.tolerance
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
        if dataclasses.is_dataclass(hint):
            value = _build(hint, value, f"{where}.{key}")
        elif typing.get_origin(hint) is list and dataclasses.is_dataclass(typing.get_args(hint)[0]):
            item = typing.get_args(hint)[0]
            value = [_build(item, v, f"{where}.{key}[{i}]") for i, v in enumerate(value)]
        kwargs[key] = value
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ConfigError(f"{where}: {e}") from None


KINDS = {"classification": {"tfidf_lr", "transformer_cls"},
         "retrieval": {"tfidf", "bm25", "dense"}, "qa": {"qa"}}


def validate(cfg: Config) -> None:
    d, c = cfg.data, cfg.conditions
    if set(d.examples) - set(TASKS) or any(n < 1 for n in d.examples.values()):
        raise ConfigError(f"data.examples: positive counts for tasks in {TASKS}")
    if not 0 <= d.qa_unanswerable_share < 1:
        raise ConfigError("data.qa_unanswerable_share must be in [0, 1)")
    if not 0 <= d.tolerance < 1:
        raise ConfigError("data.tolerance must be in [0, 1)")
    if c.sizes != sorted(set(c.sizes)) or c.sizes[0] < 1:
        raise ConfigError("conditions.sizes must be strictly increasing positive integers")
    if set(c.axes) - set(AXES):
        raise ConfigError(f"conditions.axes must be a subset of {AXES}")
    if c.base_filler not in ("random", "hard_negative"):
        raise ConfigError("conditions.base_filler must be 'random' or 'hard_negative'")
    if c.redundancy_k < 2:
        raise ConfigError("conditions.redundancy_k must be >= 2")
    if set(c.redundancy_modes) - {"copy", "paraphrase"}:
        raise ConfigError("conditions.redundancy_modes: 'copy' and/or 'paraphrase'")
    if set(c.contradiction_positions) - {"before", "after"}:
        raise ConfigError("conditions.contradiction_positions: 'before' and/or 'after'")
    if any(not 0 <= x <= 1 for x in c.depths):
        raise ConfigError("conditions.depths must be in [0, 1]")
    names = [m.name for m in cfg.models]
    if len(names) != len(set(names)):
        raise ConfigError("models: names must be unique")
    for m in cfg.models:
        if m.task not in TASKS:
            raise ConfigError(f"models.{m.name}: task must be one of {TASKS}")
        if m.kind not in KINDS[m.task]:
            raise ConfigError(f"models.{m.name}: kind for {m.task} must be in {KINDS[m.task]}")
        if m.kind == "transformer_cls" and m.long_input not in ("truncate", "chunk_mean"):
            raise ConfigError(f"models.{m.name}: long_input must be truncate or chunk_mean")
        if m.kind in ("transformer_cls", "dense", "qa") and not m.model_id:
            raise ConfigError(f"models.{m.name}: model_id is required")
    if cfg.run.examples is not None and cfg.run.examples < 1:
        raise ConfigError("run.examples must be a positive integer or null")


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = _build(Config, data, path.name)
    validate(cfg)
    return cfg


def config_to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)
