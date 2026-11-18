import pytest
import yaml

from ssb.config import ConfigError, load_config


@pytest.mark.parametrize("path", ["configs/main.yaml", "configs/smoke.yaml"])
def test_shipped_configs_load(path):
    cfg = load_config(path)
    assert cfg.models
    assert cfg.token_range == (108, 132)


def test_smoke_uses_the_main_dataset_and_one_small_model():
    main, smoke = load_config("configs/main.yaml"), load_config("configs/smoke.yaml")
    assert smoke.dataset == main.dataset and smoke.conditions == main.conditions
    assert [m.model_id for m in smoke.models] == ["Qwen/Qwen2.5-1.5B-Instruct"]
    assert smoke.run.questions == 10


def _write(tmp_path, data):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_unknown_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, {"name": "x", "seed": 1, "conditions": {"szes": [1]}}))


@pytest.mark.parametrize("conditions", [
    {"sizes": [5, 1]},
    {"depths": [1.5]},
    {"redundancy_k": [6]},
    {"axes": ["bogus"]},
    {"contradiction_positions": ["middle"]},
])
def test_invalid_conditions_are_rejected(tmp_path, conditions):
    with pytest.raises(ConfigError):
        load_config(_write(tmp_path, {"name": "x", "seed": 1, "conditions": conditions}))


def test_non_greedy_decoding_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="greedy"):
        load_config(_write(tmp_path, {"name": "x", "seed": 1,
                                      "generation": {"temperature": 0.7}}))
