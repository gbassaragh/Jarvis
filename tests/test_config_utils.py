import json

import pytest

from ai_assistant_pro.utils.config import create_default_config, load_config, save_config


def test_create_default_config_has_expected_sections():
    config = create_default_config()

    for section in ("srf", "engine", "serving", "logging"):
        assert section in config

    assert config["srf"]["top_k"] == 10
    assert config["engine"]["enable_paged_attention"] is True


def test_save_and_load_yaml_round_trip(tmp_path):
    config = {"srf": {"alpha": 0.4, "top_k": 5}, "engine": {"model_name": "gpt2"}}
    path = tmp_path / "config.yaml"

    save_config(config, path)
    loaded = load_config(path)

    assert loaded == config


def test_save_and_load_json_round_trip(tmp_path):
    config = {"srf": {"beta": 0.3}, "serving": {"port": 9000}}
    path = tmp_path / "config.json"

    save_config(config, path)
    loaded = load_config(path)

    assert loaded == config


def test_load_config_raises_for_missing_and_unknown(tmp_path):
    missing = tmp_path / "does-not-exist.yaml"
    with pytest.raises(FileNotFoundError):
        load_config(missing)

    unsupported = tmp_path / "config.txt"
    unsupported.write_text("plain text")
    with pytest.raises(ValueError):
        load_config(unsupported)
