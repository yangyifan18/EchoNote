import tomli_w
from pathlib import Path
from echonote.config import (
    Config,
    AudioConfig,
    WhisperConfig,
    LLMConfig,
    OutputConfig,
    load_config,
    save_config,
)


def test_default_config_has_expected_values():
    config = Config()
    assert config.audio.source == "mic"
    assert config.audio.sample_rate == 16000
    assert config.audio.chunk_duration == 30
    assert config.whisper.model == "large-v3"
    assert config.whisper.language == "zh"
    assert config.whisper.device == "auto"
    assert config.llm.provider == "claude"
    assert config.llm.prompt_template == "academic"
    assert config.llm.max_chars_per_chunk == 12000
    assert config.output.folder == "Meetings"
    assert config.output.keep_transcript is True
    assert config.output.keep_audio is False


def test_load_config_returns_defaults_when_no_file(tmp_config_dir):
    config = load_config(config_dir=tmp_config_dir)
    assert config.audio.source == "mic"
    assert config.whisper.model == "large-v3"


def test_save_and_load_config_roundtrip(tmp_config_dir):
    config = Config()
    config.audio.source = "system"
    config.llm.provider = "openai"
    config.llm.model = "gpt-4o"
    config.llm.prompt_template = "meeting"
    config.llm.max_chars_per_chunk = 8000
    config.output.obsidian_vault = "/Users/test/vault"

    save_config(config, config_dir=tmp_config_dir)

    loaded = load_config(config_dir=tmp_config_dir)
    assert loaded.audio.source == "system"
    assert loaded.llm.provider == "openai"
    assert loaded.llm.model == "gpt-4o"
    assert loaded.llm.prompt_template == "meeting"
    assert loaded.llm.max_chars_per_chunk == 8000
    assert loaded.output.obsidian_vault == "/Users/test/vault"


def test_load_config_merges_partial_file(tmp_config_dir):
    """A config file with only [audio] should leave other sections as defaults."""
    config_path = tmp_config_dir / "config.toml"
    partial = {"audio": {"source": "system", "chunk_duration": 60}}
    with open(config_path, "wb") as f:
        tomli_w.dump(partial, f)

    config = load_config(config_dir=tmp_config_dir)
    assert config.audio.source == "system"
    assert config.audio.chunk_duration == 60
    assert config.audio.sample_rate == 16000  # default preserved
    assert config.whisper.model == "large-v3"  # default preserved
