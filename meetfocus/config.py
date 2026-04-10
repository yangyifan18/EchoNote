"""Configuration management — load/save TOML config with sensible defaults."""

from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
import tomllib
import tomli_w

DEFAULT_CONFIG_DIR = Path.home() / ".meetfocus"


@dataclass
class AudioConfig:
    device: str = "default"
    source: str = "mic"
    chunk_duration: int = 30
    sample_rate: int = 16000


@dataclass
class WhisperConfig:
    model: str = "large-v3"
    language: str = "zh"
    device: str = "auto"


@dataclass
class LLMConfig:
    provider: str = "claude"
    model: str = "claude-sonnet-4-6"
    api_key_env: str = "ANTHROPIC_API_KEY"


@dataclass
class OutputConfig:
    obsidian_vault: str = ""
    folder: str = "Meetings"
    keep_transcript: bool = True
    keep_audio: bool = False
    filename_format: str = "{date}-{title}"


@dataclass
class SessionsConfig:
    dir: str = ""

    def __post_init__(self):
        if not self.dir:
            self.dir = str(DEFAULT_CONFIG_DIR / "sessions")


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    sessions: SessionsConfig = field(default_factory=SessionsConfig)


def _merge_section(section_cls, data: dict):
    """Create a section dataclass, applying only keys that exist in the dataclass."""
    valid_keys = {f.name for f in fields(section_cls)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return section_cls(**filtered)


def load_config(config_dir: Path = DEFAULT_CONFIG_DIR) -> Config:
    """Load config from TOML file, falling back to defaults for missing keys."""
    config_path = config_dir / "config.toml"
    if not config_path.exists():
        return Config()

    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    config = Config()
    if "audio" in data:
        config.audio = _merge_section(AudioConfig, data["audio"])
    if "whisper" in data:
        config.whisper = _merge_section(WhisperConfig, data["whisper"])
    if "llm" in data:
        config.llm = _merge_section(LLMConfig, data["llm"])
    if "output" in data:
        config.output = _merge_section(OutputConfig, data["output"])
    if "sessions" in data:
        config.sessions = _merge_section(SessionsConfig, data["sessions"])
    return config


def save_config(config: Config, config_dir: Path = DEFAULT_CONFIG_DIR) -> None:
    """Save config to TOML file."""
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"
    data = asdict(config)
    with open(config_path, "wb") as f:
        tomli_w.dump(data, f)
