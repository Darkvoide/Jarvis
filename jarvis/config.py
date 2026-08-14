"""JARVIS configuration loader.

Reads config.yaml from the project root and validates it with Pydantic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


# ── Sub-models ─────────────────────────────────────────────────────────────────


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    vision_model: Optional[str] = None
    timeout: int = 120


class STTConfig(BaseModel):
    mode: str = "continuous"  # "continuous" (hands-free) or "ptt" (push-to-talk)
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: Optional[str] = None
    vad_filter: bool = True


class TTSConfig(BaseModel):
    backend: str = "pyttsx3"   # "pyttsx3" | "edge-tts"
    rate: int = 175
    volume: float = 1.0
    edge_voice: str = "en-US-GuyNeural"


class InputConfig(BaseModel):
    ptt_key: str = "F9"
    camera_index: int = 0
    gesture_enabled: bool = True
    gesture_debounce: float = 1.5


class PermissionsConfig(BaseModel):
    allow_shell: bool = True
    allow_file_read: bool = True
    allowed_paths: list[str] = Field(default_factory=list)
    blocked_commands: list[str] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "jarvis.log"


class JarvisConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# ── Loader ─────────────────────────────────────────────────────────────────────

_config: JarvisConfig | None = None
_CONFIG_PATHS = [
    Path(__file__).parent.parent / "config.yaml",  # d:\Jarvis\config.yaml
    Path(os.getcwd()) / "config.yaml",
]


def load_config(path: Path | None = None) -> JarvisConfig:
    """Load and validate config.yaml. Results are cached after first load.

    Args:
        path: Explicit path to config.yaml. If None, searches default locations.

    Returns:
        Validated JarvisConfig instance.

    Raises:
        FileNotFoundError: If no config.yaml is found.
    """
    global _config
    if _config is not None:
        return _config

    config_path = path
    if config_path is None:
        for candidate in _CONFIG_PATHS:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None or not config_path.exists():
        # Return defaults if no file found
        _config = JarvisConfig()
        return _config

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    _config = JarvisConfig(**raw)
    return _config


def get_config() -> JarvisConfig:
    """Return the cached config, loading it first if necessary."""
    return load_config()
