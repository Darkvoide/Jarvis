"""JARVIS package."""

from jarvis.config import get_config, load_config
from jarvis.context import JarvisContext


def __getattr__(name: str):
    """Lazily import heavy modules so tests can run without all deps installed."""
    if name == "Orchestrator":
        from jarvis.orchestrator import Orchestrator
        return Orchestrator
    raise AttributeError(f"module 'jarvis' has no attribute {name!r}")


__all__ = ["get_config", "load_config", "JarvisContext", "Orchestrator"]
