"""
Central configuration for Student RAG Chat.

Every other module reads settings through this file instead of touching
os.environ directly. That keeps environment/config logic in one place and
makes it easy to unit test (see tests/test_config.py, added in Phase 15).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load variables from a local .env file if one exists. Safe no-op otherwise.
load_dotenv()

# --- Project paths -----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# --- Safe defaults (used whenever the matching env var is absent/blank) --
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "models/gemini-embedding-001"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid in a way the app
    cannot safely recover from (e.g. chunk_overlap >= chunk_size)."""


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid float for %s=%r — using default %s", name, raw, default)
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid int for %s=%r — using default %s", name, raw, default)
        return default


@dataclass(frozen=True)
class Config:
    google_api_key: str
    gemini_model: str
    embedding_model: str
    temperature: float
    chunk_size: int
    chunk_overlap: int
    top_k: int
    upload_dir: Path
    vectorstore_dir: Path

    @property
    def has_api_key(self) -> bool:
        return bool(self.google_api_key)

    def __repr__(self) -> str:  # never leak the key into logs/tracebacks
        key_display = "***set***" if self.has_api_key else "MISSING"
        return (
            f"Config(google_api_key={key_display}, gemini_model={self.gemini_model!r}, "
            f"embedding_model={self.embedding_model!r}, temperature={self.temperature}, "
            f"chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}, "
            f"top_k={self.top_k})"
        )


def load_config() -> Config:
    """
    Build a Config from environment variables.

    Deliberately does NOT raise if the Gemini API key is missing — the
    Streamlit app must still start and show a friendly "add your API key"
    banner instead of crashing. Code that is about to call Gemini should
    check `config.has_api_key` first, or call `require_api_key(config)`.

    Raises:
        ConfigError: if numeric settings are present but structurally
            invalid (e.g. chunk_overlap >= chunk_size), since that would
            silently break chunking rather than fail loudly.
    """
    cfg = Config(
        google_api_key=_get_str("GOOGLE_API_KEY", ""),
        gemini_model=_get_str("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        embedding_model=_get_str("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        temperature=_get_float("TEMPERATURE", DEFAULT_TEMPERATURE),
        chunk_size=_get_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE),
        chunk_overlap=_get_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP),
        top_k=_get_int("TOP_K", DEFAULT_TOP_K),
        upload_dir=UPLOAD_DIR,
        vectorstore_dir=VECTORSTORE_DIR,
    )

    if cfg.chunk_size <= 0:
        raise ConfigError(f"CHUNK_SIZE must be a positive integer, got {cfg.chunk_size}.")
    if cfg.chunk_overlap < 0:
        raise ConfigError(f"CHUNK_OVERLAP cannot be negative, got {cfg.chunk_overlap}.")
    if cfg.chunk_overlap >= cfg.chunk_size:
        raise ConfigError(
            f"CHUNK_OVERLAP ({cfg.chunk_overlap}) must be smaller than "
            f"CHUNK_SIZE ({cfg.chunk_size}), or chunking will loop forever."
        )
    if cfg.top_k <= 0:
        raise ConfigError(f"TOP_K must be a positive integer, got {cfg.top_k}.")

    logger.info("Configuration loaded: %r", cfg)
    return cfg


def require_api_key(cfg: Config) -> None:
    """Raise a beginner-friendly error if no Gemini API key is configured.

    Call this right before any code path that actually needs to reach
    Gemini (embeddings or chat), rather than at app startup.
    """
    if not cfg.has_api_key:
        raise ConfigError(
            "No Gemini API key found. Copy .env.example to .env, set "
            "GOOGLE_API_KEY=your_key_here, then restart the app. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
