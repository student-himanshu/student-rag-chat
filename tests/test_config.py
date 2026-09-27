"""Unit tests for src/config.py.

These run with no network access and no real Gemini API key -- they only
exercise environment-variable parsing and validation logic.
"""
from __future__ import annotations

import importlib

import pytest

import src.config as config_module


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Ensure each test starts with a blank slate, independent of whatever
    .env file happens to exist on the machine running the tests."""
    for var in (
        "GOOGLE_API_KEY",
        "GEMINI_MODEL",
        "EMBEDDING_MODEL",
        "TEMPERATURE",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
    ):
        monkeypatch.delenv(var, raising=False)
    # Prevent a real .env file from being picked up during tests.
    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: None)
    yield


def test_defaults_used_when_env_vars_absent():
    cfg = config_module.load_config()
    assert cfg.gemini_model == config_module.DEFAULT_GEMINI_MODEL
    assert cfg.embedding_model == config_module.DEFAULT_EMBEDDING_MODEL
    assert cfg.chunk_size == config_module.DEFAULT_CHUNK_SIZE
    assert cfg.chunk_overlap == config_module.DEFAULT_CHUNK_OVERLAP
    assert cfg.top_k == config_module.DEFAULT_TOP_K
    assert cfg.has_api_key is False


def test_missing_api_key_does_not_raise_on_load():
    # The app must still start and show a UI warning, not crash outright.
    cfg = config_module.load_config()
    assert cfg.google_api_key == ""
    assert cfg.has_api_key is False


def test_require_api_key_raises_friendly_error_when_missing():
    cfg = config_module.load_config()
    with pytest.raises(config_module.ConfigError, match="No Gemini API key"):
        config_module.require_api_key(cfg)


def test_require_api_key_passes_when_present(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-test-key")
    cfg = config_module.load_config()
    assert cfg.has_api_key is True
    config_module.require_api_key(cfg)  # should not raise


def test_env_overrides_are_respected(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    monkeypatch.setenv("TOP_K", "8")
    cfg = config_module.load_config()
    assert cfg.gemini_model == "gemini-3.8-flash"
    assert cfg.chunk_size == 500
    assert cfg.chunk_overlap == 50
    assert cfg.top_k == 8


def test_invalid_numeric_env_var_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
    cfg = config_module.load_config()
    assert cfg.chunk_size == config_module.DEFAULT_CHUNK_SIZE


def test_chunk_overlap_greater_than_chunk_size_raises(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "150")
    with pytest.raises(config_module.ConfigError, match="CHUNK_OVERLAP"):
        config_module.load_config()


def test_zero_top_k_raises(monkeypatch):
    monkeypatch.setenv("TOP_K", "0")
    with pytest.raises(config_module.ConfigError, match="TOP_K"):
        config_module.load_config()


def test_config_repr_never_leaks_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "super-secret-value")
    cfg = config_module.load_config()
    assert "super-secret-value" not in repr(cfg)
    assert "***set***" in repr(cfg)
