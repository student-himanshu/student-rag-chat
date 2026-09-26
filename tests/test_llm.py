"""
Tests for src.llm (Phase 8, patched post-Phase-10 live run).

client.invoke() is the only Gemini-touching call, so it's the only thing
mocked here -- same line the project has drawn since Phase 5. Response
objects are plain fakes with a `.content` attribute rather than real
AIMessage instances, since generate_answer() only ever reads `.content`
and this keeps the test file free of a langchain_core.messages import.

The TestExtractText / list-content cases below were added after a real
end-to-end run against the live Gemini API logged "Generated answer (1
chars)" for gemini-3.5-flash-lite -- a thinking-capable model that
returns AIMessage.content as a list of content blocks rather than a
plain string. Written to reproduce that shape, not guessed speculatively.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.config import Config, ConfigError
from src.llm import LLMError, generate_answer, get_llm_client


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        google_api_key="fake-key-for-tests",
        gemini_model="gemini-3.5-flash-lite",
        embedding_model="models/gemini-embedding-001",
        temperature=0.2,
        chunk_size=1000,
        chunk_overlap=150,
        top_k=4,
        upload_dir=tmp_path / "uploads",
        vectorstore_dir=tmp_path / "vectorstore",
    )


class FakeChatClient:
    """Hand-written fake standing in for ChatGoogleGenerativeAI.

    Mirrors test_embeddings.py's approach: a fake for full control over
    edge cases (empty response, exceptions, list-shaped content), since
    there's no installed langchain_google_genai in this sandbox to
    autospec against (see README's Verification report on this phase).
    """

    def __init__(self, content="A fake answer.", model="gemini-3.5-flash-lite", raise_exc=None):
        self._content = content
        self.model = model
        self._raise_exc = raise_exc
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        if self._raise_exc:
            raise self._raise_exc
        return SimpleNamespace(content=self._content)


class TestGetLlmClient:
    def test_missing_api_key_raises_config_error(self, cfg, mocker):
        mocker.patch("src.llm.ChatGoogleGenerativeAI")
        cfg_no_key = replace(cfg, google_api_key="")
        with pytest.raises(ConfigError):
            get_llm_client(cfg_no_key)

    def test_builds_client_with_config_values(self, cfg, mocker):
        mock_cls = mocker.patch("src.llm.ChatGoogleGenerativeAI")
        get_llm_client(cfg)
        mock_cls.assert_called_once_with(
            model=cfg.gemini_model,
            google_api_key=cfg.google_api_key,
            temperature=cfg.temperature,
        )


class TestGenerateAnswer:
    def test_returns_response_content(self):
        client = FakeChatClient(content="Photosynthesis converts light into energy.")
        answer = generate_answer("system instructions", "what is photosynthesis?", client)
        assert answer == "Photosynthesis converts light into energy."

    def test_sends_system_and_human_messages_separately(self):
        client = FakeChatClient()
        generate_answer("SYSTEM PROMPT HERE", "USER QUESTION HERE", client)
        assert len(client.last_messages) == 2
        assert client.last_messages[0].content == "SYSTEM PROMPT HERE"
        assert client.last_messages[1].content == "USER QUESTION HERE"

    @pytest.mark.parametrize("bad_question", ["", "   ", "\n\t"])
    def test_empty_question_raises_value_error_without_calling_client(self, bad_question):
        client = FakeChatClient()
        with pytest.raises(ValueError, match="empty"):
            generate_answer("system", bad_question, client)
        assert client.last_messages is None

    def test_client_exception_is_wrapped_as_llm_error(self):
        client = FakeChatClient(raise_exc=RuntimeError("connection reset"))
        with pytest.raises(LLMError, match="Failed to generate") as exc_info:
            generate_answer("system", "a real question", client)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.parametrize("empty_content", ["", "   ", None])
    def test_empty_response_content_raises_llm_error(self, empty_content):
        client = FakeChatClient(content=empty_content)
        with pytest.raises(LLMError, match="empty response"):
            generate_answer("system", "a real question", client)

    # --- list-of-content-blocks cases (the live "1 chars" bug) -----------

    def test_single_text_block_list_is_returned_as_plain_string(self):
        client = FakeChatClient(content=[{"type": "text", "text": "Full grounded answer."}])
        answer = generate_answer("system", "a real question", client)
        assert answer == "Full grounded answer."

    def test_thinking_blocks_are_dropped_and_only_text_kept(self):
        client = FakeChatClient(
            content=[
                {"type": "thinking", "thinking": "internal reasoning the student shouldn't see"},
                {"type": "text", "text": "The final answer."},
            ]
        )
        answer = generate_answer("system", "a real question", client)
        assert answer == "The final answer."
        assert "internal reasoning" not in answer

    def test_multiple_text_blocks_are_joined(self):
        client = FakeChatClient(content=[{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}])
        answer = generate_answer("system", "a real question", client)
        assert answer == "Part one. Part two."

    def test_list_of_plain_strings_is_joined(self):
        client = FakeChatClient(content=["Hello ", "world."])
        answer = generate_answer("system", "a real question", client)
        assert answer == "Hello world."

    def test_empty_list_content_raises_llm_error(self):
        client = FakeChatClient(content=[])
        with pytest.raises(LLMError, match="empty response"):
            generate_answer("system", "a real question", client)

    def test_list_of_only_thinking_blocks_raises_llm_error(self):
        client = FakeChatClient(content=[{"type": "thinking", "thinking": "reasoning with no final text"}])
        with pytest.raises(LLMError, match="empty response"):
            generate_answer("system", "a real question", client)