"""
Tests for src.chat_memory (Phase 12).

Same fake-client approach test_llm.py uses: client.invoke() is the only
Gemini-touching call, so it's the only thing faked. No real API key or
network access required.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.chat_memory import (
    MAX_HISTORY_TURNS,
    ChatMemoryError,
    format_history,
    resolve_question,
)


class FakeChatClient:
    """Mirrors test_llm.py's FakeChatClient: full control over the
    response shape (plain string, list-of-blocks, empty, or an
    exception) without needing an installed langchain_google_genai to
    autospec against."""

    def __init__(self, content="Rewritten question?", raise_exc=None):
        self._content = content
        self._raise_exc = raise_exc
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        if self._raise_exc:
            raise self._raise_exc
        return SimpleNamespace(content=self._content)


# --- format_history ------------------------------------------------


class TestFormatHistory:
    def test_empty_history_returns_empty_string(self):
        assert format_history([]) == ""

    def test_renders_role_and_content(self):
        history = [
            {"role": "user", "content": "What is a mitochondria?", "sources": []},
            {"role": "assistant", "content": "It's the powerhouse of the cell.", "sources": []},
        ]
        rendered = format_history(history)
        assert "User: What is a mitochondria?" in rendered
        assert "Assistant: It's the powerhouse of the cell." in rendered

    def test_truncates_to_max_turns_pairs(self):
        # 5 user/assistant pairs (10 messages); max_turns=2 keeps only the last 4 messages.
        history = []
        for i in range(5):
            history.append({"role": "user", "content": f"q{i}", "sources": []})
            history.append({"role": "assistant", "content": f"a{i}", "sources": []})
        rendered = format_history(history, max_turns=2)
        assert "q4" in rendered and "a4" in rendered
        assert "q3" in rendered and "a3" in rendered
        assert "q2" not in rendered and "a2" not in rendered

    def test_default_max_turns_constant_used(self):
        history = []
        for i in range(MAX_HISTORY_TURNS + 2):
            history.append({"role": "user", "content": f"q{i}", "sources": []})
            history.append({"role": "assistant", "content": f"a{i}", "sources": []})
        rendered = format_history(history)
        oldest_kept_index = MAX_HISTORY_TURNS + 1  # zero-indexed, last MAX_HISTORY_TURNS pairs
        assert f"q{oldest_kept_index}" in rendered
        assert "q0" not in rendered

    def test_ignores_sources_key(self):
        history = [{"role": "assistant", "content": "answer", "sources": ["something"]}]
        rendered = format_history(history)
        assert "something" not in rendered


# --- resolve_question ------------------------------------------------


class TestResolveQuestion:
    def test_empty_history_returns_question_unchanged_without_calling_client(self):
        client = FakeChatClient()
        result = resolve_question("What is mitosis?", [], client)
        assert result == "What is mitosis?"
        assert client.last_messages is None  # no Gemini call made

    @pytest.mark.parametrize("question", ["", "   ", "\n\t"])
    def test_empty_question_raises_value_error(self, question):
        client = FakeChatClient()
        history = [{"role": "user", "content": "hi", "sources": []}]
        with pytest.raises(ValueError):
            resolve_question(question, history, client)
        assert client.last_messages is None

    def test_rewrites_followup_using_history(self):
        client = FakeChatClient(content="Why is mitochondria's role in energy production useful?")
        history = [
            {"role": "user", "content": "What is a mitochondria?", "sources": []},
            {"role": "assistant", "content": "It's the powerhouse of the cell.", "sources": []},
        ]
        result = resolve_question("Why is it useful?", history, client)
        assert result == "Why is mitochondria's role in energy production useful?"

    def test_sends_system_and_human_messages_separately(self):
        client = FakeChatClient()
        history = [{"role": "user", "content": "hi", "sources": []}]
        resolve_question("what about that?", history, client)
        roles = [type(m).__name__ for m in client.last_messages]
        assert roles == ["SystemMessage", "HumanMessage"]

    def test_client_exception_is_wrapped_as_chat_memory_error(self):
        client = FakeChatClient(raise_exc=RuntimeError("boom"))
        history = [{"role": "user", "content": "hi", "sources": []}]
        with pytest.raises(ChatMemoryError):
            resolve_question("what about that?", history, client)

    @pytest.mark.parametrize("content", ["", "   ", None])
    def test_empty_response_raises_chat_memory_error(self, content):
        client = FakeChatClient(content=content)
        history = [{"role": "user", "content": "hi", "sources": []}]
        with pytest.raises(ChatMemoryError):
            resolve_question("what about that?", history, client)

    def test_list_content_with_thinking_blocks_is_normalized(self):
        client = FakeChatClient(
            content=[
                {"type": "thinking", "thinking": "reasoning about pronouns..."},
                {"type": "text", "text": "Why is mitosis important for growth?"},
            ]
        )
        history = [{"role": "user", "content": "Tell me about mitosis.", "sources": []}]
        result = resolve_question("why is it important?", history, client)
        assert result == "Why is mitosis important for growth?"

    def test_standalone_question_returned_unchanged_if_model_echoes_it(self):
        client = FakeChatClient(content="What is the capital of France?")
        history = [{"role": "user", "content": "hi", "sources": []}]
        result = resolve_question("What is the capital of France?", history, client)
        assert result == "What is the capital of France?"
