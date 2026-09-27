"""
Tests for src.prompts (Phase 8).

No Gemini call anywhere in this module -- build_system_prompt() and
format_context() are pure string assembly -- so unlike test_llm.py
nothing here needs mocking. The only external dependency is
langchain_core.documents.Document.
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.prompts import (
    VALID_MODES,
    VALID_QUESTION_TYPES,
    PromptError,
    build_system_prompt,
    format_context,
)


def make_chunk(text: str, source="a.pdf", page=1) -> Document:
    return Document(page_content=text, metadata={"source": source, "page": page})


# --- format_context ---------------------------------------------------------


class TestFormatContext:
    def test_empty_list_returns_placeholder_not_error(self):
        result = format_context([])
        assert "No relevant material" in result

    def test_single_chunk_includes_source_page_and_text(self):
        result = format_context([make_chunk("Mitochondria are the powerhouse.", "bio.pdf", 12)])
        assert "bio.pdf" in result
        assert "p. 12" in result
        assert "Mitochondria are the powerhouse." in result
        assert "[1]" in result

    def test_multiple_chunks_are_numbered_in_order(self):
        chunks = [make_chunk("first", page=1), make_chunk("second", page=2), make_chunk("third", page=3)]
        result = format_context(chunks)
        assert result.index("[1]") < result.index("[2]") < result.index("[3]")

    def test_missing_metadata_falls_back_gracefully(self):
        bare = Document(page_content="no metadata here", metadata={})
        result = format_context([bare])
        assert "unknown source" in result
        assert "p. ?" in result


# --- build_system_prompt -----------------------------------------------------


class TestBuildSystemPrompt:
    @pytest.mark.parametrize("mode", VALID_MODES)
    def test_every_valid_mode_builds_without_error(self, mode):
        prompt = build_system_prompt(mode, [make_chunk("some fact")])
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_unknown_mode_raises_prompt_error(self):
        with pytest.raises(PromptError, match="Unknown mode"):
            build_system_prompt("Sarcastic Mode", [make_chunk("x")])

    def test_context_is_embedded_in_prompt(self):
        prompt = build_system_prompt("Normal Chat", [make_chunk("Photosynthesis converts light.", "bio.pdf", 4)])
        assert "Photosynthesis converts light." in prompt
        assert "bio.pdf" in prompt

    def test_empty_context_chunks_still_builds_a_prompt(self):
        prompt = build_system_prompt("Normal Chat", [])
        assert "No relevant material" in prompt

    def test_mode_instruction_text_appears_for_each_mode(self):
        assert "simply" in build_system_prompt("Explain Simply", []).lower()
        assert "step by step" in build_system_prompt("Problem Solver", []).lower()
        assert "practice questions" in build_system_prompt("Question Generator", []).lower()

    def test_question_type_ignored_for_non_generator_modes(self):
        # Should not raise, and should not mention MCQ instructions.
        prompt = build_system_prompt("Normal Chat", [], question_type="mcq")
        assert "multiple-choice" not in prompt.lower()

    @pytest.mark.parametrize("qtype", VALID_QUESTION_TYPES)
    def test_valid_question_types_apply_for_generator_mode(self, qtype):
        prompt = build_system_prompt("Question Generator", [], question_type=qtype)
        assert prompt  # built without error; spot-check one below

    def test_mcq_question_type_mentions_multiple_choice(self):
        prompt = build_system_prompt("Question Generator", [], question_type="mcq")
        assert "multiple-choice" in prompt.lower()

    def test_question_generator_without_question_type_still_builds(self):
        prompt = build_system_prompt("Question Generator", [])
        assert "practice questions" in prompt.lower()

    def test_unknown_question_type_raises_prompt_error(self):
        with pytest.raises(PromptError, match="Unknown question_type"):
            build_system_prompt("Question Generator", [], question_type="essay")

    def test_base_instructions_present_in_every_prompt(self):
        prompt = build_system_prompt("Exam Mode", [])
        assert "cite" in prompt.lower()
        assert "ONLY" in prompt
