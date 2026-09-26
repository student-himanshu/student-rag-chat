"""
Chat memory for Student RAG Chat.

Turns a possibly-elliptical follow-up ("why is it useful?") into a
standalone question ("why is mitochondria's role in energy production
useful?") before it reaches src.retriever.retrieve() -- embeddings and
FAISS have no concept of conversation history, so an unresolved pronoun
means the wrong (or no) chunks come back for a follow-up. This is a
rewrite step in front of the existing pipeline, not a new pipeline: the
rewritten question still goes through src.rag_pipeline.answer_question()
exactly as before, unchanged.

Deliberately a single extra Gemini call using the same chat client
Phase 8/9 already build (get_llm_client) and app.py already caches in
session_state, rather than a separate model or a rule-based pronoun
resolver -- regex-matching "it"/"that"/"this" across arbitrary student
phrasing is exactly the kind of task an LLM is reliable at and
hand-rolled heuristics aren't. If that call fails, the caller (app.py)
is expected to fall back to the raw question rather than blocking the
chat -- see ChatMemoryError's docstring.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 3  # user+assistant pairs; keeps the rewrite prompt small and cheap

_CONDENSE_INSTRUCTIONS = (
    "You rewrite a student's latest chat message into a standalone question, "
    "using the conversation history only to resolve pronouns, references, or "
    "missing context (e.g. \"why is it useful?\" -> \"why is mitochondria's role "
    "in energy production useful?\"). Output ONLY the rewritten question, with "
    "no preamble, no quotes, and no explanation. If the latest message is "
    "already standalone and doesn't depend on the history, output it "
    "unchanged. Never answer the question yourself -- only rewrite it."
)


class ChatMemoryError(RuntimeError):
    """Raised when a follow-up question can't be resolved against chat
    history -- the Gemini call failed, or came back with nothing usable.

    Deliberately its own error type rather than reusing LLMError, even
    though both wrap a client.invoke() failure -- this failure is
    recoverable in a way an answer-generation failure isn't: app.py
    catches this and falls back to the raw, unresolved question rather
    than failing the whole turn, so a broken follow-up rewrite degrades
    to "no memory this turn," not "no answer at all."
    """


def format_history(chat_history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> str:
    """Render the last `max_turns` user/assistant pairs as plain text.

    Takes app.py's session_state.chat_history shape directly
    ({"role": "user"|"assistant", "content": str, ...}) rather than a
    new type, so no conversion layer sits between session_state and
    this module. Only role+content are read; a message's "sources" key
    (added in Phase 11) has nothing to do with resolving a pronoun and
    is ignored here.

    Returns "" for an empty history rather than a placeholder string --
    resolve_question() uses an empty result to skip the rewrite call
    entirely, so this has to be a plain falsy value, not text.
    """
    if not chat_history:
        return ""
    turns = chat_history[-(max_turns * 2):]
    lines = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in turns]
    return "\n".join(lines)


def _extract_text(content) -> str:
    """Normalize AIMessage.content the same way src.llm._extract_text does.

    Duplicated rather than imported -- it's a private helper of that
    module, not a shared utility, and this module has the same
    dependency on it that src.llm does: Gemini's thinking-capable
    models can return `.content` as a list of {"type": ...} blocks
    instead of a plain string (see src.llm's docstring, discovered live
    against the real API in Phase 10). If this normalization ever needs
    to change, change it in both places.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def resolve_question(
    question: str,
    chat_history: list[dict],
    llm_client: ChatGoogleGenerativeAI,
    max_turns: int = MAX_HISTORY_TURNS,
) -> str:
    """Rewrite `question` into a standalone question if it depends on
    prior turns; otherwise return it unchanged.

    Args:
        question: the student's latest message, as typed.
        chat_history: session_state.chat_history so far -- NOT including
            `question` itself (app.py calls this before appending the
            new user turn, and passes chat_history[:-1] after appending
            it, to avoid the message resolving against itself). Only
            the last `max_turns` pairs are used.
        llm_client: an existing chat client from src.llm.get_llm_client,
            reused rather than rebuilt -- same reasoning as every other
            client in this project (Phase 5/8's docstrings).
        max_turns: how many prior user/assistant pairs to include.

    Returns:
        `question` unchanged if chat_history is empty -- there's
        nothing to resolve against, and no Gemini call is made, so a
        student's very first message never costs an extra API call.
        Otherwise, the rewritten standalone question.

    Raises:
        ValueError: question is empty or whitespace-only.
        ChatMemoryError: the Gemini call failed, or returned an empty
            response. Callers should treat this as recoverable (see
            ChatMemoryError's docstring) rather than fatal.
    """
    if not question or not question.strip():
        raise ValueError("Cannot resolve an empty question.")

    history_text = format_history(chat_history, max_turns=max_turns)
    if not history_text:
        return question

    user_prompt = f"Conversation so far:\n{history_text}\n\nLatest message: {question}"
    messages = [SystemMessage(content=_CONDENSE_INSTRUCTIONS), HumanMessage(content=user_prompt)]

    try:
        response = llm_client.invoke(messages)
    except Exception as exc:
        raise ChatMemoryError(f"Failed to resolve follow-up question: {exc}") from exc

    rewritten = _extract_text(response.content).strip()
    if not rewritten:
        raise ChatMemoryError("Gemini returned an empty rewritten question.")

    if rewritten != question:
        logger.info("Resolved follow-up %r -> %r", question, rewritten)
    return rewritten
