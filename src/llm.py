"""
LLM wrapper for Student RAG Chat.

Wraps langchain_google_genai.ChatGoogleGenerativeAI to turn a built
system/user prompt pair (src.prompts) into a student-facing answer.
Deliberately mirrors src.embeddings' shape: client construction
(get_llm_client) is separated from the actual call (generate_answer) so
a Streamlit session can build the client once and reuse it across a
whole conversation, and so tests can inject a fake client instead of
hitting the real Gemini API -- same reasoning as Phase 5, same reasoning
Phase 7 leaned on when it added an optional `client` parameter to
retrieve().
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import Config, require_api_key

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the Gemini chat API can't be reached or returns
    something the rest of the pipeline can't use (empty response, a
    client-side error, etc.)."""


def get_llm_client(cfg: Config) -> ChatGoogleGenerativeAI:
    """Build a Gemini chat client from app config.

    Raises:
        ConfigError: if no Gemini API key is configured (via
            require_api_key), matching get_embeddings_client's contract
            so both client-construction functions fail the same way.
    """
    require_api_key(cfg)
    return ChatGoogleGenerativeAI(
        model=cfg.gemini_model,
        google_api_key=cfg.google_api_key,
        temperature=cfg.temperature,
    )


def _extract_text(content) -> str:
    """Normalize AIMessage.content into a plain string.

    Discovered live (not guessed in advance): against the real Gemini
    API, `gemini-3.5-flash-lite` -- a model with extended thinking /
    reasoning support -- can come back through
    `langchain_google_genai` as a *list* of content blocks (each a dict
    like {"type": "text", "text": "..."} or {"type": "thinking", ...})
    rather than a plain string. Before this fix, generate_answer() did
    `len(content)` on that list directly, so a perfectly good answer
    wrapped in a single-item list logged as "(1 chars)" -- 1 *block*,
    not 1 character -- and the raw list/dicts would have been handed
    to the Streamlit UI instead of readable text.

    This collapses either shape into plain text once, here, so every
    caller (this module's own logging, src.rag_pipeline, the Streamlit
    UI) only ever has to deal with a string. Non-"text" blocks (e.g.
    "thinking") are dropped -- students should see the answer, not the
    model's internal reasoning trace.
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


def generate_answer(system_prompt: str, user_prompt: str, client: ChatGoogleGenerativeAI) -> str:
    """Send a system/user prompt pair to Gemini and return the answer text.

    Args:
        system_prompt: mode instructions + retrieved context, built by
            src.prompts.build_system_prompt(). Kept as a separate
            SystemMessage rather than concatenated into the user prompt
            -- every LangChain chat model treats system vs human messages
            differently, and collapsing them would throw that distinction
            away for no benefit.
        user_prompt: the student's question, verbatim (already validated
            non-empty by src.retriever.retrieve() upstream, but checked
            again here since this function is also callable on its own,
            e.g. directly from tests or a future non-RAG "ask anything"
            mode).
        client: a chat client from get_llm_client.

    Returns:
        The model's answer text, always a plain string -- see
        _extract_text() for why that's not simply `response.content`.

    Raises:
        ValueError: if user_prompt is empty or whitespace-only.
        LLMError: if the API call fails, or the response has no usable
            text content once normalized.
    """
    if not user_prompt or not user_prompt.strip():
        raise ValueError("Cannot generate an answer for an empty question.")

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    try:
        response: AIMessage = client.invoke(messages)
    except Exception as exc:
        raise LLMError(f"Failed to generate an answer: {exc}") from exc

    content = _extract_text(response.content)
    if not content or not content.strip():
        raise LLMError("Gemini returned an empty response.")

    logger.info("Generated answer (%d chars) using model=%s", len(content), client.model)
    return content