"""
Prompt building for Student RAG Chat.

Turns retrieved chunks (src.retriever) plus a student mode into the
system prompt src.llm.generate_answer() sends to Gemini. Depends only on
langchain_core.documents.Document, not on src.vector_store.SearchResult --
callers pass `[r.document for r in results]`, which keeps this module
usable anywhere a plain chunk list exists, not just after a FAISS search.

Mode strings are NOT invented here -- they're copied verbatim from
app.py's already-built (currently disabled) mode selectbox, so Phase 13
can pass st.session_state.mode straight into build_system_prompt() with
no translation layer:
    "Normal Chat", "Explain Simply", "Exam Mode", "Problem Solver",
    "Summary", "Question Generator"
If app.py's selectbox options ever change, VALID_MODES here must change
with them, or Phase 13's wiring breaks silently.
"""
from __future__ import annotations

from langchain_core.documents import Document

VALID_MODES = (
    "Normal Chat",
    "Explain Simply",
    "Exam Mode",
    "Problem Solver",
    "Summary",
    "Question Generator",
)

VALID_QUESTION_TYPES = ("mcq", "short", "long")


class PromptError(RuntimeError):
    """Raised when a prompt can't be built: an unknown mode or
    question_type was passed."""


_BASE_INSTRUCTIONS = (
    "You are a study assistant. Answer the student's question using ONLY "
    "the context below, drawn from PDFs they uploaded. If the answer "
    "isn't in the context, say so plainly instead of guessing or using "
    "outside knowledge. When you state something from the context, cite "
    "its source and page in the form (source, p. N)."
)

_MODE_INSTRUCTIONS: dict[str, str] = {
    "Normal Chat": "Answer conversationally and directly, at a normal level of detail.",
    "Explain Simply": (
        "Explain the answer as simply as possible, as if to someone new to "
        "the subject. Avoid jargon; define any term you must use."
    ),
    "Exam Mode": (
        "Answer the way a strong exam response would: precise, structured, "
        "and using the terminology an instructor would expect, without "
        "extra padding."
    ),
    "Problem Solver": (
        "Work through the problem step by step, showing the reasoning for "
        "each step before moving to the next, and state the final answer "
        "clearly at the end."
    ),
    "Summary": (
        "Summarize the relevant material concisely, in your own words, "
        "covering the key points a student would need to know."
    ),
    "Question Generator": (
        "Generate practice questions from the context instead of directly "
        "answering the student's message; treat their message as a topic "
        "or instruction (e.g. 'chapter 3' or 'focus on mitosis'), not a "
        "question to answer."
    ),
}

_QUESTION_TYPE_INSTRUCTIONS: dict[str, str] = {
    "mcq": "Write multiple-choice questions, each with 4 options and the correct answer marked.",
    "short": "Write short-answer questions (answerable in a sentence or two each).",
    "long": "Write long-answer / essay-style questions that require a full paragraph or more.",
}


def format_context(chunks: list[Document]) -> str:
    """Render retrieved chunks into a numbered, citeable context block.

    Each chunk is labeled [n] (source, p. page) so the model has a ready
    citation to attach to any claim, and so a citation the model writes
    can be matched back to a specific chunk if Phase 11 ever needs to.

    Returns a placeholder string for an empty chunk list rather than
    raising -- "nothing relevant was retrieved" is a normal, expected
    state (a question genuinely not covered by the uploaded PDFs), and
    the resulting prompt should let the model say so, not fail to build.
    """
    if not chunks:
        return "(No relevant material was found in the uploaded documents.)"

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown source")
        page = chunk.metadata.get("page", "?")
        lines.append(f"[{i}] ({source}, p. {page})\n{chunk.page_content}")
    return "\n\n".join(lines)


def build_system_prompt(
    mode: str,
    context_chunks: list[Document],
    question_type: str | None = None,
) -> str:
    """Build the system prompt for a given student mode and retrieved context.

    Args:
        mode: one of VALID_MODES, matched exactly (see module docstring
            on why this isn't a looser/case-insensitive match).
        context_chunks: chunks retrieved for this question -- typically
            `[r.document for r in retrieve(...)]` -- ordered most
            relevant first. An empty list is valid input.
        question_type: only consulted when mode == "Question Generator";
            one of VALID_QUESTION_TYPES, or None for a mixed set of
            question styles. Ignored (not an error) for every other mode,
            since a student mode selector and a question-type selector
            are independent controls in the UI and shouldn't have to be
            kept in sync by the caller.

    Raises:
        PromptError: mode isn't one of VALID_MODES, or question_type is
            given but not one of VALID_QUESTION_TYPES.
    """
    if mode not in _MODE_INSTRUCTIONS:
        raise PromptError(f"Unknown mode {mode!r}. Must be one of: {', '.join(VALID_MODES)}.")
    if question_type is not None and question_type not in _QUESTION_TYPE_INSTRUCTIONS:
        raise PromptError(
            f"Unknown question_type {question_type!r}. Must be one of: "
            f"{', '.join(VALID_QUESTION_TYPES)}, or None."
        )

    mode_instruction = _MODE_INSTRUCTIONS[mode]
    if mode == "Question Generator" and question_type is not None:
        mode_instruction = f"{mode_instruction} {_QUESTION_TYPE_INSTRUCTIONS[question_type]}"

    context = format_context(context_chunks)

    return (
        f"{_BASE_INSTRUCTIONS}\n\n"
        f"Mode instructions: {mode_instruction}\n\n"
        f"Context:\n{context}"
    )
