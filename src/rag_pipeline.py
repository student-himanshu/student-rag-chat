"""
src/rag_pipeline.py
RAG pipeline orchestration for Student RAG Chat.

Chains the three stages that already exist and are independently
verified: src.retriever.retrieve() -> src.prompts.build_system_prompt()
-> src.llm.generate_answer(). This module's only real job is wiring
those three together behind a single error type and a single return
value, the same way src.retriever wired src.embeddings + src.vector_store
together behind RetrieverError -- see that module's docstring for the
precedent this follows.

Deliberately thin: no new retrieval, prompting, or generation logic
lives here. If a bug shows up in what chunks come back, what the system
prompt says, or what Gemini answers, the fix belongs in retriever.py /
prompts.py / llm.py respectively, not here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.config import Config, ConfigError
from src.llm import LLMError, generate_answer, get_llm_client
from src.prompts import PromptError, build_system_prompt
from src.retriever import RetrieverError, retrieve
from src.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class RAGPipelineError(RuntimeError):
    """Raised when a question can't be answered end-to-end.

    Wraps RetrieverError / PromptError / ConfigError / LLMError so
    Phase 10's chat UI has exactly one exception type to catch, instead
    of needing to know about (and import) three other modules' error
    types individually. The original exception is always chained via
    `from exc` so the root cause is never lost -- same convention every
    other module in this project follows.
    """


@dataclass(frozen=True)
class RAGResult:
    """The final answer plus the chunks it was grounded in.

    `sources` is exactly what retrieve() returned: SearchResult objects
    carrying {source, page, chunk_index, chunk_id} metadata plus a
    similarity score. Phase 11's citation UI can read
    `result.document.metadata["source"]` / `["page"]` straight off each
    entry without a second lookup back into the vector store.

    `mode` is carried through mainly so a caller building chat history
    (Phase 10/12) can record which mode produced a given answer without
    having to separately remember what it passed in.
    """

    answer: str
    sources: list[SearchResult]
    mode: str


def answer_question(
    query: str,
    store: VectorStore,
    cfg: Config,
    mode: str = "Normal Chat",
    question_type: str | None = None,
    k: int | None = None,
    embeddings_client: GoogleGenerativeAIEmbeddings | None = None,
    llm_client: ChatGoogleGenerativeAI | None = None,
    source_filter: set[str] | None = None,
) -> RAGResult:
    """Answer a student's question grounded in the uploaded documents.

    Runs, in order: retrieve top-k chunks for `query` from `store`,
    build a mode-specific system prompt from those chunks, then send
    `query` plus that system prompt to Gemini and return its answer
    alongside the chunks it was grounded in.

    Args:
        query: the student's question, as typed. Validated by
            retrieve() (raises before any Gemini call for an empty or
            whitespace-only question) -- not re-validated here, so
            there's exactly one place that rule lives.
        store: a VectorStore already populated via build_vector_store()
            or VectorStore.load(). An empty store is valid input: no
            chunks come back, format_context() renders that as "(No
            relevant material was found...)", and Gemini still answers
            -- typically by saying the material isn't in the documents,
            per src.prompts' base instructions. That's a normal outcome,
            not an error.
        cfg: app Config, forwarded to retrieve() (for the embeddings
            client / default top_k) and to get_llm_client() (for the
            chat client), whenever a caller doesn't supply its own
            client for either.
        mode: one of src.prompts.VALID_MODES. Defaults to "Normal Chat".
        question_type: only consulted for mode == "Question Generator";
            forwarded as-is to build_system_prompt(), which itself
            ignores it for every other mode.
        k: chunks to retrieve. Defaults to cfg.top_k (via retrieve()).
        embeddings_client: an existing embeddings client to reuse
            instead of building a fresh one. Pass the same client across
            a Streamlit session's questions -- see src.retriever's
            docstring on why this matters (client construction is
            deliberately split from the embedding calls precisely so it
            can be reused).
        llm_client: an existing chat client to reuse instead of building
            a fresh one, for the same session-reuse reason.
        source_filter: Phase 14's per-document chat filter -- restricts
            retrieval to chunks from these source filenames. Forwarded
            straight through to retrieve() / VectorStore.search(); an
            empty set retrieves nothing (which still produces a normal
            answer -- see the empty-store case below), and `None` (the
            default) searches every indexed document, matching every
            caller before Phase 14.

    Returns:
        A RAGResult with the generated answer, the SearchResults it was
        grounded in (possibly empty), and the mode that produced it.

    Raises:
        RAGPipelineError: retrieval failed (bad query, no API key,
            embedding call failed, FAISS search failed), the mode or
            question_type was invalid, no API key was configured for
            the chat client, or the Gemini chat call failed / returned
            an empty response. In every case the original exception
            (RetrieverError / PromptError / ConfigError / LLMError) is
            available via `__cause__`.
    """
    try:
        results = retrieve(
            query, store, cfg, client=embeddings_client, k=k, allowed_sources=source_filter
        )
    except RetrieverError as exc:
        raise RAGPipelineError(f"Couldn't retrieve context: {exc}") from exc

    context_chunks = [result.document for result in results]

    try:
        system_prompt = build_system_prompt(mode, context_chunks, question_type=question_type)
    except PromptError as exc:
        raise RAGPipelineError(f"Couldn't build the prompt: {exc}") from exc

    if llm_client is None:
        try:
            llm_client = get_llm_client(cfg)
        except ConfigError as exc:
            raise RAGPipelineError(str(exc)) from exc

    try:
        answer = generate_answer(system_prompt, query, llm_client)
    except (ValueError, LLMError) as exc:
        raise RAGPipelineError(f"Couldn't generate an answer: {exc}") from exc

    logger.info(
        "Answered question in mode=%r using %d source chunk(s)", mode, len(results)
    )
    return RAGResult(answer=answer, sources=results, mode=mode)
