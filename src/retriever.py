"""
src/retriever.py  
Retriever for Student RAG Chat.

Embeds a student's question with the same embedding model used for chunks
(src.embeddings.embed_query), then similarity-searches the FAISS store
(Phase 6) for the top-k most relevant chunks. Deliberately a thin
orchestration layer: src.embeddings owns the Gemini call, src.vector_store
owns the FAISS search, and this module's only real job is wiring the two
together behind a single error type, so Phase 9's rag_pipeline has one
exception to catch instead of three.
"""
from __future__ import annotations

import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import Config, ConfigError
from src.embeddings import EmbeddingError, embed_query, get_embeddings_client
from src.vector_store import SearchResult, VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


class RetrieverError(RuntimeError):
    """Raised when a question can't be embedded or searched against the store."""


def retrieve(
    query: str,
    store: VectorStore,
    cfg: Config,
    client: GoogleGenerativeAIEmbeddings | None = None,
    k: int | None = None,
    allowed_sources: set[str] | None = None,
) -> list[SearchResult]:
    """Return the k chunks most relevant to `query`, most relevant first.

    Args:
        query: the student's question, as typed -- not yet cleaned or
            otherwise transformed (Phase 12's pronoun resolution, if any,
            happens before this is called, not inside it).
        store: a VectorStore already populated via build_vector_store()
            or VectorStore.load(). An empty store is valid input.
        cfg: app Config. Used to build an embeddings client (via
            get_embeddings_client) when `client` isn't supplied, and to
            supply the default top_k when `k` isn't given.
        client: an existing GoogleGenerativeAIEmbeddings client. Pass one
            in when answering more than one question in the same
            Streamlit session -- src.embeddings' own module docstring
            says client construction is deliberately split from the
            embed calls so a session can build the client once (e.g.
            cache it in st.session_state) and reuse it. When omitted,
            retrieve() builds a fresh client from `cfg` for this call
            only, which is fine for a single question but wasteful if
            this is called once per turn of a chat.
        k: how many chunks to return. Defaults to cfg.top_k. Forwarded to
            VectorStore.search(), which already returns fewer than k
            results (down to []) rather than erroring if the store holds
            fewer chunks than requested -- that behavior is preserved
            here, not re-implemented.
        allowed_sources: Phase 14's per-document chat filter, forwarded
            straight through to VectorStore.search() -- see its
            docstring for exactly what `None` vs. a set vs. an empty set
            each mean. `None` (the default) searches every indexed
            document, matching every caller before Phase 14.

    Returns:
        SearchResults ordered most relevant first. Empty list if the
        store has no chunks yet -- not an error, since that's the normal
        state before any PDF has been processed, and the caller (Phase 9)
        is better placed to decide how to tell the student than this
        module is.

    Raises:
        RetrieverError: query is empty/whitespace-only, k isn't positive,
            no Gemini API key is configured, the embedding call fails, or
            the FAISS search fails. The original exception (ConfigError /
            ValueError / EmbeddingError / VectorStoreError) is chained
            via `from exc` so nothing about the root cause is lost.
    """
    if not query or not query.strip():
        raise RetrieverError("Cannot retrieve for an empty question.")

    effective_k = cfg.top_k if k is None else k
    if effective_k <= 0:
        raise RetrieverError(f"k must be a positive integer, got {effective_k}.")

    if client is None:
        try:
            client = get_embeddings_client(cfg)
        except ConfigError as exc:
            raise RetrieverError(str(exc)) from exc

    try:
        query_vector = embed_query(query, client)
    except (ValueError, EmbeddingError) as exc:
        raise RetrieverError(f"Couldn't embed the question: {exc}") from exc

    try:
        results = store.search(query_vector, k=effective_k, allowed_sources=allowed_sources)
    except VectorStoreError as exc:
        raise RetrieverError(f"Couldn't search the document index: {exc}") from exc

    logger.info("Retrieved %d chunk(s) for query %r (k=%d)", len(results), query, effective_k)
    return results
