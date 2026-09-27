"""
src/embeddings.py
Embeddings for Student RAG Chat.

Wraps `langchain_google_genai.GoogleGenerativeAIEmbeddings` to turn
chunk text (src.chunking.Document objects) and student questions into
vectors for Phase 6's FAISS index.

Client construction is separated from embedding calls (get_embeddings_
client vs embed_chunks/embed_query) so the client can be built once per
Streamlit session and reused, and so tests can inject a fake client
instead of hitting the real Gemini API.

Note on task_type: `GoogleGenerativeAIEmbeddings.embed_documents` and
`.embed_query` already default their `task_type` to "RETRIEVAL_DOCUMENT"
and "RETRIEVAL_QUERY" respectively when none is passed, which is exactly
the asymmetric embedding setup this app needs -- a chunk and the
question that should retrieve it are not embedded identically. We rely
on those defaults rather than re-specifying them here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import Config, require_api_key

logger = logging.getLogger(__name__)

# Google Generative AI's current max batch size for embed_content.
DEFAULT_BATCH_SIZE = 100


class EmbeddingError(RuntimeError):
    """Raised when the Gemini embeddings API can't be reached or returns
    something the rest of the pipeline can't use (wrong vector count,
    a client-side error, etc.)."""


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk paired with its embedding vector, ready for Phase 6's
    FAISS index to store."""

    chunk: Document
    vector: list[float]


def get_embeddings_client(cfg: Config) -> GoogleGenerativeAIEmbeddings:
    """Build a Gemini embeddings client from app config.

    Raises:
        ConfigError: if no Gemini API key is configured (via
            require_api_key), so callers get the same friendly error
            message used everywhere else in the app rather than a raw
            client-library exception.
    """
    require_api_key(cfg)
    return GoogleGenerativeAIEmbeddings(
        model=cfg.embedding_model,
        google_api_key=cfg.google_api_key,
    )


def embed_chunks(
    chunks: list[Document],
    client: GoogleGenerativeAIEmbeddings,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[EmbeddedChunk]:
    """Embed a list of chunks, preserving order.

    Args:
        chunks: chunk Documents from src.chunking.chunk_documents.
        client: an embeddings client from get_embeddings_client.
        batch_size: chunks sent to the API per request (Gemini's
            current max is 100; the client batches internally, this
            just controls the batch size it uses).

    Returns:
        One EmbeddedChunk per input chunk, in the same order -- callers
        can zip() this 1:1 against `chunks` if needed. Empty input
        returns an empty list without calling the API.

    Raises:
        ValueError: if any chunk's page_content is empty or whitespace
            -- embedding nothing is always a bug upstream (chunking
            should never produce an empty chunk), not something to
            silently skip and lose a citation for.
        EmbeddingError: if the API call fails, or returns a different
            number of vectors than chunks sent (would silently
            misalign chunk/vector pairs downstream otherwise).
    """
    if not chunks:
        return []

    for i, chunk in enumerate(chunks):
        if not chunk.page_content or not chunk.page_content.strip():
            raise ValueError(f"Chunk at index {i} has empty page_content; cannot embed it.")

    texts = [chunk.page_content for chunk in chunks]

    try:
        vectors = client.embed_documents(texts, batch_size=batch_size)
    except Exception as exc:
        raise EmbeddingError(f"Failed to embed {len(chunks)} chunk(s): {exc}") from exc

    if len(vectors) != len(chunks):
        raise EmbeddingError(
            f"Embeddings API returned {len(vectors)} vector(s) for "
            f"{len(chunks)} chunk(s) -- can't safely pair them up."
        )

    logger.info("Embedded %d chunk(s) using model=%s", len(chunks), client.model)
    return [EmbeddedChunk(chunk=c, vector=v) for c, v in zip(chunks, vectors)]


def embed_query(text: str, client: GoogleGenerativeAIEmbeddings) -> list[float]:
    """Embed a single student question for similarity search.

    Raises:
        ValueError: if text is empty or whitespace-only.
        EmbeddingError: if the API call fails.
    """
    if not text or not text.strip():
        raise ValueError("Cannot embed an empty question.")

    try:
        return client.embed_query(text)
    except Exception as exc:
        raise EmbeddingError(f"Failed to embed query: {exc}") from exc
