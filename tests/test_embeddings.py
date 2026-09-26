"""Unit tests for src/embeddings.py.

No real call to the Gemini API is made here -- Google's API isn't on
this environment's network allowlist, and a unit test shouldn't depend
on a live service or a real API key anyway. Instead:

  - Most tests use FakeEmbeddingsClient, a small hand-written stand-in
    that gives full control over what "the API" returns, including
    failure cases that would be awkward to trigger for real.
  - A couple of tests build the client via create_autospec against the
    *real* GoogleGenerativeAIEmbeddings class, so that if a future
    version renames or reshapes embed_documents/embed_query, these
    tests fail loudly instead of the mismatch only surfacing against
    the live API.
  - get_embeddings_client's own tests construct a real client (that part
    is safe -- see src/embeddings.py's note that construction doesn't
    touch the network) to check config wiring, without calling it.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import create_autospec

import pytest
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import Config, ConfigError
from src.embeddings import (
    EmbeddedChunk,
    EmbeddingError,
    embed_chunks,
    embed_query,
    get_embeddings_client,
)


# --- Fixture helpers ------------------------------------------------------


def _make_cfg(**overrides) -> Config:
    defaults = dict(
        google_api_key="",
        gemini_model="gemini-3.5-flash-lite",
        embedding_model="models/gemini-embedding-001",
        temperature=0.2,
        chunk_size=1000,
        chunk_overlap=150,
        top_k=4,
        upload_dir=Path("/tmp/student-rag-chat-tests/uploads"),
        vectorstore_dir=Path("/tmp/student-rag-chat-tests/vectorstore"),
    )
    defaults.update(overrides)
    return Config(**defaults)


def _make_chunk(text: str, source: str = "a.pdf", page: int = 1, chunk_index: int = 0) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
            "chunk_id": f"{source}::p{page}::c{chunk_index}",
        },
    )


class FakeEmbeddingsClient:
    """Stand-in for GoogleGenerativeAIEmbeddings with the same call
    shape but fully scripted, deterministic responses."""

    def __init__(self, vector_size: int = 8, model: str = "fake-embedding-model"):
        self.model = model
        self.vector_size = vector_size
        self.calls: list[tuple] = []

    def embed_documents(self, texts, batch_size=100):
        self.calls.append(("embed_documents", list(texts), batch_size))
        return [[float(i)] * self.vector_size for i in range(len(texts))]

    def embed_query(self, text):
        self.calls.append(("embed_query", text))
        return [1.0] * self.vector_size


class RaisingEmbeddingsClient:
    """Stand-in that always fails, to test error-wrapping."""

    model = "raising-fake-model"

    def embed_documents(self, texts, batch_size=100):
        raise RuntimeError("simulated API failure")

    def embed_query(self, text):
        raise RuntimeError("simulated API failure")


class MismatchedCountEmbeddingsClient:
    """Stand-in that returns the wrong number of vectors."""

    model = "mismatched-fake-model"

    def embed_documents(self, texts, batch_size=100):
        return [[0.0, 0.0]]  # always one vector, regardless of input size

    def embed_query(self, text):
        return [0.0, 0.0]


# --- get_embeddings_client --------------------------------------------


class TestGetEmbeddingsClient:
    def test_raises_config_error_without_api_key(self):
        cfg = _make_cfg(google_api_key="")
        with pytest.raises(ConfigError, match="No Gemini API key"):
            get_embeddings_client(cfg)

    def test_builds_client_with_configured_model_and_key(self):
        cfg = _make_cfg(google_api_key="fake-test-key", embedding_model="models/custom-embed")

        client = get_embeddings_client(cfg)

        assert isinstance(client, GoogleGenerativeAIEmbeddings)
        assert client.model == "models/custom-embed"


# --- embed_chunks: happy paths -----------------------------------------


class TestEmbedChunksHappyPath:
    def test_empty_chunks_returns_empty_list_without_calling_client(self):
        client = FakeEmbeddingsClient()

        result = embed_chunks([], client)

        assert result == []
        assert client.calls == []

    def test_returns_one_embedded_chunk_per_input_in_order(self):
        chunks = [
            _make_chunk("first chunk", chunk_index=0),
            _make_chunk("second chunk", chunk_index=1),
            _make_chunk("third chunk", chunk_index=2),
        ]
        client = FakeEmbeddingsClient(vector_size=4)

        result = embed_chunks(chunks, client)

        assert len(result) == 3
        assert all(isinstance(ec, EmbeddedChunk) for ec in result)
        assert [ec.chunk for ec in result] == chunks  # same objects, same order

    def test_vectors_align_with_the_chunk_that_produced_them(self):
        chunks = [_make_chunk("a"), _make_chunk("b"), _make_chunk("c")]
        client = FakeEmbeddingsClient(vector_size=2)

        result = embed_chunks(chunks, client)

        # FakeEmbeddingsClient returns [i.0, i.0] for the i-th text sent.
        assert result[0].vector == [0.0, 0.0]
        assert result[1].vector == [1.0, 1.0]
        assert result[2].vector == [2.0, 2.0]

    def test_passes_chunk_texts_to_client_in_order(self):
        chunks = [_make_chunk("alpha"), _make_chunk("beta")]
        client = FakeEmbeddingsClient()

        embed_chunks(chunks, client)

        _, texts_sent, _ = client.calls[0]
        assert texts_sent == ["alpha", "beta"]

    def test_batch_size_is_passed_through_to_the_client(self):
        chunks = [_make_chunk("x")]
        client = FakeEmbeddingsClient()

        embed_chunks(chunks, client, batch_size=7)

        _, _, batch_size_sent = client.calls[0]
        assert batch_size_sent == 7


# --- embed_chunks: error paths ------------------------------------------


class TestEmbedChunksErrorPaths:
    def test_empty_page_content_raises_value_error(self):
        chunks = [_make_chunk("good chunk"), _make_chunk("")]
        client = FakeEmbeddingsClient()

        with pytest.raises(ValueError, match="empty page_content"):
            embed_chunks(chunks, client)

    def test_whitespace_only_page_content_raises_value_error(self):
        chunks = [_make_chunk("   \n\t  ")]
        client = FakeEmbeddingsClient()

        with pytest.raises(ValueError, match="empty page_content"):
            embed_chunks(chunks, client)

    def test_value_error_prevents_any_api_call(self):
        chunks = [_make_chunk("good"), _make_chunk("")]
        client = FakeEmbeddingsClient()

        with pytest.raises(ValueError):
            embed_chunks(chunks, client)
        assert client.calls == []  # fails validation before touching the client

    def test_client_exception_is_wrapped_in_embedding_error(self):
        chunks = [_make_chunk("hello")]
        client = RaisingEmbeddingsClient()

        with pytest.raises(EmbeddingError, match="Failed to embed 1 chunk"):
            embed_chunks(chunks, client)

    def test_vector_count_mismatch_raises_embedding_error(self):
        chunks = [_make_chunk("a"), _make_chunk("b"), _make_chunk("c")]
        client = MismatchedCountEmbeddingsClient()

        with pytest.raises(EmbeddingError, match="3 chunk"):
            embed_chunks(chunks, client)


# --- embed_query ------------------------------------------------------


class TestEmbedQuery:
    def test_returns_vector_from_client(self):
        client = FakeEmbeddingsClient(vector_size=5)

        vector = embed_query("What is backpropagation?", client)

        assert vector == [1.0] * 5

    def test_records_the_exact_query_text_sent(self):
        client = FakeEmbeddingsClient()

        embed_query("Why is the sky blue?", client)

        _, text_sent = client.calls[0]
        assert text_sent == "Why is the sky blue?"

    def test_empty_query_raises_value_error(self):
        client = FakeEmbeddingsClient()

        with pytest.raises(ValueError, match="empty question"):
            embed_query("", client)

    def test_whitespace_only_query_raises_value_error(self):
        client = FakeEmbeddingsClient()

        with pytest.raises(ValueError, match="empty question"):
            embed_query("   ", client)

    def test_client_exception_is_wrapped_in_embedding_error(self):
        client = RaisingEmbeddingsClient()

        with pytest.raises(EmbeddingError, match="Failed to embed query"):
            embed_query("Any question", client)


# --- Signature drift checks against the real client -----------------------


class TestCallsMatchRealClientSignature:
    """create_autospec enforces the real method signatures, so these
    fail if a future langchain-google-genai version renames or
    reshapes embed_documents/embed_query out from under us."""

    def test_embed_chunks_call_matches_real_embed_documents_signature(self):
        spec_client = create_autospec(GoogleGenerativeAIEmbeddings, instance=True)
        spec_client.model = "fake-model"  # a pydantic field, not auto-specced by create_autospec
        spec_client.embed_documents.return_value = [[0.1, 0.2]]
        chunks = [_make_chunk("hello")]

        result = embed_chunks(chunks, spec_client)

        assert result[0].vector == [0.1, 0.2]
        spec_client.embed_documents.assert_called_once()

    def test_embed_query_call_matches_real_embed_query_signature(self):
        spec_client = create_autospec(GoogleGenerativeAIEmbeddings, instance=True)
        spec_client.embed_query.return_value = [0.5, 0.6]

        result = embed_query("hello", spec_client)

        assert result == [0.5, 0.6]
        spec_client.embed_query.assert_called_once_with("hello")
