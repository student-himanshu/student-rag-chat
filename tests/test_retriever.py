"""
Tests for src.retriever (Phase 7).

Only embed_query / get_embeddings_client are mocked (the Gemini-touching
calls) -- same line the project drew since Phase 5. VectorStore itself is
real (pure local FAISS compute, per test_vector_store.py's philosophy
note), so these tests exercise a real search against real vectors, not a
faked one.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
from langchain_core.documents import Document

from src.config import Config, ConfigError
from src.embeddings import EmbeddedChunk, EmbeddingError
from src.retriever import RetrieverError, retrieve
from src.vector_store import VectorStore, build_vector_store


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(
        google_api_key="fake-key-for-tests",
        gemini_model="gemini-3.5-flash-lite",
        embedding_model="models/gemini-embedding-001",
        temperature=0.2,
        chunk_size=1000,
        chunk_overlap=150,
        top_k=2,
        upload_dir=tmp_path / "uploads",
        vectorstore_dir=tmp_path / "vectorstore",
    )


@pytest.fixture
def fake_client():
    """A stand-in embeddings client. Its identity is all that matters --
    embed_query() itself is always mocked in these tests, so nothing ever
    calls a real method on this object."""
    return object()


def make_embedded_chunk(text: str, vector: list[float], source="a.pdf", page=1, chunk_index=0) -> EmbeddedChunk:
    chunk_id = f"{source}::p{page}::c{chunk_index}"
    doc = Document(
        page_content=text,
        metadata={"source": source, "page": page, "chunk_index": chunk_index, "chunk_id": chunk_id},
    )
    return EmbeddedChunk(chunk=doc, vector=vector)


@pytest.fixture
def populated_store() -> VectorStore:
    return build_vector_store(
        [
            make_embedded_chunk("closest", [1.0, 0.0], chunk_index=0),
            make_embedded_chunk("middle", [0.7, 0.7], chunk_index=1),
            make_embedded_chunk("farthest", [0.0, 5.0], chunk_index=2),
        ]
    )


class TestRetrieve:
    def test_returns_top_k_nearest_first(self, cfg, populated_store, fake_client, mocker):
        mocker.patch("src.retriever.embed_query", return_value=[1.0, 0.0])
        results = retrieve("what is X?", populated_store, cfg, client=fake_client, k=2)
        assert [r.document.page_content for r in results] == ["closest", "middle"]

    def test_defaults_k_to_config_top_k(self, cfg, populated_store, fake_client, mocker):
        # cfg.top_k == 2 in the fixture above.
        mocker.patch("src.retriever.embed_query", return_value=[1.0, 0.0])
        results = retrieve("what is X?", populated_store, cfg, client=fake_client)
        assert len(results) == 2

    def test_explicit_k_overrides_config_top_k(self, cfg, populated_store, fake_client, mocker):
        mocker.patch("src.retriever.embed_query", return_value=[1.0, 0.0])
        results = retrieve("what is X?", populated_store, cfg, client=fake_client, k=1)
        assert len(results) == 1
        assert results[0].document.page_content == "closest"

    def test_calls_embed_query_with_text_and_client(self, cfg, populated_store, fake_client, mocker):
        mock_embed = mocker.patch("src.retriever.embed_query", return_value=[1.0, 0.0])
        retrieve("what is photosynthesis?", populated_store, cfg, client=fake_client, k=1)
        # Real signature is embed_query(text, client) -- text first.
        mock_embed.assert_called_once_with("what is photosynthesis?", fake_client)

    def test_builds_client_when_none_provided(self, cfg, populated_store, fake_client, mocker):
        mock_get_client = mocker.patch("src.retriever.get_embeddings_client", return_value=fake_client)
        mock_embed = mocker.patch("src.retriever.embed_query", return_value=[1.0, 0.0])
        retrieve("what is X?", populated_store, cfg, k=1)
        mock_get_client.assert_called_once_with(cfg)
        mock_embed.assert_called_once_with("what is X?", fake_client)

    def test_provided_client_skips_get_embeddings_client(self, cfg, populated_store, fake_client, mocker):
        mock_get_client = mocker.patch("src.retriever.get_embeddings_client")
        mocker.patch("src.retriever.embed_query", return_value=[1.0, 0.0])
        retrieve("what is X?", populated_store, cfg, client=fake_client, k=1)
        mock_get_client.assert_not_called()

    def test_empty_store_returns_empty_list(self, cfg, fake_client, mocker):
        mocker.patch("src.retriever.embed_query", return_value=[1.0, 0.0])
        empty_store = VectorStore(dimension=2)
        assert retrieve("anything", empty_store, cfg, client=fake_client) == []

    @pytest.mark.parametrize("bad_query", ["", "   ", "\n\t"])
    def test_empty_or_whitespace_query_raises_before_touching_gemini(
        self, cfg, populated_store, fake_client, mocker, bad_query
    ):
        mock_get_client = mocker.patch("src.retriever.get_embeddings_client")
        mock_embed = mocker.patch("src.retriever.embed_query")
        with pytest.raises(RetrieverError, match="empty"):
            retrieve(bad_query, populated_store, cfg)
        mock_get_client.assert_not_called()
        mock_embed.assert_not_called()

    def test_non_positive_k_raises_before_touching_gemini(self, cfg, populated_store, mocker):
        mock_get_client = mocker.patch("src.retriever.get_embeddings_client")
        mock_embed = mocker.patch("src.retriever.embed_query")
        with pytest.raises(RetrieverError, match="k must be"):
            retrieve("what is X?", populated_store, cfg, k=0)
        mock_get_client.assert_not_called()
        mock_embed.assert_not_called()

    def test_missing_api_key_is_wrapped_and_chained(self, cfg, populated_store, mocker):
        mock_embed = mocker.patch("src.retriever.embed_query")
        mocker.patch(
            "src.retriever.get_embeddings_client",
            side_effect=ConfigError("No Gemini API key found."),
        )
        cfg_no_key = replace(cfg, google_api_key="")
        with pytest.raises(RetrieverError, match="API key") as exc_info:
            retrieve("what is X?", populated_store, cfg_no_key)
        assert isinstance(exc_info.value.__cause__, ConfigError)
        mock_embed.assert_not_called()

    def test_embed_query_value_error_is_wrapped(self, cfg, populated_store, fake_client, mocker):
        # embed_query() itself raises ValueError for empty/whitespace text;
        # retrieve() already guards against that, but the wrapping is
        # tested independently in case embed_query's own validation
        # rejects something retrieve()'s check lets through.
        mocker.patch("src.retriever.embed_query", side_effect=ValueError("Cannot embed an empty question."))
        with pytest.raises(RetrieverError, match="Couldn't embed") as exc_info:
            retrieve("what is X?", populated_store, cfg, client=fake_client)
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_embedding_error_is_wrapped_and_chained(self, cfg, populated_store, fake_client, mocker):
        mocker.patch("src.retriever.embed_query", side_effect=EmbeddingError("quota exceeded"))
        with pytest.raises(RetrieverError, match="Couldn't embed") as exc_info:
            retrieve("what is X?", populated_store, cfg, client=fake_client)
        assert isinstance(exc_info.value.__cause__, EmbeddingError)

    def test_vector_dimension_mismatch_is_wrapped_as_retriever_error(self, cfg, populated_store, fake_client, mocker):
        # Query vector dimension (3) doesn't match the store's (2).
        mocker.patch("src.retriever.embed_query", return_value=[1.0, 0.0, 0.0])
        with pytest.raises(RetrieverError, match="Couldn't search"):
            retrieve("what is X?", populated_store, cfg, client=fake_client)

    def test_results_carry_citation_metadata(self, cfg, populated_store, fake_client, mocker):
        mocker.patch("src.retriever.embed_query", return_value=[1.0, 0.0])
        [result] = retrieve("what is X?", populated_store, cfg, client=fake_client, k=1)
        assert result.document.metadata["source"] == "a.pdf"
        assert result.document.metadata["page"] == 1
        assert "chunk_id" in result.document.metadata


# --- allowed_sources (Phase 14) ----------------------------------------


class TestRetrieveWithAllowedSources:
    @pytest.fixture
    def multi_source_store(self) -> VectorStore:
        return build_vector_store(
            [
                make_embedded_chunk("a-close", [1.0, 0.0], source="a.pdf", chunk_index=0),
                make_embedded_chunk("b-mid", [0.9, 0.1], source="b.pdf", chunk_index=0),
            ]
        )

    def test_default_none_ignores_no_source(self, cfg, multi_source_store, fake_client, mocker):
        mocker.patch("src.retriever.embed_query", return_value=[1.0, 0.0])
        results = retrieve("what is X?", multi_source_store, cfg, client=fake_client, k=2)
        assert {r.document.metadata["source"] for r in results} == {"a.pdf", "b.pdf"}

    def test_allowed_sources_is_forwarded_to_the_store(self, cfg, multi_source_store, fake_client, mocker):
        mocker.patch("src.retriever.embed_query", return_value=[1.0, 0.0])
        results = retrieve(
            "what is X?", multi_source_store, cfg, client=fake_client, k=2, allowed_sources={"b.pdf"}
        )
        assert [r.document.metadata["source"] for r in results] == ["b.pdf"]

    def test_empty_allowed_sources_returns_empty_list(self, cfg, multi_source_store, fake_client, mocker):
        mocker.patch("src.retriever.embed_query", return_value=[1.0, 0.0])
        results = retrieve(
            "what is X?", multi_source_store, cfg, client=fake_client, allowed_sources=set()
        )
        assert results == []
