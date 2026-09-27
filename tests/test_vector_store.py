"""
Tests for src.vector_store (Phase 6).

Uses real faiss-cpu against small synthetic vectors -- no mocking, per the
project's test philosophy of exercising the real library wherever it can
actually run in this environment. Only src.embeddings' network call was
untestable that way (Phase 5); FAISS itself is pure local compute, so
there's no reason to fake it here.
"""
from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

from src.embeddings import EmbeddedChunk
from src.vector_store import (
    DOCSTORE_FILENAME,
    INDEX_FILENAME,
    SearchResult,
    SourceSummary,
    VectorStore,
    VectorStoreError,
    build_vector_store,
)


def make_chunk(text: str, source: str, page: int, chunk_index: int) -> Document:
    chunk_id = f"{source}::p{page}::c{chunk_index}"
    return Document(
        page_content=text,
        metadata={"source": source, "page": page, "chunk_index": chunk_index, "chunk_id": chunk_id},
    )


def make_embedded_chunk(text: str, vector: list[float], source="a.pdf", page=1, chunk_index=0) -> EmbeddedChunk:
    return EmbeddedChunk(chunk=make_chunk(text, source, page, chunk_index), vector=vector)


# --- build_vector_store --------------------------------------------------


class TestBuildVectorStore:
    def test_infers_dimension_from_first_vector(self):
        chunks = [make_embedded_chunk("hello", [1.0, 0.0, 0.0])]
        store = build_vector_store(chunks)
        assert store.dimension == 3
        assert len(store) == 1

    def test_empty_list_raises(self):
        with pytest.raises(VectorStoreError, match="empty"):
            build_vector_store([])

    def test_multiple_chunks(self):
        chunks = [
            make_embedded_chunk("a", [1.0, 0.0], chunk_index=0),
            make_embedded_chunk("b", [0.0, 1.0], chunk_index=1),
        ]
        store = build_vector_store(chunks)
        assert len(store) == 2


# --- VectorStore.add -------------------------------------------------------


class TestAdd:
    def test_add_empty_list_is_noop(self):
        store = VectorStore(dimension=2)
        store.add([])
        assert len(store) == 0

    def test_add_grows_store(self):
        store = VectorStore(dimension=2)
        store.add([make_embedded_chunk("a", [1.0, 0.0])])
        assert len(store) == 1
        store.add([make_embedded_chunk("b", [0.0, 1.0], chunk_index=1)])
        assert len(store) == 2

    def test_dimension_mismatch_raises(self):
        store = VectorStore(dimension=3)
        with pytest.raises(VectorStoreError, match="dimension"):
            store.add([make_embedded_chunk("a", [1.0, 0.0])])

    def test_missing_chunk_id_raises(self):
        store = VectorStore(dimension=2)
        bad_doc = Document(page_content="a", metadata={"source": "a.pdf", "page": 1})
        with pytest.raises(VectorStoreError, match="chunk_id"):
            store.add([EmbeddedChunk(chunk=bad_doc, vector=[1.0, 0.0])])

    def test_zero_or_negative_dimension_raises(self):
        with pytest.raises(VectorStoreError, match="positive"):
            VectorStore(dimension=0)
        with pytest.raises(VectorStoreError, match="positive"):
            VectorStore(dimension=-1)


# --- VectorStore.search -----------------------------------------------------


class TestSearch:
    def test_returns_nearest_first(self):
        store = VectorStore(dimension=2)
        store.add(
            [
                make_embedded_chunk("close", [1.0, 0.0], chunk_index=0),
                make_embedded_chunk("far", [0.0, 100.0], chunk_index=1),
            ]
        )
        results = store.search([1.0, 0.0], k=2)
        assert [r.document.page_content for r in results] == ["close", "far"]
        assert results[0].score < results[1].score

    def test_result_is_search_result_with_full_metadata(self):
        store = VectorStore(dimension=2)
        store.add([make_embedded_chunk("hello", [1.0, 0.0], source="notes.pdf", page=3, chunk_index=2)])
        [result] = store.search([1.0, 0.0], k=1)
        assert isinstance(result, SearchResult)
        assert result.document.metadata["chunk_id"] == "notes.pdf::p3::c2"
        assert result.document.metadata["source"] == "notes.pdf"
        assert result.document.metadata["page"] == 3

    def test_k_larger_than_store_returns_all(self):
        store = VectorStore(dimension=2)
        store.add([make_embedded_chunk("only", [1.0, 0.0])])
        results = store.search([1.0, 0.0], k=10)
        assert len(results) == 1

    def test_search_on_empty_store_returns_empty_list(self):
        store = VectorStore(dimension=2)
        assert store.search([1.0, 0.0], k=5) == []

    def test_non_positive_k_raises(self):
        store = VectorStore(dimension=2)
        store.add([make_embedded_chunk("a", [1.0, 0.0])])
        with pytest.raises(VectorStoreError, match="k must be"):
            store.search([1.0, 0.0], k=0)

    def test_query_dimension_mismatch_raises(self):
        store = VectorStore(dimension=3)
        store.add([make_embedded_chunk("a", [1.0, 0.0, 0.0])])
        with pytest.raises(VectorStoreError, match="dimension"):
            store.search([1.0, 0.0], k=1)

    def test_duplicate_vectors_both_returned(self):
        store = VectorStore(dimension=2)
        store.add(
            [
                make_embedded_chunk("first", [1.0, 0.0], chunk_index=0),
                make_embedded_chunk("second", [1.0, 0.0], chunk_index=1),
            ]
        )
        results = store.search([1.0, 0.0], k=2)
        assert len(results) == 2
        assert {r.document.page_content for r in results} == {"first", "second"}


# --- VectorStore.search with allowed_sources (Phase 14) --------------------


class TestSearchWithAllowedSources:
    @pytest.fixture
    def multi_source_store(self) -> VectorStore:
        # "far" is nearest to the query overall but belongs to an
        # excluded source -- these tests only pass if filtering happens
        # *before* picking the top k, not after.
        store = VectorStore(dimension=2)
        store.add(
            [
                make_embedded_chunk("a-close", [1.0, 0.0], source="a.pdf", chunk_index=0),
                make_embedded_chunk("a-far", [0.0, 3.0], source="a.pdf", chunk_index=1),
                make_embedded_chunk("b-mid", [0.9, 0.1], source="b.pdf", chunk_index=0),
            ]
        )
        return store

    def test_none_searches_everything(self, multi_source_store):
        results = multi_source_store.search([1.0, 0.0], k=3, allowed_sources=None)
        assert len(results) == 3

    def test_filters_to_allowed_sources_only(self, multi_source_store):
        results = multi_source_store.search([1.0, 0.0], k=3, allowed_sources={"a.pdf"})
        assert len(results) == 2
        assert all(r.document.metadata["source"] == "a.pdf" for r in results)

    def test_returns_k_nearest_among_allowed_not_top_k_then_filtered(self, multi_source_store):
        # k=1 overall nearest is "a-close" (also in a.pdf), so this
        # alone wouldn't distinguish the two strategies -- but combined
        # with the next test it does.
        results = multi_source_store.search([1.0, 0.0], k=1, allowed_sources={"b.pdf"})
        assert [r.document.page_content for r in results] == ["b-mid"]

    def test_excluded_nearest_neighbor_does_not_crowd_out_allowed_results(self, multi_source_store):
        # Only b.pdf has one chunk, but it should still come back even
        # though a.pdf's chunks are both nearer to the query overall.
        results = multi_source_store.search([1.0, 0.0], k=2, allowed_sources={"b.pdf"})
        assert [r.document.page_content for r in results] == ["b-mid"]

    def test_empty_allowed_sources_returns_empty_without_error(self, multi_source_store):
        assert multi_source_store.search([1.0, 0.0], k=3, allowed_sources=set()) == []

    def test_unknown_source_in_filter_returns_empty(self, multi_source_store):
        assert multi_source_store.search([1.0, 0.0], k=3, allowed_sources={"nope.pdf"}) == []


# --- VectorStore.list_sources (Phase 14) ------------------------------------


class TestListSources:
    def test_empty_store_returns_empty_list(self):
        assert VectorStore(dimension=2).list_sources() == []

    def test_single_source_counts_pages_and_chunks(self):
        store = VectorStore(dimension=2)
        store.add(
            [
                make_embedded_chunk("p1c0", [1.0, 0.0], source="a.pdf", page=1, chunk_index=0),
                make_embedded_chunk("p1c1", [0.0, 1.0], source="a.pdf", page=1, chunk_index=1),
                make_embedded_chunk("p2c0", [1.0, 1.0], source="a.pdf", page=2, chunk_index=0),
            ]
        )
        assert store.list_sources() == [SourceSummary(source="a.pdf", pages=2, chunks=3)]

    def test_multiple_sources_sorted_by_name(self):
        store = VectorStore(dimension=2)
        store.add(
            [
                make_embedded_chunk("b", [1.0, 0.0], source="zebra.pdf", page=1, chunk_index=0),
                make_embedded_chunk("a", [0.0, 1.0], source="apple.pdf", page=1, chunk_index=0),
            ]
        )
        assert [s.source for s in store.list_sources()] == ["apple.pdf", "zebra.pdf"]


# --- VectorStore.remove_source (Phase 14) -----------------------------------


class TestRemoveSource:
    @pytest.fixture
    def multi_source_store(self) -> VectorStore:
        store = VectorStore(dimension=2)
        store.add(
            [
                make_embedded_chunk("a0", [1.0, 0.0], source="a.pdf", chunk_index=0),
                make_embedded_chunk("a1", [0.9, 0.1], source="a.pdf", chunk_index=1),
                make_embedded_chunk("b0", [0.0, 1.0], source="b.pdf", chunk_index=0),
            ]
        )
        return store

    def test_removes_only_the_given_source(self, multi_source_store):
        removed = multi_source_store.remove_source("a.pdf")
        assert removed == 2
        assert len(multi_source_store) == 1
        remaining = multi_source_store.search([0.0, 1.0], k=10)
        assert [r.document.page_content for r in remaining] == ["b0"]

    def test_removed_source_no_longer_in_list_sources(self, multi_source_store):
        multi_source_store.remove_source("a.pdf")
        assert [s.source for s in multi_source_store.list_sources()] == ["b.pdf"]

    def test_unknown_source_returns_zero_and_is_a_noop(self, multi_source_store):
        before = len(multi_source_store)
        removed = multi_source_store.remove_source("nope.pdf")
        assert removed == 0
        assert len(multi_source_store) == before

    def test_remaining_source_still_searchable_after_removal(self, multi_source_store):
        multi_source_store.remove_source("b.pdf")
        [result] = multi_source_store.search([1.0, 0.0], k=1)
        assert result.document.metadata["source"] == "a.pdf"

    def test_further_adds_after_removal_keep_working(self, multi_source_store):
        multi_source_store.remove_source("a.pdf")
        multi_source_store.add(
            [make_embedded_chunk("c0", [0.5, 0.5], source="c.pdf", chunk_index=0)]
        )
        assert len(multi_source_store) == 2
        assert {s.source for s in multi_source_store.list_sources()} == {"b.pdf", "c.pdf"}


# --- save / load -------------------------------------------------------------


class TestSaveLoad:
    def test_roundtrip_preserves_search_results(self, tmp_path):
        store = VectorStore(dimension=2)
        store.add(
            [
                make_embedded_chunk("close", [1.0, 0.0], source="a.pdf", page=1, chunk_index=0),
                make_embedded_chunk("far", [0.0, 5.0], source="a.pdf", page=1, chunk_index=1),
            ]
        )
        store.save(tmp_path)

        loaded = VectorStore.load(tmp_path)
        assert len(loaded) == len(store)

        before = store.search([1.0, 0.0], k=2)
        after = loaded.search([1.0, 0.0], k=2)
        assert [r.document.page_content for r in before] == [r.document.page_content for r in after]
        assert [r.score for r in before] == pytest.approx([r.score for r in after])

    def test_roundtrip_preserves_metadata(self, tmp_path):
        store = VectorStore(dimension=2)
        store.add([make_embedded_chunk("hi", [1.0, 0.0], source="notes.pdf", page=7, chunk_index=3)])
        store.save(tmp_path)

        loaded = VectorStore.load(tmp_path)
        [result] = loaded.search([1.0, 0.0], k=1)
        assert result.document.metadata == {
            "source": "notes.pdf",
            "page": 7,
            "chunk_index": 3,
            "chunk_id": "notes.pdf::p7::c3",
        }

    def test_save_creates_expected_files(self, tmp_path):
        store = build_vector_store([make_embedded_chunk("a", [1.0, 0.0])])
        store.save(tmp_path)
        assert (tmp_path / INDEX_FILENAME).exists()
        assert (tmp_path / DOCSTORE_FILENAME).exists()

    def test_save_creates_directory_if_missing(self, tmp_path):
        target = tmp_path / "nested" / "vectorstore"
        store = build_vector_store([make_embedded_chunk("a", [1.0, 0.0])])
        store.save(target)
        assert target.exists()

    def test_docstore_json_is_plain_and_inspectable(self, tmp_path):
        store = build_vector_store([make_embedded_chunk("a", [1.0, 0.0])])
        store.save(tmp_path)
        payload = json.loads((tmp_path / DOCSTORE_FILENAME).read_text())
        assert payload["dimension"] == 2
        assert "a.pdf::p1::c0" in payload["chunks"]

    def test_load_missing_directory_raises(self, tmp_path):
        with pytest.raises(VectorStoreError, match="No saved vector store"):
            VectorStore.load(tmp_path / "does_not_exist")

    def test_load_partial_files_raises(self, tmp_path):
        # Only the docstore written, no index -- simulates a crash mid-save.
        (tmp_path / DOCSTORE_FILENAME).write_text("{}")
        with pytest.raises(VectorStoreError, match="No saved vector store"):
            VectorStore.load(tmp_path)

    def test_further_adds_after_load_keep_working(self, tmp_path):
        store = build_vector_store([make_embedded_chunk("a", [1.0, 0.0], chunk_index=0)])
        store.save(tmp_path)

        loaded = VectorStore.load(tmp_path)
        loaded.add([make_embedded_chunk("b", [0.0, 1.0], chunk_index=1)])
        assert len(loaded) == 2
        results = loaded.search([0.0, 1.0], k=1)
        assert results[0].document.page_content == "b"
