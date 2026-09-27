"""Unit tests for src/chunking.py.

Runs against the real langchain_text_splitters.RecursiveCharacterTextSplitter
rather than mocking it -- a version bump that changes splitting behavior
should be visible here, not hidden behind a mock that assumes the old
behavior. No network access is used.
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.chunking import chunk_documents, pages_to_documents
from src.pdf_processor import PageContent


def _words(n: int, start: int = 0) -> str:
    """A long, space-separated string of short fixed-width tokens --
    predictable splitting behavior since no single "word" ever exceeds
    a reasonable chunk_size on its own."""
    return " ".join(f"word{i:04d}" for i in range(start, start + n))


# --- pages_to_documents ------------------------------------------------


class TestPagesToDocuments:
    def test_empty_list_returns_empty_list(self):
        assert pages_to_documents([]) == []

    def test_one_document_per_page(self):
        pages = [
            PageContent(source="a.pdf", page=1, text="Hello"),
            PageContent(source="a.pdf", page=2, text="World"),
        ]

        docs = pages_to_documents(pages)

        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)

    def test_page_content_and_metadata_carried_over(self):
        pages = [PageContent(source="lecture_04.pdf", page=3, text="Some content")]

        docs = pages_to_documents(pages)

        assert docs[0].page_content == "Some content"
        assert docs[0].metadata == {"source": "lecture_04.pdf", "page": 3}


# --- chunk_documents: validation ----------------------------------------


class TestChunkDocumentsValidation:
    def test_empty_pages_returns_empty_list(self):
        assert chunk_documents([], chunk_size=1000, chunk_overlap=150) == []

    def test_zero_chunk_size_raises(self):
        pages = [PageContent(source="a.pdf", page=1, text="x")]
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_documents(pages, chunk_size=0, chunk_overlap=0)

    def test_negative_chunk_size_raises(self):
        pages = [PageContent(source="a.pdf", page=1, text="x")]
        with pytest.raises(ValueError, match="chunk_size"):
            chunk_documents(pages, chunk_size=-100, chunk_overlap=0)

    def test_negative_chunk_overlap_raises(self):
        pages = [PageContent(source="a.pdf", page=1, text="x")]
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_documents(pages, chunk_size=100, chunk_overlap=-5)

    def test_chunk_overlap_equal_to_chunk_size_raises(self):
        pages = [PageContent(source="a.pdf", page=1, text="x")]
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_documents(pages, chunk_size=100, chunk_overlap=100)

    def test_chunk_overlap_greater_than_chunk_size_raises(self):
        pages = [PageContent(source="a.pdf", page=1, text="x")]
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_documents(pages, chunk_size=100, chunk_overlap=150)


# --- chunk_documents: splitting behavior ---------------------------------


class TestChunkDocumentsSplitting:
    def test_short_page_produces_a_single_chunk(self):
        pages = [PageContent(source="a.pdf", page=1, text="Short text.")]

        chunks = chunk_documents(pages, chunk_size=1000, chunk_overlap=150)

        assert len(chunks) == 1
        assert chunks[0].page_content == "Short text."

    def test_long_page_produces_multiple_chunks_within_size(self):
        pages = [PageContent(source="a.pdf", page=1, text=_words(200))]

        chunks = chunk_documents(pages, chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 1
        assert all(len(c.page_content) <= 100 for c in chunks)

    def test_reconstructed_text_covers_original_content_in_order(self):
        # Every word placed in the source text must appear in the chunk
        # sequence, in the same order, even after splitting with overlap.
        pages = [PageContent(source="a.pdf", page=1, text=_words(50))]

        chunks = chunk_documents(pages, chunk_size=80, chunk_overlap=15)

        seen_words: list[str] = []
        for chunk in chunks:
            for word in chunk.page_content.split():
                if not seen_words or word != seen_words[-1]:
                    seen_words.append(word)
        expected_first_words = [f"word{i:04d}" for i in range(5)]
        assert seen_words[:5] == expected_first_words

    def test_chunks_never_merge_across_a_page_boundary(self):
        pages = [
            PageContent(source="a.pdf", page=1, text="Page one short text."),
            PageContent(source="a.pdf", page=2, text="Page two short text."),
        ]

        chunks = chunk_documents(pages, chunk_size=1000, chunk_overlap=150)

        assert len(chunks) == 2
        assert chunks[0].metadata["page"] == 1
        assert chunks[1].metadata["page"] == 2
        assert "Page two" not in chunks[0].page_content
        assert "Page one" not in chunks[1].page_content

    def test_large_page_split_never_bleeds_into_next_pages_chunks(self):
        pages = [
            PageContent(source="a.pdf", page=1, text=_words(100, start=0)),
            PageContent(source="a.pdf", page=2, text=_words(100, start=1000)),
        ]

        chunks = chunk_documents(pages, chunk_size=100, chunk_overlap=20)

        page1_chunks = [c for c in chunks if c.metadata["page"] == 1]
        page2_chunks = [c for c in chunks if c.metadata["page"] == 2]
        assert len(page1_chunks) > 1
        assert all("word1" not in c.page_content for c in page1_chunks)
        assert all("word0" not in c.page_content for c in page2_chunks)

    def test_output_order_is_grouped_by_page_then_position(self):
        pages = [
            PageContent(source="a.pdf", page=1, text=_words(100)),
            PageContent(source="a.pdf", page=2, text=_words(100, start=1000)),
        ]

        chunks = chunk_documents(pages, chunk_size=100, chunk_overlap=20)

        pages_seen = [c.metadata["page"] for c in chunks]
        # once page 2 starts, page 1 never reappears later in the list
        first_page2_index = pages_seen.index(2)
        assert 1 not in pages_seen[first_page2_index:]


# --- chunk_documents: metadata --------------------------------------------


class TestChunkDocumentsMetadata:
    def test_single_chunk_metadata_fields(self):
        pages = [PageContent(source="lecture_04.pdf", page=3, text="Short text.")]

        chunks = chunk_documents(pages, chunk_size=1000, chunk_overlap=150)

        assert chunks[0].metadata == {
            "source": "lecture_04.pdf",
            "page": 3,
            "chunk_index": 0,
            "chunk_id": "lecture_04.pdf::p3::c0",
        }

    def test_chunk_index_resets_per_page(self):
        long_text = _words(100)
        pages = [
            PageContent(source="a.pdf", page=1, text=long_text),
            PageContent(source="a.pdf", page=2, text=long_text),
        ]

        chunks = chunk_documents(pages, chunk_size=100, chunk_overlap=20)

        page1_indices = [c.metadata["chunk_index"] for c in chunks if c.metadata["page"] == 1]
        page2_indices = [c.metadata["chunk_index"] for c in chunks if c.metadata["page"] == 2]
        assert page1_indices == list(range(len(page1_indices)))
        assert page2_indices == list(range(len(page2_indices)))
        assert len(page1_indices) > 1  # sanity check the page actually split

    def test_chunk_id_format_matches_source_page_chunk_index(self):
        pages = [PageContent(source="notes.pdf", page=7, text=_words(60))]

        chunks = chunk_documents(pages, chunk_size=80, chunk_overlap=15)

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_id"] == f"notes.pdf::p7::c{i}"

    def test_metadata_dicts_are_independent_across_chunks(self):
        pages = [PageContent(source="a.pdf", page=1, text=_words(100))]

        chunks = chunk_documents(pages, chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 1
        assert chunks[0].metadata is not chunks[1].metadata
        chunks[0].metadata["chunk_index"] = 999
        assert chunks[1].metadata["chunk_index"] != 999
