"""Unit tests for src/pdf_processor.py.

These build tiny real PDFs in memory with PyMuPDF itself and feed them
back through the module under test, rather than mocking PyMuPDF -- that
way a real bug in how we call the library (wrong kwarg, wrong page
indexing, etc.) still fails a test. No network access is used.
"""
from __future__ import annotations

import io

import pymupdf
import pytest

from src.pdf_processor import (
    ExtractionResult,
    PageContent,
    PDFProcessingError,
    clean_text,
    extract_pdf,
    extract_pdfs,
)


# --- Fixture helpers ----------------------------------------------------


def _make_pdf_bytes(page_texts: list[str]) -> bytes:
    """Build an in-memory PDF with one page per string in page_texts.
    An empty string produces a genuinely blank page (no text inserted)."""
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def _make_encrypted_pdf_bytes(text: str = "Secret") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    buf = io.BytesIO()
    doc.save(
        buf,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-pw",
        user_pw="user-pw",
    )
    doc.close()
    return buf.getvalue()


class FakeUploadedFile(io.BytesIO):
    """Minimal stand-in for Streamlit's UploadedFile: a readable,
    seekable, in-memory stream that also carries a `.name`."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


# --- clean_text -----------------------------------------------------------


class TestCleanText:
    def test_empty_string_returns_empty_string(self):
        assert clean_text("") == ""

    def test_rejoins_hyphenated_line_break(self):
        assert clean_text("informa-\ntion") == "information"

    def test_does_not_join_hyphen_before_non_word_char(self):
        # A real end-of-sentence hyphen/dash shouldn't be eaten.
        assert clean_text("done-\n\n") == "done-"

    def test_collapses_three_or_more_newlines_to_paragraph_break(self):
        assert clean_text("para one\n\n\n\npara two") == "para one\n\npara two"

    def test_preserves_single_and_double_newlines(self):
        assert clean_text("line one\nline two") == "line one\nline two"
        assert clean_text("para one\n\npara two") == "para one\n\npara two"

    def test_strips_trailing_whitespace_per_line(self):
        assert clean_text("line one   \nline two\t\t\n") == "line one\nline two"

    def test_collapses_runs_of_spaces(self):
        assert clean_text("col1     col2") == "col1 col2"

    def test_removes_null_characters(self):
        assert clean_text("hello\x00world") == "helloworld"

    def test_strips_leading_and_trailing_overall_whitespace(self):
        assert clean_text("\n\n  hello  \n\n") == "hello"


# --- extract_pdf: happy paths ---------------------------------------------


class TestExtractPdfHappyPath:
    def test_returns_one_page_content_per_page_with_text(self, tmp_path):
        pdf_path = tmp_path / "notes.pdf"
        pdf_path.write_bytes(_make_pdf_bytes(["Page one text", "Page two text"]))

        pages = extract_pdf(pdf_path)

        assert len(pages) == 2
        assert all(isinstance(p, PageContent) for p in pages)

    def test_source_name_inferred_from_path(self, tmp_path):
        pdf_path = tmp_path / "lecture_04.pdf"
        pdf_path.write_bytes(_make_pdf_bytes(["Some content"]))

        pages = extract_pdf(pdf_path)

        assert pages[0].source == "lecture_04.pdf"

    def test_page_numbers_are_one_indexed(self, tmp_path):
        pdf_path = tmp_path / "notes.pdf"
        pdf_path.write_bytes(_make_pdf_bytes(["First", "Second", "Third"]))

        pages = extract_pdf(pdf_path)

        assert [p.page for p in pages] == [1, 2, 3]

    def test_extracted_text_is_cleaned(self, tmp_path):
        pdf_path = tmp_path / "notes.pdf"
        pdf_path.write_bytes(_make_pdf_bytes(["Hello World"]))

        pages = extract_pdf(pdf_path)

        assert "Hello World" in pages[0].text
        assert pages[0].text == pages[0].text.strip()

    def test_skips_blank_pages_rather_than_returning_empty_entries(self, tmp_path):
        pdf_path = tmp_path / "notes.pdf"
        pdf_path.write_bytes(_make_pdf_bytes(["Real content", "", "More content"]))

        pages = extract_pdf(pdf_path)

        assert len(pages) == 2
        assert all(p.text for p in pages)
        # The blank middle page must not appear as page 2 of the output.
        assert [p.text for p in pages] == ["Real content", "More content"]

    def test_accepts_str_path_not_only_pathlib_path(self, tmp_path):
        pdf_path = tmp_path / "notes.pdf"
        pdf_path.write_bytes(_make_pdf_bytes(["Content"]))

        pages = extract_pdf(str(pdf_path))

        assert len(pages) == 1

    def test_file_like_object_with_name_attribute(self):
        data = _make_pdf_bytes(["Uploaded content"])
        fake_upload = FakeUploadedFile(data, name="slides.pdf")

        pages = extract_pdf(fake_upload)

        assert pages[0].source == "slides.pdf"
        assert "Uploaded content" in pages[0].text

    def test_explicit_source_name_overrides_inferred_name(self, tmp_path):
        pdf_path = tmp_path / "on_disk_name.pdf"
        pdf_path.write_bytes(_make_pdf_bytes(["Content"]))

        pages = extract_pdf(pdf_path, source_name="display_name.pdf")

        assert pages[0].source == "display_name.pdf"

    def test_file_like_object_seek_position_does_not_matter(self):
        data = _make_pdf_bytes(["Content"])
        fake_upload = FakeUploadedFile(data, name="doc.pdf")
        fake_upload.seek(len(data))  # simulate a stream someone already read

        pages = extract_pdf(fake_upload)

        assert len(pages) == 1


# --- extract_pdf: error paths ----------------------------------------------


class TestExtractPdfErrorPaths:
    def test_corrupt_bytes_raise_pdf_processing_error(self, tmp_path):
        pdf_path = tmp_path / "corrupt.pdf"
        pdf_path.write_bytes(b"%PDF-1.7\nnot actually a valid pdf body")

        with pytest.raises(PDFProcessingError, match="Could not open"):
            extract_pdf(pdf_path)

    def test_nonexistent_path_raises_pdf_processing_error(self, tmp_path):
        missing = tmp_path / "does_not_exist.pdf"

        with pytest.raises(PDFProcessingError):
            extract_pdf(missing)

    def test_password_protected_pdf_raises_with_clear_message(self, tmp_path):
        pdf_path = tmp_path / "locked.pdf"
        pdf_path.write_bytes(_make_encrypted_pdf_bytes())

        with pytest.raises(PDFProcessingError, match="password-protected"):
            extract_pdf(pdf_path)

    def test_all_blank_pages_raises_no_extractable_text(self, tmp_path):
        pdf_path = tmp_path / "scanned.pdf"
        pdf_path.write_bytes(_make_pdf_bytes(["", "", ""]))

        with pytest.raises(PDFProcessingError, match="no extractable text"):
            extract_pdf(pdf_path)

    def test_file_like_object_without_name_and_no_source_name_raises(self):
        data = _make_pdf_bytes(["Content"])
        anonymous_stream = io.BytesIO(data)  # no .name attribute

        with pytest.raises(PDFProcessingError, match="source_name is required"):
            extract_pdf(anonymous_stream)

    def test_empty_file_raises_pdf_processing_error(self, tmp_path):
        pdf_path = tmp_path / "empty.pdf"
        pdf_path.write_bytes(b"")

        with pytest.raises(PDFProcessingError):
            extract_pdf(pdf_path)


# --- extract_pdfs (batch) --------------------------------------------------


class TestExtractPdfsBatch:
    def test_aggregates_pages_across_multiple_good_files(self, tmp_path):
        path_a = tmp_path / "a.pdf"
        path_a.write_bytes(_make_pdf_bytes(["A page one", "A page two"]))
        path_b = tmp_path / "b.pdf"
        path_b.write_bytes(_make_pdf_bytes(["B page one"]))

        result = extract_pdfs([path_a, path_b])

        assert isinstance(result, ExtractionResult)
        assert len(result.pages) == 3
        assert result.failed == []

    def test_isolates_a_failure_without_losing_the_good_files(self, tmp_path):
        good_path = tmp_path / "good.pdf"
        good_path.write_bytes(_make_pdf_bytes(["Good content"]))
        bad_path = tmp_path / "bad.pdf"
        bad_path.write_bytes(b"not a pdf at all")

        result = extract_pdfs([good_path, bad_path])

        assert len(result.pages) == 1
        assert result.pages[0].source == "good.pdf"
        assert len(result.failed) == 1
        assert result.failed[0][0] == "bad.pdf"
        assert "Could not open" in result.failed[0][1]

    def test_empty_file_list_returns_empty_result(self):
        result = extract_pdfs([])

        assert result.pages == []
        assert result.failed == []

    def test_succeeded_sources_is_sorted_and_deduplicated(self, tmp_path):
        path_b = tmp_path / "b_notes.pdf"
        path_b.write_bytes(_make_pdf_bytes(["x", "y"]))  # two pages, same source
        path_a = tmp_path / "a_notes.pdf"
        path_a.write_bytes(_make_pdf_bytes(["z"]))

        result = extract_pdfs([path_b, path_a])

        assert result.succeeded_sources == ["a_notes.pdf", "b_notes.pdf"]
