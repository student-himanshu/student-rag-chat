"""
PDF processing for Student RAG Chat.

Extracts per-page text from uploaded PDFs using PyMuPDF, cleans it, and
returns plain data structures carrying {text, source, page} -- the
metadata that chunking (Phase 4) and source citations (Phase 11) both
depend on.

Deliberately does NOT depend on langchain_core.documents.Document here:
that container is introduced when chunking wraps this raw per-page text
into chunks (Phase 4). Keeping this module's only real dependency
PyMuPDF makes it independently testable and reusable outside a
LangChain pipeline if that's ever needed.

Import convention: `import pymupdf`, not the deprecated `fitz` alias
(see README, Tech stack table).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union

import pymupdf

logger = logging.getLogger(__name__)


class PDFProcessingError(RuntimeError):
    """Raised when a PDF cannot be opened, is password-protected, or
    contains no extractable text at all (e.g. a fully scanned PDF with
    no text layer -- this app does not do OCR)."""


@dataclass(frozen=True)
class PageContent:
    """One page of cleaned text plus the metadata every later phase
    needs to build a source citation."""

    source: str  # original filename, e.g. "lecture_04.pdf"
    page: int  # 1-indexed page number -- matches what a student sees on screen
    text: str  # cleaned page text


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of processing a batch of PDFs.

    Partial failure is a normal case here, not an error: if a student
    uploads three PDFs and one is corrupt, the other two should still
    become searchable rather than the whole upload failing.
    """

    pages: list[PageContent]
    failed: list[tuple[str, str]]  # (filename, human-readable reason)

    @property
    def succeeded_sources(self) -> list[str]:
        return sorted({p.source for p in self.pages})


# --- Text cleaning ----------------------------------------------------


def clean_text(raw: str) -> str:
    """Normalize whitespace-mangled PDF extraction output.

    PyMuPDF's plain text extraction commonly produces:
      - hyphenated line-break words: "informa-\\ntion" -> should rejoin
      - runs of blank lines from page headers/footers/margins
      - stray null characters
      - trailing whitespace per line
      - runs of spaces from multi-column layouts collapsing oddly

    Keeping this predictable matters because chunking (Phase 4) splits
    on character count -- mangled whitespace shifts chunk boundaries
    and can split a sentence mid-word.
    """
    if not raw:
        return ""

    text = raw.replace("\x00", "")
    # Rejoin hyphenated line-break words: "informa-\ntion" -> "information"
    text = re.sub(r"-\n(?=\w)", "", text)
    # Collapse 3+ newlines (blank-line runs) down to a single paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Collapse runs of spaces/tabs (not newlines) into one space
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# --- Single-file extraction --------------------------------------------


def extract_pdf(
    file: Union[str, Path, BinaryIO],
    source_name: str | None = None,
) -> list[PageContent]:
    """Extract cleaned per-page text from a single PDF.

    Args:
        file: a filesystem path (str or Path), or a file-like object
            such as Streamlit's UploadedFile (behaves like BytesIO).
        source_name: display name recorded in each PageContent.source,
            used later for citations. If omitted, this is inferred from
            `file` when it's a path, or from `file.name` when it's a
            file-like object that has one (Streamlit's UploadedFile
            does). Required if neither is available.

    Returns:
        One PageContent per page that has extractable text. Pages that
        are blank or image-only are skipped rather than returned as
        empty entries -- a citation pointing to an empty page is
        useless to a student.

    Raises:
        PDFProcessingError: if the file can't be opened at all (corrupt
            data, wrong format), if it's password-protected, or if
            every page turns out to have no extractable text (most
            likely a fully scanned PDF with no OCR text layer).
    """
    resolved_name = source_name or _infer_source_name(file)
    if not resolved_name:
        raise PDFProcessingError(
            "source_name is required when `file` is not a path and has no "
            "`.name` attribute."
        )

    doc = _open_pdf(file, resolved_name)

    try:
        if doc.is_encrypted:
            raise PDFProcessingError(
                f"'{resolved_name}' is password-protected. Remove the "
                "password and re-upload -- encrypted PDFs aren't supported."
            )

        total_page_count = doc.page_count
        pages: list[PageContent] = []
        for page_index in range(total_page_count):
            raw_text = doc[page_index].get_text()
            cleaned = clean_text(raw_text)
            if cleaned:
                pages.append(
                    PageContent(source=resolved_name, page=page_index + 1, text=cleaned)
                )
    finally:
        doc.close()

    if not pages:
        raise PDFProcessingError(
            f"'{resolved_name}' has no extractable text on any page. If "
            "this is a scanned document, note that OCR isn't supported yet."
        )

    logger.info(
        "Extracted %d page(s) with text from '%s' (%d total pages)",
        len(pages),
        resolved_name,
        total_page_count,
    )
    return pages


def _infer_source_name(file: Union[str, Path, BinaryIO]) -> str | None:
    if isinstance(file, (str, Path)):
        return Path(file).name
    return getattr(file, "name", None)


def _open_pdf(file: Union[str, Path, BinaryIO], resolved_name: str) -> pymupdf.Document:
    try:
        if isinstance(file, (str, Path)):
            return pymupdf.open(file)
        # File-like object (e.g. Streamlit UploadedFile): read bytes and
        # open as a stream, since these objects aren't always seekable
        # in the way pymupdf expects.
        if hasattr(file, "seek"):
            file.seek(0)
        data = file.read()
        return pymupdf.open(stream=data, filetype="pdf")
    except PDFProcessingError:
        raise
    except Exception as exc:  # pymupdf raises its own exception types
        raise PDFProcessingError(
            f"Could not open '{resolved_name}' as a PDF: {exc}"
        ) from exc


# --- Batch extraction (multiple uploaded files) -------------------------


def extract_pdfs(files: list[Union[str, Path, BinaryIO]]) -> ExtractionResult:
    """Extract text from multiple PDFs, isolating failures per file.

    Mirrors the sidebar's `accept_multiple_files=True` uploader in
    app.py: one bad PDF in the batch is reported, not fatal to the rest.
    """
    all_pages: list[PageContent] = []
    failed: list[tuple[str, str]] = []

    for file in files:
        name = _infer_source_name(file) or "<unknown file>"
        try:
            all_pages.extend(extract_pdf(file, source_name=name))
        except PDFProcessingError as exc:
            logger.warning("Skipping '%s': %s", name, exc)
            failed.append((name, str(exc)))

    return ExtractionResult(pages=all_pages, failed=failed)
