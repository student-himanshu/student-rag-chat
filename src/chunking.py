"""
Chunking for Student RAG Chat.

Wraps per-page text (src.pdf_processor.PageContent) into LangChain
Documents, then splits those into embedding-sized chunks with
RecursiveCharacterTextSplitter -- while keeping the {source, page}
metadata that Phase 11's citations depend on.

This is the module boundary where the codebase starts depending on
langchain_core.documents.Document -- see README's architecture note on
why src/pdf_processor.py deliberately does not.
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.pdf_processor import PageContent

logger = logging.getLogger(__name__)

# Tried in order: paragraph breaks first, then lines, then sentences,
# then words, then raw characters as a last resort. Matches the
# whitespace shape clean_text() (Phase 3) normalizes PDF pages into.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def pages_to_documents(pages: list[PageContent]) -> list[Document]:
    """Wrap each PageContent into its own LangChain Document.

    One page = one Document at this stage (pre-split), carrying
    {source, page} metadata. Kept as a separate, directly testable step
    from chunk_documents() since it's also useful on its own wherever a
    plain per-page Document list is needed without splitting.
    """
    return [
        Document(page_content=page.text, metadata={"source": page.source, "page": page.page})
        for page in pages
    ]


def chunk_documents(
    pages: list[PageContent],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split per-page text into embedding-sized chunks.

    Splitting happens per page, never merging text across a page
    boundary -- a chunk that spanned two pages would make Phase 11's
    "page N" citation wrong for whichever half of it came from the
    other page. The trade-off is chunks right at a page boundary can be
    shorter than chunk_size; that's judged worth it for citation
    accuracy in a study tool.

    Each returned Document's metadata carries:
      - source: original filename (from PageContent.source)
      - page: 1-indexed page number (from PageContent.page)
      - chunk_index: 0-indexed position of this chunk within its page
      - chunk_id: f"{source}::p{page}::c{chunk_index}", a stable ID
        Phase 6's vector store can use as a FAISS docstore key

    Args:
        pages: extracted, cleaned pages from src.pdf_processor.
        chunk_size: target characters per chunk.
        chunk_overlap: characters of overlap between consecutive chunks
            from the same page.

    Returns:
        Chunks in stable order: grouped by page in input order, then by
        position within that page. Empty input returns an empty list.

    Raises:
        ValueError: if chunk_size isn't positive, or chunk_overlap is
            negative or >= chunk_size -- mirrors the guard already
            enforced in src/config.py, so this function is safe to call
            directly (e.g. in tests) without going through Config.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size}), or splitting will loop forever."
        )

    if not pages:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=DEFAULT_SEPARATORS,
    )

    chunks: list[Document] = []
    for page_doc in pages_to_documents(pages):
        page_chunks = splitter.split_documents([page_doc])
        for chunk_index, chunk in enumerate(page_chunks):
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunk_id"] = (
                f"{chunk.metadata['source']}::p{chunk.metadata['page']}::c{chunk_index}"
            )
            chunks.append(chunk)

    logger.info(
        "Chunked %d page(s) into %d chunk(s) (chunk_size=%d, chunk_overlap=%d)",
        len(pages),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
