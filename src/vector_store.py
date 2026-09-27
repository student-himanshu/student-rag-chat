"""
src/vector_store.py
FAISS vector store for Student RAG Chat.

Hand-rolled wrapper around `faiss-cpu` directly (see README's architecture
note on why: `langchain_community.vectorstores.FAISS` depends on the
sunset/archived `langchain-community` package, with no replacement partner
package published).

FAISS itself only ever knows vectors and integer ids -- it has no concept
of the chunk text or {source, page} metadata that produced a vector. This
module bridges that gap with `IndexIDMap`, which lets us assign our own
sequential integer id to each vector, plus a parallel `_docstore` dict
keyed by the `chunk_id` string that src.chunking already produces
(f"{source}::p{page}::c{chunk_index}"). A search hit's integer id is
mapped -> chunk_id -> Document, which is how a similarity-search result
gets back to something Phase 7's retriever / Phase 11's citations can use.

Similarity metric: IndexFlatL2 (Euclidean distance over the raw vectors),
not inner product / cosine. This is deliberate rather than an oversight --
it works correctly regardless of whether GoogleGenerativeAIEmbeddings
returns unit-normalized vectors, which was not independently confirmed
(see README's note on verifying library behavior before relying on it).
A lower `SearchResult.score` means more similar. If Phase 7 wants a
0-1 similarity number instead of a raw distance, convert there rather
than changing the metric here, since flipping to IndexFlatIP silently
gives wrong results if the vectors turn out not to be normalized.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document

from src.embeddings import EmbeddedChunk

logger = logging.getLogger(__name__)

INDEX_FILENAME = "index.faiss"
DOCSTORE_FILENAME = "docstore.json"


class VectorStoreError(RuntimeError):
    """Raised when the vector store can't be built, saved, loaded, or searched."""


@dataclass(frozen=True)
class SearchResult:
    """One similarity-search hit: the chunk plus its distance.

    `document` carries the same {source, page, chunk_index, chunk_id}
    metadata that src.chunking attaches, so Phase 11's citations can read
    it directly off a search result without a second lookup.
    """

    document: Document
    score: float


@dataclass(frozen=True)
class SourceSummary:
    """Per-document chunk/page counts for one indexed PDF.

    Phase 14's per-document sidebar list (app.py) is built entirely from
    these -- one row per SourceSummary, so a student can see, for every
    PDF they've uploaded, how much of it actually made it into the
    index without having to open the chunk-preview expander and count.
    """

    source: str
    pages: int
    chunks: int


class VectorStore:
    """A FAISS IndexFlatL2 (wrapped in IndexIDMap) plus a chunk_id docstore.

    Not a frozen dataclass like the project's other value objects
    (EmbeddedChunk, PageContent) -- this wraps a stateful FAISS index that
    grows via `.add()`, so it's a service object, not a value.
    """

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise VectorStoreError(f"Embedding dimension must be positive, got {dimension}.")
        self.dimension = dimension
        self._index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
        self._docstore: dict[str, Document] = {}
        self._id_to_chunk_id: dict[int, str] = {}
        self._next_id = 0

    def __len__(self) -> int:
        return len(self._docstore)

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Add embedded chunks to the index.

        Skips nothing and fails the whole call on the first bad chunk
        (unlike src.pdf_processor's batch extraction) -- a dimension
        mismatch or missing chunk_id here means something upstream is
        broken, not that one PDF among several was unreadable, so there's
        no partial-success case worth preserving.

        Raises:
            VectorStoreError: a vector's dimension doesn't match the
                store's, or a chunk is missing 'chunk_id' metadata.
        """
        if not embedded_chunks:
            return

        vectors: list[list[float]] = []
        ids: list[int] = []
        for embedded in embedded_chunks:
            vector = embedded.vector
            if len(vector) != self.dimension:
                raise VectorStoreError(
                    f"Vector for chunk_id={embedded.chunk.metadata.get('chunk_id')!r} has "
                    f"dimension {len(vector)}, expected {self.dimension}."
                )
            chunk_id = embedded.chunk.metadata.get("chunk_id")
            if not chunk_id:
                raise VectorStoreError(
                    "Chunk is missing 'chunk_id' metadata -- did it come from "
                    "src.chunking.chunk_documents()?"
                )
            faiss_id = self._next_id
            self._next_id += 1
            self._id_to_chunk_id[faiss_id] = chunk_id
            self._docstore[chunk_id] = embedded.chunk
            vectors.append(vector)
            ids.append(faiss_id)

        matrix = np.array(vectors, dtype="float32")
        id_array = np.array(ids, dtype="int64")
        self._index.add_with_ids(matrix, id_array)
        logger.info(
            "Added %d vector(s) to the index (total now %d)", len(embedded_chunks), len(self)
        )

    def search(
        self,
        query_vector: list[float],
        k: int,
        allowed_sources: set[str] | None = None,
    ) -> list[SearchResult]:
        """Return the k nearest chunks to query_vector, nearest first.

        Returns fewer than k results (down to an empty list) if the store
        holds fewer than k vectors, rather than raising -- an empty or
        small index is a normal state early on, not an error.

        Args:
            query_vector: the embedded question.
            k: how many results to return.
            allowed_sources: Phase 14's per-document chat filter. When
                given, only chunks whose `metadata["source"]` is in this
                set are eligible, and the k returned are the k *nearest
                among those*, not "the top k overall, then filtered down"
                -- the latter would silently return fewer than k (or
                zero) results the moment a student excludes even one of
                several close matches. `None` (the default) searches
                every indexed document, matching every caller before
                Phase 14. Passing an empty set is a valid way to say
                "search nothing" and short-circuits to `[]` without
                touching FAISS at all.

        Raises:
            VectorStoreError: k isn't positive, or query_vector's
                dimension doesn't match the store's.
        """
        if k <= 0:
            raise VectorStoreError(f"k must be a positive integer, got {k}.")
        if len(query_vector) != self.dimension:
            raise VectorStoreError(
                f"Query vector has dimension {len(query_vector)}, expected {self.dimension}."
            )
        if allowed_sources is not None and not allowed_sources:
            return []
        if len(self) == 0:
            return []

        # With a source filter, we can't ask FAISS for just k results --
        # the nearest k overall might all belong to excluded documents,
        # even though plenty of matches exist among the allowed ones.
        # This index is IndexFlatL2 (exact, brute-force distance over
        # every vector, not an approximate/graph index), so searching
        # everything and filtering in Python is exact, not an
        # approximation of what a real filtered search would return --
        # there's no recall trade-off being made here.
        k_search = len(self) if allowed_sources is not None else min(k, len(self))
        query = np.array([query_vector], dtype="float32")
        distances, ids = self._index.search(query, k_search)

        results: list[SearchResult] = []
        for faiss_id, distance in zip(ids[0], distances[0]):
            if faiss_id == -1:  # FAISS pads short result rows with -1
                continue
            chunk_id = self._id_to_chunk_id[int(faiss_id)]
            document = self._docstore[chunk_id]
            if allowed_sources is not None and document.metadata.get("source") not in allowed_sources:
                continue
            results.append(SearchResult(document=document, score=float(distance)))
            if len(results) == k:
                break
        return results

    def list_sources(self) -> list[SourceSummary]:
        """Return a per-document chunk/page breakdown of everything
        currently indexed, sorted by filename.

        Backs Phase 14's sidebar document list and the "search in"
        filter's available options -- both need exactly this shape
        (which files, and how much of each), and both should reflect the
        live index rather than whatever was staged in the uploader.
        """
        counts: dict[str, dict[str, object]] = {}
        for document in self._docstore.values():
            source = document.metadata.get("source", "unknown")
            info = counts.setdefault(source, {"pages": set(), "chunks": 0})
            info["pages"].add(document.metadata.get("page"))
            info["chunks"] += 1
        return sorted(
            (
                SourceSummary(source=source, pages=len(info["pages"]), chunks=info["chunks"])
                for source, info in counts.items()
            ),
            key=lambda summary: summary.source,
        )

    def remove_source(self, source: str) -> int:
        """Remove every chunk belonging to `source` from the index.

        Lets a student drop one PDF from an otherwise-built index
        without rebuilding the whole thing from scratch -- important
        once several PDFs share one store (Phase 14), since re-embedding
        every remaining document just to drop one would be slow and
        burns Gemini embedding calls for no reason.

        Uses FAISS's own `remove_ids` (supported by IndexFlatL2, which
        this store wraps) rather than rebuilding the index, so the
        vectors for every other document are untouched -- no
        re-embedding, no id reassignment, no drift in `_next_id`.

        Returns:
            The number of chunks removed. 0 if `source` isn't currently
            indexed -- not an error, since a student clicking "remove"
            on a document that's already gone (e.g. a stale rerun) is a
            harmless race, not a bug worth raising over.
        """
        ids_to_remove = [
            faiss_id
            for faiss_id, chunk_id in self._id_to_chunk_id.items()
            if self._docstore[chunk_id].metadata.get("source") == source
        ]
        if not ids_to_remove:
            return 0

        self._index.remove_ids(np.array(ids_to_remove, dtype="int64"))
        for faiss_id in ids_to_remove:
            chunk_id = self._id_to_chunk_id.pop(faiss_id)
            self._docstore.pop(chunk_id, None)

        logger.info(
            "Removed %d chunk(s) for source '%s' (%d remaining in index)",
            len(ids_to_remove),
            source,
            len(self),
        )
        return len(ids_to_remove)

    def save(self, directory: Path) -> None:
        """Persist the index and docstore to `directory` (e.g. vectorstore/).

        Two files: a native FAISS index (`index.faiss`) and a JSON
        docstore (`docstore.json`) carrying id<->chunk_id mapping plus
        each chunk's page_content/metadata. JSON over pickle so the saved
        store is human-inspectable and isn't tied to exact class
        pickling behavior across langchain_core versions.
        """
        directory.mkdir(parents=True, exist_ok=True)
        try:
            faiss.write_index(self._index, str(directory / INDEX_FILENAME))
        except Exception as exc:
            raise VectorStoreError(f"Failed to write FAISS index to {directory}: {exc}") from exc

        payload = {
            "dimension": self.dimension,
            "next_id": self._next_id,
            "id_to_chunk_id": self._id_to_chunk_id,
            "chunks": {
                chunk_id: {"page_content": doc.page_content, "metadata": doc.metadata}
                for chunk_id, doc in self._docstore.items()
            },
        }
        try:
            (directory / DOCSTORE_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            raise VectorStoreError(f"Failed to write docstore to {directory}: {exc}") from exc
        logger.info("Saved vector store (%d chunks) to %s", len(self), directory)

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        """Load a store previously written by `.save()`.

        Raises:
            VectorStoreError: the index/docstore files aren't both
                present, or either fails to parse.
        """
        index_path = directory / INDEX_FILENAME
        docstore_path = directory / DOCSTORE_FILENAME
        if not index_path.exists() or not docstore_path.exists():
            raise VectorStoreError(f"No saved vector store found in {directory}.")

        try:
            payload = json.loads(docstore_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError(f"Failed to read docstore from {directory}: {exc}") from exc

        store = cls(dimension=payload["dimension"])
        try:
            store._index = faiss.read_index(str(index_path))
        except Exception as exc:
            raise VectorStoreError(f"Failed to read FAISS index from {directory}: {exc}") from exc

        store._next_id = payload["next_id"]
        store._id_to_chunk_id = {int(k): v for k, v in payload["id_to_chunk_id"].items()}
        store._docstore = {
            chunk_id: Document(page_content=data["page_content"], metadata=data["metadata"])
            for chunk_id, data in payload["chunks"].items()
        }
        logger.info("Loaded vector store (%d chunks) from %s", len(store), directory)
        return store


def build_vector_store(embedded_chunks: list[EmbeddedChunk]) -> VectorStore:
    """Build a fresh VectorStore from a non-empty list of embedded chunks.

    The dimension is inferred from the first chunk's vector rather than
    taken as a parameter, since it's always determined by whatever
    embedding model produced the vectors -- passing it separately would
    just be one more way for the two to drift out of sync.

    Raises:
        VectorStoreError: embedded_chunks is empty (there's no vector to
            infer a dimension from).
    """
    if not embedded_chunks:
        raise VectorStoreError("Cannot build a vector store from an empty list of embedded chunks.")
    dimension = len(embedded_chunks[0].vector)
    store = VectorStore(dimension=dimension)
    store.add(embedded_chunks)
    return store
