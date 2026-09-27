"""
Student RAG Chat — Streamlit entry point.

THROUGH PHASE 14: everything from Phase 13 (upload -> extract -> chunk
-> embed -> index -> chat with citations, follow-up resolution, and
student modes), plus real multi-PDF UX. Three changes, all built on
plumbing that already existed:

  1. "Process Documents" now only extracts/chunks/embeds files that
     aren't already indexed (matched by filename), and *adds* them to
     the existing VectorStore (VectorStore.add already supported
     growing an index) instead of discarding and rebuilding it. Before
     this, re-clicking "Process Documents" after staging one more PDF
     silently re-embedded every previously-processed file too, since
     Streamlit's uploader keeps earlier files in `uploaded` across
     reruns.
  2. A per-document list in the sidebar (VectorStore.list_sources(),
     Phase 14) shows page/chunk counts per PDF with a 🗑️ button that
     calls VectorStore.remove_source() to drop just that one document
     -- no re-embedding of anything else.
  3. A "Search in" multiselect lets a student temporarily narrow chat to
     a subset of indexed documents. Forwarded as answer_question's new
     source_filter straight down to VectorStore.search's new
     allowed_sources param (Phase 14) -- no new retrieval logic, just a
     filter threaded through the same three-stage pipeline.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import logging

import streamlit as st

from src.chat_memory import ChatMemoryError, resolve_question
from src.chunking import chunk_documents
from src.config import Config, ConfigError, load_config
from src.embeddings import EmbeddingError, embed_chunks, get_embeddings_client
from src.llm import get_llm_client
from src.pdf_processor import extract_pdfs
from src.prompts import VALID_MODES
from src.rag_pipeline import RAGPipelineError, answer_question
from src.vector_store import VectorStore, VectorStoreError, build_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Friendly labels for src.prompts.VALID_QUESTION_TYPES, shown only when
# mode == "Question Generator". "Mixed" (no selection) maps to None,
# which src.prompts.build_system_prompt() already treats as "let Gemini
# vary the question style" -- not a new option, just a label for one
# that already existed.
_QUESTION_TYPE_LABELS = {"mcq": "Multiple choice", "short": "Short answer", "long": "Long answer"}
_LABEL_TO_QUESTION_TYPE = {label: qtype for qtype, label in _QUESTION_TYPE_LABELS.items()}
_MIXED_LABEL = "Mixed"

st.set_page_config(
    page_title="Student RAG Chat",
    page_icon="📚",
    layout="wide",
)


def init_session_state(cfg: Config) -> None:
    """Set up st.session_state defaults exactly once per browser session.

    Using setdefault (rather than always assigning) is what stops
    Streamlit's rerun-on-every-interaction model from wiping out chat
    history, staged uploads, or a processed vector store on every click.
    """
    defaults = {
        "chat_history": [],  # list[{"role": "user"|"assistant", "content": str, "sources": list}]
        "uploaded_files": [],  # staged UploadedFile objects, not yet processed
        "documents_processed": False,
        "extracted_pages": [],  # list[PageContent], populated by Phase 3 extraction
        "chunks": [],  # list[Document], populated by Phase 4 chunking
        "failed_files": [],  # list[(filename, reason)] from the last processing run
        "vector_store": None,  # populated by Phase 6's build_vector_store, wired in Phase 10
        "embeddings_client": None,  # cached across questions/re-processing, Phase 10
        "llm_client": None,  # cached across questions, Phase 10
        "known_sources": set(),  # sources seen in the index as of the last rerun, Phase 14
        "selected_sources": set(),  # sources currently checked in the "Search in" filter, Phase 14
        "mode": "Normal Chat",
        "question_type_label": _MIXED_LABEL,
        "top_k": cfg.top_k,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _process_documents(cfg: Config, uploaded) -> None:
    """Run extract -> chunk -> embed -> index for newly staged uploads,
    updating session_state as each stage completes. Isolated into its
    own function so render_sidebar stays readable -- this is the one
    place all of Phases 3-6 meet, now joined by Phase 14's incremental
    add.

    Phase 14 change: only files whose name isn't already in the index
    (per VectorStore.list_sources()) are extracted/chunked/embedded.
    Streamlit's file_uploader with accept_multiple_files=True keeps
    every previously staged file in `uploaded` across reruns, so without
    this check, clicking "Process Documents" again after adding just one
    more PDF would re-run every earlier Gemini embedding call too. New
    chunks are added to the existing VectorStore (VectorStore.add
    already supported growing an index in place -- Phase 6 built it that
    way from the start) rather than the whole store being thrown away
    and rebuilt, so documents indexed earlier, and any the student has
    since removed with the per-document 🗑️ button, are left untouched.
    """
    store = st.session_state.vector_store
    already_indexed = {summary.source for summary in store.list_sources()} if store else set()
    to_process = [f for f in uploaded if f.name not in already_indexed]
    skipped = [f.name for f in uploaded if f.name in already_indexed]

    if not to_process:
        st.info("Every staged file is already indexed — nothing new to process.")
        return

    with st.spinner(f"Extracting text from {len(to_process)} PDF(s)..."):
        result = extract_pdfs(to_process)

    st.session_state.extracted_pages.extend(result.pages)
    st.session_state.failed_files = result.failed

    if not result.pages:
        if result.failed:
            for name, reason in result.failed:
                st.error(f"**{name}**: {reason}")
        return

    with st.spinner("Splitting into chunks..."):
        new_chunks = chunk_documents(
            result.pages,
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
        )
    st.session_state.chunks.extend(new_chunks)

    if not cfg.has_api_key:
        st.warning(
            f"Extracted {len(result.pages)} page(s) from "
            f"{len(result.succeeded_sources)} new file(s), split into "
            f"{len(new_chunks)} chunk(s). Add a Gemini API key to embed "
            "them and enable chat.",
            icon="⚠️",
        )
        if result.failed:
            for name, reason in result.failed:
                st.error(f"**{name}**: {reason}")
        return

    try:
        if st.session_state.embeddings_client is None:
            st.session_state.embeddings_client = get_embeddings_client(cfg)
        with st.spinner(f"Embedding {len(new_chunks)} chunk(s)..."):
            embedded = embed_chunks(new_chunks, st.session_state.embeddings_client)
        with st.spinner("Updating search index..."):
            if st.session_state.vector_store is None:
                st.session_state.vector_store = build_vector_store(embedded)
            else:
                st.session_state.vector_store.add(embedded)
    except (ConfigError, ValueError, EmbeddingError, VectorStoreError) as exc:
        st.error(f"Couldn't update the search index: {exc}")
    else:
        st.session_state.documents_processed = True
        total_store = st.session_state.vector_store
        st.success(
            f"Added {len(result.pages)} page(s) from "
            f"{len(result.succeeded_sources)} new file(s) → "
            f"{len(new_chunks)} chunk(s) indexed. "
            f"{len(total_store)} chunk(s) total across "
            f"{len(total_store.list_sources())} document(s), ready to chat.",
            icon="✅",
        )

    if skipped:
        st.caption(f"Already indexed, skipped: {', '.join(skipped)}")
    if result.failed:
        for name, reason in result.failed:
            st.error(f"**{name}**: {reason}")


def _remove_document(source: str) -> None:
    """Drop one indexed PDF via VectorStore.remove_source(), and keep
    every other bit of session_state (chunk preview, follow-up filter
    selection, known-sources tracking) consistent with the smaller
    index that results -- so the sidebar and chat filter never show a
    document that was just removed.
    """
    store = st.session_state.vector_store
    if store is None:
        return
    store.remove_source(source)
    st.session_state.extracted_pages = [
        p for p in st.session_state.extracted_pages if p.source != source
    ]
    st.session_state.chunks = [
        c for c in st.session_state.chunks if c.metadata.get("source") != source
    ]
    st.session_state.known_sources.discard(source)
    st.session_state.selected_sources.discard(source)
    if len(store) == 0:
        st.session_state.vector_store = None
        st.session_state.documents_processed = False


def _sync_selected_sources(store: VectorStore | None) -> None:
    """Keep session_state.selected_sources in step with what's actually
    indexed, right before the "Search in" multiselect (below) reads it.

    A document newly added since the last rerun is auto-selected --
    otherwise a student who ticks "select all" once would have to
    re-tick it every time they add another PDF, which defeats the
    point of a filter that defaults to "everything." A document that's
    been removed is dropped from the selection outright, since keeping
    it would leave st.multiselect holding a value outside its own
    `options` list, which Streamlit rejects. Anything the student
    deliberately unchecked stays unchecked -- it's neither new nor
    removed, so neither branch below touches it.
    """
    current = {summary.source for summary in store.list_sources()} if store else set()
    known = st.session_state.known_sources
    newly_added = current - known
    removed = known - current
    selected = set(st.session_state.selected_sources)
    st.session_state.selected_sources = (selected - removed) | newly_added
    st.session_state.known_sources = current


def render_sidebar(cfg: Config) -> None:
    with st.sidebar:
        st.header("📄 Documents")
        uploaded = st.file_uploader(
            "Drag & drop PDF study material",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDFs: lecture notes, textbooks, slides.",
        )
        if uploaded:
            st.session_state.uploaded_files = uploaded
            for f in uploaded:
                st.caption(f"📎 {f.name} — {f.size / 1024:.1f} KB")

        col1, col2 = st.columns(2)
        with col1:
            process_clicked = st.button(
                "Process Documents",
                type="primary",
                use_container_width=True,
                disabled=not uploaded,
            )
        with col2:
            if st.button("Clear Documents", use_container_width=True):
                st.session_state.uploaded_files = []
                st.session_state.documents_processed = False
                st.session_state.extracted_pages = []
                st.session_state.chunks = []
                st.session_state.failed_files = []
                st.session_state.vector_store = None
                st.session_state.known_sources = set()
                st.session_state.selected_sources = set()
                st.rerun()

        if process_clicked:
            _process_documents(cfg, uploaded)

        source_summaries = st.session_state.vector_store.list_sources() if st.session_state.vector_store else []
        if source_summaries:
            st.divider()
            st.subheader(f"🗂️ Indexed documents ({len(source_summaries)})")
            for summary in source_summaries:
                doc_col, remove_col = st.columns([4, 1])
                with doc_col:
                    st.caption(f"📄 **{summary.source}** — {summary.pages} page(s), {summary.chunks} chunk(s)")
                with remove_col:
                    if st.button("🗑️", key=f"remove_{summary.source}", help=f"Remove {summary.source} from the index"):
                        _remove_document(summary.source)
                        st.rerun()

            _sync_selected_sources(st.session_state.vector_store)
            st.multiselect(
                "Search in",
                options=sorted(s.source for s in source_summaries),
                key="selected_sources",
                help="Only checked documents are searched when you ask a "
                "question. Uncheck a document to leave it indexed but "
                "excluded from chat, without removing it.",
            )

        st.divider()
        st.header("⚙️ Settings")
        st.selectbox(
            "Mode",
            list(VALID_MODES),
            key="mode",
            help="Changes how Gemini is instructed to answer -- see "
            "_MODE_INSTRUCTIONS in src/prompts.py for exactly what each "
            "mode tells it to do.",
        )
        if st.session_state.mode == "Question Generator":
            st.selectbox(
                "Question type",
                [_MIXED_LABEL] + list(_QUESTION_TYPE_LABELS.values()),
                key="question_type_label",
                help="Only applies in Question Generator mode.",
            )
        st.slider("Top-K chunks retrieved", 1, 10, key="top_k")
        st.slider(
            "Chunk size",
            200,
            2000,
            step=50,
            key="chunk_size",
            help="Applies the next time you click Process Documents.",
        )
        st.slider(
            "Chunk overlap",
            0,
            500,
            step=25,
            key="chunk_overlap",
            help="Applies the next time you click Process Documents.",
        )

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


def _render_sources(sources) -> None:
    """Render an assistant message's retrieved chunks as real citations.

    Groups by (source, page) rather than listing every chunk, since a
    page split across multiple chunks (src.chunking never merges across
    a page boundary, but a long page can still produce 2+ chunks on it)
    would otherwise show the same page twice for one answer. The first
    time a (source, page) pair is seen wins the slot -- SearchResults
    already arrive most-relevant-first from src.retriever, so keeping
    first-seen order keeps that ranking intact after deduping.

    Each entry expands to the actual chunk text so a student can check
    whether Gemini's inline "(source, p. N)" citation (see
    src.prompts._BASE_INSTRUCTIONS) is a fair reading of the source,
    rather than trusting the citation blindly. The bracket numbers here
    are this panel's own footnote numbering, not the same as the [n]
    labels src.prompts.format_context() puts in Gemini's own context --
    Gemini is instructed to cite as "(source, p. N)" plain text, never
    by bracket index, so there's nothing for these numbers to conflict
    with.
    """
    if not sources:
        return

    unique: dict[tuple[str, object], object] = {}
    for result in sources:
        meta = result.document.metadata
        key = (meta.get("source", "unknown"), meta.get("page", "?"))
        unique.setdefault(key, result)

    with st.expander(f"📚 Sources ({len(unique)})"):
        for i, ((source, page), result) in enumerate(unique.items(), start=1):
            st.markdown(f"**[{i}] {source} — page {page}**")
            preview = result.document.page_content.strip()
            st.caption(preview[:400] + ("…" if len(preview) > 400 else ""))


def render_main(cfg: Config) -> None:
    st.title("📚 Student RAG Chat")
    st.caption("Upload your study material, then ask questions grounded in it.")

    if not cfg.has_api_key:
        st.warning(
            "**No Gemini API key configured.** Copy `.env.example` to `.env` "
            "and set `GOOGLE_API_KEY=your_key_here` (free key: "
            "https://aistudio.google.com/apikey), then restart the app.",
            icon="⚠️",
        )
    else:
        st.success(f"Gemini configured — chat model: `{cfg.gemini_model}`", icon="✅")

    if not st.session_state.uploaded_files:
        st.info("Upload one or more PDFs from the sidebar to get started.")
    elif not st.session_state.documents_processed:
        st.info(
            f"{len(st.session_state.uploaded_files)} document(s) staged. "
            "Click **Process Documents** in the sidebar."
        )
    elif st.session_state.chunks:
        indexed = st.session_state.vector_store is not None
        label = (
            f"✅ {len(st.session_state.extracted_pages)} page(s) → "
            f"{len(st.session_state.chunks)} chunk(s)"
            + (" — indexed, ready to chat" if indexed else " — not yet indexed")
        )
        with st.expander(label):
            for chunk in st.session_state.chunks[:5]:
                meta = chunk.metadata
                st.markdown(f"**{meta['source']} — page {meta['page']}, chunk {meta['chunk_index']}**")
                preview = chunk.page_content[:300]
                st.caption(preview + ("…" if len(chunk.page_content) > 300 else ""))
            if len(st.session_state.chunks) > 5:
                st.caption(f"…and {len(st.session_state.chunks) - 5} more chunk(s).")

    st.divider()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_sources(message.get("sources", []))

    indexed = st.session_state.vector_store is not None
    selected = st.session_state.selected_sources
    chat_ready = cfg.has_api_key and indexed and bool(selected)

    if indexed:
        total_sources = len(st.session_state.vector_store.list_sources())
        if selected and len(selected) < total_sources:
            st.caption(f"💬 Answering using {len(selected)} of {total_sources} indexed document(s).")
        elif not selected:
            st.caption("⚠️ No documents selected — check at least one under **Search in** to chat.")

    if not chat_ready:
        if not cfg.has_api_key:
            placeholder = "Add a Gemini API key to chat..."
        elif not indexed:
            placeholder = "Process documents to chat..."
        else:
            placeholder = "Select at least one document to chat..."
    else:
        placeholder = "Ask a question about your documents..."

    question = st.chat_input(placeholder, disabled=not chat_ready)

    if question:
        # History *before* this turn -- resolve_question must not resolve
        # the question against itself, so this is captured before appending.
        prior_history = list(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "user", "content": question, "sources": []})
        try:
            if st.session_state.llm_client is None:
                st.session_state.llm_client = get_llm_client(cfg)

            try:
                resolved_question = resolve_question(
                    question, prior_history, st.session_state.llm_client
                )
            except (ValueError, ChatMemoryError) as exc:
                # A broken follow-up rewrite degrades to "no memory this
                # turn," not "no answer at all" -- see ChatMemoryError's
                # docstring. The student's own question still gets answered.
                logger.warning("Falling back to the raw question: %s", exc)
                resolved_question = question

            with st.spinner("Thinking..."):
                question_type = _LABEL_TO_QUESTION_TYPE.get(st.session_state.question_type_label)
                result = answer_question(
                    resolved_question,
                    st.session_state.vector_store,
                    cfg,
                    mode=st.session_state.mode,
                    question_type=question_type,
                    k=st.session_state.top_k,
                    embeddings_client=st.session_state.embeddings_client,
                    llm_client=st.session_state.llm_client,
                    source_filter=set(st.session_state.selected_sources),
                )
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result.answer, "sources": result.sources}
            )
        except (ConfigError, RAGPipelineError) as exc:
            logger.exception("Failed to answer question")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"⚠️ Sorry, I couldn't answer that: {exc}", "sources": []}
            )
        st.rerun()


def main() -> None:
    try:
        cfg = load_config()
    except ConfigError as exc:
        st.error(f"⚠️ Configuration error: {exc}")
        st.stop()
        return

    init_session_state(cfg)
    render_sidebar(cfg)
    render_main(cfg)


if __name__ == "__main__":
    main()
