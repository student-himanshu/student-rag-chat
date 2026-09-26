"""
Student RAG Chat — Streamlit entry point.

THROUGH PHASE 12: everything from Phase 11 (upload -> extract -> chunk
-> embed -> index -> chat with source citations), plus follow-up
resolution. Before a question reaches src.rag_pipeline.answer_question,
src.chat_memory.resolve_question() rewrites it into a standalone
question using the last few turns of chat_history -- so "why is it
useful?" retrieves and gets answered as if the student had asked "why
is mitochondria's role in energy production useful?" instead of
searching for the literal, contextless word "it". If that rewrite call
fails for any reason, app.py falls back to the raw question rather than
failing the turn -- see _process_question() below.

Still fixed/disabled this phase (see README, "Implementation phases"):
mode is always "Normal Chat" (Phase 13 enables the selector).

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
from src.rag_pipeline import RAGPipelineError, answer_question
from src.vector_store import VectorStoreError, build_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        "mode": "Normal Chat",
        "top_k": cfg.top_k,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _process_documents(cfg: Config, uploaded) -> None:
    """Run the full extract -> chunk -> embed -> index pipeline for the
    currently staged uploads, updating session_state as each stage
    completes. Isolated into its own function so render_sidebar stays
    readable -- this is the one place all of Phases 3-6 meet.
    """
    with st.spinner("Extracting text from PDF(s)..."):
        result = extract_pdfs(uploaded)

    st.session_state.extracted_pages = result.pages
    st.session_state.failed_files = result.failed
    st.session_state.documents_processed = bool(result.pages)
    st.session_state.vector_store = None  # any previous index is now stale

    if not result.pages:
        if result.failed:
            for name, reason in result.failed:
                st.error(f"**{name}**: {reason}")
        return

    with st.spinner("Splitting into chunks..."):
        st.session_state.chunks = chunk_documents(
            result.pages,
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
        )

    if not cfg.has_api_key:
        st.warning(
            f"Extracted {len(result.pages)} page(s), split into "
            f"{len(st.session_state.chunks)} chunk(s). Add a Gemini API "
            "key to embed them and enable chat.",
            icon="⚠️",
        )
        if result.failed:
            for name, reason in result.failed:
                st.error(f"**{name}**: {reason}")
        return

    try:
        if st.session_state.embeddings_client is None:
            st.session_state.embeddings_client = get_embeddings_client(cfg)
        with st.spinner(f"Embedding {len(st.session_state.chunks)} chunk(s)..."):
            embedded = embed_chunks(st.session_state.chunks, st.session_state.embeddings_client)
        with st.spinner("Building search index..."):
            st.session_state.vector_store = build_vector_store(embedded)
    except (ConfigError, ValueError, EmbeddingError, VectorStoreError) as exc:
        st.session_state.vector_store = None
        st.error(f"Couldn't build the search index: {exc}")
    else:
        st.success(
            f"Extracted {len(result.pages)} page(s) from "
            f"{len(result.succeeded_sources)} file(s) → "
            f"{len(st.session_state.chunks)} chunk(s) → indexed and "
            "ready to chat.",
            icon="✅",
        )

    if result.failed:
        for name, reason in result.failed:
            st.error(f"**{name}**: {reason}")


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
                st.rerun()

        if process_clicked:
            _process_documents(cfg, uploaded)

        st.divider()
        st.header("⚙️ Settings")
        st.selectbox(
            "Mode",
            [
                "Normal Chat",
                "Explain Simply",
                "Exam Mode",
                "Problem Solver",
                "Summary",
                "Question Generator",
            ],
            key="mode",
            disabled=True,
            help="Student modes are wired into the prompt builder already — "
            "the selector itself is enabled in Phase 13.",
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

    chat_ready = cfg.has_api_key and st.session_state.vector_store is not None
    if not chat_ready:
        placeholder = (
            "Add a Gemini API key to chat..."
            if not cfg.has_api_key
            else "Process documents to chat..."
        )
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
                result = answer_question(
                    resolved_question,
                    st.session_state.vector_store,
                    cfg,
                    mode=st.session_state.mode,
                    k=st.session_state.top_k,
                    embeddings_client=st.session_state.embeddings_client,
                    llm_client=st.session_state.llm_client,
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
