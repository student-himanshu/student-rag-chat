# Student RAG Chat

A ChatGPT-style study assistant that answers questions **grounded in PDFs you upload** — lecture notes, textbooks, slides — instead of the model's general knowledge. Every answer that draws on your documents comes with the source PDF and page number.

> **Status: Phase 14 of 18 (Multiple PDFs) complete.** Uploading several PDFs already worked pipeline-side since Phase 3 (`extract_pdfs` always accepted a list); what Phase 14 adds is everything that makes managing several of them at once usable:
>
> - **Incremental processing.** "Process Documents" now only extracts/chunks/embeds files that aren't already indexed (matched by filename) and _adds_ them to the existing `VectorStore` via `VectorStore.add()` instead of discarding and rebuilding the whole index. Previously, re-clicking "Process Documents" after staging one more PDF silently re-embedded every earlier file too, since Streamlit's `accept_multiple_files=True` uploader keeps prior selections across reruns.
> - **Per-document management.** A new sidebar section lists every indexed PDF with its page/chunk counts (`VectorStore.list_sources()`) and a 🗑️ button that removes just that one document (`VectorStore.remove_source()`, backed by FAISS's own `remove_ids` — no re-embedding of anything else).
> - **Per-document chat filter.** A "Search in" multiselect lets a student narrow a question to a subset of indexed PDFs. Threaded through as `source_filter` → `retrieve(allowed_sources=...)` → `VectorStore.search(allowed_sources=...)`, which searches every vector then filters (this index is exact/brute-force `IndexFlatL2`, so that's not an approximation) — filtering happens _before_ picking the top-k, so an excluded document's closer chunks never crowd out an included document's more distant ones.
>
> **Verified in a clean environment:** `pytest` → **211/211 passing**, `streamlit run app.py --server.headless true` → HTTP 200, no runtime errors.
>
> Ongoing per-module testing (Phase 15) and bug fixing (Phase 16) are next.

## Overview

The finished app will let a student:

- Drag-and-drop one or more PDFs and ask questions about them
- Get answers with page-level source citations, not invented ones
- Switch between modes: normal chat, simple explanations, exam-style answers, step-by-step problem solving, summaries, and question generation (MCQ / short / long)
- Ask follow-up questions that resolve pronouns ("why is _it_ useful?") using chat memory
- Be told clearly when the answer isn't in the uploaded material, instead of getting a confident guess

## Architecture

```
User
  │
  ▼
Streamlit UI  ──upload──▶  PDF Processor (PyMuPDF)
                                 │  extract text per page, keep {source, page}
                                 ▼
                           Text Cleaning
                                 │
                                 ▼
                           Chunking (RecursiveCharacterTextSplitter)
                                 │  chunk_size=1000, overlap=150, keeps metadata
                                 ▼
                           Embeddings (Gemini embedding model)
                                 │
                                 ▼
                           FAISS vector index  ◀── saved/loaded from vectorstore/
                                 │
User question ──▶ Question embedding ──▶ similarity search ──▶ top-K chunks
                                 │
                                 ▼
                           Prompt builder (context + question + mode)
                                 │
                                 ▼
                           Gemini chat model
                                 │
                                 ▼
                     Answer + source citations ──▶ Streamlit chat UI
```

### Why a hand-rolled FAISS wrapper instead of `langchain_community.vectorstores.FAISS`?

The spec calls for LangChain as the RAG framework and FAISS as the vector store. Historically those were glued together via `langchain-community`. That package was **officially sunset and archived in 2026**, and — confirmed by its own maintainers in the migration discussion — **no standalone FAISS partner package has been published to replace it.** Depending on a frozen, deprecated package for the core vector-store layer would violate the project's own rule against outdated APIs.

Instead, `src/vector_store.py` (Phase 6) will call `faiss-cpu` directly and manage the chunk/metadata docstore ourselves. This is a small amount of extra code, gives full control over the save/load format, and — per the project's extensibility requirement — makes it _easier_, not harder, to swap in Chroma or Qdrant later, since our own interface won't be tied to LangChain's vectorstore base class at all.

Everywhere else, real LangChain packages are used as intended:

- `langchain_google_genai` — `ChatGoogleGenerativeAI` (LLM) and `GoogleGenerativeAIEmbeddings` (embeddings)
- `langchain_text_splitters` — `RecursiveCharacterTextSplitter` (chunking)
- `langchain_core.documents.Document` — the chunk/metadata container passed between stages

## Tech stack

| Layer        | Choice                                                      | Verified version         |
| ------------ | ----------------------------------------------------------- | ------------------------ |
| UI           | Streamlit                                                   | 1.64.0                   |
| Language     | Python                                                      | 3.11+ (tested on 3.12.3) |
| PDF parsing  | PyMuPDF (`import pymupdf`, not the deprecated `fitz` alias) | 1.28.2                   |
| Chunking     | `langchain-text-splitters`                                  | 1.1.2                    |
| Embeddings   | Gemini, via `langchain-google-genai`                        | 4.4.0                    |
| LLM          | Gemini, via `langchain-google-genai`                        | 4.4.0                    |
| Vector store | `faiss-cpu` (direct, own wrapper)                           | 1.15.1                   |
| Config       | `python-dotenv`                                             | 1.2.3                    |
| Testing      | `pytest` + `pytest-mock`                                    | 9.1.1 / 3.15.1           |

**Model choice, and why:** Google's own model docs (checked live, last updated 2026‑09‑24) recommend new projects default to `gemini-3.5-flash-lite` (fast, cheapest, current-generation stable) or `gemini-3.8-flash` for higher quality; the older `gemini-2.5-*` family is being deprecated for new projects. This app defaults to **`gemini-3.5-flash-lite`** for chat (a student tool should be cheap to run) and **`gemini-embedding-001`** for embeddings (the current stable embedding model). Both are configurable via `.env` — see [Environment variables](#environment-variables).

## Project structure

```
student-rag-chat/
│
├── app.py                  # Streamlit entry point
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── pytest.ini
│
├── data/uploads/            # staged PDF uploads (gitignored, folder kept)
├── vectorstore/              # saved FAISS index + metadata (gitignored, folder kept)
│
├── src/
│   ├── __init__.py
│   ├── config.py             # ✅ implemented — env loading, defaults, validation
│   ├── pdf_processor.py      # ⏳ Phase 3
│   ├── embeddings.py         # ⏳ Phase 5
│   ├── vector_store.py       # ⏳ Phase 6
│   ├── retriever.py          # ⏳ Phase 7
│   ├── llm.py                # ⏳ Phase 8
│   ├── prompts.py            # ⏳ Phase 8
│   ├── rag_pipeline.py       # ⏳ Phase 9
│   └── chat_memory.py        # ⏳ Phase 12
│
└── tests/
    ├── __init__.py
    └── test_config.py        # ✅ implemented — 9 tests, all passing
```

## Installation (Windows)

```bat
:: 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

:: 2. Install dependencies
pip install -r requirements.txt

:: 3. Set up your environment file
copy .env.example .env
:: then open .env and paste your key after GOOGLE_API_KEY=
:: get a free key at https://aistudio.google.com/apikey
```

## Environment variables

All read from `.env` (see `.env.example`). Every one is optional except the API key — safe defaults are defined in `src/config.py`.

| Variable          | Default                          | Purpose                       |
| ----------------- | -------------------------------- | ----------------------------- |
| `GOOGLE_API_KEY`  | _(required for any Gemini call)_ | Your Gemini API key           |
| `GEMINI_MODEL`    | `gemini-3.5-flash-lite`          | Chat model                    |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001`    | Embedding model               |
| `TEMPERATURE`     | `0.2`                            | LLM sampling temperature      |
| `CHUNK_SIZE`      | `1000`                           | Characters per chunk          |
| `CHUNK_OVERLAP`   | `150`                            | Overlap between chunks        |
| `TOP_K`           | `4`                              | Chunks retrieved per question |

The app **never crashes on a missing API key** — it starts normally and shows a warning banner instead, so you can still see the UI before setting one up. Invalid _numeric_ settings (e.g. `CHUNK_OVERLAP` ≥ `CHUNK_SIZE`) do raise a clear startup error, since that would silently break chunking rather than fail loudly.

## Running locally

```bat
streamlit run app.py
```

At Phase 1, this opens an app shell: a working PDF drag-and-drop uploader, a config/API-key status banner, and a disabled chat input / settings panel (each control shows which phase enables it). No PDF is parsed and no Gemini call is made yet.

## RAG pipeline

Documented in full once `rag_pipeline.py` exists (Phase 9). The conceptual flow is in [Architecture](#architecture) above.

## Testing

```bat
pytest
```

Currently: 9 tests in `tests/test_config.py`, covering default values, missing/invalid env vars, the friendly missing-API-key error, and that the API key is never leaked via `repr(config)`. All pass with no network access and no real API key required. More test files land alongside the modules they test (see [Verification report](#verification-report--phase-1)).

## Troubleshooting

| Symptom                                      | Cause                                         | Fix                                                                         |
| -------------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------- |
| Yellow "No Gemini API key configured" banner | `.env` missing or `GOOGLE_API_KEY` blank      | `copy .env.example .env`, fill in the key, restart                          |
| Startup error mentioning `CHUNK_OVERLAP`     | Overlap ≥ chunk size in `.env`                | Lower `CHUNK_OVERLAP` below `CHUNK_SIZE`                                    |
| `ModuleNotFoundError`                        | Dependencies not installed in the active venv | Confirm `venv\Scripts\activate` ran, then `pip install -r requirements.txt` |

## Implementation phases

| Phase | Scope                | Status                                                                                                                                                                                                                                                                                                                                                                                  |
| ----- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | Project setup        | ✅ Done                                                                                                                                                                                                                                                                                                                                                                                 |
| 2     | Configuration        | ✅ Done                                                                                                                                                                                                                                                                                                                                                                                 |
| 3     | PDF processing       | ✅ Done                                                                                                                                                                                                                                                                                                                                                                                 |
| 4     | Chunking             | ✅ Done                                                                                                                                                                                                                                                                                                                                                                                 |
| 5     | Embeddings           | ✅ Done                                                                                                                                                                                                                                                                                                                                                                                 |
| 6     | FAISS vector store   | ✅ Done                                                                                                                                                                                                                                                                                                                                                                                 |
| 7     | Retriever            | ✅ Done                                                                                                                                                                                                                                                                                                                                                                                 |
| 8     | Gemini LLM + prompts | ✅ Done                                                                                                                                                                                                                                                                                                                                                                                 |
| 9     | RAG pipeline         | ✅ Done — `src/rag_pipeline.py` chains retrieve → prompt → generate behind one error type (`RAGPipelineError`) and one return value (`RAGResult`)                                                                                                                                                                                                                                       |
| 10    | Streamlit chat       | ✅ Done — `app.py`'s "Process Documents" builds a real embedded FAISS index; `st.chat_input` enabled once that index + an API key both exist                                                                                                                                                                                                                                            |
| 11    | Source citations     | ✅ Done — each assistant message gets an expandable "📚 Sources (N)" panel: `SearchResult`s deduped by `(source, page)`, most-relevant-first, each showing the actual retrieved chunk text so a citation can be checked, not just trusted. See `_render_sources()` in `app.py`.                                                                                                         |
| 12    | Chat memory          | ✅ Done — `src/chat_memory.py`'s `resolve_question()` rewrites follow-ups into standalone questions via one extra Gemini call, using the same cached `llm_client`; falls back to the raw question on any failure (`ChatMemoryError`/`ValueError`), so a broken rewrite never blocks an answer. 17 new tests.                                                                            |
| 13    | Student modes        | ✅ Done — mode `st.selectbox` enabled, sourced from `src.prompts.VALID_MODES` (single source of truth, no hardcoded duplicate list); "Question Generator" reveals a question-type selector (mcq/short/long/mixed) that maps to `answer_question(question_type=...)`. No new modules — pure `app.py` UI change onto pipeline plumbing that already existed.                              |
| 14    | Multiple PDFs        | ✅ Done — incremental add (`VectorStore.add` onto an existing index instead of a rebuild), per-document list with remove (`VectorStore.list_sources()` / `remove_source()`), and a "Search in" chat filter (`source_filter` → `allowed_sources` in `retrieve()` / `VectorStore.search()`). See `_process_documents()`, `_remove_document()` and `_sync_selected_sources()` in `app.py`. |
| 15    | Testing              | ⏳ Next (ongoing per-module)                                                                                                                                                                                                                                                                                                                                                            |
| 16    | Bug fixing           | ⏳                                                                                                                                                                                                                                                                                                                                                                                      |
| 17    | README completion    | ⏳                                                                                                                                                                                                                                                                                                                                                                                      |
| 18    | Final verification   | ⏳                                                                                                                                                                                                                                                                                                                                                                                      |

## Future improvements

- Swap FAISS for Chroma/Qdrant (interface is already decoupled — see architecture note above)
- Optional local LLM provider (Ollama) alongside Gemini, behind a shared `LLMProvider` interface
- React + FastAPI frontend as an alternative to Streamlit

## License

Unlicensed / personal project template — add a license before distributing.

.venv\Scripts\Activate.ps1
uv run streamlit run app.py
