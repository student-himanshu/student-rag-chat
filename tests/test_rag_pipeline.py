"""
Tests for src.rag_pipeline (Phase 9).

Follows the project's established split: FAISS (via a real VectorStore)
is exercised for real, same as test_retriever.py and test_vector_store.py
do -- only the two Gemini-touching calls (embed_query, chat client.invoke)
are faked, since those are the only things that truly can't run in this
sandbox. No mocking of retrieve()/build_system_prompt()/generate_answer()
themselves: this suite is meant to catch a real wiring bug between those
three, which patching them out would hide.
"""
from __future__ import annotations

from dataclasses import replace

import pytest
from langchain_core.documents import Document

from src.config import Config, ConfigError
from src.embeddings import EmbeddedChunk
from src.llm import LLMError
from src.prompts import PromptError
from src.rag_pipeline import RAGPipelineError, RAGResult, answer_question
from src.retriever import RetrieverError
from src.vector_store import VectorStore, build_vector_store


# --- fixtures & fakes --------------------------------------------------


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


def make_chunk(text: str, source: str, page: int, chunk_index: int) -> Document:
    chunk_id = f"{source}::p{page}::c{chunk_index}"
    return Document(
        page_content=text,
        metadata={"source": source, "page": page, "chunk_index": chunk_index, "chunk_id": chunk_id},
    )


@pytest.fixture
def store() -> VectorStore:
    """A real, populated VectorStore -- two chunks, 2-dimensional vectors
    chosen so a query of [1.0, 0.0] finds "close" before "far"."""
    chunks = [
        EmbeddedChunk(
            chunk=make_chunk("Mitochondria are the powerhouse of the cell.", "bio.pdf", 12, 0),
            vector=[1.0, 0.0],
        ),
        EmbeddedChunk(
            chunk=make_chunk("Unrelated content about the French Revolution.", "history.pdf", 3, 0),
            vector=[0.0, 5.0],
        ),
    ]
    return build_vector_store(chunks)


class FakeEmbeddingsClient:
    """Stands in for GoogleGenerativeAIEmbeddings. Only embed_query is
    ever called by this pipeline (never embed_documents), so that's all
    this fake implements."""

    def __init__(self, vector=None, raise_exc=None):
        self._vector = vector if vector is not None else [1.0, 0.0]
        self._raise_exc = raise_exc
        self.queries: list[str] = []

    def embed_query(self, text):
        self.queries.append(text)
        if self._raise_exc:
            raise self._raise_exc
        return self._vector


class FakeChatClient:
    """Stands in for ChatGoogleGenerativeAI -- mirrors test_llm.py's fake."""

    def __init__(self, content="A fake grounded answer.", model="gemini-3.5-flash-lite", raise_exc=None):
        self._content = content
        self.model = model
        self._raise_exc = raise_exc
        self.last_messages = None
        self.call_count = 0

    def invoke(self, messages):
        self.call_count += 1
        self.last_messages = messages
        if self._raise_exc:
            raise self._raise_exc

        class _Response:
            def __init__(self, content):
                self.content = content

        return _Response(self._content)


# --- happy path ----------------------------------------------------------


class TestAnswerQuestionHappyPath:
    def test_returns_answer_and_sources(self, cfg, store):
        embeddings = FakeEmbeddingsClient(vector=[1.0, 0.0])
        chat = FakeChatClient(content="Mitochondria produce ATP through respiration.")

        result = answer_question(
            "What do mitochondria do?",
            store,
            cfg,
            embeddings_client=embeddings,
            llm_client=chat,
        )

        assert isinstance(result, RAGResult)
        assert result.answer == "Mitochondria produce ATP through respiration."
        assert result.mode == "Normal Chat"
        assert len(result.sources) == 2
        assert result.sources[0].document.page_content.startswith("Mitochondria")

    def test_defaults_k_to_config_top_k(self, cfg, store):
        embeddings = FakeEmbeddingsClient()
        chat = FakeChatClient()

        result = answer_question("Any question", store, cfg, embeddings_client=embeddings, llm_client=chat)

        assert len(result.sources) == cfg.top_k

    def test_explicit_k_overrides_config_top_k(self, cfg, store):
        embeddings = FakeEmbeddingsClient()
        chat = FakeChatClient()

        result = answer_question(
            "Any question", store, cfg, k=1, embeddings_client=embeddings, llm_client=chat
        )

        assert len(result.sources) == 1

    def test_sources_carry_full_citation_metadata(self, cfg, store):
        embeddings = FakeEmbeddingsClient(vector=[1.0, 0.0])
        chat = FakeChatClient()

        result = answer_question(
            "What do mitochondria do?", store, cfg, k=1, embeddings_client=embeddings, llm_client=chat
        )

        [top] = result.sources
        assert top.document.metadata["source"] == "bio.pdf"
        assert top.document.metadata["page"] == 12
        assert top.document.metadata["chunk_id"] == "bio.pdf::p12::c0"

    def test_retrieved_chunks_are_embedded_in_the_system_prompt(self, cfg, store):
        embeddings = FakeEmbeddingsClient(vector=[1.0, 0.0])
        chat = FakeChatClient()

        answer_question("What do mitochondria do?", store, cfg, embeddings_client=embeddings, llm_client=chat)

        system_message = chat.last_messages[0]
        assert "Mitochondria are the powerhouse" in system_message.content
        assert "bio.pdf" in system_message.content

    def test_query_is_sent_as_the_human_message_verbatim(self, cfg, store):
        embeddings = FakeEmbeddingsClient()
        chat = FakeChatClient()

        answer_question(
            "What do mitochondria do?", store, cfg, embeddings_client=embeddings, llm_client=chat
        )

        human_message = chat.last_messages[1]
        assert human_message.content == "What do mitochondria do?"

    def test_mode_and_question_type_are_forwarded_to_the_prompt(self, cfg, store):
        embeddings = FakeEmbeddingsClient()
        chat = FakeChatClient()

        result = answer_question(
            "Give me questions",
            store,
            cfg,
            mode="Question Generator",
            question_type="mcq",
            embeddings_client=embeddings,
            llm_client=chat,
        )

        assert result.mode == "Question Generator"
        system_message = chat.last_messages[0]
        assert "multiple-choice" in system_message.content.lower()

    def test_empty_store_still_produces_an_answer(self, cfg):
        empty_store = VectorStore(dimension=2)
        embeddings = FakeEmbeddingsClient()
        chat = FakeChatClient(content="That isn't covered in your uploaded documents.")

        result = answer_question(
            "What's in chapter 9?", empty_store, cfg, embeddings_client=embeddings, llm_client=chat
        )

        assert result.sources == []
        assert "No relevant material" in chat.last_messages[0].content
        assert result.answer == "That isn't covered in your uploaded documents."


# --- client reuse vs. on-demand construction ------------------------------


class TestClientConstruction:
    def test_provided_embeddings_client_is_used_instead_of_building_one(self, cfg, store, mocker):
        get_embeddings_client_mock = mocker.patch("src.retriever.get_embeddings_client")
        embeddings = FakeEmbeddingsClient(vector=[1.0, 0.0])
        chat = FakeChatClient()

        answer_question("A question", store, cfg, embeddings_client=embeddings, llm_client=chat)

        get_embeddings_client_mock.assert_not_called()
        assert embeddings.queries == ["A question"]

    def test_provided_llm_client_skips_get_llm_client(self, cfg, store, mocker):
        get_llm_client_mock = mocker.patch("src.rag_pipeline.get_llm_client")
        embeddings = FakeEmbeddingsClient()
        chat = FakeChatClient()

        answer_question("A question", store, cfg, embeddings_client=embeddings, llm_client=chat)

        get_llm_client_mock.assert_not_called()
        assert chat.call_count == 1

    def test_missing_llm_client_builds_one_via_get_llm_client(self, cfg, store, mocker):
        embeddings = FakeEmbeddingsClient()
        chat = FakeChatClient()
        get_llm_client_mock = mocker.patch("src.rag_pipeline.get_llm_client", return_value=chat)

        answer_question("A question", store, cfg, embeddings_client=embeddings)

        get_llm_client_mock.assert_called_once_with(cfg)
        assert chat.call_count == 1


# --- error wrapping --------------------------------------------------------


class TestErrorWrapping:
    def test_empty_query_is_wrapped_as_rag_pipeline_error(self, cfg, store):
        embeddings = FakeEmbeddingsClient()
        chat = FakeChatClient()

        with pytest.raises(RAGPipelineError) as exc_info:
            answer_question("   ", store, cfg, embeddings_client=embeddings, llm_client=chat)

        assert isinstance(exc_info.value.__cause__, RetrieverError)
        assert chat.call_count == 0

    def test_embedding_failure_is_wrapped_and_chained(self, cfg, store):
        embeddings = FakeEmbeddingsClient(raise_exc=RuntimeError("network down"))
        chat = FakeChatClient()

        with pytest.raises(RAGPipelineError) as exc_info:
            answer_question("A question", store, cfg, embeddings_client=embeddings, llm_client=chat)

        assert isinstance(exc_info.value.__cause__, RetrieverError)
        assert chat.call_count == 0

    def test_unknown_mode_is_wrapped_as_rag_pipeline_error(self, cfg, store):
        embeddings = FakeEmbeddingsClient()
        chat = FakeChatClient()

        with pytest.raises(RAGPipelineError) as exc_info:
            answer_question(
                "A question", store, cfg, mode="Sarcastic Mode", embeddings_client=embeddings, llm_client=chat
            )

        assert isinstance(exc_info.value.__cause__, PromptError)
        assert chat.call_count == 0

    def test_unknown_question_type_is_wrapped_as_rag_pipeline_error(self, cfg, store):
        embeddings = FakeEmbeddingsClient()
        chat = FakeChatClient()

        with pytest.raises(RAGPipelineError) as exc_info:
            answer_question(
                "A question",
                store,
                cfg,
                mode="Question Generator",
                question_type="essay",
                embeddings_client=embeddings,
                llm_client=chat,
            )

        assert isinstance(exc_info.value.__cause__, PromptError)
        assert chat.call_count == 0

    def test_missing_api_key_for_llm_client_is_wrapped(self, cfg, store):
        cfg_no_key = replace(cfg, google_api_key="")
        embeddings = FakeEmbeddingsClient()

        with pytest.raises(RAGPipelineError) as exc_info:
            answer_question("A question", store, cfg_no_key, embeddings_client=embeddings)

        assert isinstance(exc_info.value.__cause__, ConfigError)

    def test_llm_client_exception_is_wrapped_as_rag_pipeline_error(self, cfg, store):
        embeddings = FakeEmbeddingsClient()
        chat = FakeChatClient(raise_exc=RuntimeError("connection reset"))

        with pytest.raises(RAGPipelineError) as exc_info:
            answer_question("A question", store, cfg, embeddings_client=embeddings, llm_client=chat)

        assert isinstance(exc_info.value.__cause__, LLMError)

    def test_empty_llm_response_is_wrapped_as_rag_pipeline_error(self, cfg, store):
        embeddings = FakeEmbeddingsClient()
        chat = FakeChatClient(content="   ")

        with pytest.raises(RAGPipelineError) as exc_info:
            answer_question("A question", store, cfg, embeddings_client=embeddings, llm_client=chat)

        assert isinstance(exc_info.value.__cause__, LLMError)
