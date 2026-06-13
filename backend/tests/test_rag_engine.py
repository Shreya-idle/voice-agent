import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_engine import RAGEngine


@pytest.fixture
def mock_embeddings():
    with patch('rag_engine.HuggingFaceInferenceAPIEmbeddings') as mock:
        yield mock


@pytest.fixture
def mock_chroma():
    with patch('rag_engine.Chroma') as mock:
        yield mock


@pytest.fixture
def mock_settings():
    with patch('rag_engine.settings') as mock:
        mock.huggingface_api_key = "hf_test_token_123"
        yield mock


class TestRAGEngine:
    def test_init_existing_store(self, mock_embeddings, mock_chroma, mock_settings):
        with patch('os.path.exists', return_value=True), \
             patch('os.scandir', return_value=[Mock()]):
            engine = RAGEngine(persist_directory="dummy_dir")
            assert engine.vector_store is not None
            mock_chroma.assert_called_once()

    def test_init_new_store(self, mock_embeddings, mock_chroma, mock_settings):
        with patch('os.path.exists', return_value=False), \
             patch('rag_engine.RAGEngine._build_vector_store') as mock_build:
            engine = RAGEngine(persist_directory="new_dir")
            mock_build.assert_called_once()

    def test_query_with_rerank_disabled(self, mock_embeddings, mock_chroma, mock_settings):
        """When rerank=False, query() should call similarity_search directly."""
        engine = RAGEngine()
        engine.vector_store = Mock()
        engine.vector_store.similarity_search.return_value = [Mock(page_content="test result")]

        results = engine.query("test query", rerank=False)
        assert len(results) == 1
        assert results[0].page_content == "test result"
        engine.vector_store.similarity_search.assert_called_with("test query", k=3)

    def test_query_with_rerank_enabled(self, mock_embeddings, mock_chroma, mock_settings):
        """When rerank=True (default), query() should fetch top_n candidates then rerank."""
        engine = RAGEngine()
        engine.vector_store = Mock()
        mock_docs = [Mock(page_content=f"doc{i}") for i in range(5)]
        engine.vector_store.similarity_search.return_value = mock_docs

        with patch.object(engine, 'rerank_documents', return_value=mock_docs[:3]) as mock_rerank:
            results = engine.query("test query", k=3, rerank=True, top_n=5)
            engine.vector_store.similarity_search.assert_called_with("test query", k=5)
            mock_rerank.assert_called_once_with("test query", mock_docs, k=3)
            assert len(results) == 3

    def test_ensure_data_file_exists(self, mock_embeddings, mock_chroma, mock_settings):
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs') as mock_mkdir, \
             patch('rag_engine.RAGEngine._build_vector_store'), \
             patch('builtins.open', MagicMock()) as mock_open:
            engine = RAGEngine(data_path="dummy/path.txt")
            engine._ensure_data_file_exists()
            mock_mkdir.assert_called()
            mock_open.assert_called()

    def test_load_documents_txt(self, mock_embeddings, mock_chroma, mock_settings):
        with patch('rag_engine.TextLoader') as mock_loader:
            engine = RAGEngine(data_path="test.txt")
            engine._load_documents()
            mock_loader.assert_called_with("test.txt")

    def test_load_documents_pdf(self, mock_embeddings, mock_chroma, mock_settings):
        with patch('rag_engine.PyPDFLoader') as mock_loader, \
             patch('os.path.exists', return_value=True):
            engine = RAGEngine(data_path="test.pdf")
            engine._load_documents()
            mock_loader.assert_called_with("test.pdf")

    def test_chunk_documents(self, mock_embeddings, mock_chroma, mock_settings):
        with patch('rag_engine.RecursiveCharacterTextSplitter') as mock_splitter, \
             patch('rag_engine.RAGEngine._build_vector_store'):
            engine = RAGEngine()
            mock_docs = [Mock()]
            engine._chunk_documents(mock_docs)
            mock_splitter.assert_called_once()

    def test_build_vector_store(self, mock_embeddings, mock_chroma, mock_settings):
        with patch('os.path.exists', return_value=False), \
             patch('rag_engine.RAGEngine._ensure_data_file_exists'), \
             patch('rag_engine.RAGEngine._load_documents', return_value=[Mock()]), \
             patch('rag_engine.RAGEngine._chunk_documents', return_value=[Mock()]):
            engine = RAGEngine()
            mock_chroma.from_documents.assert_called()

    def test_query_no_vector_store(self, mock_embeddings, mock_chroma, mock_settings):
        engine = RAGEngine()
        engine.vector_store = None
        assert engine.query("test") == []

    def test_ensure_data_file_exists_pdf_not_found(self, mock_embeddings, mock_chroma, mock_settings):
        with patch('os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                RAGEngine(data_path="missing.pdf")

    def test_init_offline_fallback(self, mock_embeddings, mock_chroma, mock_settings):
        """Verify that RAGEngine initializes successfully and builds vector store even if embeddings API fails with network error."""
        import requests
        mock_instance = mock_embeddings.return_value
        mock_instance.embed_documents.side_effect = requests.exceptions.ConnectionError("DNS resolution failed")
        
        # Simulate Chroma calling the embedding function
        def fake_from_documents(documents, embedding, **kwargs):
            embedding.embed_documents([d.page_content for d in documents])
            return mock_chroma
            
        mock_chroma.from_documents.side_effect = fake_from_documents
        
        with patch('os.path.exists', return_value=False), \
             patch('rag_engine.RAGEngine._ensure_data_file_exists'), \
             patch('rag_engine.RAGEngine._load_documents', return_value=[Mock(page_content="hello")]), \
             patch('rag_engine.RAGEngine._chunk_documents', return_value=[Mock(page_content="hello")]):
            
            engine = RAGEngine()
            assert engine.vector_store is not None
            assert engine.embeddings._fallback is True
            mock_chroma.from_documents.assert_called_once()




class TestReranking:
    """Tests for the API-based cross-encoder reranking method."""

    def test_rerank_empty_documents(self, mock_embeddings, mock_chroma, mock_settings):
        engine = RAGEngine()
        result = engine.rerank_documents("query", [], k=3)
        assert result == []

    def test_rerank_no_api_key(self, mock_embeddings, mock_chroma, mock_settings):
        """Without an API key, reranking should fall back to returning docs in original order."""
        mock_settings.huggingface_api_key = None
        engine = RAGEngine()
        assert engine.api_key is None

        docs = [Mock(page_content="doc1"), Mock(page_content="doc2")]
        result = engine.rerank_documents("query", docs, k=1)
        assert len(result) == 1
        assert result[0].page_content == "doc1"

    def test_rerank_success_dict_scores(self, mock_embeddings, mock_chroma, mock_settings):
        """Successful reranking with dict-format scores from the API."""
        engine = RAGEngine()
        engine.api_key = "hf_test_token"

        docs = [Mock(page_content="low relevance"), Mock(page_content="high relevance")]

        mock_response = Mock()
        mock_response.status_code = 200
        # API returns list of dicts with label/score (text-classification format)
        mock_response.json.return_value = [
            {"label": "LABEL_0", "score": 0.2},
            {"label": "LABEL_0", "score": 0.9},
        ]

        with patch('rag_engine.requests.post', return_value=mock_response):
            result = engine.rerank_documents("query", docs, k=2)
            assert len(result) == 2
            # Higher score should come first
            assert result[0].page_content == "high relevance"
            assert result[1].page_content == "low relevance"

    def test_rerank_success_nested_list_scores(self, mock_embeddings, mock_chroma, mock_settings):
        """Successful reranking with nested list format from the API."""
        engine = RAGEngine()
        engine.api_key = "hf_test_token"

        docs = [Mock(page_content="doc_a"), Mock(page_content="doc_b")]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [{"label": "LABEL_0", "score": 0.3}],
            [{"label": "LABEL_0", "score": 0.8}],
        ]

        with patch('rag_engine.requests.post', return_value=mock_response):
            result = engine.rerank_documents("query", docs, k=1)
            assert len(result) == 1
            assert result[0].page_content == "doc_b"

    def test_rerank_success_float_scores(self, mock_embeddings, mock_chroma, mock_settings):
        """Successful reranking with raw float scores from the API."""
        engine = RAGEngine()
        engine.api_key = "hf_test_token"

        docs = [Mock(page_content="first"), Mock(page_content="second")]

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [0.1, 0.95]

        with patch('rag_engine.requests.post', return_value=mock_response):
            result = engine.rerank_documents("query", docs, k=2)
            assert result[0].page_content == "second"

    def test_rerank_api_error_fallback(self, mock_embeddings, mock_chroma, mock_settings):
        """On API error, reranking should fall back gracefully."""
        engine = RAGEngine()
        engine.api_key = "hf_test_token"

        docs = [Mock(page_content="doc1"), Mock(page_content="doc2")]

        mock_response = Mock()
        mock_response.status_code = 503
        mock_response.text = "Model is loading"

        with patch('rag_engine.requests.post', return_value=mock_response):
            result = engine.rerank_documents("query", docs, k=2)
            assert len(result) == 2
            assert result[0].page_content == "doc1"  # Original order preserved

    def test_rerank_network_exception_fallback(self, mock_embeddings, mock_chroma, mock_settings):
        """On network exception, reranking should fall back gracefully."""
        engine = RAGEngine()
        engine.api_key = "hf_test_token"

        docs = [Mock(page_content="doc1")]

        with patch('rag_engine.requests.post', side_effect=ConnectionError("No network")):
            result = engine.rerank_documents("query", docs, k=1)
            assert len(result) == 1
            assert result[0].page_content == "doc1"


class TestSummarization:
    """Tests for the API-based BART summarization method."""

    def test_summarize_empty_context(self, mock_embeddings, mock_chroma, mock_settings):
        engine = RAGEngine()
        assert engine.summarize_context("") == ""
        assert engine.summarize_context("   ") == ""

    def test_summarize_no_api_key(self, mock_embeddings, mock_chroma, mock_settings):
        """Without an API key, summarize should return raw context."""
        mock_settings.huggingface_api_key = None
        engine = RAGEngine()
        assert engine.api_key is None

        result = engine.summarize_context("This is a long context about voice agents.")
        assert result == "This is a long context about voice agents."

    def test_summarize_success(self, mock_embeddings, mock_chroma, mock_settings):
        """Successful summarization returns the summary_text from the API."""
        engine = RAGEngine()
        engine.api_key = "hf_test_token"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"summary_text": "A concise summary."}]

        with patch('rag_engine.requests.post', return_value=mock_response):
            result = engine.summarize_context("Very long text about many things...")
            assert result == "A concise summary."

    def test_summarize_unexpected_format_fallback(self, mock_embeddings, mock_chroma, mock_settings):
        """If API returns unexpected format, fall back to raw context."""
        engine = RAGEngine()
        engine.api_key = "hf_test_token"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = "unexpected string"

        with patch('rag_engine.requests.post', return_value=mock_response):
            result = engine.summarize_context("Original context.")
            assert result == "Original context."

    def test_summarize_api_error_fallback(self, mock_embeddings, mock_chroma, mock_settings):
        """On API error, summarize should return raw context."""
        engine = RAGEngine()
        engine.api_key = "hf_test_token"

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch('rag_engine.requests.post', return_value=mock_response):
            result = engine.summarize_context("Original context.")
            assert result == "Original context."

    def test_summarize_network_exception_fallback(self, mock_embeddings, mock_chroma, mock_settings):
        """On network exception, summarize should return raw context."""
        engine = RAGEngine()
        engine.api_key = "hf_test_token"

        with patch('rag_engine.requests.post', side_effect=ConnectionError("No network")):
            result = engine.summarize_context("Original context.")
            assert result == "Original context."


class TestFallbackEmbeddings:
    def test_fallback_embeddings_success(self):
        from rag_engine import FallbackEmbeddings
        mock_primary = Mock()
        mock_primary.embed_documents.return_value = [[0.1, 0.2]]
        mock_primary.embed_query.return_value = [0.1, 0.2]
        
        fallback = FallbackEmbeddings(mock_primary, dimension=2)
        
        assert fallback.embed_documents(["test"]) == [[0.1, 0.2]]
        assert fallback.embed_query("test") == [0.1, 0.2]
        assert fallback._fallback is False

    def test_fallback_embeddings_failure(self):
        from rag_engine import FallbackEmbeddings
        mock_primary = Mock()
        mock_primary.embed_documents.side_effect = Exception("Connection refused")
        mock_primary.embed_query.side_effect = Exception("Connection refused")
        
        fallback = FallbackEmbeddings(mock_primary, dimension=2)
        
        docs_res = fallback.embed_documents(["test"])
        assert docs_res == [[0.0, 0.0]]
        assert fallback._fallback is True
        
        mock_primary.reset_mock()
        query_res = fallback.embed_query("test")
        assert query_res == [0.0, 0.0]
        mock_primary.embed_query.assert_not_called()
