import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_engine import RAGEngine

@pytest.fixture
def mock_embeddings():
    with patch('rag_engine.HuggingFaceEmbeddings') as mock:
        yield mock

@pytest.fixture
def mock_chroma():
    with patch('rag_engine.Chroma') as mock:
        yield mock

class TestRAGEngine:
    def test_init_existing_store(self, mock_embeddings, mock_chroma):
        with patch('os.path.exists', return_value=True), \
             patch('os.scandir', return_value=[Mock()]):
            engine = RAGEngine(persist_directory="dummy_dir")
            assert engine.vector_store is not None
            mock_chroma.assert_called_once()

    def test_init_new_store(self, mock_embeddings, mock_chroma):
        with patch('os.path.exists', return_value=False), \
             patch('rag_engine.RAGEngine._build_vector_store') as mock_build:
            engine = RAGEngine(persist_directory="new_dir")
            mock_build.assert_called_once()

    def test_query(self, mock_embeddings, mock_chroma):
        engine = RAGEngine()
        engine.vector_store = Mock()
        engine.vector_store.similarity_search.return_value = [Mock(page_content="test result")]
        
        results = engine.query("test query")
        assert len(results) == 1
        assert results[0].page_content == "test result"
        engine.vector_store.similarity_search.assert_called_with("test query", k=3)

    def test_ensure_data_file_exists(self, mock_embeddings, mock_chroma):
        with patch('os.path.exists', return_value=False), \
             patch('os.makedirs') as mock_mkdir, \
             patch('rag_engine.RAGEngine._build_vector_store'), \
             patch('builtins.open', MagicMock()) as mock_open:
            engine = RAGEngine(data_path="dummy/path.txt")
            engine._ensure_data_file_exists()
            mock_mkdir.assert_called()
            mock_open.assert_called()

    def test_load_documents_txt(self, mock_embeddings, mock_chroma):
        with patch('rag_engine.TextLoader') as mock_loader:
            engine = RAGEngine(data_path="test.txt")
            engine._load_documents()
            mock_loader.assert_called_with("test.txt")

    def test_load_documents_pdf(self, mock_embeddings, mock_chroma):
        with patch('rag_engine.PyPDFLoader') as mock_loader:
            engine = RAGEngine(data_path="test.pdf")
            engine._load_documents()
            mock_loader.assert_called_with("test.pdf")

    def test_chunk_semantically(self, mock_embeddings, mock_chroma):
        with patch('rag_engine.SemanticChunker') as mock_splitter:
            engine = RAGEngine()
            mock_docs = [Mock()]
            engine._chunk_semantically(mock_docs)
            mock_splitter.assert_called_once()

    def test_build_vector_store(self, mock_embeddings, mock_chroma):
        with patch('os.path.exists', return_value=False), \
             patch('rag_engine.RAGEngine._load_documents', return_value=[Mock()]), \
             patch('rag_engine.RAGEngine._chunk_semantically', return_value=[Mock()]):
            engine = RAGEngine()
            mock_chroma.from_documents.assert_called()
