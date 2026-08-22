import os
import requests
from typing import List, Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from config import settings
from dotenv import load_dotenv

load_dotenv()

class FallbackEmbeddings(Embeddings):
    """
    Wraps a primary Embeddings instance and fails clearly if it is unavailable.

    Returning zero vectors makes every document equally similar and silently
    turns RAG into arbitrary retrieval. A visible failure is safer than an
    answer that appears grounded but is not.
    """
    def __init__(self, primary: Embeddings, dimension: int = 384):
        self.primary = primary
        self.dimension = dimension
        self._fallback = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return self.primary.embed_documents(texts)
        except Exception as e:
            self._fallback = True
            raise RuntimeError("Embedding provider is unavailable; RAG cannot safely retrieve context.") from e

    def embed_query(self, text: str) -> List[float]:
        try:
            return self.primary.embed_query(text)
        except Exception as e:
            self._fallback = True
            raise RuntimeError("Embedding provider is unavailable; RAG cannot safely retrieve context.") from e

class RAGEngine:
    def __init__(
        self, 
        data_path: str = "./articulo.pdf", 
        persist_directory: Optional[str] = None
    ):
        self.data_path = data_path
        self.persist_directory = persist_directory
        
        # Load API key defensively
        self.api_key = settings.huggingface_api_key
        if self.api_key == "hf_YOUR_TOKEN_HERE" or not self.api_key:
            self.api_key = None
            
        primary_embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=self.api_key or "",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        self.embeddings = FallbackEmbeddings(primary_embeddings)
        self.vector_store: Optional[Chroma] = None
        self._setup_vector_store()

    def _setup_vector_store(self) -> None:
        if self.persist_directory and self._has_existing_store():
            print("Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            if self.persist_directory:
                print("Creating new vector store and persisting to disk...")
            else:
                print("Creating new in-memory vector store from source document...")
            self._build_vector_store()

    def _has_existing_store(self) -> bool:
        if not self.persist_directory:
            return False
        return os.path.exists(self.persist_directory) and any(os.scandir(self.persist_directory))

    def _build_vector_store(self) -> None:
        self._ensure_data_file_exists()
        
        documents = self._load_documents()
        chunks = self._chunk_documents(documents)
        
        chroma_kwargs = {}
        if self.persist_directory:
            chroma_kwargs["persist_directory"] = self.persist_directory

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            **chroma_kwargs
        )

    def _ensure_data_file_exists(self) -> None:
        if not os.path.exists(self.data_path):
            if self.data_path.lower().endswith(".pdf"):
                raise FileNotFoundError(f"Required PDF file not found: {self.data_path}")
            dirname = os.path.dirname(self.data_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(self.data_path, "w") as f:
                f.write("Welcome to the Voice Agent Knowledge Base.")

    def _load_documents(self) -> List[Document]:
        if self.data_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(self.data_path)
        else:
            loader = TextLoader(self.data_path)
        return loader.load()

    def _chunk_documents(self, documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        return text_splitter.split_documents(documents)

    def rerank_documents(self, query: str, documents: List[Document], k: int = 3) -> List[Document]:
        """
        Rerank documents using Hugging Face Inference API cross-encoder model.
        Falls back to vector similarity ranking if the API fails or is not configured.
        """
        if not documents:
            return []
        if not self.api_key:
            print("Warning: Hugging Face API Key is missing. Skipping reranking, returning vector search order.")
            return documents[:k]

        url = "https://api-inference.huggingface.co/models/cross-encoder/ms-marco-MiniLM-L-6-v2"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": [
                {"text": query, "text_pair": doc.page_content}
                for doc in documents
            ]
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code == 200:
                scores_data = response.json()
                scored_docs = []
                for i, item in enumerate(scores_data):
                    if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
                        score = item[0].get("score", 0.0)
                    elif isinstance(item, dict):
                        score = item.get("score", 0.0)
                    elif isinstance(item, (int, float)):
                        score = float(item)
                    else:
                        score = 0.0
                    scored_docs.append((score, documents[i]))

                # Sort documents by reranker score descending
                scored_docs.sort(key=lambda x: x[0], reverse=True)
                return [doc for _, doc in scored_docs[:k]]
            else:
                print(f"Warning: Hugging Face Inference API error ({response.status_code}): {response.text}. Skipping reranking.")
                return documents[:k]
        except Exception as e:
            print(f"Warning: Failed to rerank documents via Hugging Face API: {e}. Skipping reranking.")
            return documents[:k]

    def summarize_context(self, context: str, max_length: int = 150) -> str:
        """
        Summarize raw retrieved context using Hugging Face Inference API summarizer.
        Falls back to raw context if the API fails or is not configured.
        """
        if not context.strip():
            return ""
        if not self.api_key:
            print("Warning: Hugging Face API Key is missing. Skipping summarization, returning raw context.")
            return context

        url = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Calculate min_length safely
        min_len = min(30, len(context) // 2)
        if min_len < 10:
            min_len = 10
            
        payload = {
            "inputs": context,
            "parameters": {
                "max_length": max_length,
                "min_length": min_len,
                "do_sample": False
            }
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                summary_data = response.json()
                if isinstance(summary_data, list) and len(summary_data) > 0 and isinstance(summary_data[0], dict):
                    return summary_data[0].get("summary_text", context)
                return context
            else:
                print(f"Warning: Hugging Face Summarizer API error ({response.status_code}): {response.text}. Skipping summarization.")
                return context
        except Exception as e:
            print(f"Warning: Failed to summarize context via Hugging Face API: {e}. Skipping summarization.")
            return context

    def query(self, text: str, k: int = 3, rerank: bool = True, top_n: int = 10) -> List[Document]:
        if not self.vector_store:
            return []
        
        if rerank:
            # Retrieve a larger pool of candidates for reranking
            candidates = self.vector_store.similarity_search(text, k=top_n)
            return self.rerank_documents(text, candidates, k=k)
        else:
            return self.vector_store.similarity_search(text, k=k)

if __name__ == "__main__":
    engine = RAGEngine()
    results = engine.query("What is the Voice Agent?")
    for res in results:
        print(f"Match: {res.page_content}")
