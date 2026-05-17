import os
from typing import List, Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    def __init__(
        self, 
        data_path: str = "../data/knowledge_base.txt", 
        persist_directory: str = "./chroma_db"
    ):
        self.data_path = data_path
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store: Optional[Chroma] = None
        self._setup_vector_store()

    def _setup_vector_store(self) -> None:
        if self._has_existing_store():
            print("Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector store...")
            self._build_vector_store()

    def _has_existing_store(self) -> bool:
        return os.path.exists(self.persist_directory) and any(os.scandir(self.persist_directory))

    def _build_vector_store(self) -> None:
        self._ensure_data_file_exists()
        
        documents = self._load_documents()
        chunks = self._chunk_semantically(documents)
        
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

    def _ensure_data_file_exists(self) -> None:
        if not os.path.exists(self.data_path):
            if self.data_path.lower().endswith(".pdf"):
                raise FileNotFoundError(f"Required PDF file not found: {self.data_path}")
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            with open(self.data_path, "w") as f:
                f.write("Welcome to the Voice Agent Knowledge Base.")

    def _load_documents(self) -> List[Document]:
        if self.data_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(self.data_path)
        else:
            loader = TextLoader(self.data_path)
        return loader.load()

    def _chunk_semantically(self, documents: List[Document]) -> List[Document]:
        semantic_splitter = SemanticChunker(
            self.embeddings, 
            breakpoint_threshold_type="percentile"
        )
        return semantic_splitter.split_documents(documents)

    def query(self, text: str, k: int = 3) -> List[Document]:
        if not self.vector_store:
            return []
        return self.vector_store.similarity_search(text, k=k)

if __name__ == "__main__":
    engine = RAGEngine()
    results = engine.query("What is the Voice Agent?")
    for res in results:
        print(f"Match: {res.page_content}")
