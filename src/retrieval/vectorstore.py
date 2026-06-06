"""ChromaDB storage and similarity search."""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langfuse import observe
from src.retrieval.embeddings import EmbeddingService

class VectorStore:
    """ChromaDB wrapper — handles storage and similarity search."""

    def __init__(self, persist_dir: str = "./chroma_db", embedding_service: EmbeddingService = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_service.model,
        )

    def add_documents(self, chunks: list[Document]) -> None:
        """Add chunks to the vector store"""
        self.store.add_documents(chunks)

    def delete_documents(self, filters: dict = None) -> None:
        """Delete documents matching the metadata filter."""
        if not filters:
            raise ValueError("Filters required — cannot delete without specifying criteria.")
        self.store._collection.delete(where=filters)

    @observe()
    def search(self, query: str, k: int = 3, filters: dict = None) -> list[Document]:
        """Search for similar chunks and optional metadata filtering"""
        search_kwargs = {"k": k}
        if filters:
            search_kwargs["filter"] = filters
        return self.store.similarity_search(query, **search_kwargs)