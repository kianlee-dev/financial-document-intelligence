"""Embedding generation using sentence-transformers."""

from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingService:
    """Wraps embedding model. Swap model by changing model_name."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = HuggingFaceEmbeddings(model_name=model_name)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts for ingestion."""
        return self.model.embed_documents(texts)
    
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query for retrieval."""
        return self.model.embed_query(text)
