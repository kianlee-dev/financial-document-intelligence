"""CLI script to run the ingestion pipeline."""

from pathlib import Path
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.vectorstore import VectorStore

if __name__ == "__main__":
    doc_path = Path("data")
    documents = load_documents(doc_path)
    print(f"Documents loaded: {len(documents)}")
    chunks = chunk_documents(documents)
    print(f"Chunks created: {len(chunks)}")
    vector_store = VectorStore()
    vector_store.add_documents(chunks)
    query = "What is the revenue of Apple in 2025"
    result = vector_store.search(query)
    print(f"Search Result: {result}")