"""PDF loading with pypdf — extracts text with page metadata."""
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

def load_documents(data_dir: Path) -> list[Document]:
    """Load all PDFs from data_dir, attach metadata from config."""
    
    metadata_path = data_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata_config = json.load(f)

    all_documents = []

    for pdf_path in data_dir.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        file_metadata = metadata_config.get(pdf_path.name, {})

        for page in pages:
            page.metadata.update(file_metadata)

        all_documents.extend(pages)
    return all_documents