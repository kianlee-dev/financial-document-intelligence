"""Prompt assembly — system prompt, chunk formatting, query construction."""

from langfuse import observe
from langchain_core.documents import Document

def get_system_prompt() -> str:
    """Return the system prompt for financial document Q&A."""
    return (
        "You are a financial document analyst. "
        "Answer questions based only on the provided context. "
        "Cite your sources with company name, document type, year, "
        "and page number. If the context doesn't contain enough "
        "information to answer, say so — do not make up data."
    )

def format_chunks_with_sources(chunks: list[Document]) -> str:
    """Format chunks with [Source: company, type, year, Page N] labels."""
    context_parts = []
    for chunk in chunks:
        header = (
            f"[Source: {chunk.metadata['company_name']}, "
            f"{chunk.metadata['document_type']}, "
            f"{chunk.metadata['report_year']}, "
            f"Page {chunk.metadata['page']}]"
        )
        context_parts.append(f"{header}\n{chunk.page_content}")

    return "\n\n".join(context_parts)


@observe()
def get_prompt(query: str, chunks: list[Document]) -> str:
    """Assemble context chunks and query into a formatted prompt."""
    context = format_chunks_with_sources(chunks)
    return f"Context:\n{context}\n\nQuestion: {query}"
