"""Agent tools — search_documents and get_metadata.

Each tool is an @observe()-traced callable. TOOL_SCHEMAS holds the
Anthropic-format schemas sent to the model; TOOL_REGISTRY maps tool names to
the callables the tools node dispatches to.
"""

import json
from pathlib import Path

from langfuse import observe

from src.context.prompt_builder import format_chunks_with_sources
from src.retrieval.vectorstore import VectorStore

METADATA_PATH = Path(__file__).resolve().parents[2] / "data" / "metadata.json"

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    """Lazily build a single VectorStore so import stays cheap and testable."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def _build_filters(company_name: str | None, report_year: int | None) -> dict | None:
    """Compose a ChromaDB metadata filter from the optional arguments."""
    clauses = []
    if company_name:
        clauses.append({"company_name": company_name})
    if report_year:
        clauses.append({"report_year": report_year})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


@observe()
def search_documents(
    query: str,
    company_name: str | None = None,
    report_year: int | None = None,
) -> str:
    """Retrieve relevant chunks from the vector store, optionally filtered."""
    filters = _build_filters(company_name, report_year)
    chunks = _get_store().search(query, filters=filters)
    if not chunks:
        return "No matching documents found."
    return format_chunks_with_sources(chunks)


@observe()
def get_metadata() -> str:
    """Return the available companies, years, and document types."""
    try:
        with open(METADATA_PATH) as f:
            metadata = json.load(f)
    except FileNotFoundError:
        return "Error: No metadata file found. No documents have been ingested."
    except json.JSONDecodeError:
        return "Error: The file exists, but is empty or contains invalid JSON."
    
    docs = list(metadata.values())
    companies = sorted({d["company_name"] for d in docs})
    years = sorted({d["report_year"] for d in docs})
    doc_types = sorted({d["document_type"] for d in docs})

    lines = ["Available documents:"]
    for d in docs:
        lines.append(
            f"- {d['company_name']} — {d['document_type']} — {d['report_year']}"
        )
    lines.append("")
    lines.append(f"Companies: {', '.join(companies)}")
    lines.append(f"Years: {', '.join(str(y) for y in years)}")
    lines.append(f"Document types: {', '.join(doc_types)}")
    return "\n".join(lines)


TOOL_SCHEMAS = [
    {
        "name": "search_documents",
        "description": (
            "Search the financial document collection for passages relevant to "
            "a query. Optionally filter by company name and/or report year to "
            "narrow the search to a specific document. Returns passages labelled "
            "with their source (company, document type, year, page) for citation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for, e.g. 'total net sales'.",
                },
                "company_name": {
                    "type": "string",
                    "description": (
                        "Optional. Restrict to one company, e.g. 'Apple'. Call "
                        "get_metadata first to see exact available names."
                    ),
                },
                "report_year": {
                    "type": "integer",
                    "description": "Optional. Restrict to one report year, e.g. 2025.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_metadata",
        "description": (
            "List the documents available in the collection: company names, "
            "report years, and document types. Use this to discover what can be "
            "searched and to get exact company names for filtering."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


TOOL_REGISTRY = {
    "search_documents": search_documents,
    "get_metadata": get_metadata,
}
