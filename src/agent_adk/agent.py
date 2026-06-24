"""Google ADK agent — same tools and prompt, different framework."""

import asyncio

from google.adk import Agent

from src.agent.tools import search_documents as _search, get_metadata as _get_metadata
from src.agent.tools import build_filters, get_store
from src.context.prompt_builder import format_chunks_with_sources

GEMINI_INSTRUCTION = (
    "You are a financial analyst. Answer questions using the provided tools. "
    "Cite sources with company name, document type, year, and page number. "
    "If the context is insufficient, say so. Do not fabricate data. "
    "For comparisons, use compare_companies. Do not call search_documents for comparisons. "
    "Search once per company. Do not repeat searches."
)


async def search_documents(query: str, company_name: str = None, report_year: int = None) -> str:
    """Search financial documents for a single company.

    Args:
        query: The search query describing what financial information to find.
        company_name: Optional company name to filter results.
        report_year: Optional report year to filter results.

    Returns:
        Formatted text chunks with source citations.
    """
    result = await asyncio.to_thread(_search, query, company_name, report_year)
    return result[:3000]


async def compare_companies(query: str, company_a: str, company_b: str) -> str:
    """Compare financial data between two companies in a single search.

    Use this instead of calling search_documents twice.

    Args:
        query: What to compare (e.g. "net income", "total revenue").
        company_a: First company name.
        company_b: Second company name.

    Returns:
        Combined search results from both companies with source citations.
    """
    def _do_compare():
        results = []
        for company in [company_a, company_b]:
            filters = build_filters(company, None)
            chunks = get_store().search(query, filters=filters)
            if chunks:
                results.append(f"--- {company} ---\n{format_chunks_with_sources(chunks)[:1500]}")
            else:
                results.append(f"--- {company} ---\nNo matching documents found.")
        return "\n\n".join(results)

    return await asyncio.to_thread(_do_compare)


async def get_metadata() -> str:
    """List all available companies, years, and document types in the collection.

    Call this when the user asks what companies or data are available.

    Returns:
        A formatted list of available documents.
    """
    return await asyncio.to_thread(_get_metadata)


root_agent = Agent(
    name="financial_analyst",
    model="gemini-3.5-flash",
    instruction=GEMINI_INSTRUCTION,
    tools=[search_documents, compare_companies, get_metadata],
)