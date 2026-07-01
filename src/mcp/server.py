from mcp.server.fastmcp import FastMCP
from src.agent.tools import search_documents as _search, get_metadata as _get_metadata
from typing import Optional

mcp = FastMCP("Financial Document Intelligence")

@mcp.tool()
def search_documents(query: str, company_name: Optional[str] = None, report_year: Optional[int] = None) -> str:
    """Search the financial document collection for passages relevant to a query.
    Optionally filter by company name and/or report year."""
    return _search(query, company_name, report_year)

@mcp.tool()
def get_metadata() -> str:
    """List the documents available in the collection: company names, report years, and document types."""
    return _get_metadata()

if __name__ == "__main__":
    mcp.run(transport="stdio")