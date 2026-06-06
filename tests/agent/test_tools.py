"""Unit tests for agent tools with a mocked VectorStore."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

import src.agent.tools as tools


def _doc(company="Apple", dtype="10-K", year=2025, page=1, text="content"):
    return Document(
        page_content=text,
        metadata={
            "company_name": company,
            "document_type": dtype,
            "report_year": year,
            "page": page,
        },
    )


def test_build_filters_none():
    assert tools._build_filters(None, None) is None


def test_build_filters_company_only():
    assert tools._build_filters("Apple", None) == {"company_name": "Apple"}


def test_build_filters_year_only():
    assert tools._build_filters(None, 2025) == {"report_year": 2025}


def test_build_filters_both():
    assert tools._build_filters("Apple", 2025) == {
        "$and": [{"company_name": "Apple"}, {"report_year": 2025}]
    }


def test_search_documents_passes_filters_and_formats(monkeypatch):
    fake = MagicMock()
    fake.search.return_value = [_doc()]
    monkeypatch.setattr(tools, "_store", fake)

    out = tools.search_documents("net sales", company_name="Apple", report_year=2025)

    fake.search.assert_called_once_with(
        "net sales",
        filters={"$and": [{"company_name": "Apple"}, {"report_year": 2025}]},
    )
    assert "[Source: Apple, 10-K, 2025, Page 1]" in out


def test_search_documents_no_filters(monkeypatch):
    fake = MagicMock()
    fake.search.return_value = [_doc()]
    monkeypatch.setattr(tools, "_store", fake)

    tools.search_documents("anything")

    fake.search.assert_called_once_with("anything", filters=None)


def test_search_documents_no_results(monkeypatch):
    fake = MagicMock()
    fake.search.return_value = []
    monkeypatch.setattr(tools, "_store", fake)

    assert tools.search_documents("nothing") == "No matching documents found."


def test_get_metadata_lists_all_companies():
    md = tools.get_metadata()
    for company in ["Apple", "JPMorgan Chase", "HSBC", "AIA Group", "Sony"]:
        assert company in md


def test_registry_and_schemas_match():
    assert set(tools.TOOL_REGISTRY) == {"search_documents", "get_metadata"}
    schema_names = {s["name"] for s in tools.TOOL_SCHEMAS}
    assert schema_names == {"search_documents", "get_metadata"}
    search_schema = next(
        s for s in tools.TOOL_SCHEMAS if s["name"] == "search_documents"
    )
    assert search_schema["input_schema"]["required"] == ["query"]

def test_get_metadata_empty(monkeypatch, tmp_path):
    fake_file = tmp_path / "metadata.json"
    fake_file.write_text('{}')
    monkeypatch.setattr(tools, "METADATA_PATH", fake_file)
    result = tools.get_metadata()
    assert "Available documents:" in result
    assert "Apple" not in result

def test_get_metadata_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "METADATA_PATH", tmp_path / "nonexistent.json")
    result = tools.get_metadata()
    assert "No metadata file found" in result