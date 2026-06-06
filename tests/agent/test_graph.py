"""Unit tests for the agent graph with a mocked LLM."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.documents import Document
from langgraph.graph import END

import src.agent.graph as graph
import src.agent.tools as tools


def _tool_use(block_id, name, inp):
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=inp)


def _text(value):
    return SimpleNamespace(type="text", text=value)


def _doc():
    return Document(
        page_content="value",
        metadata={
            "company_name": "Apple",
            "document_type": "10-K",
            "report_year": 2025,
            "page": 1,
        },
    )

def _doc2():
    return Document(
        page_content="value",
        metadata={
            "company_name": "HSBC",
            "document_type": "Annual Report",
            "report_year": 2025,
            "page": 1,
        },
    )

def test__route_to_tools_on_tool_use():
    state = {"messages": [{"role": "assistant", "content": [_tool_use("t1", "get_metadata", {})]}]}
    assert graph._route(state) == "tools"


def test__route_to_end_on_text():
    state = {"messages": [{"role": "assistant", "content": [_text("done")]}]}
    assert graph._route(state) == END


def test__tools_node_dispatches_and_wraps_result(monkeypatch):
    fake = MagicMock()
    fake.search.return_value = [_doc()]
    monkeypatch.setattr(tools, "_store", fake)

    state = {
        "messages": [
            {
                "role": "assistant",
                "content": [_tool_use("t1", "search_documents", {"query": "sales"})],
            }
        ]
    }
    out = graph._tools_node(state)

    msg = out["messages"][0]
    assert msg["role"] == "user"
    result = msg["content"][0]
    assert result["type"] == "tool_result"
    assert result["tool_use_id"] == "t1"
    assert "[Source: Apple" in result["content"]


def test_run_multi_turn_search_then_answer(monkeypatch):
    fake = MagicMock()
    fake.search.return_value = [_doc()]
    monkeypatch.setattr(tools, "_store", fake)

    fake_llm = MagicMock()
    fake_llm.generate_with_tools.side_effect = [
        SimpleNamespace(content=[_tool_use("u1", "search_documents", {"query": "net sales"})]),
        SimpleNamespace(content=[_text("Apple net sales were 391B.")]),
    ]
    monkeypatch.setattr(graph, "_llm", fake_llm)

    answer = graph.run("What were Apple net sales?")

    assert "Apple net sales" in answer["response"]
    assert fake_llm.generate_with_tools.call_count == 2


def test_run_direct_answer_no_tool(monkeypatch):
    fake_llm = MagicMock()
    fake_llm.generate_with_tools.side_effect = [
        SimpleNamespace(content=[_text("Direct answer.")]),
    ]
    monkeypatch.setattr(graph, "_llm", fake_llm)

    answer = graph.run("hello")

    assert answer["response"] == "Direct answer."
    assert fake_llm.generate_with_tools.call_count == 1

def test_run_cross_company_comparison(monkeypatch):
    fake = MagicMock()
    fake.search.side_effect = [[_doc()], [_doc2()]]
    monkeypatch.setattr(tools, "_store", fake)

    fake_llm = MagicMock()
    fake_llm.generate_with_tools.side_effect = [
        SimpleNamespace(content=[_tool_use("u1", "search_documents", {"query": "net sales", "company_name": "Apple"})]),
        SimpleNamespace(content=[_tool_use("u2", "search_documents", {"query": "net sales", "company_name": "HSBC"})]),
        SimpleNamespace(content=[_text("Apple net sales were 391B vs HSBC net sales were 200B.")]),
    ]
    monkeypatch.setattr(graph, "_llm", fake_llm)

    answer = graph.run("What were Apple net sales vs HSBC net sales?")

    assert "Apple net sales" in answer["response"]
    assert "HSBC net sales" in answer["response"]
    assert fake_llm.generate_with_tools.call_count == 3

def test_run_missing_company(monkeypatch):
    fake = MagicMock()
    fake.search.return_value = []
    monkeypatch.setattr(tools, "_store", fake)

    fake_llm = MagicMock()
    fake_llm.generate_with_tools.side_effect = [
        SimpleNamespace(content=[_tool_use("u1", "search_documents", {"query": "net sales", "company_name": "Tesla"})]),
        SimpleNamespace(content=[_text("Tesla is not available in the document collection.")]),
    ]
    monkeypatch.setattr(graph, "_llm", fake_llm)

    answer = graph.run("What were Tesla net sales")

    assert "Tesla" in answer["response"]
    assert "not available" in answer["response"]
    assert fake_llm.generate_with_tools.call_count == 2
