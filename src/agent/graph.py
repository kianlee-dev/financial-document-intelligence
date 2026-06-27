"""LangGraph StateGraph for the tool-calling agent.

Two nodes loop until the model answers without requesting a tool:

    agent --(tool_use)--> tools --> agent
    agent --(no tool_use)--> END

Messages are kept in Anthropic content-block format (not LangChain messages),
so the graph is hand-wired rather than using langgraph-prebuilt's ToolNode.
"""

import operator
from typing import Annotated, TypedDict

from langfuse import observe, get_client
from langgraph.graph import END, StateGraph

from src.agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS
from src.context.prompt_builder import get_system_prompt
from src.llm.client import LLMClient
from src.agent.reflexion import verify
import os


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    retrieved_chunks: Annotated[list, operator.add]
    reflexion_verdict: dict


_llm: LLMClient | None = None


def _get_llm() -> LLMClient:
    """Lazily build a single LLMClient so import stays cheap and testable."""
    global _llm
    if _llm is None:
        _llm = LLMClient()
    return _llm


def _is_tool_use(block) -> bool:
    return getattr(block, "type", None) == "tool_use"


@observe()
def _agent_node(state: AgentState) -> dict:
    """Ask the model what to do next; append its reply to the messages."""
    response = _get_llm().generate_with_tools(
        state["messages"], get_system_prompt(), TOOL_SCHEMAS
    )
    return {"messages": [{"role": "assistant", "content": response.content}]}


@observe()
def _tools_node(state: AgentState) -> dict:
    """Execute every tool_use block in the last reply, return tool_result blocks."""
    last = state["messages"][-1]
    results = []
    chunks_list = []
    for block in last["content"]:
        if _is_tool_use(block):
            try:
                output = TOOL_REGISTRY[block.name](**block.input)
                if block.name == "search_documents":
                    chunks_list.append(output)
            except TypeError as e:
                output = f"Tool call failed: {e}"
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                }
            )
    return {"messages": [{"role": "user", "content": results}], "retrieved_chunks": chunks_list}

@observe()
def _reflexion_node(state: AgentState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"messages": []}
    last = state["messages"][-1]
    texts = [
        block.text
        for block in last["content"]
        if getattr(block, "type", None) == "text"
    ]
    answer = "\n".join(texts)
    query = state["messages"][0]["content"]
    result = verify(query, answer, chunks)
    return {"messages": [{"role": "assistant", "content": result["response"]}], "reflexion_verdict": result["verdict"]}


def _route(state: AgentState) -> str:
    """Continue to the tools node if the model requested a tool, else stop."""
    last = state["messages"][-1]
    if any(_is_tool_use(block) for block in last["content"]):
        return "tools"
    return END


def build_graph(reflexion = None):
    """Build and compile the agent StateGraph."""
    if reflexion is None:
        reflexion = os.environ.get("REFLEXION_ENABLED", "false").lower() == "true"
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", _tools_node)
    graph.set_entry_point("agent")
    if reflexion:
        graph.add_node("reflexion", _reflexion_node)
        graph.add_conditional_edges("agent", _route, {"tools": "tools", END: "reflexion"})
        graph.add_edge("reflexion", END)
    else:
        graph.add_conditional_edges("agent", _route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


@observe()
def run(query: str, reflexion = None) -> dict:
    """Run the agent on a single query and return the final answer text."""
    langfuse = get_client()
    graph = build_graph(reflexion)
    state = graph.invoke({"messages": [{"role": "user", "content": query}]})
    final = state["messages"][-1]
    if isinstance(final["content"], str):
        response_text = final["content"]
    else:
        texts = [
            block.text
            for block in final["content"]
            if getattr(block, "type", None) == "text"
        ]
        response_text = "\n".join(texts)

    trace_id = langfuse.get_current_trace_id()

    chunks = state.get("retrieved_chunks", [])

    verdict = state.get("reflexion_verdict", {})
    grounded = verdict.get("grounded", True)
    flagged_claims = verdict.get("flagged_claims", [])

    return {"response": response_text, "trace_id": trace_id, "retrieved_chunks": chunks, "grounded": grounded, "flagged_claims": flagged_claims}