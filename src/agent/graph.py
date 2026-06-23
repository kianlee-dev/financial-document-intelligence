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


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


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
    for block in last["content"]:
        if _is_tool_use(block):
            output = TOOL_REGISTRY[block.name](**block.input)
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                }
            )
    return {"messages": [{"role": "user", "content": results}]}


def _route(state: AgentState) -> str:
    """Continue to the tools node if the model requested a tool, else stop."""
    last = state["messages"][-1]
    if any(_is_tool_use(block) for block in last["content"]):
        return "tools"
    return END


def build_graph():
    """Build and compile the agent StateGraph."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", _agent_node)
    graph.add_node("tools", _tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


@observe()
def run(query: str) -> dict:
    """Run the agent on a single query and return the final answer text."""
    langfuse = get_client()
    graph = build_graph()
    state = graph.invoke({"messages": [{"role": "user", "content": query}]})
    final = state["messages"][-1]
    texts = [
        block.text
        for block in final["content"]
        if getattr(block, "type", None) == "text"
    ]

    trace_id = langfuse.get_current_trace_id()

    # Map tool_use IDs to names, collect content from search_documents results only

    tool_map = {}
    chunks = []

    for message in state["messages"]:
        if message["role"] == "assistant":
            for block in message["content"]:
                if _is_tool_use(block):
                    tool_map[block.id] = block.name
        elif message["role"] == "user" and isinstance(message["content"], list):
            for block in message["content"]:
                if block.get("type") == "tool_result":
                    if tool_map.get(block["tool_use_id"]) == "search_documents":
                        chunks.append(block["content"])
    return {"response": "\n".join(texts), "trace_id": trace_id, "retrieved_chunks": chunks}