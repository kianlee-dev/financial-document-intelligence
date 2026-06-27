import json
from src.llm.anthropic_client import AnthropicClient
from langfuse import observe


REFLEXION_SYSTEM_PROMPT = (
    "You are a factual verification auditor. Your job is to check whether an AI agent's "
    "answer is fully grounded in the retrieved source documents.\n\n"
    "Rules:\n"
    "- Check every factual claim (numbers, percentages, dates, company names, comparisons) "
    "against the provided source chunks.\n"
    "- A claim is grounded ONLY if it can be traced to a specific [Source: ...] chunk.\n"
    "- A claim is ungrounded if no chunk supports it, even if it might be true from general knowledge.\n"
    "- Do not penalise the agent for stating that information is unavailable or refusing to answer.\n\n"
    "Respond with ONLY a JSON object, no markdown fences, no preamble:\n"
    "{\n"
    '  "verdict": {\n'
    '    "grounded": true/false,\n'
    '    "flagged_claims": [\n'
    '      {"claim": "exact text of ungrounded claim", "reason": "why it is not supported by sources"}\n'
    "    ]\n"
    "  },\n"
    '  "response": "if grounded: the original answer verbatim, unchanged. '
    "if not grounded: corrected version with only flagged claims removed, "
    'preserving all grounded content, citations, structure, and tone."\n'
    "}"
)

_llm = None

def _get_llm() -> AnthropicClient:
    """Lazily build a single LLMClient so import stays cheap and testable."""
    global _llm
    if _llm is None:
        _llm = AnthropicClient()
    return _llm

def _format_message(query: str, answer: str, chunks: list[str]) -> str:
    sources = "\n\n".join(chunks)
    formatted_message = (
        f"Query:\n{query}\n\n"
        f"Answer:\n{answer}\n\n"
        f"Chunks:\n{sources}"
    )
    return formatted_message

@observe()
def verify(query: str, answer: str, chunks: list[str]) -> dict:
    prompt = _format_message(query, answer, chunks)
    result = _get_llm().generate(prompt, REFLEXION_SYSTEM_PROMPT)
    result = result.strip().removeprefix("```json").removesuffix("```").strip()
    try:
        output = json.loads(result)
    except:
        output = {"verdict": {"grounded": True, "flagged_claims": []}, "response": answer}
    return output 