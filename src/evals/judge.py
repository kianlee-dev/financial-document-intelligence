import json
from src.llm.anthropic_client import AnthropicClient

_llm = None

def _get_llm() -> AnthropicClient:
    """Lazily build a single LLMClient so import stays cheap and testable."""
    global _llm
    if _llm is None:
        _llm = AnthropicClient()
    return _llm

def judge(query: str, expected: str, actual: str, chunks: list[str]) -> dict:
    """Score a response on relevance, accuracy, and faithfulness (1-5 each)."""
    sources = "\n".join(chunks) if chunks else "None"
    prompt = (
        f"Query:\n{query}\n\n"
        f"Expected Answer:\n{expected}\n\n"
        f"Actual Response:\n{actual}\n\n"
        f"Retrieved Sources:\n{sources}"
    )
    result = _get_llm().generate(prompt, get_judge_prompt())
    result = result.strip().removeprefix("```json").removesuffix("```").strip()
    return json.loads(result)


def get_judge_prompt() -> str:
    """Return the system prompt for evaluation judge"""
    return (
        "You are an evaluation judge. Score the actual response against the expected answer objectively. "
        "Do not help, explain, or add commentary. Return ONLY a JSON object, no preamble, no markdown fences.\n\n"
        "Scoring dimensions (each 1-5):\n"
        "- relevance: Does the response address the question asked? 1 = completely off-topic, 5 = directly answers the question.\n"
        "- accuracy: Do the factual claims match the expected answer? 1 = all facts wrong, 5 = all facts correct. "
        "A response can be relevant but inaccurate if it answers the right question with wrong numbers.\n"
        "- faithfulness: Is every claim in the response supported by the retrieved sources? "
        "1 = fabricated claims with no source support, 5 = every claim traceable to a retrieved source. "
        "If no sources were retrieved, score based on whether the response appropriately declines to answer.\n\n"
        'Return exactly this format:\n'
        '{"relevance": int, "relevance_reason": "one sentence", "accuracy": int, '
        '"accuracy_reason": "one sentence", "faithfulness": int, "faithfulness_reason": "one sentence"}'
    )