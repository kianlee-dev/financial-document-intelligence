"""Real-API smoke test. Skipped unless ANTHROPIC_API_KEY is set.

Requires an ingested ChromaDB collection. Run explicitly with:
    pytest tests/agent/test_smoke.py -m smoke
"""

import os

import pytest

pytestmark = pytest.mark.smoke


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_agent_answers_single_company():
    from src.agent import run

    answer = run("What was Apple's total net sales in fiscal year 2025?")

    assert answer["response"].strip()
    assert "Apple" in answer["response"]
