"""Unit tests for the reflexion node with a mocked LLM."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import src.agent.reflexion as reflexion


def test_verify_json_parse_failure_fallback(monkeypatch):
    fake_llm = MagicMock()
    fake_llm.generate.return_value = "not valid json"
    monkeypatch.setattr(reflexion, "_llm", fake_llm)
    verdict = reflexion.verify("some query", "original answer", ["chunk1"])
    assert verdict["verdict"]["grounded"] == True
    assert not verdict["verdict"]["flagged_claims"]
    assert verdict["response"] == "original answer"