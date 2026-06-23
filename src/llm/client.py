"""LLM client factory — routes to Anthropic or OpenAI backend."""

import os


class LLMClient:
    """Factory class — returns AnthropicClient or OpenAIClient based on LLM_BACKEND."""
    def __new__(cls):
        backend = os.environ.get("LLM_BACKEND", "anthropic")
        if backend == "openai":
            from src.llm.openai_client import OpenAIClient
            return OpenAIClient()
        from src.llm.anthropic_client import AnthropicClient
        return AnthropicClient()