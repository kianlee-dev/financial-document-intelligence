"""Claude API wrapper — handles LLM calls."""

from anthropic import Anthropic
from anthropic.types import Message
from langfuse import observe

class LLMClient:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = Anthropic()
        self.model = model

    @observe()
    def generate(self, prompt: str,  system_prompt: str = "") -> str:
        """Send prompt to LLM API, return response text."""
        response = self.client.messages.create(model= self.model,
        max_tokens=1024,
        system = system_prompt,
        messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    @observe()
    def generate_with_tools(self, messages: list, system_prompt: str, tools: list) -> Message:
        """Send a conversation with tool schemas to the LLM API.

        Returns the raw Anthropic Message; the caller reads .content (text and
        tool_use blocks) and .stop_reason to drive the agent loop.
        """
        return self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )
