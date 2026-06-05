"""Claude API wrapper — handles LLM calls."""

from anthropic import Anthropic

class LLMClient:
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.client = Anthropic()
        self.model = model

    def generate(self, prompt: str,  system_prompt: str = "") -> str:
        """Send prompt to LLM API, return response text."""
        response = self.client.messages.create(model= self.model,
        max_tokens=1024,
        system = system_prompt,
        messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
