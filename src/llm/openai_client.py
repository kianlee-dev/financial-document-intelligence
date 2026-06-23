"""OpenAI API wrapper — handles LLM calls."""

from openai import OpenAI
from langfuse import observe
import os
from types import SimpleNamespace
import json

class OpenAIClient:
    def __init__(self, model: str = None, base_url="http://localhost:11434/v1", api_key: str = "ollama"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model or os.environ.get("LOCAL_MODEL", "mistral")

    @observe()
    def generate(self, prompt: str,  system_prompt: str = "") -> str:
        """Send prompt to LLM API, return response text."""
        response = self.client.chat.completions.create(model= self.model,
        max_tokens=1024,
        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


    def _convert_messages(self, messages: list) -> list:
        """Convert Anthropic-format messages to OpenAI-format."""
        converted = []
        for msg in messages:
            # Case 1: plain string content (user query) — pass through
            if isinstance(msg["content"], str):
                converted.append(msg)
                continue

            # Case 2: assistant message with tool_use blocks
            if msg["role"] == "assistant":
                tool_calls = []
                text = None
                for block in msg["content"]:
                    if getattr(block, "type", None) == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            }
                        })
                    elif getattr(block, "type", None) == "text":
                        text = block.text
                converted.append({
                    "role": "assistant",
                    "content": text,
                    "tool_calls": tool_calls if tool_calls else None,
                })
                continue

            # Case 3: user message with tool_result list
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for block in msg["content"]:
                    if block.get("type") == "tool_result":
                        converted.append({
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": block["content"],
                        })
                continue

        return converted
    

    @observe()
    def generate_with_tools(self, messages: list, system_prompt: str, tools: list) -> SimpleNamespace:
        """Send a conversation with tool schemas to the LLM API."""
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                }
            })

        converted = [{"role": "system", "content": system_prompt}] + self._convert_messages(messages)

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            tools=openai_tools,
            messages=converted,
        )

        content = []
        message = response.choices[0].message

        if message.content:
            content.append(SimpleNamespace(type="text", text=message.content))

        if message.tool_calls:
            for tc in message.tool_calls:
                content.append(SimpleNamespace(
                    type="tool_use",
                    name=tc.function.name,
                    id=tc.id,
                    input=json.loads(tc.function.arguments)
                ))
        return SimpleNamespace(content=content)
   