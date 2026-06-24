"""ADK agent query script — run queries through the Google ADK agent."""

import asyncio
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from src.agent_adk.agent import root_agent


async def run_single_query(query: str):
    """Run a single query with a fresh runner and session."""
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name="financial-adk", session_service=session_service)
    session = await session_service.create_session(app_name="financial-adk", user_id="test")
    content = types.Content(role="user", parts=[types.Part(text=query)])
    async for event in runner.run_async(user_id="test", session_id=session.id, new_message=content):
        if event.is_final_response() and event.content:
            print(f"Q: {query}")
            print(f"A: {event.content.parts[0].text[:300]}")
            print("---")


async def main():
    queries = [
        "What was Apple's total net sales in fiscal year 2025?",
        "What companies are available?",
        "What was Tesla's revenue in 2025?",
        "Compare HSBC and JPMorgan net income in 2025",
    ]

    for query in queries:
        await run_single_query(query)
        await asyncio.sleep(2) 

if __name__ == "__main__":
    asyncio.run(main())