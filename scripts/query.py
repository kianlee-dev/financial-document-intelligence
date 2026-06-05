"""Q&A query script — retrieve chunks, build prompt, call LLM API."""

from src.context.prompt_builder import get_system_prompt, get_prompt
from src.llm.client import LLMClient
from src.retrieval.vectorstore import VectorStore

if __name__ == "__main__":
    vector_store = VectorStore()
    llm = LLMClient()
    system_prompt = get_system_prompt()

    queries = [
        "What was Apple's total net sales in fiscal year 2025?",
        "Compare HSBC and JPMorgan's net income in 2025",
        "What are Apple's main risk factors?",
        "What was Tesla's revenue in 2025?",
    ]

    for query in queries:
        chunks = vector_store.search(query)
        prompt = get_prompt(query, chunks)
        response = llm.generate(prompt, system_prompt)
        print(f"\nQ: {query}")
        print(f"A: {response}")
        print("-" * 80)
