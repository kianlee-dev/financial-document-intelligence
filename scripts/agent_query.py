"""Agent query script — run sample queries through the tool-calling agent."""

from src.agent import run

if __name__ == "__main__":
    queries = [
        "What was Apple's total net sales in fiscal year 2025?",        # single-company
        #"Compare HSBC and JPMorgan's net income in 2025",              # cross-company
        #"What companies and years are available in the collection?",   # metadata lookup
        #"What was Tesla's revenue in 2025?",                            # unanswerable
    ]

    for query in queries:
        result = run(query)
        print(f"\nQ: {query}")
        print(f"A: {result['response']}")
        print("-" * 80)
