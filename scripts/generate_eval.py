from src.agent import run

queries = [
    "What was AIA Group's annualised new premiums in 2025?",
    "What was Apple's total revenue from the iPhone segment in 2025?",
    "What was JPMorgan's net income in 2025?",
    "What was HSBC's common equity tier 1 ratio in 2025?",
    "What was Sony's total revenue in fiscal year 2025?",
    "What was Apple's research and development expenditure in 2025?",
    "What was AIA Group's value of new business in 2025?",
    "What was JPMorgan's return on equity in 2025?",
    "Compare JPMorgan and HSBC's total assets in 2025",
    "What are AIA Group's main risk factors?",
    "What is HSBC's strategy for wealth management?",
    "What was Amazon's net income in 2025?",
]

for i, q in enumerate(queries, 1):
    output = run(q)
    print(f"{i}. Q: {q}")
    print(f"   A: {output['response'][:300]}")
    print(f"   Chunks: {len(output['retrieved_chunks'])}")
    print("===")