from src.agent import run
from src.evals.test_dataset import EVAL_DATASET
from src.evals.judge import judge
from langfuse import get_client

def run_eval() -> dict:
    """Run the full eval suite and return average scores."""
    score_list = []
    langfuse = get_client()
    precisions = []
    hallucinations = 0

    for data in EVAL_DATASET:
        query = data["query"]
        expected = data["expected_answer"]

        # Agent call — returns response, trace_id, and retrieved chunks
        output = run(query)
        actual = output["response"]
        chunks = output["retrieved_chunks"]

        # LLM-as-judge — second API call scores relevance, accuracy, faithfulness (1-5)
        scores = judge(query, expected, actual, chunks)
        score_list.append(scores)

        # Attach scores to Langfuse trace for dashboard filtering
        for dim in ["relevance", "accuracy", "faithfulness"]:
            langfuse.create_score(
                name=dim,
                value=scores[dim],
                trace_id=output["trace_id"],
                data_type="NUMERIC",
                comment=scores[f"{dim}_reason"],
            )

        # Retrieval precision@k
        if data["expected_company"]:
            matches = sum(1 for chunk in chunks if f"[Source: {data['expected_company']}" in chunk)
            precision = matches / len(chunks) if chunks else 0
            precisions.append(precision)

        # Hallucination count
        if scores["faithfulness"] < 3:
            hallucinations += 1

    # --- Averages ---
    avg_relevance = sum(s["relevance"] for s in score_list) / len(score_list)
    avg_accuracy = sum(s["accuracy"] for s in score_list) / len(score_list)
    avg_faithfulness = sum(s["faithfulness"] for s in score_list) / len(score_list)
    avg_precision = sum(precisions) / len(precisions) if precisions else 0

    # --- Summary table ---
    print(f"\n{'='*80}")
    print(f"{'Query':<60} {'Rel':>4} {'Acc':>4} {'Fai':>4}")
    print(f"{'='*80}")
    for data, scores in zip(EVAL_DATASET, score_list):
        q = data["query"][:57] + "..." if len(data["query"]) > 57 else data["query"]
        print(f"{q:<60} {scores['relevance']:>4} {scores['accuracy']:>4} {scores['faithfulness']:>4}")
    print(f"{'='*80}")
    print(f"{'Averages':<60} {avg_relevance:>4.1f} {avg_accuracy:>4.1f} {avg_faithfulness:>4.1f}")
    print(f"Precision@k: {avg_precision:.2f} | Hallucinations: {hallucinations}")
    print(f"Thresholds: Avg >= 3.5 | Precision >= 0.7 | Hallucinations == 0")

    return {"relevance": avg_relevance, "accuracy": avg_accuracy, "faithfulness": avg_faithfulness, "precision": avg_precision, "hallucinations": hallucinations}

if __name__ == "__main__":
    run_eval()