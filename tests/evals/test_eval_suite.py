import pytest
import os
from src.evals.runner import run_eval

pytestmark = [
       pytest.mark.eval,
       pytest.mark.skipif(
           not os.environ.get("ANTHROPIC_API_KEY"),
           reason="ANTHROPIC_API_KEY not set",
       )
   ]

@pytest.fixture(scope="module")
def eval_results():
    """Run the full eval suite once, share results across all test functions."""
    return run_eval()

def test_relevance_threshold(eval_results):
    assert eval_results["relevance"] >= 3.5

def test_accuracy_threshold(eval_results):
    assert eval_results["accuracy"] >= 3.5

def test_faithfulness_threshold(eval_results):
    assert eval_results["faithfulness"] >= 3.5

def test_precision_threshold(eval_results):
    assert eval_results["precision"] >= 0.7

def test_zero_hallucinations(eval_results):
    assert eval_results["hallucinations"] == 0