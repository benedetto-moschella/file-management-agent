"""
Script for evaluating the FileAgent's performance on a predefined dataset.
"""
import json
import shutil
from pathlib import Path
from typing import Dict, Any

from agent.agent_core import FileAgent


def _run_single_test(case: Dict[str, Any], case_index: int) -> Dict[str, Any]:
    """Sets up, runs, and evaluates a single test case."""
    query = case["query"]
    expected_keyword = case["expected_keyword"]
    setup_files = case.get("setup_files", [])

    print(f"Query: {query}")

    # Crea una workspace temporanea e isolata per ogni test
    test_workspace_path = Path(f"./temp_eval_workspace_{case_index}")
    if test_workspace_path.exists():
        shutil.rmtree(test_workspace_path)
    test_workspace_path.mkdir()

    if setup_files:
        print(f"Setting up {len(setup_files)} file(s) for this test case...")
        for file_info in setup_files:
            (test_workspace_path / file_info["filename"]).write_text(
                file_info["content"], encoding="utf-8"
            )

    agent = FileAgent(base_path=str(test_workspace_path))
    response = agent.plan(query)
    actual_output = response.get("output", "")

    print(f"Agent Output: {actual_output}")

    test_passed = expected_keyword.lower() in actual_output.lower()
    if test_passed:
        print("✅ Result: PASSED")
    else:
        print(f"❌ Result: FAILED (Expected to find '{expected_keyword}')")

    shutil.rmtree(test_workspace_path)

    return {
        "query": query,
        "expected": expected_keyword,
        "actual": actual_output,
        "passed": test_passed
    }


def _print_summary(total_cases: int, passed_count: int):
    """Prints the final evaluation summary."""
    accuracy = (passed_count / total_cases) * 100 if total_cases else 0
    print("\n--- 🏁 Evaluation Finished ---")
    print(f"Total Test Cases: {total_cases}")
    print(f"Passed: {passed_count}")
    print(f"Accuracy: {accuracy:.2f}%")


def run_evaluation():
    """
    Loads the evaluation dataset, runs the agent on each query,
    and reports the accuracy.
    """
    print("🚀 Starting agent evaluation...")

    dataset_path = Path("evaluation_dataset.jsonl")
    if not dataset_path.exists():
        print(f"❌ Error: Evaluation dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = [json.loads(line) for line in f]

    results = []
    for i, case in enumerate(test_cases):
        print(f"\n--- Running Test Case {i+1}/{len(test_cases)} ---")
        result = _run_single_test(case, i)
        results.append(result)

    passed_count = sum(1 for r in results if r["passed"])
    _print_summary(len(test_cases), passed_count)


if __name__ == "__main__":
    run_evaluation()
    