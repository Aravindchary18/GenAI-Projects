import json
import time
from pathlib import Path
import os
from dotenv import load_dotenv
from services.agent_service import agent


load_dotenv()

if not os.getenv("TAVILY_AKEY"):
    raise ValueError("TAVILY_AKEY not found in .env")

DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "web_search.dataset.py"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_PATH = RESULTS_DIR / "web_eval.json"



def get_message_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict)
        )

    return str(content)


def evaluate_test(test):

    start_time = time.perf_counter()

    try:
        # Invoke agent directly to bypass semantic cache.
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": test["question"]
                    }
                ]
            },
            config={"recursion_limit": 6}
        )

        elapsed = round(time.perf_counter() - start_time, 2)

        messages = response.get("messages", [])

        # Check whether the LLM requested a tool call.
        tool_calls = []

        for message in messages:
            if getattr(message, "tool_calls", None):
                tool_calls.extend(message.tool_calls)

        # Check whether the tool returned results.
        tool_messages = [
            message for message in messages
            if getattr(message, "type", None) == "tool"
        ]

        # Extract final AI answer.
        answer = ""

        for message in reversed(messages):
            if (
                getattr(message, "type", None) == "ai"
                and not getattr(message, "tool_calls", None)
                and message.content
            ):
                answer = get_message_text(message.content)
                break

        # Keyword matching.
        answer_lower = answer.lower()
        expected = test["expected_keywords"]

        matched = [
            keyword for keyword in expected
            if keyword.lower() in answer_lower
        ]

        missing = [
            keyword for keyword in expected
            if keyword.lower() not in answer_lower
        ]

        # Calculate keyword coverage.
        coverage = (
            len(matched) / len(expected) * 100
            if expected else 0
        )

        # Confirm actual tool execution.
        tool_used = bool(tool_calls) and bool(tool_messages)

        # Pass/fail.
        passed = (
            tool_used
            and bool(answer)
            and len(matched) >= test["min_keyword_matches"]
        )

        return {
            "id": test["id"],
            "question": test["question"],
            "answer": answer,
            "tool_used": tool_used,
            "tool_call_count": len(tool_calls),
            "tool_result_count": len(tool_messages),
            "matched_keywords": matched,
            "missing_keywords": missing,
            "keyword_coverage": round(coverage, 2),
            "latency_seconds": elapsed,
            "passed": passed,
            "error": None
        }

    except Exception as error:

        return {
            "id": test["id"],
            "question": test["question"],
            "answer": "",
            "tool_used": False,
            "tool_call_count": 0,
            "tool_result_count": 0,
            "matched_keywords": [],
            "missing_keywords": test["expected_keywords"],
            "keyword_coverage": 0,
            "latency_seconds": round(
                time.perf_counter() - start_time, 2
            ),
            "passed": False,
            "error": str(error)
        }


def main():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    results = []

    for index, test in enumerate(dataset, start=1):

        print(f"\nRunning {index}/{len(dataset)}: {test['id']}")

        result = evaluate_test(test)
        results.append(result)

        print("Tool used:", result["tool_used"])
        print("Keyword coverage:", result["keyword_coverage"], "%")
        print("Latency:", result["latency_seconds"], "seconds")
        print("Passed:", result["passed"])

        if result["error"]:
            print("Error:", result["error"])

    total = len(results)
    passed = sum(result["passed"] for result in results)
    tool_used = sum(result["tool_used"] for result in results)

    average_coverage = (
        sum(result["keyword_coverage"] for result in results) / total
        if total else 0
    )

    average_latency = (
        sum(result["latency_seconds"] for result in results) / total
        if total else 0
    )

    summary = {
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total * 100, 2) if total else 0,
        "tool_invocation_rate": round(
            tool_used / total * 100, 2
        ) if total else 0,
        "average_keyword_coverage": round(average_coverage, 2),
        "average_latency_seconds": round(average_latency, 2),
        "results": results
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    print("\n========== WEB SEARCH EVALUATION ==========")
    print(f"Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Pass rate: {summary['pass_rate']}%")
    print(f"Tool invocation rate: {summary['tool_invocation_rate']}%")
    print(f"Average keyword coverage: {average_coverage:.2f}%")
    print(f"Average latency: {average_latency:.2f}s")
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()