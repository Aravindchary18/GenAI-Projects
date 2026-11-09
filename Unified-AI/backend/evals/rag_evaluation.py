import os

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
from data.rag_dataset import GOLDEN_DATASET_RAG
from services.rag_service import rag_pipeline





def evaluate_rag_output(answer, expected_keywords):

    answer = answer.lower()

    matched = []
    missing = []

    for keyword in expected_keywords:

        if keyword.lower() in answer:
            matched.append(keyword)

        else:
            missing.append(keyword)

    score = (
        len(matched) / max(1, len(expected_keywords))
    ) * 100

    return {
        "score": round(score, 2),
        "matched_keywords": matched,
        "missing_keywords": missing
    }


def run_rag_evaluation():

    document_id = os.getenv("RAG_EVAL_DOCUMENT_ID")

    if not document_id:
        raise ValueError(
            "Please set RAG_EVAL_DOCUMENT_ID in your .env file."
        )

    results = []

    for case in GOLDEN_DATASET_RAG:

        print(f"\nEvaluating: {case['id']}")

        # CALL EXISTING RAG PIPELINE
        response = rag_pipeline(
            case["question"],
            document_id
        )

        # COLLECT STREAMING RESPONSE
        answer = "".join(response)

        # EVALUATE RESPONSE
        evaluation = evaluate_rag_output(
            answer,
            case["expected_keywords"]
        )

        results.append({

            "id": case["id"],

            "score": evaluation["score"],

            "passed": evaluation["score"] >= 70,

            "matched_keywords": evaluation["matched_keywords"],

            "missing_keywords": evaluation["missing_keywords"]

        })

    total_tests = len(results)

    passed = sum(
        1 for result in results
        if result["passed"]
    )

    accuracy = (passed / max(1, total_tests)) * 100

    return {

        "accuracy": round(accuracy, 2),

        "total_tests": total_tests,

        "passed": passed,

        "failed": total_tests - passed,

        "results": results

    }


if __name__ == "__main__":

    import json

    print(
        f"RAG EVALUATION:\n"
        f"{json.dumps(run_rag_evaluation(), indent=4)}"
    )