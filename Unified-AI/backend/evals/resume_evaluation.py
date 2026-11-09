from data.golden_dataset_resume import GOLDEN_DATASET_RESUME
from services.resume_analysis_service import analyze_resume
import json
import re
from rapidfuzz import fuzz
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

def fuzz_score(llm_items,expected_items):

    total_score = 0

    for expected_item in expected_items:
        
        best_score = 0

        for llm_item in llm_items:

            score = fuzz.ratio(expected_item.lower(),llm_item.lower())

            best_score = max(best_score,score)
        
        total_score += best_score

    return total_score/(max(1,len(expected_items)))


def evaluate_re_llm_output(llm_items,expected):

    skill_score = fuzz_score(
        llm_items.get("skills",[]),
        expected.get("expected_skills",[])
    )

   

    level_score = fuzz_score(
        [llm_items.get("level","")],
        [expected.get("expected_level","")]
    )


    final_score = (
        skill_score*0.85 + level_score*0.15
    )

    return {
        "final_score": round(final_score,2),

        "skill_score": round(skill_score,2),

        "level_score": round(level_score,2),

    }

def run_re_evaluation():
    
    results = []

    for case in GOLDEN_DATASET_RESUME:

        raw_output = analyze_resume(case["input"])

        clean_output = re.sub(
            r"^\s*```(?:json)?\s*|\s*```\s*$",
            "",
            raw_output,
            flags=re.IGNORECASE
        ).strip()

        llm_output = json.loads(clean_output)

        score = evaluate_re_llm_output(llm_output,case)

        results.append({
            "id": case["id"],

            "score": score["final_score"]
        })

    passed = sum(
            1 for r in results 
            if r["score"]>=70
        )

    accuracy = (passed/len(results))*100

    return{

           "accuracy": round(accuracy,2),

            "total_tests": len(results),

            "passed": passed,

            "failed": len(results)-passed,

            "results": results
        }

if __name__ == "__main__":
    print(f"RESUME EVALUATION:\n{json.dumps(run_re_evaluation(),indent = 4)}")

