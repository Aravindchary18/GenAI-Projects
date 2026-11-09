
import json
from rapidfuzz import fuzz
from data.golden_dataset_skill_gap import GOLDEN_DATASET_SKILL_GAP
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
from services.skill_gap_service import analyze_skill_gap

def fuzz_score(llm_items,expected_items):

    total_score = 0

    for expected_item in expected_items:
        
        best_score = 0

        for llm_item in llm_items:

            score = fuzz.ratio(expected_item.lower(),llm_item.lower())

            best_score = max(best_score,score)
        
        total_score += best_score

    return total_score/(max(1,len(expected_items)))


def evaluate_sg_llm_output(llm_items,expected_items):

    matched_score = fuzz_score(
        llm_items.get("matched_skills",""),
        expected_items.get("expected_matched_skills","")    
    )

    missing_score = fuzz_score(
        llm_items.get("missing_skills",""),
        expected_items.get("expected_missing_skills","")    
    )


    low,high = expected_items["expected_ats_range"]

  
    ats_score_value = llm_items.get("ats_score",0)
  
    ats_score = 100 if (low<=ats_score_value<=high) else 0

    final_score = (matched_score*0.4 + missing_score*0.4+ ats_score*0.2)

    return {
        "final_score": round(final_score,2),

        "matched_score": round(matched_score,2),

        "missing_score": round(missing_score,2),

        "ats_score": ats_score
    }


def run_sg_evaluation():
    results=[]

    for case in GOLDEN_DATASET_SKILL_GAP:

        llm_output = json.loads(analyze_skill_gap(case["resume"] , case["job_description"]))
        
       

        score = evaluate_sg_llm_output(llm_output,case)
        
        results.append({
            "id": case["id"],
            "score": score["final_score"]
        })

    passed=sum(1 for r in results if r["score"]>=70)
    
    accuracy = (passed / len(results))*100
    
    return {
        "accuracy": round(accuracy, 2),
        "total_tests": len(results),
        "passed": passed,
        "failed": len(results)-passed,
        "results": results
    }


if __name__ == "__main__":

    print(f"SKILL GAP EVALUATION:\n{json.dumps(run_sg_evaluation(),indent = 4)}")
