from data.golden_dataset_roadmap import GOLDEN_DATASET_ROADMAP

import json

from services.roadmap_service import analyze_roadmap

from rapidfuzz import fuzz

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

def fuzz_ratio_score(llm_items,expected_items):

    total_score = 0

    for expected_item in expected_items:
        
        best_score = 0

        for llm_item in llm_items:

            score = fuzz.ratio(expected_item.lower(),llm_item.lower())

            best_score = max(best_score,score)
        
        total_score += best_score

    return total_score/(max(1,len(expected_items)))

def fuzz_token_ratio_score(llm_items,expected_items):

    total_score = 0

    for expected_item in expected_items:
        
        best_score = 0

        for llm_item in llm_items:

            score = fuzz.token_set_ratio(expected_item.lower(),llm_item.lower())

            best_score = max(best_score,score)
        
        total_score += best_score

    return total_score/(max(1,len(expected_items)))


def evaluate_rm_llm_output(llm_output,expected):

    

    skills_score = fuzz_ratio_score(
        llm_output.get("current_assessment",[]),
        expected.get("expected_current_assessment_contains",[])
    )
    phases = set()

    for item in llm_output.get("roadmap_phases",[]):
        if isinstance(item,dict) and "phase" in item :
            phases.add(item["phase"])
        

    expected_phases = set(expected["expected_phases"])

    phase_score = fuzz_ratio_score(
        phases,
        expected_phases
    )

    llm_roadmap = set()
    for key in llm_output.get("roadmap_phases",[]):
        if isinstance(key, dict):

            llm_roadmap.update(key.get("topics",[]))
            llm_roadmap.update(key.get("projects",[]))            
            goal = key.get("goal","")
            if goal:
                llm_roadmap.add(goal)

    llm_roadmap.update(llm_output.get("interview_preparation",[]))
    llm_roadmap.update(llm_output.get("portfolio_projects",[]))

    for item in (llm_output.get("learning_resources",[])):
        if isinstance(item,str):
            llm_roadmap.add(item)
        
        elif isinstance(item,dict):
                name = item.get("name")
                if name :
                    llm_roadmap.add(name)
                
            
    expected_roadmap = set(expected["expected_roadmap_contains"])

    roadmap_score = fuzz_token_ratio_score(
        llm_roadmap,
        expected_roadmap
    )

    final_roadmap_score = (skills_score*0.3 + phase_score*0.2 + roadmap_score*0.5)

    return {
        "final_roadmap_score": round(final_roadmap_score, 2),
        "skills_score": round(skills_score,2),
        "phase_score": round(phase_score,2),
        "roadmap_score": round(roadmap_score,2),
    }

def run_rm_evaluation():
    results=[]

    for test in GOLDEN_DATASET_ROADMAP:
        llm_output = json.loads(analyze_roadmap(
            test["current_skills"],
            test["target_role"],
            test["experience_years"]
        ))
       
        score = evaluate_rm_llm_output(llm_output,test)

        results.append({
            "id": test["id"],
            "score": score["final_roadmap_score"]
        })
    passed = sum(1 for r in results if r["score"]>=70)

    accuracy = (passed / len(results))*100

    return {
        "accuracy": round(accuracy,2),
        "total_tests": len(results),
        "passed":passed,
        "failed": len(results)-passed,
        "results":results
    }

if __name__ == "__main__":
    print(f"ROADMAP EVALUATION:\n{json.dumps(run_rm_evaluation(),indent = 4)}")