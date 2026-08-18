import requests

import json

def analyze_skill_gap(resume_data, job_description):

    prompt =f"""
You are a strict JSON generator.

RULES:
- Output ONLY valid JSON
- No explanation
- No text before or after JSON

Return format:

{{
  "matched_skills": [],
  "missing_skills": [],
  "ats_score": 0,
  "resume_improvements": [],
  "learning_recommendations": []
}}

Resume:
{resume_data}

Job:
{job_description}
"""
    
    response = requests.post(
        "http://ollama:11434/api/generate",
        json = {
            "model":"qwen2.5-coder:3b-instruct",
            "prompt": prompt,
            "stream":False,
            "format": "json",

            "options": {
                "temperature": 0
            }

        }
    )
   
    
    result = response.json()["response"]
    

    result = json.loads(result)

    return result