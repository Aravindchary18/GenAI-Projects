from langchain_openai import ChatOpenAI
import os

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
    
    bllm = ChatOpenAI(
            model="qwen-3.8-27b",
            api_key=os.environ["bharat_akey"],
            base_url="https://bharatcode.ai/api/model/v1",
            temperature=0,
            timeout=180,
            max_retries=0
            )
        
    result = bllm.invoke(prompt)
        
    return result.content