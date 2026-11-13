from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama



ollama_url = (
    "http://ollama:11434"
    if os.name != "nt"
    else "http://localhost:11434"
)

bllm = ChatOpenAI(
    model="qwen-3.8-27b",
    api_key=os.environ["bharat_akey"],
    base_url="https://bharatcode.ai/api/model/v1",
    temperature=0,
    timeout=180,
    max_retries=0
)
local_llm = ChatOllama(
    model = "qwen3:4b",
    base_url=ollama_url,
    temperature=0,
    validate_model_on_init=False,
    reasoning=False,
    timeout=180
)

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
    
    llm = local_llm.with_fallbacks([bllm])
        
    response = llm.invoke(prompt)
        
    return response.content