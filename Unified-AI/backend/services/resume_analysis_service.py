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

def analyze_resume(resume_text):

    prompt = f"""
you are an expert AI career mentor.

Analyze the resume below.

Return only valid json. Do not add explainations or texts.

Json Format : 
{{
    "skills": [],
    "level":"",
    "projects": [],
    "education": [],
    "experience": [],
    "strengths": [],
    "weaknesses": [],
    "resume improvements": [],   
    "score": 0
}}

Resume :
{resume_text}
"""
    

    llm = local_llm.with_fallbacks([bllm])

    response = llm.invoke(prompt)

    return response.content