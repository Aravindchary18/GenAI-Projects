from langchain_openai import ChatOpenAI
import os

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