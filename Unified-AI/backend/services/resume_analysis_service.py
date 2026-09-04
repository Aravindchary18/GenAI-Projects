import requests
import json

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
    response=requests.post(
         "http://ollama:11434/api/generate",
         json={
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

   