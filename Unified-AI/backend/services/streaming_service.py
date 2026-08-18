import requests
import json 
import time

def stream_llm(prompt):

    response = requests.post(
        "http://ollama:11434/api/generate",
        json={
            "model": "qwen2.5-coder:3b-instruct",
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0
            }

        },
        stream=True
    ) 

    for line in response.iter_lines():


        if line:

            try:

                data = json.loads(line.decode("utf-8"))
            except:
                continue

            if "response" in data:
                yield data["response"]

            if data.get("done"):
                break
  

