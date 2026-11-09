from langchain_ollama import ChatOllama 
import os
from langchain_openai import ChatOpenAI

ollama_url = (
    "http://ollama:11434"
    if os.name != "nt"
    else "http://localhost:11434"
)

bharath_llm = ChatOpenAI(
    model="qwen-3.8-27b",
    api_key=os.environ["bharat_akey"],
    base_url="https://bharatcode.ai/api/model/v1",
    temperature=0,
    timeout=180,
    max_retries=0
)

local_llm = ChatOllama(
       model="qwen2.5-coder:3b-instruct",
       base_url=ollama_url,
       temperature=0,
       validate_model_on_init=True,
       timeout=180
) 

llm = bharath_llm.with_fallbacks([local_llm])

def ans_call(prompt):

    for chunk in llm.stream(prompt):
        if chunk.content:
            yield chunk.content
