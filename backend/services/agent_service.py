from langchain.agents import create_agent 
from langchain_ollama import ChatOllama 
from tools.web_search import web_search_tool
from utils.prompts import WEB_SEARCH_SYSTEM_PROMPT
from services.cache_service import ( get_cached_answer , add_to_cache )

llm = ChatOllama(
    model = "qwen2.5:3b-instruct",
    base_url="http://ollama:11434",
    temperature=0,
    validate_model_on_init=True,
    reasoning=False,
    timeout=180
)


agent = create_agent(
    model=llm,
    tools=[web_search_tool],
    system_prompt = WEB_SEARCH_SYSTEM_PROMPT
)

def ask_agent(question:  str)->str:
    cached = get_cached_answer(question)
    if cached:
        return cached
   
    try:
        response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": question,
                            }
                        ]
                    },
                    config={
                        "recursion_limit": 6
                    }
        
                )
        messages = response.get("messages",[])

        if not messages:
        
            raise ValueError("No response generated.")
        
                
        answer = None

        for msg in reversed(messages):
            if msg.type =="ai" and msg.content:
                answer = msg.content
                break
        
        if not answer:
            raise ValueError("No AI response generated.")
        
                
        add_to_cache(question, answer)
        
        return answer
        
    except Exception:
        
        raise 