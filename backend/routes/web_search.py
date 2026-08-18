from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.agent_service import ask_agent

router = APIRouter(
    prefix="/websearch",
    tags=["web search"]
)

class WebSearchRequest(BaseModel):
    question : str

@router.post("/search")
def web_search(request: WebSearchRequest):
    try:
        answer = ask_agent(request.question)

        return {
            "message": "Web search completed successfully.",
            "result": answer
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Web search failed: {str(e)}"
        )

