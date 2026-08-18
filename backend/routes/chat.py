from fastapi import APIRouter, HTTPException

from pydantic import BaseModel

from fastapi.responses import StreamingResponse

from services.rag_service import rag_pipeline

# ==========================================
# ROUTER
# ==========================================

router = APIRouter()


# ==========================================
# REQUEST MODEL
# ==========================================

class ChatRequest(BaseModel):

    question : str

    document_id: str


# ==========================================
# CHAT ENDPOINT
# ==========================================

@router.post("/chat")

def chat(request:ChatRequest):

    try:

        # EXTRACT QUESTION FROM OBJECT
    
        question = request.question

        document_id = request.document_id

        return StreamingResponse(
            rag_pipeline(
                question,
                document_id
            ),
            media_type="text/plain"
        )


        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"chat failed: {str(e)}"
        )
        