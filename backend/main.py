from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.upload import router as upload_router
from routes.chat import router as chat_router
from routes.career_mentor import router as  career_router 
from services.cache_service import create_cache_db
from routes.web_search import router as web_search_router
app = FastAPI()



# Routers
app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(career_router)
app.include_router(web_search_router)

@app.on_event("startup")
def startup():
    create_cache_db()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root
@app.get("/")
def root():
    
    return {"message": "Unified AI Workspace is running"}