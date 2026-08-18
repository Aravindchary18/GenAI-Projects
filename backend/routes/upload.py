from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
)

import pdfplumber

from qdrant_client import QdrantClient

from qdrant_client.models import (
   
    PointStruct,
    VectorParams,
    Distance,
    Filter
)
from utils.chunking import get_text_splitter

from services.embedding_service import generate_embeddings

from utils.logging_utils import log_latency

import uuid

import os

router=APIRouter()

qdrant = QdrantClient(
    url="http://qdrant:6333"
)

COLLECTION_NAME="documents"

if not qdrant.collection_exists(COLLECTION_NAME):
   
    qdrant.create_collection(
        
        collection_name=COLLECTION_NAME,
        
        vectors_config=VectorParams(size=384,
                                    distance=Distance.COSINE)

                             )
os.makedirs("uploads",
            exist_ok=True)

@router.post("/upload")
@log_latency
async def upload_file(file:UploadFile=File(...)):

    try:

        if not file.filename.lower().endswith(".pdf"):        
           raise HTTPException(
            status_code=400,
            detail="only pdf's are allowed"
        )   
    
        file_path=(f"uploads/{uuid.uuid4()}_{file.filename}")
        
        with open(file_path,"wb") as f:
            
            while chunk:=await  file.read(1024*1024):
            
                f.write(chunk)
    
        full_text =  ""

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()

                if text:
                    full_text += text + "\n"

        if not full_text.strip():
            raise HTTPException(
                status_code=400,
                detail="pdf does not contain text"
            )
        
        splitter=get_text_splitter()
        
        chunks=splitter.split_text(
            full_text
        )

        embeddings=generate_embeddings(
            chunks
        )

        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter()
        )
    
        document_id=str(
            uuid.uuid4()
        )

        points=[]

        for idx,(chunk,embedding) in enumerate(zip(chunks,embeddings)):
            
            points.append(
                
                PointStruct(
                    id=str(uuid.uuid4()),

                    vector=embedding.tolist(),

                    payload={
                        "text":chunk,
                        "source":file.filename,
                        'document_id':document_id,
                        "chunk_index":idx

                    }
                )
            )
        qdrant.upsert(

            collection_name=COLLECTION_NAME,

            points=points

        )
        return {

            "message":"file uploaded successfully",

            "filename":file.filename,

            "document_id":document_id,

            "chunks length":len(chunks)
        }
    

    
    except HTTPException:
        raise
    
    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )

    finally:

        if os.path.exists(file_path):
            os.remove(file_path)
    