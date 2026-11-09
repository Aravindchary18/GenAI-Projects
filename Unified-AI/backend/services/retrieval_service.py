from fastapi import HTTPException

from qdrant_client import QdrantClient

from services.embedding_service import (
      generate_embeddings
)

from qdrant_client.models import (
   Filter,
   FieldCondition,
   MatchValue   
)

import os

# ==========================================
# CONNECT TO QDRANT VECTOR DATABASE
# ==========================================

qdrant_url = "http://localhost:6333" if os.name=="nt" else "http://qdrant:6333"

qdrant = QdrantClient(
    url=qdrant_url 
)

# ==========================================
# COLLECTION NAME
# ==========================================

COLLECTION_NAME = "documents"

# RETRIEVAL FUNCTION

def retrieve(query,document_id,top_k=5):
    
    try:
        
      # CONVERT QUESTION INTO VECTOR
      query_embedding = generate_embeddings([query])[0]
      # SEARCH SIMILAR VECTORS

      results = qdrant.query_points(
            collection_name=COLLECTION_NAME,

            query=query_embedding.tolist(),

            query_filter = Filter(
               must=[
                  FieldCondition(
                     key="document_id",

                     match=MatchValue(
                        value=document_id
                     )
                  )
               ]
            ),

            limit=top_k

      )
      # RETURN BEST MATCHES
      
      return results.points
    
    except Exception as e:
       
       raise HTTPException(
          
          status_code=500,
          detail=f"Retrieval failed: {str(e)}"
       )