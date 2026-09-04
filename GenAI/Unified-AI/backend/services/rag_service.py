from services.hybrid_retrieval_service import hybrid_retrieve

from utils.prompts import SYSTEM_PROMPT

from services.reranker_service import rerank

from services.streaming_service import stream_llm

# ==========================================
# BUILD CONTEXT FROM RETRIEVED CHUNKS
# ==========================================

def build_context(results):

    texts = []

    for r in results:

        if isinstance(r,tuple):

            r = r[0]

        payload = getattr(r, "payload", { })

        text = payload.get("text","")

        source = payload.get("source","unknown")

        chunk_index = payload.get("chunk_index","unknown")

        texts.append(f"FILE: {source}\n {text}\n chunk index: {chunk_index}")

    return "\n\n".join(texts)
    
  
# ==========================================
# MAIN RAG PIPELINE
# ==========================================

def rag_pipeline(question,document_id):
    
    # --------------------------------------
    # RETRIEVE TOP MATCHING CHUNKS
    # --------------------------------------

    retrieved_chunks= hybrid_retrieve(
        question,
        document_id,
        top_k=10
    )

    #---------------------------------------
    # RERANK CHUNKS USING CROSS-ENCODER FOR BETTER RELEVANCE
    #---------------------------------------
    
    reranked_chunks = rerank(
        question,
        retrieved_chunks
    )

    # --------------------------------------
    # CONVERT CHUNKS INTO CONTEXT
    # --------------------------------------

    context=build_context(reranked_chunks)

    # --------------------------------------
    # CREATE FINAL PROMPT
    # --------------------------------------

    prompt = f"""{SYSTEM_PROMPT}


Context:
{context}


Question:
{question}


Answer:
"""


    return stream_llm(prompt)
