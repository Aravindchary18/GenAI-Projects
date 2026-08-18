from services.bm25_service import bm25_search

from services.retrieval_service import retrieve

def hybrid_retrieve(ques,document_id,top_k=10):

    vector_chunks = retrieve(
        ques,
        document_id,
        top_k=top_k
    )

    keyword_chunks = bm25_search(
        ques,
        document_id,
        top_R=top_k
    )

    combined_chunks = (
        vector_chunks + keyword_chunks
    )

    unique_chunks=[]

    seen=set()

    for chunk in combined_chunks:

        chunk_id = getattr(chunk,"id",None)

        if chunk_id not in seen:

            seen.add(chunk_id)

            unique_chunks.append(
                chunk
            )

    return unique_chunks