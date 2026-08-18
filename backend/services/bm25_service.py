from rank_bm25 import BM25Okapi

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue
)


COLLECTION_NAME="documents"

qdrant = QdrantClient(
     url="http://qdrant:6333"
)

def bm25_search(que,document_id,top_R=10):

    chunks , _ = qdrant.scroll(

        collection_name=COLLECTION_NAME,

        scroll_filter=Filter(
            must=[
                FieldCondition(

                    key="document_id",

                    match=MatchValue(

                        value=document_id
                    )


                )
            ]
        ),

        limit=10000,

        with_payload=True
    )

    chunks_texts = [
        t.payload.get("text","")
        for t in chunks
    ]

    split_chunk=[
        s.split()
        for s in chunks_texts
    ]

    bm25=BM25Okapi(split_chunk)

    split_que = que.split()

    scores = bm25.get_scores(split_que)

    ranked =  sorted(
        zip(chunks,scores),
        key=lambda x:x[1],
        reverse = True
    )
    return [
        chunk
        for chunk,score in ranked[:top_R]
    ]
