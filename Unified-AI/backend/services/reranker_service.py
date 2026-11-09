from sentence_transformers import CrossEncoder
from pathlib import Path
Model_Path = "/models/bge-reranker-base"

if not Path(Model_Path).exists():
    Model_Path = "BAAI/bge-reranker-base"

reranker=CrossEncoder(
      Model_Path
)

def rerank(query,results,top_k=5):

    pairs = []

    for r in results:

        text = r.payload.get("text","")

        pairs.append(
        (query,text)
        )
    
    scores = reranker.predict(pairs)

    scored_results=list(zip(results,scores))

    scored_results.sort(
        key = lambda x : x[1],
        reverse = True
    )

    return [
        result
        for result , score in scored_results[:top_k]
    ]