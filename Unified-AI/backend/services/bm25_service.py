
from __future__ import annotations

import os
import re
from collections import OrderedDict
from threading import RLock
from typing import Sequence, Union

from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

COLLECTION_NAME = "documents"
DOCUMENT_ID_FIELD = "document_id"
TEXT_FIELD = "text"
SCROLL_PAGE_SIZE = 512
MAX_CACHED_DOCUMENTS = 24
RRF_K = 60

qdrant_url = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333" if os.name == "nt" else "http://qdrant:6333",
)
qdrant = QdrantClient(url=qdrant_url, timeout=30)

# Cache entries hold (Qdrant points, BM25Okapi index). The bounded LRU avoids
# rebuilding the same corpus on every question without allowing unlimited growth.
_INDEX_CACHE: OrderedDict[str, tuple[list, BM25Okapi]] = OrderedDict()
_CACHE_LOCK = RLock()
DocumentIds = Union[str, Sequence[str]]


def normalize_document_ids(document_ids: DocumentIds) -> list[str]:

    if isinstance(document_ids, str):
        values = [document_ids]
    else:
        values = list(document_ids)
    normalized = list(dict.fromkeys(str(value).strip() for value in values if value is not None and str(value).strip()))
    if not normalized:
        raise ValueError("At least one non-empty document_id is required.")
    return normalized


def _tokenize(text: str) -> list[str]:
   
    return re.findall(r"[\w'-]+", (text or "").casefold(), flags=re.UNICODE)


def _load_document_index(document_id: str) -> tuple[list, BM25Okapi] | None:
    
    with _CACHE_LOCK:
        cached = _INDEX_CACHE.get(document_id)
        if cached is not None:
            _INDEX_CACHE.move_to_end(document_id)
            return cached

        all_points: list = []
        offset = None
        query_filter = Filter(
            must=[
                FieldCondition(
                    key=DOCUMENT_ID_FIELD,
                    match=MatchValue(value=document_id),
                )
            ]
        )

        while True:
            points, next_offset = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=query_filter,
                limit=SCROLL_PAGE_SIZE,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
            if next_offset is None:
                break
            offset = next_offset

        usable_points = [
            point for point in all_points
            if _tokenize((point.payload or {}).get(TEXT_FIELD, ""))
        ]
        if not usable_points:
            return None

        tokenized_corpus = [
            _tokenize((point.payload or {}).get(TEXT_FIELD, ""))
            for point in usable_points
        ]
        index = BM25Okapi(tokenized_corpus)
        entry = (usable_points, index)
        _INDEX_CACHE[document_id] = entry
        _INDEX_CACHE.move_to_end(document_id)

        while len(_INDEX_CACHE) > MAX_CACHED_DOCUMENTS:
            _INDEX_CACHE.popitem(last=False)

        return entry



def _search_one_document(query: str, document_id: str, top_R: int) -> list:
    entry = _load_document_index(document_id)
    if entry is None:
        return []
    points, index = entry
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []
    scores = index.get_scores(query_tokens)
    ranked = sorted(
        zip(points, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [point for point, _score in ranked[:top_R]]


def bm25_search(que: str, document_id: DocumentIds, top_R: int = 10) -> list:
   
    if not isinstance(que, str) or not que.strip():
        raise ValueError("BM25 query must be a non-empty string.")
    if not isinstance(top_R, int) or top_R < 1:
        raise ValueError("top_R must be a positive integer.")

    document_ids = normalize_document_ids(document_id)
    if len(document_ids) == 1:
        return _search_one_document(que, document_ids[0], top_R)

    fused: dict[str, dict] = {}
    for doc_id in document_ids:
        ranked_points = _search_one_document(que, doc_id, top_R)
        for rank, point in enumerate(ranked_points, start=1):
            point_id = str(point.id)
            item = fused.setdefault(
                point_id,
                {"point": point, "score": 0.0},
            )
            item["score"] += 1.0 / (RRF_K + rank)

    ordered = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    return [item["point"] for item in ordered[:top_R]]
