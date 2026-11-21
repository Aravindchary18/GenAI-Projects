
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from security.assistant_auth import require_assistant_token
from services.memory_service import (
    delete_memory,
    get_memory,
    list_memories,
    save_memory,
    update_memory,
)

router = APIRouter(
    prefix="/assistant/memory",
    tags=["personal memory"],
    dependencies=[Depends(require_assistant_token)],
)

Category = Literal["profile", "preference", "instruction", "project", "other"]


class MemoryWriteRequest(BaseModel):
    category: Category = "profile"
    memory_key: str = Field(min_length=1, max_length=120)
    memory_value: str = Field(min_length=1, max_length=5000)


@router.post("", status_code=status.HTTP_201_CREATED)
def create_or_replace_memory(request: MemoryWriteRequest):
    """Save an explicit memory; same category/key updates the existing record."""
    try:
        return save_memory(
            category=request.category,
            memory_key=request.memory_key,
            memory_value=request.memory_value,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("")
def get_memory_list(
    q: str | None = Query(default=None, max_length=200),
    category: Category | None = None,
    limit: int = Query(default=50, ge=1, le=100),
):
    try:
        memories = list_memories(query=q, category=category, limit=limit)
        return {"count": len(memories), "memories": memories}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{memory_id}")
def get_memory_by_id(memory_id: str):
    memory = get_memory(memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return memory


@router.put("/{memory_id}")
def replace_memory(memory_id: str, request: MemoryWriteRequest):
    try:
        memory = update_memory(
            memory_id,
            category=request.category,
            memory_key=request.memory_key,
            memory_value=request.memory_value,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return memory


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def forget_memory(memory_id: str):
    if not delete_memory(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
