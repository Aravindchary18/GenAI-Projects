
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StrictRequestModel(BaseModel):
    """Base request model that rejects unexpected fields."""

    class Config:
        extra = "forbid"


class AutomationRequest(StrictRequestModel):
    """One explicitly requested MCP action; permission is checked downstream."""

    tool_name: str = Field(min_length=1, max_length=100)
    arguments: dict[str, Any] = Field(default_factory=dict)


class AssistantTaskRequest(StrictRequestModel):
    """Start a bounded document, research or structured automation task."""

    user_request: str = Field(min_length=1, max_length=12_000)
    selected_document_ids: list[str] = Field(default_factory=list, max_items=20)
    automation_request: AutomationRequest | None = None


class AssistantResumeRequest(StrictRequestModel):
    """Resume a paused task; this payload is not itself an approval."""

    acknowledgement: dict[str, Any] | None = None

