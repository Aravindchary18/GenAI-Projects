
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from security.assistant_auth import require_assistant_token
from services.approval_service import (
    ApprovalAlreadyDecidedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
    create_approval,
    decide_approval,
    get_approval,
    list_approvals,
)

router = APIRouter(
    prefix="/assistant/approvals",
    tags=["action approvals"],
    dependencies=[Depends(require_assistant_token)],
)


class ApprovalCreateRequest(BaseModel):
    action_type: str = Field(min_length=2, max_length=80)
    summary: str = Field(min_length=1, max_length=500)
    action_payload: dict = Field(default_factory=dict)
    expires_in_seconds: int = Field(default=900, ge=60, le=86400)


class ApprovalDecisionRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    reason: str | None = Field(default=None, max_length=500)


def _public_approval(record: dict) -> dict:
    """Do not return the raw action payload from approval list/detail routes."""
    return {
        key: value
        for key, value in record.items()
        if key != "action_payload"
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def request_approval(request: ApprovalCreateRequest):
    try:
        created = create_approval(
            action_type=request.action_type,
            summary=request.summary,
            action_payload=request.action_payload,
            expires_in_seconds=request.expires_in_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _public_approval(created)


@router.get("")
def get_approval_list(
    approval_status: Literal["pending", "approved", "rejected", "expired"] = Query(
        default="pending", alias="status"
    ),
    limit: int = Query(default=50, ge=1, le=100),
):
    try:
        approvals = list_approvals(status=approval_status, limit=limit)
        return {"count": len(approvals), "approvals": approvals}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/{approval_id}")
def get_approval_by_id(approval_id: str):
    approval = get_approval(approval_id, include_payload=False)
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found.")
    return _public_approval(approval)


@router.post("/{approval_id}/decision")
def record_approval_decision(
    approval_id: str,
    request: ApprovalDecisionRequest,
):
    try:
        result = decide_approval(
            approval_id,
            decision=request.decision,
            reason=request.reason,
        )
    except ApprovalNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ApprovalExpiredError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except ApprovalAlreadyDecidedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _public_approval(result)

