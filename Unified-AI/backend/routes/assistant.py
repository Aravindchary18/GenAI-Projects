
from __future__ import annotations

import re
from typing import Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Path, status

from schemas.assistant_schemas import AssistantResumeRequest, AssistantTaskRequest
from security.assistant_auth import require_assistant_token

router = APIRouter(
    prefix="/assistant/tasks",
    tags=["assistant workflow"],
    dependencies=[Depends(require_assistant_token)],
)

_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _load_graph_functions() -> tuple[Callable[..., dict[str, Any]], ...]:
    """Lazy-load LangGraph so unrelated API routes can still import independently."""
    from agents.graph import get_assistant_state, invoke_assistant, resume_assistant

    return invoke_assistant, get_assistant_state, resume_assistant


def _safe_interrupts(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert LangGraph interrupt objects to small JSON-safe summaries."""
    output: list[dict[str, Any]] = []
    for item in result.get("__interrupt__", []) or []:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            output.append({str(key): val for key, val in value.items() if isinstance(val, (str, int, float, bool, type(None)))})
        else:
            output.append({"message": str(value)[:1000]})
    return output


def _public_task(result: dict[str, Any], thread_id: str | None = None) -> dict[str, Any]:
    """Return only the supported workflow fields, not arbitrary checkpoint internals."""
    source = dict(result or {})
    resolved_thread = str(source.get("thread_id") or thread_id or "")
    return {
        "thread_id": resolved_thread,
        "request_id": source.get("request_id"),
        "task_status": source.get("task_status", "unknown"),
        "final_response": source.get("final_response"),
        "requested_specialists": source.get("requested_specialists", []),
        "completed_specialists": source.get("completed_specialists", []),
        "specialist_results": source.get("specialist_results", []),
        "tool_history": source.get("tool_history", []),
        "pending_approval": source.get("pending_approval"),
        "approval_status": source.get("approval_status"),
        "errors": source.get("errors", []),
        "interrupts": _safe_interrupts(source),
    }


def _thread_id(value: str = Path(..., min_length=1, max_length=128)) -> str:
    if not _THREAD_ID_RE.fullmatch(value):
        raise HTTPException(status_code=422, detail="Invalid thread_id format.")
    return value


def _workflow_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ImportError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Assistant workflow dependencies are unavailable. Install backend/assistant_requirements.txt.",
        )
    # Do not leak tracebacks, credentials, provider payloads or internal paths.
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="The assistant workflow could not complete this request. Check backend logs using the request timestamp.",
    )


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def start_assistant_task(request: AssistantTaskRequest) -> dict[str, Any]:
    """Start a new task and return its thread ID and latest workflow state."""
    invoke_assistant, _, _ = _load_graph_functions()
    try:
        automation = request.automation_request.model_dump() if request.automation_request else None
        result = invoke_assistant(
            request.user_request,
            selected_document_ids=request.selected_document_ids,
            automation_request=automation,
        )
        return _public_task(result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise _workflow_error(exc) from exc


@router.get("/{thread_id}")
def read_assistant_task(thread_id: str) -> dict[str, Any]:
    """Read the latest checkpoint for a known task in this single-owner deployment."""
    resolved = _thread_id(thread_id)
    _, get_assistant_state, _ = _load_graph_functions()
    try:
        state = get_assistant_state(resolved)
        if state is None:
            raise HTTPException(status_code=404, detail="Assistant task not found.")
        return _public_task(state, resolved)
    except HTTPException:
        raise
    except Exception as exc:
        raise _workflow_error(exc) from exc


@router.post("/{thread_id}/resume")
def resume_assistant_task(
    thread_id: str,
    request: AssistantResumeRequest,
) -> dict[str, Any]:
    """Resume the same checkpoint; the graph re-checks the approval database."""
    resolved = _thread_id(thread_id)
    _, _, resume_assistant = _load_graph_functions()
    try:
        result = resume_assistant(
            resolved,
            acknowledgement=request.acknowledgement or {"acknowledged": True},
        )
        return _public_task(result, resolved)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise _workflow_error(exc) from exc

