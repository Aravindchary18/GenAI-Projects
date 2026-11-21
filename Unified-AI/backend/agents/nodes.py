
from __future__ import annotations

from typing import Any

from agents.state import AssistantState
from agents.tool_registry import TOOL_POLICIES


def _discover_mcp_tools() -> list[dict[str, Any]]:
    """Discover the real MCP catalogue through the Batch-4 client."""
    from agents.mcp_client import discover_tools_sync

    return discover_tools_sync()


def classify_request(state: AssistantState) -> dict[str, Any]:
    """Choose specialists using transparent routing rules, not an LLM guess."""
    request = str(state.get("user_request", "")).strip()
    if not request:
        return {
            "requested_specialists": [],
            "task_status": "failed",
            "errors": ["user_request must not be empty."],
            "final_response": "Please provide a non-empty request.",
        }

    lower = request.casefold()
    requested: list[str] = []
    if state.get("selected_document_ids") or any(
        term in lower for term in ("document", "pdf", "uploaded file", "uploaded files", "these files", "compare files")
    ):
        requested.append("document")
    if any(
        term in lower
        for term in (
            "research", "web", "website", "online", "latest", "current", "today",
            "news", "deadline", "price", "source", "look up", "find information",
        )
    ):
        requested.append("research")
    if isinstance(state.get("automation_request"), dict) and state.get("automation_request"):
        requested.append("automation")

    # Preserve an explicit caller-supplied specialist list only when it is valid.
    requested = list(dict.fromkeys(name for name in requested if name in {"document", "research", "automation"}))
    if not requested:
        message = (
            "This workflow currently supports uploaded-document Q&A, fresh web research, "
            "and explicitly structured browser/Gmail/memory actions. No supported specialist was selected."
        )
        return {
            "requested_specialists": [],
            "task_status": "blocked",
            "errors": [message],
            "final_response": message,
        }

    try:
        catalog = _discover_mcp_tools()
        available = {str(item.get("name")) for item in catalog if isinstance(item, dict)}
        needed: set[str] = set()
        if "document" in requested:
            needed.update({"list_uploaded_documents", "search_uploaded_documents"})
        if "research" in requested:
            needed.add("research_web_fresh")
        if "automation" in requested:
            action = state.get("automation_request") or {}
            tool_name = str(action.get("tool_name", ""))
            if tool_name not in TOOL_POLICIES:
                raise ValueError(f"Automation tool '{tool_name}' is not in the tool allowlist.")
            needed.add(tool_name)
        missing = sorted(needed - available)
        if missing:
            raise RuntimeError("Required MCP tool(s) were not discovered: " + ", ".join(missing))
        return {
            "workflow_id": "UNIFIED_AI_PERSONAL_ASSISTANT",
            "requested_specialists": requested,
            "available_mcp_tools": catalog,
            "task_status": "running",
        }
    except Exception as exc:
        message = f"Supervisor could not validate the MCP tool catalogue: {exc}"
        return {
            "requested_specialists": [],
            "task_status": "failed",
            "errors": [message],
            "final_response": message,
        }


def next_specialist(state: AssistantState) -> str:
    """Return the next requested specialist in a fixed, bounded order."""
    completed = set(state.get("completed_specialists") or [])
    for specialist in state.get("requested_specialists") or []:
        if specialist not in completed:
            return str(specialist)
    return "synthesize"


def synthesize_response(state: AssistantState) -> dict[str, Any]:
    """Build a transparent response from completed specialist records."""
    results = state.get("specialist_results") or []
    if not results:
        response = state.get("final_response") or "No specialist produced a result."
        return {"final_response": response, "task_status": state.get("task_status", "failed")}

    sections: list[str] = []
    statuses = []
    for result in results:
        specialist = str(result.get("specialist", "specialist")).title()
        status = str(result.get("status", "unknown"))
        statuses.append(status)
        content = str(result.get("answer") or result.get("error") or "No details returned.").strip()
        sections.append(f"{specialist} ({status}):\n{content}")

    if any(value == "failed" for value in statuses):
        overall = "failed" if all(value == "failed" for value in statuses) else "completed"
    elif any(value == "blocked" for value in statuses):
        overall = "blocked"
    else:
        overall = "completed"
    prefix = "Completed with some specialist failures." if overall == "completed" and "failed" in statuses else ""
    response = "\n\n".join(([prefix] if prefix else []) + sections)
    if state.get("errors"):
        response += "\n\nNotes:\n- " + "\n- ".join(str(item) for item in state["errors"])
    return {"final_response": response, "task_status": overall}
