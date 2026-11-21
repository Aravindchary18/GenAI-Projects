from __future__ import annotations

import json

from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import interrupt

from agents.state import (
    AssistantState,
    SpecialistResult,
)

from agents.tool_registry import (
    validate_tool_request,
)


def _mcp_call(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:

    from agents.mcp_client import call_tool_sync

    return call_tool_sync(
        tool_name,
        arguments or {},
    )


def _parse_json_text(value: str) -> dict[str, Any]:

    try:
        parsed = json.loads(value)

    except (TypeError, json.JSONDecodeError):
        return {"answer": str(value)}

    return (
        parsed
        if isinstance(parsed, dict)
        else {"value": parsed}
    )


def _record_tool(
    tool_name: str,
    result: dict[str, Any],
) -> dict[str, Any]:

    return {
        "tool_name": tool_name,
        "status": "success",
        "arguments_summary": {},
        "result_summary": str(
            result.get("text", "")
        )[:1200],
    }


def _document_node(
    state: AssistantState,
) -> dict[str, Any]:

    question = str(
        state.get("user_request", "")
    ).strip()

    selected_ids = (
        state.get("selected_document_ids") or []
    )

    try:

        listed = _mcp_call(
            "list_uploaded_documents",
            {},
        )

        list_payload = _parse_json_text(
            listed.get("text", "")
        )

        available = list_payload.get(
            "documents",
            [],
        )

        available_ids = {
            str(item.get("document_id"))
            for item in available
            if isinstance(item, dict)
            and item.get("document_id")
        }

        if selected_ids:

            ids = list(
                dict.fromkeys(
                    str(value).strip()
                    for value in selected_ids
                    if str(value).strip()
                )
            )

            unknown = [
                value
                for value in ids
                if value not in available_ids
            ]

            if unknown:
                raise ValueError(
                    "Unknown document IDs: "
                    + ", ".join(unknown)
                )

        else:

            ids = [
                str(item["document_id"])
                for item in available
                if item.get("document_id")
            ]

        if not ids:
            raise ValueError(
                "No uploaded documents are available."
            )

        if len(ids) > 20:
            raise ValueError(
                "At most 20 documents can be searched."
            )

        searched = _mcp_call(
            "search_uploaded_documents",
            {
                "question": question,
                "document_ids": ids,
            },
        )

        payload = _parse_json_text(
            searched.get("text", "")
        )

        answer = str(
            payload.get("answer")
            or searched.get("text", "")
        ).strip()

        result: SpecialistResult = {
            "specialist": "document",
            "status": "success",
            "answer": answer,
            "tool_names": [
                "list_uploaded_documents",
                "search_uploaded_documents",
            ],
            "document_ids": ids,
        }

        return {
            "specialist_results": [result],
            "tool_history": [
                _record_tool(
                    "list_uploaded_documents",
                    listed,
                ),
                _record_tool(
                    "search_uploaded_documents",
                    searched,
                ),
            ],
            "completed_specialists": ["document"],
        }

    except Exception as exc:

        message = f"Document specialist failed: {exc}"

        return {
            "specialist_results": [
                {
                    "specialist": "document",
                    "status": "failed",
                    "error": message,
                }
            ],
            "errors": [message],
            "completed_specialists": ["document"],
        }


def _research_node(
    state: AssistantState,
) -> dict[str, Any]:

    query = str(
        state.get("user_request", "")
    ).strip()

    try:

        result = _mcp_call(
            "research_web_fresh",
            {"query": query},
        )

        answer = str(
            result.get("text", "")
        ).strip()

        if not answer:
            raise RuntimeError(
                "Research returned an empty answer."
            )

        return {
            "specialist_results": [
                {
                    "specialist": "research",
                    "status": "success",
                    "answer": answer,
                    "tool_names": [
                        "research_web_fresh"
                    ],
                }
            ],
            "tool_history": [
                _record_tool(
                    "research_web_fresh",
                    result,
                )
            ],
            "completed_specialists": ["research"],
        }

    except Exception as exc:

        message = f"Research specialist failed: {exc}"

        return {
            "specialist_results": [
                {
                    "specialist": "research",
                    "status": "failed",
                    "error": message,
                }
            ],
            "errors": [message],
            "completed_specialists": ["research"],
        }


def _prepare_automation(
    state: AssistantState,
) -> dict[str, Any]:

    request = state.get("automation_request") or {}

    if not isinstance(request, dict):

        message = (
            "automation_request must be a JSON object."
        )

        return {
            "automation_ready": False,
            "automation_result": {
                "status": "blocked",
                "error": message,
            },
        }

    tool_name = request.get("tool_name")

    arguments = request.get("arguments") or {}

    discovered_names = {
        str(item.get("name"))
        for item in (
            state.get("available_mcp_tools") or []
        )
        if isinstance(item, dict) and item.get("name")
    }

    try:

        policy, clean_arguments = validate_tool_request(
            str(tool_name or ""),
            arguments,
            discovered_tool_names=discovered_names,
        )

        user_request = str(
            state.get("user_request", "")
        ).strip().casefold()

        if not policy.user_must_request_explicitly:
            raise PermissionError(
                "Action must be explicitly requested."
            )

        if policy.explicit_request_terms and not any(
            cue in user_request
            for cue in policy.explicit_request_terms
        ):
            raise PermissionError(
                "The user's message does not explicitly request this action."
            )

        if policy.requires_approval:

            from services.approval_service import (
                create_approval,
                get_approval,
            )

            draft_id = clean_arguments["draft_id"]

            supplied_approval_id = clean_arguments.get(
                "approval_id"
            )

            if supplied_approval_id:

                approval = get_approval(
                    supplied_approval_id,
                    include_payload=True,
                )

                if not approval:
                    raise PermissionError(
                        "Approval record was not found."
                    )

                payload = approval.get(
                    "action_payload"
                ) or {}

                if (
                    approval.get("action_type") != "gmail.send"
                    or payload.get("draft_id") != draft_id
                ):
                    raise PermissionError(
                        "Approval does not authorize this draft."
                    )

            else:

                approval = create_approval(
                    action_type="gmail.send",
                    summary=f"Send Gmail draft {draft_id}",
                    action_payload={
                        "draft_id": draft_id
                    },
                    expires_in_seconds=900,
                )

                supplied_approval_id = str(
                    approval["approval_id"]
                )

            status = str(
                approval.get("status", "pending")
            )

            clean_arguments["approval_id"] = (
                supplied_approval_id
            )

            if status not in {
                "pending",
                "approved",
            }:
                raise PermissionError(
                    f"Gmail approval is {status}."
                )

            return {
                "automation_request": {
                    "tool_name": tool_name,
                    "arguments": clean_arguments,
                },
                "pending_approval": {
                    "approval_id": supplied_approval_id,
                    "action_type": "gmail.send",
                    "summary": f"Send Gmail draft {draft_id}",
                    "status": status,
                    "expires_at": approval.get("expires_at"),
                },
                "approval_status": status,
                "automation_ready": status == "approved",
                "task_status": (
                    "awaiting_approval"
                    if status == "pending"
                    else "running"
                ),
            }

        return {
            "automation_request": {
                "tool_name": tool_name,
                "arguments": clean_arguments,
            },
            "automation_ready": True,
            "task_status": "running",
        }

    except Exception as exc:

        message = f"Automation request blocked: {exc}"

        return {
            "automation_ready": False,
            "automation_result": {
                "status": "blocked",
                "error": message,
            },
            "specialist_results": [
                {
                    "specialist": "automation",
                    "status": "blocked",
                    "error": message,
                }
            ],
            "errors": [message],
            "completed_specialists": ["automation"],
            "task_status": "blocked",
        }


def _route_after_prepare(
    state: AssistantState,
) -> str:

    if (
        state.get("automation_result", {}).get("status")
        == "blocked"
    ):
        return "finish"

    pending = state.get("pending_approval") or {}

    if pending.get("status") == "pending":
        return "wait"

    if state.get("automation_ready"):
        return "execute"

    return "finish"


def _wait_for_approval(
    state: AssistantState,
) -> dict[str, Any]:

    pending = state.get("pending_approval") or {}

    resume_value = interrupt({
        "kind": "approval_required",
        "approval_id": pending.get("approval_id"),
        "action_type": pending.get("action_type"),
        "summary": pending.get("summary"),
        "expires_at": pending.get("expires_at"),
        "instructions": (
            "Approve the exact action through the Batch-2 approval API, "
            "then resume this workflow."
        ),
    })

    from services.approval_service import get_approval

    approval = get_approval(
        str(pending.get("approval_id", "")),
        include_payload=True,
    )

    if (
        not approval
        or approval.get("status") != "approved"
        or approval.get("action_type") != "gmail.send"
        or (
            (approval.get("action_payload") or {}).get("draft_id")
            != (
                state.get("automation_request", {}).get(
                    "arguments", {}
                ) or {}
            ).get("draft_id")
        )
    ):

        status = str(
            (approval or {}).get("status", "not_found")
        )

        message = (
            "Gmail send was not executed because approval "
            f"status is {status}."
        )

        return {
            "approval_resume_value": resume_value,
            "approval_status": status,
            "automation_ready": False,
            "automation_result": {
                "status": "blocked",
                "error": message,
            },
            "specialist_results": [
                {
                    "specialist": "automation",
                    "status": "blocked",
                    "error": message,
                }
            ],
            "errors": [message],
            "completed_specialists": ["automation"],
            "task_status": "blocked",
        }

    return {
        "approval_resume_value": resume_value,
        "approval_status": "approved",
        "automation_ready": True,
        "task_status": "running",
    }


def _execute_automation(
    state: AssistantState,
) -> dict[str, Any]:

    request = state.get("automation_request") or {}

    tool_name = str(
        request.get("tool_name", "")
    )

    arguments = request.get("arguments") or {}

    try:

        policy, clean_arguments = validate_tool_request(
            tool_name,
            arguments,
            discovered_tool_names={
                str(item.get("name"))
                for item in (
                    state.get("available_mcp_tools") or []
                )
                if isinstance(item, dict) and item.get("name")
            },
        )

        if policy.requires_approval:

            from services.approval_service import get_approval

            approval_id = str(
                clean_arguments.get("approval_id", "")
            )

            approval = get_approval(
                approval_id,
                include_payload=True,
            )

            if (
                not approval
                or approval.get("status") != "approved"
                or approval.get("action_type") != "gmail.send"
                or (
                    approval.get("action_payload") or {}
                ).get("draft_id")
                != clean_arguments.get("draft_id")
            ):
                raise PermissionError(
                    "The exact Gmail draft has no current approval."
                )

        result = _mcp_call(
            tool_name,
            clean_arguments,
        )

        parsed = _parse_json_text(
            result.get("text", "")
        )

        if (
            tool_name == "fill_public_form_fields"
            and parsed.get("submitted") is not False
        ):
            raise RuntimeError(
                "Form-fill verification failed."
            )

        if (
            tool_name == "create_gmail_draft"
            and parsed.get("status")
            != "draft_created_not_sent"
        ):
            raise RuntimeError(
                "Gmail draft verification failed."
            )

        if (
            tool_name == "send_gmail_draft_after_approval"
            and parsed.get("status")
            != "sent_after_approval"
        ):
            raise RuntimeError(
                "Gmail send verification failed."
            )

        if (
            tool_name == "download_public_pdf"
            and not parsed.get("saved")
        ):
            raise RuntimeError(
                "PDF download verification failed."
            )

        result_summary = {
            "status": "verified",
            "tool_name": tool_name,
            "result": parsed,
        }

        specialist: SpecialistResult = {
            "specialist": "automation",
            "status": "success",
            "answer": _automation_summary(
                tool_name,
                parsed,
            ),
            "tool_names": [tool_name],
            "details": parsed,
        }

        return {
            "automation_result": result_summary,
            "specialist_results": [specialist],
            "tool_history": [
                _record_tool(tool_name, result)
            ],
            "completed_specialists": ["automation"],
            "task_status": "running",
        }

    except Exception as exc:

        message = f"Automation execution failed: {exc}"

        return {
            "automation_result": {
                "status": "failed",
                "error": message,
            },
            "specialist_results": [
                {
                    "specialist": "automation",
                    "status": "failed",
                    "error": message,
                }
            ],
            "errors": [message],
            "completed_specialists": ["automation"],
            "task_status": "failed",
        }


def _automation_summary(
    tool_name: str,
    payload: dict[str, Any],
) -> str:

    if tool_name == "fill_public_form_fields":

        return (
            "Form fields were populated. The form was NOT submitted; "
            "review and submit manually if appropriate."
        )

    if tool_name == "create_gmail_draft":

        return (
            "Gmail draft created "
            f"(draft ID: {payload.get('draft_id', 'unknown')}); "
            "it was NOT sent."
        )

    if tool_name == "send_gmail_draft_after_approval":

        return (
            "The exact Gmail draft was sent after approval."
        )

    if tool_name == "download_public_pdf":

        return (
            "Public PDF saved as "
            f"{payload.get('download_id', 'downloaded file')} "
            f"(size: {payload.get('size_bytes', 'unknown')} bytes)."
        )

    if tool_name == "play_youtube_video":

        return (
            "Temporary muted headless playback was attempted. "
            "This does not play audio on the user's desktop."
        )

    return str(
        payload.get("title")
        or payload.get("url")
        or payload.get("status")
        or "Automation tool completed."
    )


def build_document_subgraph():

    builder = StateGraph(AssistantState)

    builder.add_node(
        "retrieve_documents",
        _document_node,
    )

    builder.add_edge(
        START,
        "retrieve_documents",
    )

    builder.add_edge(
        "retrieve_documents",
        END,
    )

    return builder.compile()


def build_research_subgraph():

    builder = StateGraph(AssistantState)

    builder.add_node(
        "research_web",
        _research_node,
    )

    builder.add_edge(
        START,
        "research_web",
    )

    builder.add_edge(
        "research_web",
        END,
    )

    return builder.compile()


def build_automation_subgraph():

    builder = StateGraph(AssistantState)

    builder.add_node(
        "prepare_action",
        _prepare_automation,
    )

    builder.add_node(
        "wait_for_approval",
        _wait_for_approval,
    )

    builder.add_node(
        "execute_action",
        _execute_automation,
    )

    builder.add_edge(
        START,
        "prepare_action",
    )

    builder.add_conditional_edges(
        "prepare_action",
        _route_after_prepare,
        {
            "wait": "wait_for_approval",
            "execute": "execute_action",
            "finish": END,
        },
    )

    builder.add_conditional_edges(
        "wait_for_approval",
        lambda state: (
            "execute"
            if state.get("automation_ready")
            else "finish"
        ),
        {
            "execute": "execute_action",
            "finish": END,
        },
    )

    builder.add_edge(
        "execute_action",
        END,
    )

    return builder.compile()


# ------------------------------------------------------------------
# New LLM-driven Gmail approval helpers
# ------------------------------------------------------------------


def prepare_llm_gmail_send_approval(
    state: AssistantState,
) -> dict[str, Any]:
    """Create an approval record for an LLM-requested Gmail send."""

    messages = list(
        state.get("messages") or []
    )

    assistant_message = next(
        (
            item
            for item in reversed(messages)
            if isinstance(item, dict)
            and item.get("role") == "assistant"
        ),
        {},
    )

    calls = (
        assistant_message.get("tool_calls") or []
    )

    send_calls = [
        call
        for call in calls
        if isinstance(call, dict)
        and call.get("name")
        == "send_gmail_draft_after_approval"
    ]

    if len(calls) != 1 or len(send_calls) != 1:

        return _llm_send_blocked_result(
            calls,
            (
                "The assistant must request one tool at a time. "
                "Retry the Gmail send as a separate action."
            ),
        )

    call = send_calls[0]

    call_id = str(
        call.get("id") or ""
    )

    arguments = (
        call.get("args")
        if isinstance(call.get("args"), dict)
        else {}
    )

    try:

        discovered_names = {
            str(item.get("name"))
            for item in (
                state.get("available_mcp_tools") or []
            )
            if isinstance(item, dict) and item.get("name")
        }

        if "approval_id" in arguments:
            raise PermissionError(
                "The LLM cannot provide an approval ID."
            )

        policy, clean_arguments = validate_tool_request(
            "send_gmail_draft_after_approval",
            arguments,
            discovered_tool_names=discovered_names,
        )

        request_text = str(
            state.get("user_request") or ""
        ).casefold()

        if not policy.user_must_request_explicitly:
            raise PermissionError(
                "Gmail sending must be explicitly requested."
            )

        if policy.explicit_request_terms and not any(
            term in request_text
            for term in policy.explicit_request_terms
        ):
            raise PermissionError(
                "The user did not explicitly request sending."
            )

        draft_id = clean_arguments["draft_id"]

        from services.approval_service import create_approval

        approval = create_approval(
            action_type="gmail.send",
            summary=f"Send Gmail draft {draft_id}",
            action_payload={
                "draft_id": draft_id
            },
            expires_in_seconds=900,
        )

        approval_id = str(
            approval["approval_id"]
        )

        status = str(
            approval.get("status", "pending")
        )

        if status != "pending":
            raise PermissionError(
                "A new send approval was expected to be pending."
            )

        return {
            "pending_approval": {
                "approval_id": approval_id,
                "action_type": "gmail.send",
                "summary": f"Send Gmail draft {draft_id}",
                "status": "pending",
                "expires_at": approval.get("expires_at"),
                "draft_id": draft_id,
                "tool_call_id": call_id,
            },
            "approval_status": "pending",
            "task_status": "awaiting_approval",
        }

    except Exception as exc:

        return _llm_send_blocked_result(
            calls,
            (
                "Gmail send approval could not be prepared: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


def _llm_send_blocked_result(
    calls: list[dict[str, Any]],
    message: str,
) -> dict[str, Any]:
    """Return blocked tool results without executing the send."""

    tool_messages = []

    for call in calls:

        if not isinstance(call, dict):
            continue

        tool_messages.append({
            "role": "tool",
            "tool_call_id": str(
                call.get("id") or "missing-tool-call-id"
            ),
            "name": str(
                call.get("name") or "unknown_tool"
            ),
            "content": message,
        })

    return {
        "messages": tool_messages,
        "pending_approval": {},
        "approval_status": "blocked",
        "task_status": "running",
        "errors": [message],
        "tool_history": [
            {
                "tool_name": "send_gmail_draft_after_approval",
                "status": "blocked",
                "arguments_summary": {},
                "result_summary": message[:600],
            }
        ],
    }


def wait_for_llm_gmail_send_approval(
    state: AssistantState,
) -> dict[str, Any]:
    """Pause until the existing approval DB authorizes the exact draft."""

    pending = state.get("pending_approval") or {}

    approval_id = str(
        pending.get("approval_id") or ""
    )

    draft_id = str(
        pending.get("draft_id") or ""
    )

    tool_call_id = str(
        pending.get("tool_call_id") or ""
    )

    resume_value = interrupt({
        "kind": "approval_required",
        "approval_id": approval_id,
        "action_type": "gmail.send",
        "summary": pending.get("summary"),
        "expires_at": pending.get("expires_at"),
        "instructions": (
            "Approve this exact draft through the existing Batch-2 "
            "approval API/UI and then resume the task."
        ),
    })

    from services.approval_service import get_approval

    approval = get_approval(
        approval_id,
        include_payload=True,
    )

    valid = bool(
        approval
        and approval.get("status") == "approved"
        and approval.get("action_type") == "gmail.send"
        and (
            approval.get("action_payload") or {}
        ).get("draft_id") == draft_id
    )

    if not valid:

        status = str(
            (approval or {}).get("status", "not_found")
        )

        message = (
            "Gmail was not sent because the exact draft "
            "does not have a current approved record. "
            f"Approval status: {status}."
        )

        return {
            "messages": [
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": "send_gmail_draft_after_approval",
                    "content": message,
                }
            ],
            "pending_approval": {},
            "approval_status": status,
            "approval_resume_value": resume_value,
            "task_status": "running",
            "tool_history": [
                {
                    "tool_name": "send_gmail_draft_after_approval",
                    "status": "blocked",
                    "arguments_summary": {
                        "draft_id": draft_id
                    },
                    "result_summary": message,
                }
            ],
        }

    return {
        "pending_approval": {
            **pending,
            "status": "approved",
        },
        "approval_status": "approved",
        "approval_resume_value": resume_value,
        "task_status": "running",
    }