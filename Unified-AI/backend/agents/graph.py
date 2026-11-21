from __future__ import annotations

import atexit
import sqlite3
import threading
import uuid

from pathlib import Path
from typing import Any

from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.types import Command

from agents.llm_nodes import (
    assistant_llm_node,
    execute_mcp_tool_calls_node,
    route_after_llm,
)

from agents.nodes import (
    classify_request,
    next_specialist,
    synthesize_response,
)

from agents.specialists import (
    build_automation_subgraph,
    build_document_subgraph,
    build_research_subgraph,
    prepare_llm_gmail_send_approval,
    wait_for_llm_gmail_send_approval,
)

from agents.state import AssistantState


_CHECKPOINTER_LOCK = threading.RLock()

_CHECKPOINTER = None

_CHECKPOINT_CONNECTION: sqlite3.Connection | None = None

_COMPILED_GRAPH = None


def _close_checkpoint_connection() -> None:

    global _CHECKPOINT_CONNECTION

    if _CHECKPOINT_CONNECTION is not None:

        try:
            _CHECKPOINT_CONNECTION.close()

        except Exception:
            pass

        _CHECKPOINT_CONNECTION = None


atexit.register(_close_checkpoint_connection)


def _get_checkpointer():

    global _CHECKPOINTER
    global _CHECKPOINT_CONNECTION

    with _CHECKPOINTER_LOCK:

        if _CHECKPOINTER is not None:
            return _CHECKPOINTER

        import os

        os.environ.setdefault(
            "LANGGRAPH_STRICT_MSGPACK",
            "true",
        )

        from langgraph.checkpoint.sqlite import SqliteSaver

        from services.assistant_store import database_path

        path: Path = database_path()

        _CHECKPOINT_CONNECTION = sqlite3.connect(
            str(path),
            check_same_thread=False,
            timeout=30,
        )

        _CHECKPOINT_CONNECTION.execute(
            "PRAGMA busy_timeout = 30000"
        )

        _CHECKPOINTER = SqliteSaver(
            _CHECKPOINT_CONNECTION
        )

        _CHECKPOINTER.setup()

        return _CHECKPOINTER


def _route_initial_request(
    state: AssistantState,
) -> str:
    """Use LLM by default; preserve structured legacy requests."""

    structured_action = state.get(
        "automation_request"
    )

    if (
        isinstance(structured_action, dict)
        and structured_action
    ):
        return "legacy"

    return "llm"


def _route_after_approval_preparation(
    state: AssistantState,
) -> str:

    pending = state.get("pending_approval") or {}

    return (
        "wait"
        if pending.get("status") == "pending"
        else "llm"
    )


def _route_after_approval_wait(
    state: AssistantState,
) -> str:

    pending = state.get("pending_approval") or {}

    return (
        "execute"
        if pending.get("status") == "approved"
        else "llm"
    )


def build_assistant_graph():

    global _COMPILED_GRAPH

    with _CHECKPOINTER_LOCK:

        if _COMPILED_GRAPH is not None:
            return _COMPILED_GRAPH

        builder = StateGraph(AssistantState)

        # Existing deterministic workflow.

        builder.add_node(
            "supervisor",
            classify_request,
        )

        builder.add_node(
            "document_specialist",
            build_document_subgraph(),
        )

        builder.add_node(
            "research_specialist",
            build_research_subgraph(),
        )

        builder.add_node(
            "automation_specialist",
            build_automation_subgraph(),
        )

        builder.add_node(
            "synthesize",
            synthesize_response,
        )

        # New LLM-driven workflow.

        builder.add_node(
            "llm_assistant",
            assistant_llm_node,
        )

        builder.add_node(
            "llm_execute_tools",
            execute_mcp_tool_calls_node,
        )

        builder.add_node(
            "llm_prepare_gmail_approval",
            prepare_llm_gmail_send_approval,
        )

        builder.add_node(
            "llm_wait_gmail_approval",
            wait_for_llm_gmail_send_approval,
        )

        builder.add_conditional_edges(
            START,
            _route_initial_request,
            {
                "legacy": "supervisor",
                "llm": "llm_assistant",
            },
        )

        # Preserve all existing specialist routing.

        builder.add_conditional_edges(
            "supervisor",
            next_specialist,
            {
                "document": "document_specialist",
                "research": "research_specialist",
                "automation": "automation_specialist",
                "synthesize": "synthesize",
            },
        )

        for node_name in (
            "document_specialist",
            "research_specialist",
            "automation_specialist",
        ):

            builder.add_conditional_edges(
                node_name,
                next_specialist,
                {
                    "document": "document_specialist",
                    "research": "research_specialist",
                    "automation": "automation_specialist",
                    "synthesize": "synthesize",
                },
            )

        builder.add_edge(
            "synthesize",
            END,
        )

        # LLM chooses the next MCP tool or finishes.

        builder.add_conditional_edges(
            "llm_assistant",
            route_after_llm,
            {
                "tools": "llm_execute_tools",
                "approval": "llm_prepare_gmail_approval",
                "end": END,
            },
        )

        builder.add_edge(
            "llm_execute_tools",
            "llm_assistant",
        )

        # Gmail sending always goes through approval.

        builder.add_conditional_edges(
            "llm_prepare_gmail_approval",
            _route_after_approval_preparation,
            {
                "wait": "llm_wait_gmail_approval",
                "llm": "llm_assistant",
            },
        )

        builder.add_conditional_edges(
            "llm_wait_gmail_approval",
            _route_after_approval_wait,
            {
                "execute": "llm_execute_tools",
                "llm": "llm_assistant",
            },
        )

        _COMPILED_GRAPH = builder.compile(
            checkpointer=_get_checkpointer()
        )

        return _COMPILED_GRAPH


def invoke_assistant(
    user_request: str,
    *,
    selected_document_ids: list[str] | None = None,
    automation_request: dict[str, Any] | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Start an assistant task."""

    if (
        not isinstance(user_request, str)
        or not user_request.strip()
    ):
        raise ValueError(
            "user_request must be a non-empty string."
        )

    if selected_document_ids is not None:

        if (
            not isinstance(selected_document_ids, list)
            or len(selected_document_ids) > 20
        ):
            raise ValueError(
                "selected_document_ids must be a list of at most 20 IDs."
            )

        if any(
            not isinstance(item, str) or not item.strip()
            for item in selected_document_ids
        ):
            raise ValueError(
                "Every selected document ID must be non-empty."
            )

    resolved_thread = thread_id or str(uuid.uuid4())

    initial: AssistantState = {

        "workflow_id": "UNIFIED_AI_PERSONAL_ASSISTANT",

        "request_id": str(uuid.uuid4()),

        "user_request": user_request.strip(),

        "selected_document_ids": selected_document_ids or [],

        "automation_request": automation_request or {},

        "completed_specialists": [],

        "specialist_results": [],

        "tool_history": [],

        "errors": [],

        "messages": [
            {
                "role": "user",
                "content": user_request.strip(),
            }
        ],

        "llm_iterations": 0,

        "tool_call_count": 0,

        "llm_done": False,

        "task_status": "running",
    }

    config = {
        "configurable": {
            "thread_id": resolved_thread,
        },
        "recursion_limit": 50,
    }

    result = build_assistant_graph().invoke(
        initial,
        config=config,
    )

    return {
        "thread_id": resolved_thread,
        **result,
    }


def resume_assistant(
    thread_id: str,
    *,
    acknowledgement: Any = None,
) -> dict[str, Any]:
    """Resume a paused task; acknowledgement does not grant approval."""

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError(
            "thread_id must be non-empty."
        )

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": 50,
    }

    result = build_assistant_graph().invoke(
        Command(
            resume=(
                acknowledgement
                if acknowledgement is not None
                else {"acknowledged": True}
            )
        ),
        config=config,
    )

    return {
        "thread_id": thread_id,
        **result,
    }


def get_assistant_state(
    thread_id: str,
) -> dict[str, Any] | None:
    """Read the latest saved workflow checkpoint."""

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError(
            "thread_id must be non-empty."
        )

    snapshot = build_assistant_graph().get_state(
        {
            "configurable": {
                "thread_id": thread_id,
            }
        }
    )

    return (
        dict(snapshot.values)
        if snapshot and snapshot.values
        else None
    )