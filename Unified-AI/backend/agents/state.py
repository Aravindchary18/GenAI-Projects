
from __future__ import annotations

import operator

from typing import Annotated, Any, Literal

from typing_extensions import TypedDict


WORKFLOW_ID = "UNIFIED_AI_PERSONAL_ASSISTANT"


SpecialistName = Literal[
    "document",
    "research",
    "automation",
]


TaskStatus = Literal[
    "running",
    "completed",
    "awaiting_approval",
    "blocked",
    "failed",
]


class SpecialistResult(TypedDict, total=False):

    specialist: str
    status: str
    answer: str
    tool_name: str
    tool_names: list[str]
    document_ids: list[str]
    approval_id: str
    details: dict[str, Any]
    error: str


class ToolCallRecord(TypedDict, total=False):

    tool_name: str
    status: str
    arguments_summary: dict[str, Any]
    result_summary: str
    error: str


class AssistantState(TypedDict, total=False):

    workflow_id: str
    request_id: str
    user_request: str

    selected_document_ids: list[str]

    automation_request: dict[str, Any]

    requested_specialists: list[SpecialistName]

    completed_specialists: Annotated[
        list[str],
        operator.add,
    ]

    available_mcp_tools: list[dict[str, Any]]

    specialist_results: Annotated[
        list[SpecialistResult],
        operator.add,
    ]

    tool_history: Annotated[
        list[ToolCallRecord],
        operator.add,
    ]

    task_status: TaskStatus

    pending_approval: dict[str, Any]

    approval_status: str

    approval_resume_value: Any

    automation_ready: bool

    automation_result: dict[str, Any]

    final_response: str

    errors: Annotated[
        list[str],
        operator.add,
    ]

    # New LLM conversation and tool-execution state.

    messages: Annotated[
        list[dict[str, Any]],
        operator.add,
    ]

    llm_iterations: int

    tool_call_count: int

    llm_done: bool