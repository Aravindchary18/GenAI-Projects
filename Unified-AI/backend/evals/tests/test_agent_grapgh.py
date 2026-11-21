from __future__ import annotations

import os
import tempfile
import unittest

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

import agents.graph as graph_module

from agents.graph import invoke_assistant
from agents.llm_nodes import (
    route_after_llm,
    execute_mcp_tool_calls_node,
)


TOOL_NAMES = {
    "research_web_fresh",
    "open_public_website",
    "list_uploaded_documents",
    "search_uploaded_documents",
    "create_gmail_draft",
    "send_gmail_draft_after_approval",
}

CATALOGUE = [
    {
        "name": name,
        "description": "Test MCP tool",
        "input_schema": {"type": "object"},
    }
    for name in TOOL_NAMES
]


class FakeChatModel:
    """Return predefined LLM responses without calling a real provider."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.seen_messages = []

    def invoke(self, messages):
        self.seen_messages.append(list(messages))

        if not self.responses:
            raise AssertionError(
                "The graph requested more LLM responses than expected."
            )

        return self.responses.pop(0)


def _reset_graph():
    """Reset graph/checkpointer singletons between isolated tests."""

    graph_module._close_checkpoint_connection()

    graph_module._CHECKPOINTER = None
    graph_module._COMPILED_GRAPH = None


@contextmanager
def mock_llm_environment(model):
    """Mock LLM and MCP dependencies; no external services are called."""

    with (
        patch(
            "agents.llm_nodes.build_llm_tool_schemas",
            return_value=([], CATALOGUE, TOOL_NAMES),
        ),
        patch(
            "agents.llm_nodes.get_assistant_tool_model",
            return_value=model,
        ),
        patch(
            "agents.llm_nodes.execute_mcp_tool_call"
        ) as execute_tool,
        patch(
            "agents.llm_nodes.tool_result_text",
            side_effect=lambda result: str(
                result.get("text", "")
            ),
        ),
    ):
        yield execute_tool


def make_tool_call(
    name,
    arguments,
    call_id="call-1",
):
    return {
        "id": call_id,
        "name": name,
        "args": arguments,
        "type": "tool_call",
    }


class AssistantGraphTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.db_path = str(
            Path(self.temp_dir.name) / "assistant_test.sqlite3"
        )

        self.environment = patch.dict(
            os.environ,
            {
                "UNIFIED_AI_DB_PATH": self.db_path,
                "ASSISTANT_OWNER_ID": "graph-test-owner",
            },
        )

        self.environment.start()
        _reset_graph()

    def tearDown(self):
        _reset_graph()
        self.environment.stop()
        self.temp_dir.cleanup()

    # ---------------------------------------------------------
    # TEST 1: LLM selects an MCP tool and returns a final answer
    # ---------------------------------------------------------

    def test_llm_selects_mcp_tool_and_returns_final_answer(self):

        model = FakeChatModel(
            AIMessage(
                content="",
                tool_calls=[
                    make_tool_call(
                        "research_web_fresh",
                        {"query": "Latest AI developments"},
                    )
                ],
            ),
            AIMessage(
                content="The research found three relevant developments."
            ),
        )

        with mock_llm_environment(model) as execute_tool:

            execute_tool.return_value = {
                "text": "Research found three relevant developments."
            }

            result = invoke_assistant(
                "Research the latest AI developments."
            )

        execute_tool.assert_called_once()

        self.assertEqual(
            execute_tool.call_args.args[0],
            "research_web_fresh",
        )

        self.assertEqual(
            result["task_status"],
            "completed",
        )

        self.assertIn(
            "three relevant developments",
            result["final_response"],
        )

    # ---------------------------------------------------------
    # TEST 2: LLM performs multiple sequential MCP tool calls
    # ---------------------------------------------------------

    def test_llm_performs_multiple_sequential_tool_calls(self):

        model = FakeChatModel(
            AIMessage(
                content="",
                tool_calls=[
                    make_tool_call(
                        "research_web_fresh",
                        {"query": "AI agent frameworks"},
                        "call-1",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    make_tool_call(
                        "open_public_website",
                        {"url": "https://example.com"},
                        "call-2",
                    )
                ],
            ),
            AIMessage(
                content="Research and website inspection are complete."
            ),
        )

        with mock_llm_environment(model) as execute_tool:

            execute_tool.side_effect = [
                {"text": "Research result with website information."},
                {"text": "Website title: Example Domain."},
            ]

            result = invoke_assistant(
                "Research AI agent frameworks and inspect the website."
            )

        self.assertEqual(
            execute_tool.call_count,
            2,
        )

        self.assertEqual(
            execute_tool.call_args_list[0].args[0],
            "research_web_fresh",
        )

        self.assertEqual(
            execute_tool.call_args_list[1].args[0],
            "open_public_website",
        )

        self.assertEqual(
            result["task_status"],
            "completed",
        )

        self.assertIn(
            "inspection are complete",
            result["final_response"],
        )

        self.assertEqual(
            len(model.responses),
            0,
        )

    # ---------------------------------------------------------
    # TEST 3: Tool failure is returned to the LLM
    # ---------------------------------------------------------

    def test_tool_failure_is_returned_to_llm(self):

        model = FakeChatModel(
            AIMessage(
                content="",
                tool_calls=[
                    make_tool_call(
                        "research_web_fresh",
                        {"query": "Latest AI information"},
                    )
                ],
            ),
            AIMessage(
                content="The research tool failed, so I cannot verify the information."
            ),
        )

        with mock_llm_environment(model) as execute_tool:

            execute_tool.side_effect = RuntimeError(
                "Simulated MCP failure"
            )

            result = invoke_assistant(
                "Research the latest AI information."
            )

        self.assertEqual(
            result["task_status"],
            "completed",
        )

        # The second LLM call must receive the tool failure.
        second_turn = model.seen_messages[1]

        failure_messages = [
            message
            for message in second_turn
            if isinstance(message, ToolMessage)
        ]

        self.assertTrue(failure_messages)

        self.assertIn(
            "Simulated MCP failure",
            str(failure_messages[-1].content),
        )

    # ---------------------------------------------------------
    # TEST 4: Gmail send is routed through approval first
    # ---------------------------------------------------------

    def test_gmail_send_routes_to_approval_first(self):

        state = {
            "llm_done": False,
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        make_tool_call(
                            "send_gmail_draft_after_approval",
                            {"draft_id": "draft-123"},
                            "send-call-1",
                        )
                    ],
                }
            ],
            "pending_approval": {},
        }

        route = route_after_llm(state)

        self.assertEqual(
            route,
            "approval",
        )

    # ---------------------------------------------------------
    # TEST 5: Gmail cannot execute without matching approval
    # ---------------------------------------------------------

    def test_gmail_send_is_blocked_without_approval(self):

        state = {
            "user_request": "Send my Gmail draft.",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        make_tool_call(
                            "send_gmail_draft_after_approval",
                            {"draft_id": "draft-123"},
                            "send-call-1",
                        )
                    ],
                }
            ],
            "available_mcp_tools": CATALOGUE,
            "selected_document_ids": [],
            "pending_approval": {},
            "tool_call_count": 0,
            "tool_history": [],
        }

        with patch(
            "agents.llm_nodes.execute_mcp_tool_call"
        ) as execute_tool:

            result = execute_mcp_tool_calls_node(state)

        execute_tool.assert_not_called()

        self.assertEqual(
            result["tool_history"][0]["status"],
            "blocked_or_failed",
        )

        self.assertIn(
            "no verified approval",
            result["messages"][0]["content"],
        )

    # ---------------------------------------------------------
    # TEST 6: Rejected Gmail approval never executes
    # ---------------------------------------------------------

    def test_rejected_gmail_send_never_executes(self):

        state = {
            "user_request": "Send my Gmail draft.",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        make_tool_call(
                            "send_gmail_draft_after_approval",
                            {"draft_id": "draft-123"},
                            "send-call-1",
                        )
                    ],
                }
            ],
            "available_mcp_tools": CATALOGUE,
            "selected_document_ids": [],
            "pending_approval": {
                "status": "rejected",
                "tool_call_id": "send-call-1",
                "draft_id": "draft-123",
                "approval_id": "approval-123",
            },
            "tool_call_count": 0,
            "tool_history": [],
        }

        with patch(
            "agents.llm_nodes.execute_mcp_tool_call"
        ) as execute_tool:

            result = execute_mcp_tool_calls_node(state)

        execute_tool.assert_not_called()

        self.assertEqual(
            result["tool_history"][0]["status"],
            "blocked_or_failed",
        )

    # ---------------------------------------------------------
    # TEST 7: Approved Gmail send uses the exact approval
    # ---------------------------------------------------------

    def test_approved_gmail_send_uses_matching_approval(self):

        state = {
            "user_request": "Send my Gmail draft.",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        make_tool_call(
                            "send_gmail_draft_after_approval",
                            {"draft_id": "draft-123"},
                            "send-call-1",
                        )
                    ],
                }
            ],
            "available_mcp_tools": CATALOGUE,
            "selected_document_ids": [],
            "pending_approval": {
                "status": "approved",
                "tool_call_id": "send-call-1",
                "draft_id": "draft-123",
                "approval_id": "approval-123",
            },
            "tool_call_count": 0,
            "tool_history": [],
        }

        with (
            patch(
                "agents.llm_nodes.execute_mcp_tool_call",
                return_value={"text": "Draft sent successfully."},
            ) as execute_tool,
            patch(
                "agents.llm_nodes.tool_result_text",
                return_value="Draft sent successfully.",
            ),
        ):

            result = execute_mcp_tool_calls_node(state)

        execute_tool.assert_called_once()

        self.assertEqual(
            execute_tool.call_args.kwargs["approval_id"],
            "approval-123",
        )

        self.assertEqual(
            result["approval_status"],
            "consumed",
        )

        self.assertEqual(
            result["pending_approval"],
            {},
        )

    # ---------------------------------------------------------
    # TEST 8: Approval for another draft cannot authorize sending
    # ---------------------------------------------------------

    def test_approval_for_different_draft_is_rejected(self):

        state = {
            "user_request": "Send my Gmail draft.",
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        make_tool_call(
                            "send_gmail_draft_after_approval",
                            {"draft_id": "draft-123"},
                            "send-call-1",
                        )
                    ],
                }
            ],
            "available_mcp_tools": CATALOGUE,
            "selected_document_ids": [],
            "pending_approval": {
                "status": "approved",
                "tool_call_id": "send-call-1",
                "draft_id": "different-draft",
                "approval_id": "approval-123",
            },
            "tool_call_count": 0,
            "tool_history": [],
        }

        with patch(
            "agents.llm_nodes.execute_mcp_tool_call"
        ) as execute_tool:

            result = execute_mcp_tool_calls_node(state)

        execute_tool.assert_not_called()

        self.assertEqual(
            result["tool_history"][0]["status"],
            "blocked_or_failed",
        )


if __name__ == "__main__":
    unittest.main()