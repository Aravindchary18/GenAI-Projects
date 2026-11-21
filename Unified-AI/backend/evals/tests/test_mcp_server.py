
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mcp import Client

from agents.mcp_client import call_tool, discover_tools
from mcp_server import mcp_server

EXPECTED_TOOLS = {
    "list_uploaded_documents",
    "search_uploaded_documents",
    "research_web_fresh",
    "remember_user_fact",
    "search_user_memory",
    "update_user_memory",
    "forget_user_memory",
    "open_public_website",
    "play_youtube_video",
    "download_public_pdf",
    "fill_public_form_fields",
    "create_gmail_draft",
    "send_gmail_draft_after_approval",
}


class MCPServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_discovery_exposes_expected_tools_and_v2_schemas(self):
        async with Client(mcp_server) as client:
            page = await client.list_tools(cache_mode="bypass")

        tools_by_name = {tool.name: tool for tool in page.tools}
        self.assertTrue(EXPECTED_TOOLS.issubset(tools_by_name))
        self.assertIn("type", tools_by_name["search_uploaded_documents"].input_schema)
        self.assertIn("properties", tools_by_name["search_uploaded_documents"].input_schema)

    async def test_client_helper_returns_current_v2_input_schema(self):
        tools = await discover_tools()
        tools_by_name = {tool["name"]: tool for tool in tools}
        self.assertTrue(EXPECTED_TOOLS.issubset(tools_by_name))
        self.assertIn("input_schema", tools_by_name["search_uploaded_documents"])
        self.assertNotIn("inputSchema", tools_by_name["search_uploaded_documents"])

    async def test_tool_call_is_made_through_mcp_client(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "assistant.sqlite3")
            with patch.dict(
                os.environ,
                {
                    "UNIFIED_AI_DB_PATH": db_path,
                    "ASSISTANT_OWNER_ID": "mcp-test-owner",
                },
            ):
                result = await call_tool(
                    "search_user_memory",
                    {"query": "", "limit": 10},
                )
                self.assertEqual(result["tool_name"], "search_user_memory")
                payload = json.loads(result["text"])
                self.assertEqual(payload["memories"], [])

    async def test_memory_save_update_search_and_forget_via_protocol(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "assistant.sqlite3")
            environment = {
                "UNIFIED_AI_DB_PATH": db_path,
                "ASSISTANT_OWNER_ID": "mcp-memory-test-owner",
            }
            with patch.dict(os.environ, environment):
                async with Client(mcp_server) as client:
                    saved_result = await client.call_tool(
                        "remember_user_fact",
                        {
                            "category": "preference",
                            "memory_key": "preferred_editor",
                            "memory_value": "VS Code",
                        },
                    )
                    self.assertFalse(saved_result.is_error)
                    saved = json.loads(saved_result.content[0].text)
                    memory_id = saved["memory"]["memory_id"]

                    updated_result = await client.call_tool(
                        "update_user_memory",
                        {
                            "memory_id": memory_id,
                            "memory_key": "preferred_editor",
                            "memory_value": "PyCharm",
                            "category": "preference",
                        },
                    )
                    self.assertFalse(updated_result.is_error)
                    updated = json.loads(updated_result.content[0].text)
                    self.assertEqual(updated["memory"]["memory_value"], "PyCharm")

                    search_result = await client.call_tool(
                        "search_user_memory",
                        {"query": "PyCharm", "limit": 10},
                    )
                    searched = json.loads(search_result.content[0].text)
                    self.assertEqual(len(searched["memories"]), 1)

                    forgotten_result = await client.call_tool(
                        "forget_user_memory",
                        {"memory_id": memory_id},
                    )
                    forgotten = json.loads(forgotten_result.content[0].text)
                    self.assertTrue(forgotten["deleted"])

    async def test_client_refuses_undiscovered_tool(self):
        from agents.mcp_client import call_tool

        with self.assertRaisesRegex(ValueError, "was not discovered"):
            await call_tool("arbitrary_unregistered_tool", {})


if __name__ == "__main__":
    unittest.main()
