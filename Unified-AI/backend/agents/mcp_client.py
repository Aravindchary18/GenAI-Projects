
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mcp import Client

from mcp_server import mcp_server


async def _list_all_tools(client: Client) -> list[Any]:
    """Read every tools/list page, bypassing potentially stale catalogue cache."""
    tools: list[Any] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()

    while True:
        page = await client.list_tools(cursor=cursor, cache_mode="bypass")
        tools.extend(page.tools)
        next_cursor = page.next_cursor
        if next_cursor is None:
            break
        if next_cursor in seen_cursors:
            raise RuntimeError("MCP server repeated a pagination cursor.")
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    return tools


async def discover_tools() -> list[dict[str, Any]]:
    """Discover all tools and return their names, descriptions and input schemas."""
    async with Client(mcp_server) as client:
        tools = await _list_all_tools(client)
        return [
            {
                "name": item.name,
                "description": item.description or "",
                # MCP Python SDK v2 renamed protocol model attributes to snake_case.
                "input_schema": item.input_schema,
                "output_schema": item.output_schema,
            }
            for item in tools
        ]


async def call_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Refresh the catalogue, require the requested tool to be discovered, then call it."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("MCP tool name must not be empty.")
    if arguments is not None and not isinstance(arguments, dict):
        raise ValueError("MCP tool arguments must be a JSON object.")

    async with Client(mcp_server) as client:
        tools = await _list_all_tools(client)
        discovered = {item.name for item in tools}
        if name not in discovered:
            raise ValueError(
                f"MCP tool '{name}' was not discovered on the connected server."
            )

        result = await client.call_tool(name, arguments or {})

    if result.is_error:
        text_parts = [
            item.text
            for item in (result.content or [])
            if getattr(item, "type", None) == "text"
        ]
        detail = "\n".join(text_parts) or "No error details were returned."
        raise RuntimeError(f"MCP tool '{name}' failed: {detail}")

    text_parts = [
        item.text
        for item in (result.content or [])
        if getattr(item, "type", None) == "text"
    ]
    return {
        "tool_name": name,
        "text": "\n".join(text_parts),
        "structured_content": result.structured_content,
    }


def _run_coroutine(coroutine):
    """Run async MCP helpers from synchronous LangGraph nodes safely."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    # A sync node may be called from a thread that already owns an event loop.
    # Running a separate loop in a worker thread avoids nesting asyncio.run().
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()


def discover_tools_sync() -> list[dict[str, Any]]:
    """Synchronous wrapper used by LangGraph supervisor nodes."""
    return _run_coroutine(discover_tools())


def call_tool_sync(
    name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synchronous wrapper used by LangGraph automation nodes."""
    return _run_coroutine(call_tool(name, arguments))

