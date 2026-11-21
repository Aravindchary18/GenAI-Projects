
from __future__ import annotations

from typing import Any, Sequence


def get_assistant_chat_model():
    

    from services.streaming_service import llm

    return llm


def get_assistant_tool_model(
    tool_schemas: Sequence[dict[str, Any]]
):

    if not tool_schemas:
        raise ValueError(
            "At least one MCP tool schema is required."
        )

    from services.streaming_service import llm

    primary = getattr(llm, "runnable", None)

    fallbacks = list(
        getattr(llm, "fallbacks", []) or []
    )

    # Support a single chat model as well.
    if primary is None:

        if not callable(getattr(llm, "bind_tools", None)):
            raise TypeError(
                "The configured LLM does not support tool calling."
            )

        return llm.bind_tools(list(tool_schemas))

    if not callable(getattr(primary, "bind_tools", None)):
        raise TypeError(
            "The configured primary model does not support tool calling."
        )

    bound_primary = primary.bind_tools(
        list(tool_schemas)
    )

    bound_fallbacks = []

    for fallback in fallbacks:

        bind_tools = getattr(fallback, "bind_tools", None)

        if not callable(bind_tools):
            raise TypeError(
                "A fallback model does not support tool calling."
            )

        bound_fallbacks.append(
            bind_tools(list(tool_schemas))
        )

    if bound_fallbacks:

        return bound_primary.with_fallbacks(
            bound_fallbacks
        )

    return bound_primary