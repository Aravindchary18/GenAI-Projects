
from __future__ import annotations

from typing import Any


def _extract_final_answer(response: Any) -> str:
    """Extract the final assistant message from a LangChain agent result."""
    if not isinstance(response, dict):
        raise RuntimeError("The web research agent returned an unexpected result.")

    messages = response.get("messages") or []
    for message in reversed(messages):
        message_type = getattr(message, "type", None)
        content = getattr(message, "content", None)
        if message_type == "ai" and content:
            if isinstance(content, str):
                return content.strip()
            # Preserve structured/multimodal content without losing source text.
            return str(content).strip()

    raise RuntimeError("The web research agent returned no final answer.")


def fresh_web_research(question: str) -> str:
    """Run the existing Tavily agent directly, bypassing semantic answer cache."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Research question must not be empty.")
    if len(question) > 12000:
        raise ValueError("Research question cannot exceed 12,000 characters.")

    # Lazy import keeps module import independent of Tavily and LLM credentials.
    from services.agent_service import agent

    response = agent.invoke(
        {"messages": [{"role": "user", "content": question.strip()}]},
        config={"recursion_limit": 8},
    )
    return _extract_final_answer(response)

