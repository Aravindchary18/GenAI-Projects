
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet


@dataclass(frozen=True)
class ToolPolicy:
    """Static policy metadata for one MCP tool."""

    name: str
    required_arguments: FrozenSet[str]
    allowed_arguments: FrozenSet[str]
    risk: str
    effect: str
    requires_approval: bool = False
    user_must_request_explicitly: bool = True
    explicit_request_terms: FrozenSet[str] = frozenset()


TOOL_POLICIES: dict[str, ToolPolicy] = {
    # Read-only document and research tools. These are used only by specialists.
    "list_uploaded_documents": ToolPolicy(
        "list_uploaded_documents", frozenset(), frozenset(), "read_only", "read"
    ),
    "search_uploaded_documents": ToolPolicy(
        "search_uploaded_documents",
        frozenset({"question", "document_ids"}),
        frozenset({"question", "document_ids"}),
        "read_only",
        "read",
    ),
    "research_web_fresh": ToolPolicy(
        "research_web_fresh", frozenset({"query"}), frozenset({"query"}), "read_only", "read"
    ),
    # Explicitly requested memory lifecycle tools from Batch 4.
    "remember_user_fact": ToolPolicy(
        "remember_user_fact",
        frozenset({"category", "memory_key", "memory_value"}),
        frozenset({"category", "memory_key", "memory_value"}),
        "user_memory_write",
        "write_memory",
        explicit_request_terms=frozenset({"remember", "save", "store"}),
    ),
    "search_user_memory": ToolPolicy(
        "search_user_memory", frozenset(), frozenset({"query", "limit"}), "read_only", "read"
    ),
    "update_user_memory": ToolPolicy(
        "update_user_memory",
        frozenset({"memory_id", "memory_key", "memory_value", "category"}),
        frozenset({"memory_id", "memory_key", "memory_value", "category"}),
        "user_memory_write",
        "write_memory",
        explicit_request_terms=frozenset({"update", "change", "edit"}),
    ),
    "forget_user_memory": ToolPolicy(
        "forget_user_memory", frozenset({"memory_id"}), frozenset({"memory_id"}),
        "destructive", "delete", explicit_request_terms=frozenset({"forget", "delete", "remove"})
    ),
    # Browser tools are bounded by Batch-3 URL validation and size checks.
    "open_public_website": ToolPolicy(
        "open_public_website", frozenset({"url"}), frozenset({"url"}), "read_only", "read",
        explicit_request_terms=frozenset({"open", "visit", "browse", "go to"})
    ),
    "play_youtube_video": ToolPolicy(
        "play_youtube_video", frozenset({"url"}), frozenset({"url"}), "read_only", "temporary_playback",
        explicit_request_terms=frozenset({"play", "watch", "youtube"})
    ),
    "download_public_pdf": ToolPolicy(
        "download_public_pdf", frozenset({"url"}), frozenset({"url"}), "workspace_write", "download",
        explicit_request_terms=frozenset({"download", "save"})
    ),
    "fill_public_form_fields": ToolPolicy(
        "fill_public_form_fields",
        frozenset({"url", "fields"}),
        frozenset({"url", "fields"}),
        "external_form_fill",
        "fill_without_submit",
        explicit_request_terms=frozenset({"fill", "populate"}),
    ),
    # Gmail draft creation does not send; actual delivery is approval-gated.
    "create_gmail_draft": ToolPolicy(
        "create_gmail_draft",
        frozenset({"to", "subject", "body"}),
        frozenset({"to", "subject", "body", "cc"}),
        "external_write",
        "create_draft_only",
        explicit_request_terms=frozenset({"draft", "compose", "email"}),
    ),
    "send_gmail_draft_after_approval": ToolPolicy(
        "send_gmail_draft_after_approval",
        frozenset({"draft_id"}),
        frozenset({"draft_id", "approval_id"}),
        "external_side_effect",
        "send_email",
        requires_approval=True,
        explicit_request_terms=frozenset({"send"}),
    ),
}


def get_tool_policy(tool_name: str) -> ToolPolicy:
    """Return a policy only for explicitly supported MCP tools."""
    policy = TOOL_POLICIES.get(tool_name)
    if policy is None:
        raise ValueError(f"Tool '{tool_name}' is not in the assistant allowlist.")
    return policy


def validate_tool_request(
    tool_name: str,
    arguments: dict[str, Any] | None,
    *,
    discovered_tool_names: set[str] | None = None,
) -> tuple[ToolPolicy, dict[str, Any]]:
    """Validate the allowlist, discovered catalogue and exact argument surface.

    `discovered_tool_names` must come from the real MCP tools/list response.
    The Gmail approval ID is injected by the workflow after the user approves;
    therefore it is deliberately not a required input at preparation time.
    """
    policy = get_tool_policy(tool_name)
    if discovered_tool_names is not None and tool_name not in discovered_tool_names:
        raise ValueError(f"Tool '{tool_name}' was not discovered on the MCP server.")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")

    supplied = set(arguments)
    missing = set(policy.required_arguments) - supplied
    unexpected = supplied - set(policy.allowed_arguments)
    if missing:
        raise ValueError(f"Missing required tool arguments: {', '.join(sorted(missing))}.")
    if unexpected:
        raise ValueError(f"Unexpected tool arguments: {', '.join(sorted(unexpected))}.")

    clean = dict(arguments)
    if "url" in clean and (not isinstance(clean["url"], str) or len(clean["url"]) > 4096):
        raise ValueError("url must be a string no longer than 4096 characters.")
    if "question" in clean and (not isinstance(clean["question"], str) or not clean["question"].strip()):
        raise ValueError("question must be a non-empty string.")
    if "query" in clean and (not isinstance(clean["query"], str) or not clean["query"].strip()):
        raise ValueError("query must be a non-empty string.")
    if "document_ids" in clean and (
        not isinstance(clean["document_ids"], list)
        or not clean["document_ids"]
        or len(clean["document_ids"]) > 20
        or any(not isinstance(item, str) or not item.strip() for item in clean["document_ids"])
    ):
        raise ValueError("document_ids must contain 1 to 20 non-empty strings.")
    if "fields" in clean and (
        not isinstance(clean["fields"], dict)
        or not clean["fields"]
        or len(clean["fields"]) > 30
        or any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) for k, v in clean["fields"].items())
    ):
        raise ValueError("fields must be a non-empty label-to-string mapping (maximum 30 fields).")
    if tool_name == "send_gmail_draft_after_approval":
        draft_id = clean.get("draft_id")
        if not isinstance(draft_id, str) or not draft_id.strip() or len(draft_id) > 256:
            raise ValueError("draft_id must be a non-empty string no longer than 256 characters.")
        if "approval_id" in clean and (
            not isinstance(clean["approval_id"], str) or not clean["approval_id"].strip()
        ):
            raise ValueError("approval_id must be a non-empty string when supplied.")
    return policy, clean

