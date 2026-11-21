
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from services.assistant_store import database_connection

ApprovalDecision = Literal["approved", "rejected"]
ALLOWED_STATUSES = frozenset({"pending", "approved", "rejected", "expired"})
_ACTION_TYPE_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,79}$")
MAX_SUMMARY_LENGTH = 500
MAX_PAYLOAD_BYTES = 64 * 1024
MIN_TTL_SECONDS = 60
MAX_TTL_SECONDS = 24 * 60 * 60


class ApprovalNotFoundError(LookupError):
    """The approval does not exist for the configured owner."""


class ApprovalExpiredError(ValueError):
    """The approval expired before a decision was recorded."""


class ApprovalAlreadyDecidedError(ValueError):
    """An approval is terminal once approved or rejected."""


def _now_dt() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _owner_id(owner_id: str | None = None) -> str:
    value = (owner_id or os.getenv("ASSISTANT_OWNER_ID", "local-user")).strip()
    if not value or len(value) > 120:
        raise ValueError("owner_id must contain 1 to 120 characters.")
    return value


def init_approval_db(db_path: str | Path | None = None) -> None:
    """Create the approvals table and indexes without changing other tables."""
    with database_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_action_approvals (
                approval_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                action_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                action_payload_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending','approved','rejected','expired')),
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                decided_at TEXT,
                decision_reason TEXT
            )
            """
        )
        connection.execute(
            """CREATE INDEX IF NOT EXISTS idx_assistant_approvals_owner_status
               ON assistant_action_approvals(owner_id, status, created_at DESC)"""
        )
        connection.execute(
            """CREATE INDEX IF NOT EXISTS idx_assistant_approvals_expiry
               ON assistant_action_approvals(status, expires_at)"""
        )


def _expire_pending(connection) -> None:
    now = _iso(_now_dt())
    connection.execute(
        """UPDATE assistant_action_approvals
           SET status = 'expired'
           WHERE status = 'pending' AND expires_at <= ?""",
        (now,),
    )


def _row_to_dict(row, *, include_payload: bool) -> dict[str, Any] | None:
    if row is None:
        return None
    result = dict(row)
    raw_payload = result.pop("action_payload_json")
    if include_payload:
        result["action_payload"] = json.loads(raw_payload)
    return result


def create_approval(
    *,
    action_type: str,
    summary: str,
    action_payload: dict[str, Any],
    expires_in_seconds: int = 900,
    owner_id: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create a pending approval record; do not execute the proposed action."""
    action_type = action_type.strip().lower()
    summary = summary.strip()
    if not _ACTION_TYPE_RE.fullmatch(action_type):
        raise ValueError("action_type must be a lowercase identifier such as gmail.send.")
    if not summary or len(summary) > MAX_SUMMARY_LENGTH:
        raise ValueError(f"summary must contain 1 to {MAX_SUMMARY_LENGTH} characters.")
    if not isinstance(action_payload, dict):
        raise ValueError("action_payload must be a JSON object.")
    if expires_in_seconds < MIN_TTL_SECONDS or expires_in_seconds > MAX_TTL_SECONDS:
        raise ValueError(
            f"expires_in_seconds must be between {MIN_TTL_SECONDS} and {MAX_TTL_SECONDS}."
        )

    try:
        payload_json = json.dumps(
            action_payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("action_payload must contain JSON-serialisable values.") from exc
    if len(payload_json.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"action_payload cannot exceed {MAX_PAYLOAD_BYTES} bytes.")

    now = _now_dt()
    approval_id = str(uuid.uuid4())
    created_at = _iso(now)
    expires_at = _iso(now + timedelta(seconds=expires_in_seconds))
    init_approval_db(db_path)

    with database_connection(db_path, immediate=True) as connection:
        connection.execute(
            """INSERT INTO assistant_action_approvals (
                   approval_id, owner_id, action_type, summary,
                   action_payload_json, status, created_at, expires_at
               ) VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (
                approval_id,
                _owner_id(owner_id),
                action_type,
                summary,
                payload_json,
                created_at,
                expires_at,
            ),
        )
        row = connection.execute(
            """SELECT * FROM assistant_action_approvals
               WHERE approval_id = ?""",
            (approval_id,),
        ).fetchone()
    result = _row_to_dict(row, include_payload=True)
    if result is None:
        raise RuntimeError("Approval was created but could not be read back.")
    return result


def get_approval(
    approval_id: str,
    *,
    include_payload: bool = False,
    owner_id: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    init_approval_db(db_path)
    with database_connection(db_path, immediate=True) as connection:
        _expire_pending(connection)
        row = connection.execute(
            """SELECT * FROM assistant_action_approvals
               WHERE approval_id = ? AND owner_id = ?""",
            (approval_id, _owner_id(owner_id)),
        ).fetchone()
    return _row_to_dict(row, include_payload=include_payload)


def list_approvals(
    *,
    status: str = "pending",
    limit: int = 50,
    owner_id: str | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"status must be one of: {', '.join(sorted(ALLOWED_STATUSES))}.")
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100.")
    init_approval_db(db_path)
    with database_connection(db_path, immediate=True) as connection:
        _expire_pending(connection)
        rows = connection.execute(
            """SELECT * FROM assistant_action_approvals
               WHERE owner_id = ? AND status = ?
               ORDER BY created_at DESC LIMIT ?""",
            (_owner_id(owner_id), status, limit),
        ).fetchall()
    return [
        _row_to_dict(row, include_payload=False)
        for row in rows
    ]


def decide_approval(
    approval_id: str,
    *,
    decision: ApprovalDecision,
    reason: str | None = None,
    owner_id: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Make a one-time decision on a pending approval, without executing it."""
    if decision not in {"approved", "rejected"}:
        raise ValueError("decision must be 'approved' or 'rejected'.")
    if reason is not None:
        reason = reason.strip()
        if len(reason) > 500:
            raise ValueError("reason cannot exceed 500 characters.")
    resolved_owner = _owner_id(owner_id)
    now = _iso(_now_dt())
    init_approval_db(db_path)

    expired_during_decision = False
    updated = None
    with database_connection(db_path, immediate=True) as connection:
        row = connection.execute(
            """SELECT status, expires_at FROM assistant_action_approvals
               WHERE approval_id = ? AND owner_id = ?""",
            (approval_id, resolved_owner),
        ).fetchone()
        if row is None:
            raise ApprovalNotFoundError("Approval not found.")
        if row["status"] != "pending":
            if row["status"] == "expired":
                raise ApprovalExpiredError("Approval has expired.")
            raise ApprovalAlreadyDecidedError(
                f"Approval is already {row['status']} and cannot be changed."
            )
        if row["expires_at"] <= now:
            connection.execute(
                """UPDATE assistant_action_approvals SET status = 'expired'
                   WHERE approval_id = ? AND owner_id = ? AND status = 'pending'""",
                (approval_id, resolved_owner),
            )
            # Do not raise inside the transaction: that would roll back the
            # terminal expired status. Raise only after the context commits.
            expired_during_decision = True
        else:
            connection.execute(
                """UPDATE assistant_action_approvals
                   SET status = ?, decided_at = ?, decision_reason = ?
                   WHERE approval_id = ? AND owner_id = ? AND status = 'pending'""",
                (decision, now, reason, approval_id, resolved_owner),
            )
            updated = connection.execute(
                """SELECT * FROM assistant_action_approvals
                   WHERE approval_id = ? AND owner_id = ?""",
                (approval_id, resolved_owner),
            ).fetchone()

    if expired_during_decision:
        raise ApprovalExpiredError("Approval has expired.")

    result = _row_to_dict(updated, include_payload=False)
    if result is None:
        raise ApprovalNotFoundError("Approval not found after decision.")
    return result

