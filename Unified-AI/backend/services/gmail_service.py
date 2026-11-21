
from __future__ import annotations

import base64
import os
import re
import secrets
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from services.assistant_store import database_connection, database_path

GMAIL_SCOPE = "https://www.googleapis.com/auth/gmail.compose"
GMAIL_SCOPES = [GMAIL_SCOPE]
_STATE_TTL_SECONDS = 600


def _workspace() -> Path:
    root = Path(os.getenv("ASSISTANT_WORKSPACE_DIR", "workspace")).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _token_path() -> Path:
    return _workspace() / "gmail_token.json"


def _client_secret_path() -> Path:
    raw = os.getenv("GOOGLE_OAUTH_CLIENT_FILE", "").strip()
    if not raw:
        raise RuntimeError("Set GOOGLE_OAUTH_CLIENT_FILE to your Google OAuth client JSON path.")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError("Google OAuth client JSON file was not found at GOOGLE_OAUTH_CLIENT_FILE.")
    return path


def _redirect_uri() -> str:
    return os.getenv(
        "GMAIL_OAUTH_REDIRECT_URI",
        "http://localhost:8000/assistant/integrations/gmail/callback",
    ).strip()


def init_oauth_state_table() -> None:
    with database_connection() as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS gmail_oauth_states (
                state TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL
            )"""
        )


def _save_oauth_state(state: str) -> None:
    import time

    init_oauth_state_table()
    now = int(time.time())
    with database_connection(immediate=True) as connection:
        connection.execute(
            "DELETE FROM gmail_oauth_states WHERE created_at < ?",
            (now - _STATE_TTL_SECONDS,),
        )
        connection.execute(
            "INSERT INTO gmail_oauth_states(state, created_at) VALUES (?, ?)",
            (state, now),
        )


def _consume_oauth_state(state: str) -> bool:
    import time

    init_oauth_state_table()
    now = int(time.time())
    with database_connection(immediate=True) as connection:
        row = connection.execute(
            "SELECT created_at FROM gmail_oauth_states WHERE state = ?",
            (state,),
        ).fetchone()
        connection.execute("DELETE FROM gmail_oauth_states WHERE state = ?", (state,))
    return bool(row and row["created_at"] >= now - _STATE_TTL_SECONDS)


def build_authorization_url() -> dict[str, str]:
    """Create a short-lived OAuth state and return Google's consent URL."""
    from google_auth_oauthlib.flow import Flow

    state = secrets.token_urlsafe(32)
    flow = Flow.from_client_secrets_file(
        str(_client_secret_path()),
        scopes=GMAIL_SCOPES,
        state=state,
    )
    flow.redirect_uri = _redirect_uri()
    authorization_url, returned_state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    _save_oauth_state(returned_state)
    return {"authorization_url": authorization_url, "state": returned_state}


def complete_authorization(authorization_response: str, state: str) -> dict[str, str]:
    """Validate OAuth state, exchange the code and save credentials privately."""
    from google_auth_oauthlib.flow import Flow

    if not state or not _consume_oauth_state(state):
        raise ValueError("OAuth state is invalid, expired or already used.")

    flow = Flow.from_client_secrets_file(
        str(_client_secret_path()),
        scopes=GMAIL_SCOPES,
        state=state,
    )
    flow.redirect_uri = _redirect_uri()
    flow.fetch_token(authorization_response=authorization_response)
    credentials_json = flow.credentials.to_json()

    destination = _token_path()
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(credentials_json, encoding="utf-8")
    try:
        temporary.chmod(0o600)
    except OSError:
        pass
    temporary.replace(destination)
    return {"connected": True, "token_file": "stored in private assistant workspace"}


def _load_credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_file = _token_path()
    if not token_file.is_file():
        raise RuntimeError("Gmail is not connected. Start OAuth at /assistant/integrations/gmail/connect.")

    credentials = Credentials.from_authorized_user_file(str(token_file), GMAIL_SCOPES)
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        temporary = token_file.with_suffix(".tmp")
        temporary.write_text(credentials.to_json(), encoding="utf-8")
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        temporary.replace(token_file)
    if not credentials.valid:
        raise RuntimeError("Gmail credentials are invalid; reconnect the Gmail account.")
    return credentials


def gmail_connection_status() -> dict[str, Any]:
    token_file = _token_path()
    if not token_file.is_file():
        return {"connected": False}
    try:
        credentials = _load_credentials()
        return {"connected": bool(credentials.valid), "token_stored": True}
    except Exception:
        return {"connected": False, "reconnect_required": True}


def build_raw_message(to: str, subject: str, body: str, cc: str | None = None) -> str:
    """Build Gmail API URL-safe base64 RFC 2822 message data."""
    for field_name, value in (("to", to), ("subject", subject), ("body", body)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must not be empty.")
        if "\r" in value or "\n" in value and field_name in {"to", "subject"}:
            raise ValueError(f"{field_name} contains invalid header characters.")
    if len(to) > 1000 or len(subject) > 500 or len(body) > 100000:
        raise ValueError("Email content exceeds the configured size limit.")

    recipient_pattern = re.compile(r"^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$")
    recipients = [part.strip() for part in to.split(",") if part.strip()]
    if not recipients or any(not recipient_pattern.match(item) for item in recipients):
        raise ValueError("Provide valid comma-separated recipient email addresses.")

    message = EmailMessage()
    message["To"] = ", ".join(recipients)
    message["Subject"] = subject.strip()
    if cc:
        if "\r" in cc or "\n" in cc:
            raise ValueError("cc contains invalid header characters.")
        cc_recipients = [part.strip() for part in cc.split(",") if part.strip()]
        if any(not recipient_pattern.match(item) for item in cc_recipients):
            raise ValueError("Provide valid comma-separated CC addresses.")
        message["Cc"] = ", ".join(cc_recipients)
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")


def create_gmail_draft(*, to: str, subject: str, body: str, cc: str | None = None) -> dict[str, str]:
    """Create a Gmail draft; this operation does not send the message."""
    from googleapiclient.discovery import build

    raw = build_raw_message(to, subject, body, cc)
    service = build("gmail", "v1", credentials=_load_credentials(), cache_discovery=False)
    created = service.users().drafts().create(
        userId="me",
        body={"message": {"raw": raw}},
    ).execute()
    return {
        "draft_id": str(created.get("id", "")),
        "message_id": str((created.get("message") or {}).get("id", "")),
        "status": "draft_created_not_sent",
    }


def send_draft_after_approval(*, draft_id: str, approval_id: str) -> dict[str, str]:
    """Send only the draft named in an approved, matching Batch-2 approval."""
    # Check the approval before importing/calling the Gmail client. This keeps
    # the authorization gate testable without Google credentials or SDK access.
    from services.approval_service import get_approval

    approval = get_approval(approval_id, include_payload=True)
    if not approval or approval.get("status") != "approved":
        raise PermissionError("A matching approved action-approval record is required.")
    payload = approval.get("action_payload") or {}
    if approval.get("action_type") != "gmail.send" or payload.get("draft_id") != draft_id:
        raise PermissionError("Approval does not authorize sending this Gmail draft.")

    from googleapiclient.discovery import build

    service = build("gmail", "v1", credentials=_load_credentials(), cache_discovery=False)
    sent = service.users().drafts().send(
        userId="me",
        body={"id": draft_id},
    ).execute()
    return {
        "message_id": str(sent.get("id", "")),
        "thread_id": str(sent.get("threadId", "")),
        "status": "sent_after_approval",
    }
