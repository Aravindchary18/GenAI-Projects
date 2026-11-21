
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.requests import Request

from security.assistant_auth import require_assistant_token
from services.approval_service import create_approval
from services.gmail_service import (
    build_authorization_url,
    complete_authorization,
    create_gmail_draft,
    gmail_connection_status,
    send_draft_after_approval,
)
from tools.browser_tools import (
    download_pdf,
    fill_form_fields,
    open_website,
    play_youtube_video,
    workspace_dir,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/assistant/integrations", tags=["assistant integrations"])
_PRIVATE = [Depends(require_assistant_token)]


class BrowserOpenRequest(BaseModel):
    url: str = Field(min_length=8, max_length=4096)


class BrowserFillRequest(BaseModel):
    url: str = Field(min_length=8, max_length=4096)
    fields: dict[str, str] = Field(min_length=1, max_length=30)


class GmailDraftRequest(BaseModel):
    to: str = Field(min_length=3, max_length=1000)
    subject: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=100000)
    cc: str | None = Field(default=None, max_length=1000)


class SendDraftRequest(BaseModel):
    approval_id: str = Field(min_length=10, max_length=80)


@router.get("/gmail/status", dependencies=_PRIVATE)
def gmail_status():
    return gmail_connection_status()


@router.get("/gmail/connect", dependencies=_PRIVATE)
def gmail_connect():
    try:
        return build_authorization_url()
    except Exception as exc:
        logger.exception("Could not start Gmail OAuth flow")
        raise HTTPException(
            status_code=503,
            detail="Gmail OAuth is not configured or could not be started. Check backend logs.",
        ) from exc


@router.get("/gmail/callback")
def gmail_callback(request: Request, state: str = Query(..., min_length=20)):
    """Google OAuth redirect. One-time state validation prevents login CSRF."""
    if request.query_params.get("error"):
        raise HTTPException(status_code=400, detail="Gmail OAuth was cancelled or denied.")
    if not request.query_params.get("code"):
        raise HTTPException(status_code=400, detail="Google did not return an authorization code.")
    try:
        result = complete_authorization(str(request.url), state)
        return {"message": "Gmail connected successfully. You may close this page.", **result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Gmail OAuth callback failed")
        raise HTTPException(status_code=502, detail="Gmail OAuth failed. Check backend logs.") from exc


@router.post("/gmail/drafts", dependencies=_PRIVATE)
def gmail_create_draft(request: GmailDraftRequest):
    try:
        return create_gmail_draft(
            to=request.to,
            subject=request.subject,
            body=request.body,
            cc=request.cc,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Gmail draft creation failed")
        raise HTTPException(status_code=502, detail="Gmail draft creation failed. Check backend logs.") from exc


@router.post("/gmail/drafts/{draft_id}/request-send-approval", dependencies=_PRIVATE)
def gmail_request_send_approval(draft_id: str):
    try:
        record = create_approval(
            action_type="gmail.send",
            summary=f"Send Gmail draft {draft_id}",
            action_payload={"draft_id": draft_id},
            expires_in_seconds=900,
        )
        return {
            "approval_id": record["approval_id"],
            "status": record["status"],
            "expires_at": record["expires_at"],
            "message": "Review and approve this action before sending the draft.",
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/gmail/drafts/{draft_id}/send", dependencies=_PRIVATE)
def gmail_send_draft(draft_id: str, request: SendDraftRequest):
    try:
        return send_draft_after_approval(
            draft_id=draft_id,
            approval_id=request.approval_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Gmail draft sending failed")
        raise HTTPException(status_code=502, detail="Gmail draft sending failed. Check backend logs.") from exc


@router.post("/browser/open", dependencies=_PRIVATE)
async def browser_open(request: BrowserOpenRequest):
    try:
        return await open_website(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Browser navigation failed")
        raise HTTPException(status_code=502, detail="Browser navigation failed. Check backend logs.") from exc


@router.post("/browser/youtube", dependencies=_PRIVATE)
async def browser_youtube(request: BrowserOpenRequest):
    try:
        return await play_youtube_video(request.url)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("YouTube playback initiation failed")
        raise HTTPException(status_code=502, detail="YouTube playback initiation failed. Check backend logs.") from exc


@router.post("/browser/download-pdf", dependencies=_PRIVATE)
def browser_download_pdf(request: BrowserOpenRequest):
    try:
        result = download_pdf(request.url)
        path = Path(result.pop("path"))
        result["download_id"] = path.name
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PDF download failed")
        raise HTTPException(status_code=502, detail="PDF download failed. Check backend logs.") from exc


@router.post("/browser/fill-form", dependencies=_PRIVATE)
async def browser_fill_form(request: BrowserFillRequest):
    try:
        return await fill_form_fields(request.url, request.fields)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Browser form fill failed")
        raise HTTPException(status_code=502, detail="Form fill failed. Check backend logs.") from exc


@router.get("/downloads", dependencies=_PRIVATE)
def list_downloads():
    folder = workspace_dir() / "downloads"
    folder.mkdir(parents=True, exist_ok=True)
    files = [
        {"filename": item.name, "size_bytes": item.stat().st_size}
        for item in sorted(folder.glob("*.pdf"), key=lambda value: value.stat().st_mtime, reverse=True)
        if item.is_file()
    ]
    return {"count": len(files), "files": files}


@router.get("/downloads/{filename}", dependencies=_PRIVATE)
def get_download(filename: str):
    # Resolve the basename inside the downloads folder; reject traversal paths.
    if Path(filename).name != filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid download filename.")
    target = (workspace_dir() / "downloads" / filename).resolve()
    base = (workspace_dir() / "downloads").resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Downloaded PDF not found.")
    return FileResponse(target, media_type="application/pdf", filename=filename)

