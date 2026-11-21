
from __future__ import annotations

import ipaddress
import os
import re
import socket
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests

MAX_PDF_BYTES = 30 * 1024 * 1024
REQUEST_TIMEOUT = (8, 35)


def workspace_dir() -> Path:
    """Resolve the private assistant workspace; create it if absent."""
    path = Path(os.getenv("ASSISTANT_WORKSPACE_DIR", "workspace")).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _host_is_public(host: str, *, resolve_dns: bool = True) -> bool:
    host = host.strip("[]").rstrip(".").lower()
    if not host or host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return False

    try:
        candidate = ipaddress.ip_address(host)
        return candidate.is_global
    except ValueError:
        pass

    if not resolve_dns:
        return True

    try:
        records = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return False
    if not records:
        return False

    # Reject the entire host if any resolved address is non-public. This errs
    # on the safe side for mixed DNS answers and private-network rebinding.
    for record in records:
        address = record[4][0].split("%", 1)[0]
        try:
            if not ipaddress.ip_address(address).is_global:
                return False
        except ValueError:
            return False
    return True


def validate_public_url(url: str, *, resolve_dns: bool = True) -> str:
    """Return a normalised URL or raise ValueError for unsafe destinations."""
    if not isinstance(url, str) or len(url) > 4096:
        raise ValueError("URL must be a string no longer than 4096 characters.")
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are supported.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("URL must have a hostname and must not include credentials.")
    if not _host_is_public(parsed.hostname, resolve_dns=resolve_dns):
        raise ValueError("Local, private, reserved or unresolvable destinations are blocked.")
    return parsed.geturl()


def _safe_filename(value: str) -> str:
    name = Path(value or "download.pdf").name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
    if not name:
        name = "download.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:120]


async def _new_page():
    """Create a short-lived, isolated Chromium page and guard all requests."""
    from playwright.async_api import async_playwright

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    context_options = {
        "accept_downloads": True,
        # Context-level interception covers every page, including popups.
        # Blocking service workers ensures their requests cannot bypass routing.
        "service_workers": "block",
    }
    storage_state = os.getenv("PLAYWRIGHT_STORAGE_STATE", "").strip()
    if storage_state and Path(storage_state).is_file():
        context_options["storage_state"] = storage_state

    context = None
    try:
        context = await browser.new_context(**context_options)

        async def guard_request(route):
            try:
                validate_public_url(route.request.url)
            except ValueError:
                await route.abort("blockedbyclient")
            else:
                await route.continue_()

        # Use BrowserContext.route(), not Page.route(): the latter misses the
        # first request of a popup. Service workers are disabled above because
        # Playwright documents that service-worker requests bypass route handlers.
        await context.route("**/*", guard_request)
        page = await context.new_page()
        return playwright, browser, context, page
    except Exception:
        if context is not None:
            await context.close()
        await browser.close()
        await playwright.stop()
        raise


async def open_website(url: str) -> dict:
    """Navigate to a public website and return a compact text snapshot."""
    safe_url = validate_public_url(url)
    playwright, browser, context, page = await _new_page()
    try:
        response = await page.goto(safe_url, wait_until="domcontentloaded", timeout=30000)
        title = await page.title()
        try:
            text = await page.locator("body").inner_text(timeout=8000)
        except Exception:
            text = ""
        return {
            "url": page.url,
            "status": response.status if response else None,
            "title": title[:300],
            "text": text[:7000],
            "truncated": len(text) > 7000,
        }
    finally:
        await context.close()
        await browser.close()
        await playwright.stop()


async def play_youtube_video(url: str) -> dict:
    """Open a YouTube video and attempt muted autoplay in headless Chromium."""
    safe_url = validate_public_url(url)
    host = urlparse(safe_url).hostname or ""
    if not (host == "youtu.be" or host == "youtube.com" or host.endswith(".youtube.com")):
        raise ValueError("Only YouTube video URLs are accepted by this tool.")

    playwright, browser, context, page = await _new_page()
    try:
        await page.goto(safe_url, wait_until="domcontentloaded", timeout=35000)
        await page.wait_for_selector("video", timeout=20000)
        playback = await page.locator("video").first.evaluate(
            "async (video) => { video.muted = true; try { await video.play(); return true; } catch (_) { return false; } }"
        )
        return {
            "url": page.url,
            "title": (await page.title())[:300],
            "playback_started": bool(playback),
            "playback_persisted": False,
            "muted": True,
            "note": (
                "Playback was attempted in a temporary headless browser. "
                "The browser session closes when this tool returns, so playback "
                "does not continue and is not audible on the user's desktop. "
                "Persistent desktop playback requires a separate local headed companion."
            ),
        }
    finally:
        await context.close()
        await browser.close()
        await playwright.stop()


def download_pdf(url: str) -> dict:
    """Download a public PDF with redirect checks and a strict size limit."""
    current = validate_public_url(url)
    session = requests.Session()
    response = None
    try:
        for _ in range(6):
            current = validate_public_url(current)
            response = session.get(
                current,
                stream=True,
                allow_redirects=False,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "UnifiedAIWorkspace/1.0"},
            )
            if response.is_redirect:
                target = response.headers.get("Location")
                response.close()
                if not target:
                    raise ValueError("PDF server returned a redirect without a Location header.")
                from urllib.parse import urljoin
                current = urljoin(current, target)
                continue
            response.raise_for_status()
            break
        else:
            raise ValueError("Too many redirects while downloading the PDF.")

        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_PDF_BYTES:
            raise ValueError("PDF exceeds the 30 MiB download limit.")

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_PDF_BYTES:
                raise ValueError("PDF exceeds the 30 MiB download limit.")
            chunks.append(chunk)
        data = b"".join(chunks)
        if not data.startswith(b"%PDF-"):
            raise ValueError(f"Downloaded content is not a PDF (Content-Type: {content_type or 'unknown'}).")

        filename = _safe_filename(Path(urlparse(current).path).name or "download.pdf")
        target = workspace_dir() / "downloads"
        target.mkdir(parents=True, exist_ok=True)
        saved_path = target / f"{uuid.uuid4().hex[:12]}_{filename}"
        saved_path.write_bytes(data)
        return {
            "saved": True,
            "filename": filename,
            "path": str(saved_path),
            "size_bytes": total,
            "content_type": content_type,
            "source_url": current,
        }
    finally:
        if response is not None:
            response.close()
        session.close()


async def fill_form_fields(url: str, fields: dict[str, str]) -> dict:
    """Fill visible form controls by accessible label; deliberately do not submit."""
    safe_url = validate_public_url(url)
    if not isinstance(fields, dict) or not fields or len(fields) > 30:
        raise ValueError("Provide between 1 and 30 label/value fields.")
    if any(not str(label).strip() or len(str(value)) > 10000 for label, value in fields.items()):
        raise ValueError("Form labels must be non-empty and values cannot exceed 10,000 characters.")

    playwright, browser, context, page = await _new_page()
    try:
        await page.goto(safe_url, wait_until="domcontentloaded", timeout=30000)
        filled: list[str] = []
        missing: list[str] = []
        for label, value in fields.items():
            locator = page.get_by_label(str(label), exact=False)
            if await locator.count() == 0:
                missing.append(str(label))
                continue
            await locator.first.fill(str(value), timeout=5000)
            filled.append(str(label))
        # Never click submit, navigate to payment/confirmation, or trigger form actions.
        return {
            "url": page.url,
            "filled_labels": filled,
            "missing_labels": missing,
            "submitted": False,
            "note": "Fields were populated only; review the page before submitting manually.",
        }
    finally:
        await context.close()
        await browser.close()
        await playwright.stop()

