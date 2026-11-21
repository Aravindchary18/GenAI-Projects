
from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, status


_MIN_TOKEN_LENGTH = 32


def require_assistant_token(
    supplied_token: str | None = Header(default=None, alias="X-Assistant-Token"),
) -> None:
    """Require a configured token; missing server configuration fails closed."""
    expected_token = os.getenv("ASSISTANT_API_TOKEN", "").strip()
    if len(expected_token) < _MIN_TOKEN_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Private assistant APIs are disabled. Configure a random "
                "ASSISTANT_API_TOKEN of at least 32 characters in the backend environment."
            ),
        )

    if not supplied_token or not secrets.compare_digest(
        supplied_token, expected_token
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing assistant API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
