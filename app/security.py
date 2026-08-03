from __future__ import annotations

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings


api_key_header = APIKeyHeader(name=settings.api_key_header, auto_error=False)


def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    """Validate the production-style API key header."""
    if not api_key or not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing or invalid {settings.api_key_header} header.",
        )
    return api_key
