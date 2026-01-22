"""
API Key Authentication Middleware
=================================

Provides API key validation for protecting backend endpoints.

Usage:
    from src.backend.app.middleware.api_key_auth import require_api_key, get_api_key_optional

    # Require API key for endpoint
    @router.get("/protected")
    async def protected_endpoint(api_key: ApiKey = Depends(require_api_key)):
        return {"user": api_key.user_email}

    # Optional API key (for endpoints that work with or without auth)
    @router.get("/optional")
    async def optional_endpoint(api_key: Optional[ApiKey] = Depends(get_api_key_optional)):
        if api_key:
            return {"user": api_key.user_email}
        return {"user": "anonymous"}
"""

import os
import time
from datetime import datetime
from typing import Optional
from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.database.models import ApiKey, ApiKeyUsage

from ..utils.backend_logger import get_logger
logger = get_logger("api_key_auth")

# API key header name
API_KEY_HEADER = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)

# Database connection
_engine = None
_SessionLocal = None

# Paths that don't require authentication
PUBLIC_PATHS = {
    "/",
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
}

# Path prefixes that don't require authentication
PUBLIC_PATH_PREFIXES = [
    "/docs",
    "/redoc",
    "/api/report",  # Public report endpoints (no auth required)
]

# Scope requirements for path prefixes
# Keys with specific scopes can only access matching endpoints
# Keys with no scopes (null/empty) have full access
PATH_SCOPE_MAP = {
    "/api/media": "media:read",
    "/api/analysis": "analysis:read",
    "/api/query": "query:read",
    "/api/keys": "admin",
}


def _get_required_scope(path: str) -> Optional[str]:
    """Get the required scope for a path, if any"""
    for prefix, scope in PATH_SCOPE_MAP.items():
        if path.startswith(prefix):
            return scope
    return None


def _check_scope(api_key: 'ApiKey', required_scope: str) -> bool:
    """Check if API key has the required scope"""
    # No scopes defined = full access
    if not api_key.scopes:
        return True
    # Check if required scope is in key's scopes
    return required_scope in api_key.scopes


def _get_db_session() -> Session:
    """Get database session for API key validation"""
    global _engine, _SessionLocal

    if _SessionLocal is None:
        from src.backend.app.config.database import get_database_url
        _engine = create_engine(get_database_url(), pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine)

    return _SessionLocal()


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _is_public_path(path: str) -> bool:
    """Check if path is public (no auth required)"""
    if path in PUBLIC_PATHS:
        return True
    for prefix in PUBLIC_PATH_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def _find_api_key_by_raw_key(db: Session, raw_key: str) -> Optional[ApiKey]:
    """
    Find an API key by its raw value, supporting both bcrypt and legacy SHA-256 hashes.

    For bcrypt hashes: Uses key_prefix for efficient lookup, then verifies with bcrypt.
    For legacy SHA-256: Falls back to direct hash lookup for backward compatibility.

    Args:
        db: Database session
        raw_key: The raw API key value

    Returns:
        The matching ApiKey or None if not found
    """
    key_prefix = raw_key[:8]

    # First, try to find by prefix and verify with bcrypt
    # This handles new bcrypt-hashed keys efficiently
    candidates = db.query(ApiKey).filter(
        ApiKey.key_prefix == key_prefix
    ).all()

    for candidate in candidates:
        if candidate.verify_key(raw_key):
            return candidate

    # Fallback: try legacy SHA-256 direct lookup
    # This handles old keys that haven't been rehashed
    legacy_hash = ApiKey._hash_key_sha256(raw_key)
    legacy_key = db.query(ApiKey).filter(
        ApiKey.key_hash == legacy_hash
    ).first()

    return legacy_key


async def validate_api_key(
    request: Request,
    api_key_value: Optional[str] = Depends(api_key_header)
) -> Optional[ApiKey]:
    """
    Validate API key from header and return the ApiKey object.
    Returns None if no key provided (for optional auth).
    Raises HTTPException if key is invalid or rate limited.

    Supports both bcrypt (new) and SHA-256 (legacy) hashed keys.
    """
    if not api_key_value:
        return None

    db = _get_db_session()
    try:
        # Find the API key using prefix lookup + verification
        api_key = _find_api_key_by_raw_key(db, api_key_value)

        if not api_key:
            logger.warning(f"Invalid API key attempted from {_get_client_ip(request)}")
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"}
            )

        # Check rate limits and status
        allowed, reason = api_key.check_rate_limit()
        if not allowed:
            logger.warning(f"API key {api_key.key_prefix}... rate limited: {reason}")
            raise HTTPException(
                status_code=429,
                detail=reason
            )

        # Record usage
        api_key.record_usage()
        db.commit()

        # Log successful auth
        logger.debug(f"API key {api_key.key_prefix}... authenticated for {api_key.user_email}")

        return api_key

    finally:
        db.close()


async def require_api_key(
    request: Request,
    api_key: Optional[ApiKey] = Depends(validate_api_key)
) -> ApiKey:
    """
    Dependency that requires a valid API key.
    Use this for endpoints that must be authenticated.
    """
    # Allow public paths without auth
    if _is_public_path(request.url.path):
        return None

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Include X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"}
        )

    return api_key


async def get_api_key_optional(
    request: Request,
    api_key: Optional[ApiKey] = Depends(validate_api_key)
) -> Optional[ApiKey]:
    """
    Dependency that optionally validates API key.
    Use this for endpoints that work with or without auth.
    """
    return api_key


def validate_project_access(request: Request, projects: list) -> None:
    """
    Validate that the API key in the request has access to the specified projects.

    Should be called early in endpoint handlers that accept project filters.

    Args:
        request: FastAPI Request object (must have api_key in scope from middleware)
        projects: List of project names being accessed

    Raises:
        HTTPException 403 if access denied
    """
    # Get api_key from request scope (set by middleware)
    api_key = getattr(request.state, 'api_key', None)
    if not api_key:
        # Try from scope (ASGI middleware style)
        api_key = request.scope.get('api_key')

    if not api_key:
        # No API key means public access or already validated
        return

    if not projects:
        # No projects specified in request
        return

    # Check project access
    allowed, reason = api_key.check_project_access(projects)
    if not allowed:
        # Security event: Project access denied
        logger.warning(f"[SECURITY] Project access denied for key {api_key.key_prefix}...: {reason}")
        raise HTTPException(status_code=403, detail=reason)


def log_api_request(
    api_key: ApiKey,
    request: Request,
    status_code: int,
    response_time_ms: int,
    response_size_bytes: Optional[int] = None
):
    """
    Log an API request to the usage audit table.
    Call this after the request completes.
    """
    db = _get_db_session()
    try:
        usage = ApiKeyUsage(
            api_key_id=api_key.id,
            endpoint=str(request.url.path),
            method=request.method,
            client_ip=_get_client_ip(request),
            user_agent=request.headers.get("User-Agent", "")[:500],
            status_code=status_code,
            response_time_ms=response_time_ms,
            response_size_bytes=response_size_bytes
        )
        db.add(usage)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log API usage: {e}")
    finally:
        db.close()


class ApiKeyMiddleware:
    """
    Middleware that enforces API key authentication on all requests.

    This is more aggressive than using dependencies - it blocks ALL requests
    without a valid API key (except for public paths).

    Usage:
        app.add_middleware(ApiKeyMiddleware)
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        path = request.url.path

        # Allow public paths
        if _is_public_path(path):
            await self.app(scope, receive, send)
            return

        # Get API key from header
        api_key_value = request.headers.get(API_KEY_HEADER)

        if not api_key_value:
            response = HTTPException(
                status_code=401,
                detail="API key required. Include X-API-Key header."
            )
            await self._send_error(send, 401, "API key required. Include X-API-Key header.")
            return

        # Validate the key
        db = _get_db_session()
        try:
            # Find API key using prefix lookup + verification (supports bcrypt and legacy SHA-256)
            api_key = _find_api_key_by_raw_key(db, api_key_value)

            if not api_key:
                # Security event: Invalid API key attempt
                logger.warning(f"[SECURITY] Invalid API key attempt from {_get_client_ip(request)} for {path}")
                await self._send_error(send, 401, "Invalid API key")
                return

            allowed, reason = api_key.check_rate_limit()
            if not allowed:
                # Security event: Rate limit exceeded
                logger.warning(f"[SECURITY] Rate limit exceeded for key {api_key.key_prefix}... from {_get_client_ip(request)}: {reason}")
                await self._send_error(send, 429, reason)
                return

            # Check scope permissions
            required_scope = _get_required_scope(path)
            if required_scope and not _check_scope(api_key, required_scope):
                # Security event: Scope permission denied
                logger.warning(f"[SECURITY] Scope denied for key {api_key.key_prefix}... from {_get_client_ip(request)}: {path} requires '{required_scope}'")
                await self._send_error(send, 403, f"Access denied. This API key does not have '{required_scope}' permission.")
                return

            # Record usage and continue
            api_key.record_usage()
            db.commit()

            # Store api_key for project validation in endpoints
            # Note: Project validation happens at the endpoint level since projects
            # are specified in the request body, not the URL

            # Store api_key in scope for later use
            scope["api_key"] = api_key
            scope["api_key_id"] = api_key.id

        finally:
            db.close()

        # Process request with timing
        start_time = time.time()

        # Capture response for logging
        response_started = False
        response_status = 500

        async def send_wrapper(message):
            nonlocal response_started, response_status
            if message["type"] == "http.response.start":
                response_started = True
                response_status = message["status"]
            await send(message)

        await self.app(scope, receive, send_wrapper)

        # Log the request
        response_time_ms = int((time.time() - start_time) * 1000)
        if "api_key" in scope:
            log_api_request(
                scope["api_key"],
                request,
                response_status,
                response_time_ms
            )

    async def _send_error(self, send, status_code: int, detail: str):
        """Send an error response"""
        import json
        body = json.dumps({"detail": detail}).encode()

        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(body)).encode()],
            ],
        })
        await send({
            "type": "http.response.body",
            "body": body,
        })
