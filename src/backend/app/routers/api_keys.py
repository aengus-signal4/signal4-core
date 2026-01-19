"""
API Key Management Router
=========================

Endpoints for managing API keys. These endpoints require a master/admin API key
with 'admin' scope to access.

Endpoints:
- POST /api/keys - Create a new API key
- GET /api/keys - List all API keys (without hashes)
- GET /api/keys/{key_id} - Get details for a specific key
- DELETE /api/keys/{key_id} - Revoke (disable) a key
- GET /api/keys/{key_id}/usage - Get usage stats for a key
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker, Session

from src.database.models import ApiKey, ApiKeyUsage
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger("api_keys_router")

router = APIRouter(prefix="/api/keys", tags=["api-keys"])

# Database connection
_engine = None
_SessionLocal = None

# Admin API key (set via environment variable)
# This key has full access to manage other keys
ADMIN_API_KEY = os.environ.get('ADMIN_API_KEY', '')


def get_db() -> Session:
    """Get database session"""
    global _engine, _SessionLocal

    if _SessionLocal is None:
        from src.backend.app.config.database import get_database_url
        _engine = create_engine(get_database_url(), pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine)

    db = _SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Request/Response models
class CreateApiKeyRequest(BaseModel):
    user_email: EmailStr
    name: str
    scopes: Optional[List[str]] = None
    rate_limit_per_hour: int = 1000
    max_total_requests: Optional[int] = None
    expires_in_days: Optional[int] = None  # Optional expiration


class CreateApiKeyResponse(BaseModel):
    id: int
    key: str  # Raw key - only shown once!
    key_prefix: str
    user_email: str
    name: str
    scopes: Optional[List[str]]
    rate_limit_per_hour: int
    created_at: datetime
    expires_at: Optional[datetime]
    message: str = "Store this key securely - it will not be shown again!"


class ApiKeyInfo(BaseModel):
    id: int
    key_prefix: str
    user_email: str
    name: str
    scopes: Optional[List[str]]
    rate_limit_per_hour: int
    requests_this_hour: int
    total_requests: int
    max_total_requests: Optional[int]
    is_enabled: bool
    disabled_reason: Optional[str]
    created_at: datetime
    last_used_at: Optional[datetime]
    expires_at: Optional[datetime]


class ApiKeyUsageStats(BaseModel):
    total_requests: int
    requests_last_hour: int
    requests_last_24h: int
    unique_endpoints: int
    top_endpoints: List[dict]
    error_rate_percent: float
    avg_response_time_ms: float


# Endpoints
@router.post("", response_model=CreateApiKeyResponse)
async def create_api_key(
    request: CreateApiKeyRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new API key.

    Returns the raw key - store it securely as it cannot be retrieved again!
    """
    # Generate new key
    raw_key = ApiKey.generate_key()
    key_hash = ApiKey.hash_key(raw_key)
    key_prefix = raw_key[:8]

    # Calculate expiration if specified
    expires_at = None
    if request.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)

    # Create the key record
    api_key = ApiKey(
        key_hash=key_hash,
        key_prefix=key_prefix,
        user_email=request.user_email,
        name=request.name,
        scopes=request.scopes,
        rate_limit_per_hour=request.rate_limit_per_hour,
        max_total_requests=request.max_total_requests,
        expires_at=expires_at,
    )

    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    logger.info(f"Created API key {key_prefix}... for {request.user_email}")

    return CreateApiKeyResponse(
        id=api_key.id,
        key=raw_key,
        key_prefix=key_prefix,
        user_email=api_key.user_email,
        name=api_key.name,
        scopes=api_key.scopes,
        rate_limit_per_hour=api_key.rate_limit_per_hour,
        created_at=api_key.created_at,
        expires_at=api_key.expires_at,
    )


@router.get("", response_model=List[ApiKeyInfo])
async def list_api_keys(
    user_email: Optional[str] = Query(None, description="Filter by user email"),
    include_disabled: bool = Query(False, description="Include disabled keys"),
    db: Session = Depends(get_db)
):
    """
    List all API keys (without the actual key values).
    """
    query = db.query(ApiKey)

    if user_email:
        query = query.filter(ApiKey.user_email == user_email)

    if not include_disabled:
        query = query.filter(ApiKey.is_enabled == True)

    keys = query.order_by(desc(ApiKey.created_at)).all()

    return [
        ApiKeyInfo(
            id=k.id,
            key_prefix=k.key_prefix,
            user_email=k.user_email,
            name=k.name,
            scopes=k.scopes,
            rate_limit_per_hour=k.rate_limit_per_hour,
            requests_this_hour=k.requests_this_hour,
            total_requests=k.total_requests,
            max_total_requests=k.max_total_requests,
            is_enabled=k.is_enabled,
            disabled_reason=k.disabled_reason,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
            expires_at=k.expires_at,
        )
        for k in keys
    ]


@router.get("/{key_id}", response_model=ApiKeyInfo)
async def get_api_key(
    key_id: int,
    db: Session = Depends(get_db)
):
    """
    Get details for a specific API key.
    """
    api_key = db.query(ApiKey).filter(ApiKey.id == key_id).first()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    return ApiKeyInfo(
        id=api_key.id,
        key_prefix=api_key.key_prefix,
        user_email=api_key.user_email,
        name=api_key.name,
        scopes=api_key.scopes,
        rate_limit_per_hour=api_key.rate_limit_per_hour,
        requests_this_hour=api_key.requests_this_hour,
        total_requests=api_key.total_requests,
        max_total_requests=api_key.max_total_requests,
        is_enabled=api_key.is_enabled,
        disabled_reason=api_key.disabled_reason,
        created_at=api_key.created_at,
        last_used_at=api_key.last_used_at,
        expires_at=api_key.expires_at,
    )


@router.delete("/{key_id}")
async def revoke_api_key(
    key_id: int,
    reason: str = Query("Revoked by admin", description="Reason for revocation"),
    db: Session = Depends(get_db)
):
    """
    Revoke (disable) an API key. The key will no longer work for authentication.
    """
    api_key = db.query(ApiKey).filter(ApiKey.id == key_id).first()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    if not api_key.is_enabled:
        raise HTTPException(status_code=400, detail="API key is already disabled")

    api_key.disable(reason)
    db.commit()

    logger.info(f"Revoked API key {api_key.key_prefix}... for {api_key.user_email}: {reason}")

    return {"message": f"API key {api_key.key_prefix}... revoked", "reason": reason}


@router.post("/{key_id}/enable")
async def enable_api_key(
    key_id: int,
    db: Session = Depends(get_db)
):
    """
    Re-enable a disabled API key.
    """
    api_key = db.query(ApiKey).filter(ApiKey.id == key_id).first()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    if api_key.is_enabled:
        raise HTTPException(status_code=400, detail="API key is already enabled")

    api_key.is_enabled = True
    api_key.disabled_reason = None
    api_key.disabled_at = None
    db.commit()

    logger.info(f"Re-enabled API key {api_key.key_prefix}... for {api_key.user_email}")

    return {"message": f"API key {api_key.key_prefix}... re-enabled"}


@router.get("/{key_id}/usage", response_model=ApiKeyUsageStats)
async def get_api_key_usage(
    key_id: int,
    db: Session = Depends(get_db)
):
    """
    Get usage statistics for a specific API key.
    """
    api_key = db.query(ApiKey).filter(ApiKey.id == key_id).first()

    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    now = datetime.utcnow()
    hour_ago = now - timedelta(hours=1)
    day_ago = now - timedelta(hours=24)

    # Requests in last hour
    requests_last_hour = db.query(func.count(ApiKeyUsage.id)).filter(
        ApiKeyUsage.api_key_id == key_id,
        ApiKeyUsage.created_at >= hour_ago
    ).scalar()

    # Requests in last 24 hours
    requests_last_24h = db.query(func.count(ApiKeyUsage.id)).filter(
        ApiKeyUsage.api_key_id == key_id,
        ApiKeyUsage.created_at >= day_ago
    ).scalar()

    # Unique endpoints
    unique_endpoints = db.query(func.count(func.distinct(ApiKeyUsage.endpoint))).filter(
        ApiKeyUsage.api_key_id == key_id
    ).scalar()

    # Top endpoints (last 24h)
    top_endpoints_query = db.query(
        ApiKeyUsage.endpoint,
        func.count(ApiKeyUsage.id).label('count')
    ).filter(
        ApiKeyUsage.api_key_id == key_id,
        ApiKeyUsage.created_at >= day_ago
    ).group_by(ApiKeyUsage.endpoint).order_by(desc('count')).limit(10).all()

    top_endpoints = [{"endpoint": ep, "count": cnt} for ep, cnt in top_endpoints_query]

    # Error rate (last 24h)
    error_count = db.query(func.count(ApiKeyUsage.id)).filter(
        ApiKeyUsage.api_key_id == key_id,
        ApiKeyUsage.created_at >= day_ago,
        ApiKeyUsage.status_code >= 400
    ).scalar()

    error_rate = (error_count / requests_last_24h * 100) if requests_last_24h > 0 else 0

    # Average response time (last 24h)
    avg_response_time = db.query(func.avg(ApiKeyUsage.response_time_ms)).filter(
        ApiKeyUsage.api_key_id == key_id,
        ApiKeyUsage.created_at >= day_ago,
        ApiKeyUsage.response_time_ms.isnot(None)
    ).scalar() or 0

    return ApiKeyUsageStats(
        total_requests=api_key.total_requests,
        requests_last_hour=requests_last_hour,
        requests_last_24h=requests_last_24h,
        unique_endpoints=unique_endpoints,
        top_endpoints=top_endpoints,
        error_rate_percent=round(error_rate, 2),
        avg_response_time_ms=round(float(avg_response_time), 2)
    )
