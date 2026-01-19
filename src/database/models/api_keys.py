"""
API key management models for authentication and access control.

Contains:
- ApiKey: API key storage with rate limiting
- ApiKeyUsage: Usage audit log
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Tuple, Optional
import uuid
import hashlib

from .base import Base


class ApiKey(Base):
    """
    API keys for authenticating external access to the backend API.

    Each key is associated with a user (by email) and tracks usage for
    rate limiting and audit purposes. Keys can be disabled instantly.

    Security features:
    - Keys are hashed (SHA-256) before storage - raw key only shown once at creation
    - Per-key rate limits with automatic disable on threshold
    - Full audit trail via ApiKeyUsage
    - Scoped permissions (which endpoints/actions allowed)

    Usage:
        # Create key (returns raw key only once)
        raw_key = ApiKey.generate_key()
        api_key = ApiKey(
            key_hash=ApiKey.hash_key(raw_key),
            user_email="user@example.com",
            name="User's app"
        )

        # Validate key
        api_key = session.query(ApiKey).filter(
            ApiKey.key_hash == ApiKey.hash_key(provided_key),
            ApiKey.is_enabled == True
        ).first()
    """
    __tablename__ = 'api_keys'

    id = Column(Integer, primary_key=True)

    # Key identification (hash only - raw key never stored)
    key_hash = Column(String(64), unique=True, nullable=False, index=True)
    key_prefix = Column(String(8), nullable=False)  # First 8 chars for identification (e.g., "sk_a1b2...")

    # User identification
    user_email = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)  # Human-readable name for the key

    # Permissions (JSON array of allowed scopes)
    # e.g., ["media:read", "query:read", "analysis:read"]
    # Empty array or null = all permissions
    scopes = Column(JSONB, nullable=True)

    # Project restrictions (JSON array of allowed project names)
    # e.g., ["project-a", "project-b"]
    # Empty array or null = all projects allowed
    allowed_projects = Column(JSONB, nullable=True)

    # Rate limiting
    rate_limit_per_hour = Column(Integer, default=1000, nullable=False)
    requests_this_hour = Column(Integer, default=0, nullable=False)
    hour_window_start = Column(DateTime, nullable=True)

    # Lifetime limits (null = unlimited)
    max_total_requests = Column(Integer, nullable=True)
    total_requests = Column(Integer, default=0, nullable=False)

    # Status
    is_enabled = Column(Boolean, default=True, nullable=False, index=True)
    disabled_reason = Column(String(255), nullable=True)  # Why it was disabled
    disabled_at = Column(DateTime, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)  # Optional expiration

    # Relationships
    usage_logs = relationship("ApiKeyUsage", back_populates="api_key", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_api_key_email_enabled', 'user_email', 'is_enabled'),
    )

    @staticmethod
    def generate_key() -> str:
        """Generate a new API key. Returns the raw key (store securely, shown only once)."""
        return f"sk_{uuid.uuid4().hex}{uuid.uuid4().hex[:16]}"

    @staticmethod
    def hash_key(raw_key: str) -> str:
        """Hash a raw API key for storage/lookup."""
        return hashlib.sha256(raw_key.encode()).hexdigest()

    def check_rate_limit(self) -> Tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits. Returns (allowed, reason).
        Also resets the hourly window if needed.
        """
        now = datetime.utcnow()

        # Check if key is enabled
        if not self.is_enabled:
            return False, self.disabled_reason or "API key is disabled"

        # Check expiration
        if self.expires_at and now > self.expires_at:
            return False, "API key has expired"

        # Check lifetime limit
        if self.max_total_requests and self.total_requests >= self.max_total_requests:
            return False, "Lifetime request limit exceeded"

        # Reset hourly window if needed
        if self.hour_window_start is None or (now - self.hour_window_start).total_seconds() >= 3600:
            self.hour_window_start = now
            self.requests_this_hour = 0

        # Check hourly rate limit
        if self.requests_this_hour >= self.rate_limit_per_hour:
            return False, f"Hourly rate limit ({self.rate_limit_per_hour}) exceeded"

        return True, None

    def record_usage(self):
        """Record a successful request. Call after check_rate_limit() returns True."""
        self.requests_this_hour += 1
        self.total_requests += 1
        self.last_used_at = datetime.utcnow()

    def disable(self, reason: str):
        """Disable this API key."""
        self.is_enabled = False
        self.disabled_reason = reason
        self.disabled_at = datetime.utcnow()

    def check_project_access(self, projects: list) -> Tuple[bool, Optional[str]]:
        """
        Check if this API key has access to the requested projects.

        Args:
            projects: List of project names being accessed

        Returns:
            (allowed, reason) - True if allowed, False with reason if not
        """
        # No project restrictions = full access
        if not self.allowed_projects:
            return True, None

        # Check each requested project
        for project in projects:
            if project not in self.allowed_projects:
                return False, f"Access denied to project '{project}'. This API key is restricted to: {', '.join(self.allowed_projects)}"

        return True, None


class ApiKeyUsage(Base):
    """
    Audit log for API key usage. Records every request for monitoring and debugging.

    Use for:
    - Debugging access issues
    - Identifying abuse patterns
    - Usage analytics per user/key
    - Compliance/audit requirements

    Note: Consider partitioning or archiving old records for large-scale usage.
    """
    __tablename__ = 'api_key_usage'

    id = Column(Integer, primary_key=True)
    api_key_id = Column(Integer, ForeignKey('api_keys.id', ondelete='CASCADE'), nullable=False, index=True)

    # Request details
    endpoint = Column(String(255), nullable=False)  # e.g., "/api/media/content/abc123"
    method = Column(String(10), nullable=False)  # GET, POST, etc.

    # Client info
    client_ip = Column(String(45), nullable=False)  # IPv6 can be up to 45 chars
    user_agent = Column(String(500), nullable=True)

    # Response info
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Integer, nullable=True)

    # Optional: request/response size for bandwidth tracking
    request_size_bytes = Column(Integer, nullable=True)
    response_size_bytes = Column(Integer, nullable=True)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationship
    api_key = relationship("ApiKey", back_populates="usage_logs")

    __table_args__ = (
        # For querying recent usage by key
        Index('idx_api_usage_key_time', 'api_key_id', 'created_at'),
        # For querying by endpoint
        Index('idx_api_usage_endpoint', 'endpoint'),
        # For cleanup of old records
        Index('idx_api_usage_created', 'created_at'),
    )
