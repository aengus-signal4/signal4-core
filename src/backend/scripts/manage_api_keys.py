#!/usr/bin/env python3
"""
API Key Management CLI
======================

Command-line tool for managing API keys.

Usage:
    # Create a new API key
    uv run python -m src.backend.scripts.manage_api_keys create --email user@example.com --name "User's App"

    # List all keys
    uv run python -m src.backend.scripts.manage_api_keys list

    # List keys for a specific user
    uv run python -m src.backend.scripts.manage_api_keys list --email user@example.com

    # Get details for a key
    uv run python -m src.backend.scripts.manage_api_keys info 1

    # Revoke a key
    uv run python -m src.backend.scripts.manage_api_keys revoke 1 --reason "User requested"

    # Re-enable a key
    uv run python -m src.backend.scripts.manage_api_keys enable 1

    # Show usage stats
    uv run python -m src.backend.scripts.manage_api_keys usage 1
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from dotenv import load_dotenv
load_dotenv(override=True)

from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker

from src.database.models import ApiKey, ApiKeyUsage


def get_db_session():
    """Get database session"""
    password = os.getenv('POSTGRES_PASSWORD')
    if not password:
        raise ValueError("POSTGRES_PASSWORD environment variable is required")

    DATABASE_URL = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'signal4')}:{password}"
        f"@{os.getenv('POSTGRES_HOST', '10.0.0.4')}:{os.getenv('POSTGRES_PORT', '5432')}"
        f"/{os.getenv('POSTGRES_DB', 'av_content')}"
    )
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()


def create_key(args):
    """Create a new API key"""
    db = get_db_session()

    # Parse expiration
    expires_at = None
    if args.expires_days:
        expires_at = datetime.utcnow() + timedelta(days=args.expires_days)

    # Parse scopes
    scopes = None
    if args.scopes:
        scopes = [s.strip() for s in args.scopes.split(',')]

    # Generate key
    raw_key = ApiKey.generate_key()
    key_hash = ApiKey.hash_key(raw_key)
    key_prefix = raw_key[:8]

    api_key = ApiKey(
        key_hash=key_hash,
        key_prefix=key_prefix,
        user_email=args.email,
        name=args.name,
        scopes=scopes,
        rate_limit_per_hour=args.rate_limit,
        max_total_requests=args.max_requests,
        expires_at=expires_at,
    )

    db.add(api_key)
    db.commit()
    db.refresh(api_key)
    db.close()

    print("\n" + "=" * 60)
    print("API KEY CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nKey ID:     {api_key.id}")
    print(f"User:       {api_key.user_email}")
    print(f"Name:       {api_key.name}")
    print(f"Rate Limit: {api_key.rate_limit_per_hour}/hour")
    if expires_at:
        print(f"Expires:    {expires_at.strftime('%Y-%m-%d %H:%M UTC')}")
    print("\n" + "-" * 60)
    print("YOUR API KEY (save this - it will not be shown again!):")
    print("-" * 60)
    print(f"\n  {raw_key}\n")
    print("=" * 60 + "\n")


def list_keys(args):
    """List all API keys"""
    db = get_db_session()

    query = db.query(ApiKey)
    if args.email:
        query = query.filter(ApiKey.user_email == args.email)
    if not args.all:
        query = query.filter(ApiKey.is_enabled == True)

    keys = query.order_by(desc(ApiKey.created_at)).all()
    db.close()

    if not keys:
        print("No API keys found.")
        return

    table_data = []
    for k in keys:
        status = "✓ Active" if k.is_enabled else f"✗ Disabled ({k.disabled_reason})"
        last_used = k.last_used_at.strftime('%Y-%m-%d %H:%M') if k.last_used_at else "Never"
        expires = k.expires_at.strftime('%Y-%m-%d') if k.expires_at else "Never"

        table_data.append([
            k.id,
            k.key_prefix + "...",
            k.user_email,
            k.name[:20] + "..." if len(k.name) > 20 else k.name,
            f"{k.total_requests:,}",
            f"{k.rate_limit_per_hour}/h",
            last_used,
            status
        ])

    headers = ["ID", "Key", "Email", "Name", "Requests", "Limit", "Last Used", "Status"]
    print("\n" + tabulate(table_data, headers=headers, tablefmt="simple"))
    print(f"\nTotal: {len(keys)} key(s)\n")


def key_info(args):
    """Get detailed info for a key"""
    db = get_db_session()
    api_key = db.query(ApiKey).filter(ApiKey.id == args.key_id).first()
    db.close()

    if not api_key:
        print(f"API key {args.key_id} not found.")
        return

    print("\n" + "=" * 50)
    print(f"API KEY #{api_key.id}")
    print("=" * 50)
    print(f"Key Prefix:      {api_key.key_prefix}...")
    print(f"User Email:      {api_key.user_email}")
    print(f"Name:            {api_key.name}")
    print(f"Status:          {'Active' if api_key.is_enabled else 'DISABLED'}")
    if not api_key.is_enabled:
        print(f"Disabled Reason: {api_key.disabled_reason}")
        print(f"Disabled At:     {api_key.disabled_at}")
    print(f"\nScopes:          {api_key.scopes or 'All (no restrictions)'}")
    print(f"Rate Limit:      {api_key.rate_limit_per_hour}/hour")
    print(f"This Hour:       {api_key.requests_this_hour}")
    print(f"Total Requests:  {api_key.total_requests:,}")
    if api_key.max_total_requests:
        print(f"Max Requests:    {api_key.max_total_requests:,}")
    print(f"\nCreated:         {api_key.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"Last Used:       {api_key.last_used_at.strftime('%Y-%m-%d %H:%M UTC') if api_key.last_used_at else 'Never'}")
    print(f"Expires:         {api_key.expires_at.strftime('%Y-%m-%d %H:%M UTC') if api_key.expires_at else 'Never'}")
    print("=" * 50 + "\n")


def revoke_key(args):
    """Revoke a key"""
    db = get_db_session()
    api_key = db.query(ApiKey).filter(ApiKey.id == args.key_id).first()

    if not api_key:
        print(f"API key {args.key_id} not found.")
        db.close()
        return

    if not api_key.is_enabled:
        print(f"API key {args.key_id} is already disabled.")
        db.close()
        return

    api_key.disable(args.reason)
    db.commit()
    db.close()

    print(f"\n✓ API key {api_key.key_prefix}... ({api_key.user_email}) has been revoked.")
    print(f"  Reason: {args.reason}\n")


def enable_key(args):
    """Re-enable a key"""
    db = get_db_session()
    api_key = db.query(ApiKey).filter(ApiKey.id == args.key_id).first()

    if not api_key:
        print(f"API key {args.key_id} not found.")
        db.close()
        return

    if api_key.is_enabled:
        print(f"API key {args.key_id} is already enabled.")
        db.close()
        return

    api_key.is_enabled = True
    api_key.disabled_reason = None
    api_key.disabled_at = None
    db.commit()
    db.close()

    print(f"\n✓ API key {api_key.key_prefix}... ({api_key.user_email}) has been re-enabled.\n")


def show_usage(args):
    """Show usage statistics for a key"""
    db = get_db_session()
    api_key = db.query(ApiKey).filter(ApiKey.id == args.key_id).first()

    if not api_key:
        print(f"API key {args.key_id} not found.")
        db.close()
        return

    now = datetime.utcnow()
    hour_ago = now - timedelta(hours=1)
    day_ago = now - timedelta(hours=24)

    # Stats
    requests_hour = db.query(func.count(ApiKeyUsage.id)).filter(
        ApiKeyUsage.api_key_id == args.key_id,
        ApiKeyUsage.created_at >= hour_ago
    ).scalar()

    requests_day = db.query(func.count(ApiKeyUsage.id)).filter(
        ApiKeyUsage.api_key_id == args.key_id,
        ApiKeyUsage.created_at >= day_ago
    ).scalar()

    errors_day = db.query(func.count(ApiKeyUsage.id)).filter(
        ApiKeyUsage.api_key_id == args.key_id,
        ApiKeyUsage.created_at >= day_ago,
        ApiKeyUsage.status_code >= 400
    ).scalar()

    avg_time = db.query(func.avg(ApiKeyUsage.response_time_ms)).filter(
        ApiKeyUsage.api_key_id == args.key_id,
        ApiKeyUsage.created_at >= day_ago,
        ApiKeyUsage.response_time_ms.isnot(None)
    ).scalar() or 0

    # Top endpoints
    top_endpoints = db.query(
        ApiKeyUsage.endpoint,
        func.count(ApiKeyUsage.id).label('count')
    ).filter(
        ApiKeyUsage.api_key_id == args.key_id,
        ApiKeyUsage.created_at >= day_ago
    ).group_by(ApiKeyUsage.endpoint).order_by(desc('count')).limit(10).all()

    db.close()

    print("\n" + "=" * 50)
    print(f"USAGE STATS: {api_key.key_prefix}... ({api_key.user_email})")
    print("=" * 50)
    print(f"Total Requests:    {api_key.total_requests:,}")
    print(f"Last Hour:         {requests_hour:,}")
    print(f"Last 24 Hours:     {requests_day:,}")
    print(f"Errors (24h):      {errors_day:,} ({errors_day/requests_day*100:.1f}%)" if requests_day > 0 else "Errors (24h):      0")
    print(f"Avg Response Time: {avg_time:.0f}ms")
    print(f"\nTop Endpoints (24h):")
    for ep, count in top_endpoints:
        print(f"  {count:5,}  {ep}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="API Key Management CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new API key")
    create_parser.add_argument("--email", "-e", required=True, help="User email")
    create_parser.add_argument("--name", "-n", required=True, help="Key name/description")
    create_parser.add_argument("--rate-limit", "-r", type=int, default=1000, help="Rate limit per hour (default: 1000)")
    create_parser.add_argument("--max-requests", "-m", type=int, help="Maximum total requests (unlimited if not set)")
    create_parser.add_argument("--expires-days", "-x", type=int, help="Expiration in days (never if not set)")
    create_parser.add_argument("--scopes", "-s", help="Comma-separated list of scopes")

    # List command
    list_parser = subparsers.add_parser("list", help="List API keys")
    list_parser.add_argument("--email", "-e", help="Filter by user email")
    list_parser.add_argument("--all", "-a", action="store_true", help="Include disabled keys")

    # Info command
    info_parser = subparsers.add_parser("info", help="Get key details")
    info_parser.add_argument("key_id", type=int, help="Key ID")

    # Revoke command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke a key")
    revoke_parser.add_argument("key_id", type=int, help="Key ID")
    revoke_parser.add_argument("--reason", "-r", default="Revoked by admin", help="Reason for revocation")

    # Enable command
    enable_parser = subparsers.add_parser("enable", help="Re-enable a key")
    enable_parser.add_argument("key_id", type=int, help="Key ID")

    # Usage command
    usage_parser = subparsers.add_parser("usage", help="Show usage stats")
    usage_parser.add_argument("key_id", type=int, help="Key ID")

    args = parser.parse_args()

    if args.command == "create":
        create_key(args)
    elif args.command == "list":
        list_keys(args)
    elif args.command == "info":
        key_info(args)
    elif args.command == "revoke":
        revoke_key(args)
    elif args.command == "enable":
        enable_key(args)
    elif args.command == "usage":
        show_usage(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
