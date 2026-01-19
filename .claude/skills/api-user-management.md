# API User Management Skill

This skill provides guidance for managing external API users who need access to the Signal4 backend API.

## Overview

External users can be granted scoped API access with rate limiting and expiration. Currently, external users should only receive `media:read` access, which limits them to the media endpoint.

## Access Levels

| Level | Rate Limit | Expiration | Use Case |
|-------|------------|------------|----------|
| basic | 100/hour | 30 days | Trial access, testing |
| moderate | 500/hour | 90 days | Regular collaborators |
| full | 1000/hour | Never | Trusted partners |

## Available Scopes

| Scope | Endpoints | Description |
|-------|-----------|-------------|
| `media:read` | `/api/media/*` | Audio/video content access |
| `analysis:read` | `/api/analysis/*` | RAG/search operations |
| `query:read` | `/api/query/*` | Segment queries |
| `admin` | `/api/keys/*` | API key management |

**For external users, only grant `media:read` scope.**

## CLI Commands

```bash
# Create a new API key
uv run python -m src.backend.scripts.manage_api_keys create \
  --email "user@example.com" \
  --name "User Name - Media Access" \
  --rate-limit 500 \
  --expires-days 90 \
  --scopes "media:read"

# List all keys
uv run python -m src.backend.scripts.manage_api_keys list

# View key details
uv run python -m src.backend.scripts.manage_api_keys info <key_id>

# View usage stats
uv run python -m src.backend.scripts.manage_api_keys usage <key_id>

# Revoke a key
uv run python -m src.backend.scripts.manage_api_keys revoke <key_id> --reason "Reason here"

# Re-enable a key
uv run python -m src.backend.scripts.manage_api_keys enable <key_id>
```

## Welcome Note Template

After creating an API key, send the user this welcome note (customize the placeholders):

---

Hi {NAME},

Here's your Signal4 Media API access:

**API Key:** `{API_KEY}`

**Endpoint:**
```
https://api.signal4.ca/api/media/content/{content_id}
```

**Example Request:**
```bash
curl -H "X-API-Key: {API_KEY}" \
  "https://api.signal4.ca/api/media/content/{content_id}?start_time=0&end_time=60&media_type=audio"
```

**Parameters:**
| Parameter | Required | Description |
|-----------|----------|-------------|
| `content_id` | Yes | The content identifier |
| `start_time` | No | Start time in seconds |
| `end_time` | No | End time in seconds |
| `media_type` | No | `audio`, `video`, or `auto` (default) |
| `format` | No | `webm` or `mp4` (default: webm) |

**Your Limits:**
- {RATE_LIMIT} requests per hour
- Expires: {EXPIRY_DATE}

**Important:** Store this key securely - it cannot be retrieved again.

Questions? Contact aengus@signal4.ca

---

## Monitoring Users

Check usage patterns regularly:

```bash
# View all active keys
uv run python -m src.backend.scripts.manage_api_keys list

# Check specific user's usage
uv run python -m src.backend.scripts.manage_api_keys usage <key_id>
```

Watch for:
- High error rates (may indicate misuse or bugs)
- Approaching rate limits
- Keys near expiration

## Revoking Access

If a user's access needs to be revoked:

```bash
uv run python -m src.backend.scripts.manage_api_keys revoke <key_id> --reason "Access terminated - reason"
```

The key is immediately disabled. The user will receive 401 errors on subsequent requests.
