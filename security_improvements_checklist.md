# Security Improvements Checklist

Based on the security audit conducted on 2026-01-19.

## Completed

- [x] **Remove hardcoded database credentials** - Created centralized `src/backend/app/config/database.py`, updated all backend services/scripts to use env vars
- [x] **Generate secure database password** - 32-char cryptographically random password
- [x] **Update PostgreSQL password** - Changed on database server (10.0.0.4)
- [x] **Deploy credentials to workers** - Pushed `.env` to worker0, worker3, worker4, worker5

## Immediate Priority (Week 1)

### Input Validation
- [x] **Validate `content_id` parameter** in `src/backend/app/routers/media.py`
  - Added regex validation: `^[a-zA-Z0-9_.-]+$`
  - Rejects path traversal attempts (`../`, `..\\`, `/`)
  - Added `validate_content_id()` function called in `get_media()` endpoint

### Security Headers
- [x] **Add security headers middleware** in `src/backend/app/main.py`
  - Added `X-Content-Type-Options: nosniff`
  - Added `X-Frame-Options: DENY`
  - Added `Strict-Transport-Security: max-age=31536000; includeSubDomains`
  - Added `X-XSS-Protection: 1; mode=block`
  - Added `Referrer-Policy: strict-origin-when-cross-origin`

### CORS Configuration
- [x] **Tighten CORS settings** in `src/backend/app/main.py`
  - Changed `allow_methods=["*"]` to `allow_methods=["GET", "POST", "OPTIONS"]`
  - Changed `allow_headers=["*"]` to `allow_headers=["Content-Type", "X-API-Key", "Authorization"]`
  - Localhost origins only included when `DEBUG=true`

### API Documentation
- [x] **Protect Swagger UI** in production
  - `/docs` and `/redoc` disabled when `DEBUG=false` (production default)

## Short Term (Week 2)

### Error Handling
- [x] **Sanitize error messages** - Don't expose internal details to clients
  - `src/backend/app/routers/query.py` - Returns "Internal server error" instead of exception details
  - `src/backend/app/routers/analysis.py` - Returns "Analysis failed. Please try again." for streaming, "Internal server error" for batch
  - `src/backend/app/main.py` - Global exception handler no longer exposes `str(exc)`

### Database Connection Pooling
- [x] **Configure connection pooling** properly in `src/backend/app/database/connection.py`
  - `pool_size=20` - Maintain 20 connections
  - `max_overflow=40` - Allow up to 40 additional under load
  - `pool_pre_ping=True` - Detect stale connections
  - `pool_recycle=3600` - Recycle after 1 hour
  - `pool_timeout=30` - Wait up to 30s for connection

### Rate Limiting
- [x] **Rate limiting already in place** via API key model
  - Per-key rate limits stored in database (`rate_limit_per_hour`, `requests_this_hour`)
  - Automatic hourly window reset
  - Lifetime request limits supported
  - Note: For Redis-based distributed rate limiting, Redis infrastructure would need to be added

### Authorization
- [x] **Implement project-level RBAC**
  - Added `allowed_projects` field to `ApiKey` model in `src/database/models.py`
  - Added `check_project_access()` method to validate project permissions
  - Added `validate_project_access()` helper in `src/backend/app/middleware/api_key_auth.py`
  - Validation added to `/api/query` and `/api/analysis` endpoints
  - Keys with `null` or empty `allowed_projects` have full access

## Medium Term (Month 1)

### Secrets Management
- [ ] **Rotate compromised keys** (exposed in git history)
  - RESEND_API_KEY
  - BACKEND_API_KEY
  - YouTube API keys
  - Consider using `git filter-branch` or BFG to remove from history

- [ ] **Implement secrets management service**
  - Options: HashiCorp Vault, AWS Secrets Manager, or similar
  - Automate credential rotation

### Frontend Security
- [x] **Add security headers** in `frontend/next.config.ts`
  - X-Content-Type-Options: nosniff
  - X-Frame-Options: DENY
  - X-XSS-Protection: 1; mode=block
  - Referrer-Policy: strict-origin-when-cross-origin
  - Permissions-Policy: camera=(), microphone=(), geolocation=()
- [x] **Remove `.env` files from git** - Already excluded via `.gitignore`, not in history
- [x] **Add authentication to frontend proxy routes**
  - `/api/media/content/[contentId]` - requires session
  - `/api/media/stream/[contentId]` - requires session
  - `/api/analysis/stream` - requires session

### Logging & Monitoring
- [x] **Implement security event logging** with `[SECURITY]` prefix in logs
  - Invalid API key attempts
  - Rate limit violations
  - Scope permission denials
  - Project access denials

- [ ] **Add security unit tests**
  - Input validation bypass attempts
  - Authentication edge cases
  - Authorization boundary tests

## Cloudflare Recommendations

If Cloudflare is in front of both domains:

- [ ] Use `CF-Connecting-IP` header instead of `X-Forwarded-For`
- [ ] Enable Bot Fight Mode in Cloudflare dashboard
- [ ] Configure WAF rules for common attack patterns
- [ ] Add Rate Limiting rules at the edge
- [ ] Consider Cloudflare Access for admin endpoints (`/docs`, `/api/keys`)

## Pending Migrations

Run to apply RBAC changes:
```bash
cd ~/signal4/core
uv run alembic upgrade head
```

Migration: `b6187fea062d_add_allowed_projects_to_api_keys.py`

## Files Modified

| File | Changes |
|------|---------|
| `src/backend/app/main.py` | Security headers, CORS, docs protection, error sanitization |
| `src/backend/app/middleware/api_key_auth.py` | RBAC enforcement, project validation, security logging |
| `src/backend/app/routers/media.py` | Input validation (content_id) |
| `src/backend/app/routers/query.py` | Error sanitization, project RBAC |
| `src/backend/app/routers/analysis.py` | Error sanitization, project RBAC |
| `src/backend/app/database/connection.py` | Connection pooling optimization |
| `src/database/models.py` | Added `allowed_projects` to ApiKey, `check_project_access()` method |
| `frontend/next.config.ts` | Security headers |
| `frontend/src/app/api/media/*/route.ts` | Session authentication |
| `frontend/src/app/api/analysis/stream/route.ts` | Session authentication |

## Workers Pending `.env` Update

When these workers come online:
```bash
# worker1
cat ~/.signal4/core/.env | ssh signal4@10.0.0.9 "cat > /Users/signal4/signal4/core/.env"

# worker2
cat ~/.signal4/core/.env | ssh signal4@10.0.0.195 "cat > /Users/signal4/signal4/core/.env"
```

---
*Generated: 2026-01-19*
