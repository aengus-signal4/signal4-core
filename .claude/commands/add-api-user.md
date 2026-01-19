Add a new API user with media-only access.

Arguments: $ARGUMENTS (format: "Name email@example.com [basic|moderate|full]")

Parse the arguments to extract name, email, and access level (default: moderate).

Access levels:
- basic: 100/hour, 30 days
- moderate: 500/hour, 90 days
- full: 1000/hour, no expiry

Run this command to create the key:

```bash
cd ~/signal4/core && uv run python -m src.backend.scripts.manage_api_keys create \
  --email "<extracted_email>" \
  --name "<extracted_name> - Media Access" \
  --rate-limit <rate_for_level> \
  --expires-days <days_for_level_or_omit_for_full> \
  --scopes "media:read"
```

After creating the key, generate a welcome message for the user:

```
Hi <name>,

Your Signal4 Media API access:

API Key: <the_generated_key>

Endpoint: https://api.signal4.ca/api/media/content/{content_id}

Example:
curl -H "X-API-Key: YOUR_KEY" \
  "https://api.signal4.ca/api/media/content/CONTENT_ID?start_time=0&end_time=60"

Parameters:
- content_id (required) - Content identifier
- start_time/end_time (optional) - Time range in seconds
- media_type - "audio" or "video" (default: auto)
- format - "webm" or "mp4" (default: webm)

Limits: <rate>/hour, expires <date or "never">

Store this key securely - it cannot be retrieved again.
```
