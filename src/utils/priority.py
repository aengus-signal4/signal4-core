from datetime import datetime, timezone

def calculate_priority_by_date(publish_date: datetime, project_priority: int = 1) -> int:
    """Calculate priority based on publish date and project priority.

    Content from the past 30 days receives a recency boost that supersedes project priority,
    ensuring recent content is always processed first regardless of project.

    Args:
        publish_date: The publication date of the content
        project_priority: The priority of the project (1-3, with 3 being highest)

    Returns:
        An integer priority value where higher values mean higher priority

    Priority bands:
        - Recent (< 30 days): 10,000,000 - 10,999,999 (processed first)
        - Older, priority 3:  3,000,000 - 3,999,999
        - Older, priority 2:  2,000,000 - 2,999,999
        - Older, priority 1:  1,000,000 - 1,999,999
    """
    if not publish_date:
        return project_priority * 1000000

    # Ensure timezone-aware
    if publish_date.tzinfo is None:
        publish_date = publish_date.replace(tzinfo=timezone.utc)

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    days_since_epoch = (publish_date - epoch).days

    # Base of 20000 puts 2024-2025 at highest priority, scale for large gaps
    date_priority = (days_since_epoch - 20000) * 1000

    # Add project priority as a multiplier to create distinct bands
    final_priority = date_priority + (project_priority * 1000000)

    # Apply recency boost: content from past 30 days gets +10M priority
    # This ensures recent content is always processed first, regardless of project
    now = datetime.now(timezone.utc)
    days_old = (now - publish_date).days
    if days_old < 30:
        final_priority += 10000000  # 1e7 boost for recent content

    return max(1, min(final_priority, 999999999)) 