"""
Reporting Module
================

Daily report generation and email distribution.
"""

from .email_service import (
    send_daily_broadcast,
    send_test_email,
    get_audience_stats,
)

__all__ = [
    'send_daily_broadcast',
    'send_test_email',
    'get_audience_stats',
]
