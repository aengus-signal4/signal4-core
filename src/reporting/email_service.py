"""
Email Service
=============

Handles sending daily report emails via Resend Broadcasts API.
"""

import os
import resend
from typing import Optional
from datetime import datetime

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('email_service')


def get_resend_config():
    """Get Resend configuration from environment."""
    api_key = os.environ.get('RESEND_API_KEY')
    audience_id = os.environ.get('RESEND_AUDIENCE_ID')

    if not api_key:
        raise ValueError("RESEND_API_KEY environment variable is required")
    if not audience_id:
        raise ValueError("RESEND_AUDIENCE_ID environment variable is required")

    return {
        'api_key': api_key,
        'audience_id': audience_id,
    }


def send_daily_broadcast(
    html_content: str,
    subject: Optional[str] = None,
    from_email: str = "Signal4 <reports@signal4.ca>",
    dry_run: bool = False
) -> dict:
    """
    Send daily report broadcast to all subscribers.

    Args:
        html_content: The HTML email content
        subject: Email subject (defaults to "Signal4 Daily Report - {date}")
        from_email: From address
        dry_run: If True, don't actually send (for testing)

    Returns:
        dict with broadcast_id and status
    """
    config = get_resend_config()
    resend.api_key = config['api_key']

    # Generate subject if not provided
    if not subject:
        date_str = datetime.utcnow().strftime("%B %d, %Y")
        subject = f"Signal4 Daily Report - {date_str}"

    logger.info(f"Preparing broadcast: '{subject}' to audience {config['audience_id']}")

    if dry_run:
        logger.info("[DRY RUN] Would send broadcast with:")
        logger.info(f"  Subject: {subject}")
        logger.info(f"  From: {from_email}")
        logger.info(f"  Audience ID: {config['audience_id']}")
        logger.info(f"  Content length: {len(html_content)} characters")
        return {
            'broadcast_id': 'dry_run',
            'status': 'dry_run',
            'subject': subject,
        }

    try:
        # Create the broadcast
        broadcast = resend.Broadcasts.create({
            "audience_id": config['audience_id'],
            "from": from_email,
            "subject": subject,
            "html": html_content,
        })

        broadcast_id = broadcast.get('id')
        logger.info(f"Created broadcast: {broadcast_id}")

        # Send the broadcast
        send_result = resend.Broadcasts.send(broadcast_id)
        logger.info(f"Broadcast sent: {send_result}")

        return {
            'broadcast_id': broadcast_id,
            'status': 'sent',
            'subject': subject,
        }

    except Exception as e:
        logger.error(f"Failed to send broadcast: {e}")
        raise


def send_test_email(
    to_email: str,
    html_content: str,
    subject: Optional[str] = None,
    from_email: str = "Signal4 <reports@signal4.ca>"
) -> dict:
    """
    Send a test email to a single address (not a broadcast).

    Args:
        to_email: Recipient email address
        html_content: The HTML email content
        subject: Email subject
        from_email: From address

    Returns:
        dict with email_id and status
    """
    config = get_resend_config()
    resend.api_key = config['api_key']

    if not subject:
        date_str = datetime.utcnow().strftime("%B %d, %Y")
        subject = f"[TEST] Signal4 Daily Report - {date_str}"

    logger.info(f"Sending test email to: {to_email}")

    try:
        result = resend.Emails.send({
            "from": from_email,
            "to": [to_email],
            "subject": subject,
            "html": html_content,
        })

        email_id = result.get('id')
        logger.info(f"Test email sent: {email_id}")

        return {
            'email_id': email_id,
            'status': 'sent',
            'to': to_email,
            'subject': subject,
        }

    except Exception as e:
        logger.error(f"Failed to send test email: {e}")
        raise


def get_audience_stats() -> dict:
    """
    Get statistics about the subscriber audience.

    Returns:
        dict with subscriber count and other stats
    """
    config = get_resend_config()
    resend.api_key = config['api_key']

    try:
        # Get audience details
        audience = resend.Audiences.get(config['audience_id'])

        # Get contacts list
        contacts = resend.Contacts.list(config['audience_id'])

        active_count = 0
        unsubscribed_count = 0

        for contact in contacts.get('data', []):
            if contact.get('unsubscribed'):
                unsubscribed_count += 1
            else:
                active_count += 1

        return {
            'audience_id': config['audience_id'],
            'audience_name': audience.get('name', 'Unknown'),
            'active_subscribers': active_count,
            'unsubscribed': unsubscribed_count,
            'total_contacts': active_count + unsubscribed_count,
        }

    except Exception as e:
        logger.error(f"Failed to get audience stats: {e}")
        raise
