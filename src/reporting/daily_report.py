"""
Daily Report CLI
================

Generates and sends the daily report email to subscribers.

Usage:
    uv run python -m src.reporting.daily_report --send
    uv run python -m src.reporting.daily_report --send --dry-run
    uv run python -m src.reporting.daily_report --preview
    uv run python -m src.reporting.daily_report --test-email you@example.com
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('daily_report')


def load_template() -> str:
    """Load the HTML email template."""
    template_path = Path(__file__).parent / 'templates' / 'daily_report.html'
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return template_path.read_text()


def get_report_data(projects: list[str] | None = None) -> dict:
    """Fetch report data from the database."""
    from src.backend.app.services.report_service import ReportService

    service = ReportService()
    return service.get_daily_report(projects=projects)


def format_number(n: int) -> str:
    """Format large numbers with K/M suffixes."""
    if n >= 1000000:
        return f"{n / 1000000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}K"
    return str(n)


def format_duration(minutes: float | None) -> str:
    """Format duration in minutes to human-readable string."""
    if not minutes:
        return ""
    if minutes < 60:
        return f"{int(minutes)}m"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}m"


def format_date(date_str: str | None) -> str:
    """Format date string to human-readable."""
    if not date_str:
        return ""
    try:
        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return date.strftime("%b %d")
    except ValueError:
        return date_str


def render_template(template: str, data: dict) -> str:
    """Render the HTML template with report data."""
    # Extract stats
    stats = data.get('stats', {})
    totals = stats.get('totals', {})

    # Format report date
    report_date = datetime.utcnow().strftime("%B %d, %Y")

    # Start with basic substitutions
    html = template
    html = html.replace('{{report_date}}', report_date)
    html = html.replace('{{total_episodes}}', format_number(totals.get('episodes', 0)))
    html = html.replace('{{total_channels}}', format_number(totals.get('channels', 0)))
    html = html.replace('{{total_speakers}}', format_number(totals.get('speakers', 0)))
    html = html.replace('{{total_hours}}', format_number(int(totals.get('duration_hours', 0))))
    html = html.replace('{{year}}', str(datetime.utcnow().year))

    # Handle episodes list (simple mustache-like replacement)
    recent = data.get('recent_content', {})
    episodes = recent.get('episodes', [])[:5]  # Top 5 episodes

    episodes_html = ""
    for ep in episodes:
        episodes_html += f'''
        <div class="episode">
          <span class="episode-title">{ep.get('title', 'Untitled')}</span>
          <div class="episode-meta">
            {ep.get('channel_name', 'Unknown')} &bull; {format_date(ep.get('publish_date'))} &bull; {format_duration(ep.get('duration_minutes'))}
          </div>
        </div>
        '''

    # Replace episodes section
    episodes_section_start = html.find('{{#episodes}}')
    episodes_section_end = html.find('{{/episodes}}') + len('{{/episodes}}')
    if episodes_section_start != -1 and episodes_section_end > episodes_section_start:
        html = html[:episodes_section_start] + episodes_html + html[episodes_section_end:]

    # Handle channels list
    channels_data = data.get('top_channels', {})
    channels = channels_data.get('channels', [])[:3]  # Top 3 channels

    channels_html = ""
    for ch in channels:
        channels_html += f'''
        <div class="channel">
          <span class="channel-name">{ch.get('name', 'Unknown')}</span>
          <div class="channel-stats">{ch.get('episode_count', 0)} episodes &bull; {ch.get('total_duration_hours', 0):.1f}h of content</div>
        </div>
        '''

    # Replace channels section
    channels_section_start = html.find('{{#channels}}')
    channels_section_end = html.find('{{/channels}}') + len('{{/channels}}')
    if channels_section_start != -1 and channels_section_end > channels_section_start:
        html = html[:channels_section_start] + channels_html + html[channels_section_end:]

    # Unsubscribe URL (Resend will replace this automatically, but we provide a placeholder)
    html = html.replace('{{unsubscribe_url}}', '{{{unsubscribe_url}}}')  # Triple braces for Resend

    return html


def generate_report(projects: list[str] | None = None) -> tuple[str, dict]:
    """Generate the report HTML and return with data."""
    logger.info("Fetching report data...")
    data = get_report_data(projects=projects)

    logger.info("Loading template...")
    template = load_template()

    logger.info("Rendering template...")
    html = render_template(template, data)

    return html, data


def main():
    parser = argparse.ArgumentParser(description="Generate and send daily report email")
    parser.add_argument('--send', action='store_true', help='Send the report to subscribers')
    parser.add_argument('--dry-run', action='store_true', help='Preview what would be sent (no actual send)')
    parser.add_argument('--preview', action='store_true', help='Generate and print HTML to stdout')
    parser.add_argument('--test-email', type=str, help='Send test email to a specific address')
    parser.add_argument('--projects', type=str, help='Comma-separated list of projects to include')
    parser.add_argument('--output', type=str, help='Save HTML to file')

    args = parser.parse_args()

    # Parse projects
    projects = None
    if args.projects:
        projects = [p.strip() for p in args.projects.split(',') if p.strip()]

    try:
        # Generate the report
        html, data = generate_report(projects=projects)
        stats = data.get('stats', {}).get('totals', {})

        logger.info(f"Report generated successfully:")
        logger.info(f"  Episodes: {stats.get('episodes', 0)}")
        logger.info(f"  Channels: {stats.get('channels', 0)}")
        logger.info(f"  Speakers: {stats.get('speakers', 0)}")
        logger.info(f"  Hours: {stats.get('duration_hours', 0):.1f}")

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(html)
            logger.info(f"HTML saved to: {output_path}")

        # Preview mode - print HTML
        if args.preview:
            print(html)
            return

        # Test email mode
        if args.test_email:
            from src.reporting.email_service import send_test_email
            result = send_test_email(
                to_email=args.test_email,
                html_content=html
            )
            logger.info(f"Test email sent: {result}")
            print(f"\nTASK_SUMMARY: Test email sent to {args.test_email}")
            return

        # Send mode
        if args.send:
            from src.reporting.email_service import send_daily_broadcast, get_audience_stats

            # Get audience stats first
            try:
                audience = get_audience_stats()
                logger.info(f"Audience stats: {audience['active_subscribers']} active subscribers")
            except Exception as e:
                logger.warning(f"Could not get audience stats: {e}")

            # Send the broadcast
            result = send_daily_broadcast(
                html_content=html,
                dry_run=args.dry_run
            )

            if args.dry_run:
                logger.info("[DRY RUN] Broadcast prepared but not sent")
                print(f"\nTASK_SUMMARY: [DRY RUN] Report generated, broadcast NOT sent")
            else:
                logger.info(f"Broadcast sent successfully: {result}")
                print(f"\nTASK_SUMMARY: Daily report sent to subscribers (broadcast_id: {result.get('broadcast_id')})")
            return

        # No action specified
        if not args.preview and not args.send and not args.test_email and not args.output:
            logger.info("No action specified. Use --preview, --send, --test-email, or --output")
            parser.print_help()

    except Exception as e:
        logger.error(f"Failed to generate/send report: {e}", exc_info=True)
        print(f"\nTASK_SUMMARY: ERROR - {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
