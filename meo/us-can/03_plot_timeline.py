#!/usr/bin/env python
"""
US-Canada MEO Analysis - Timeline Plot

Creates a sentiment timeline showing how top US podcasts discuss Canada,
with sentiment relative to the period average.

Input: .cache/canada_sentiment_data.json (from 01_extract_data.py)
Output: outputs/03_timeline.png, outputs/03_timeline_tv.png
"""

import sys
import json
from pathlib import Path
from datetime import date, datetime
from typing import List, Dict
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import uniform_filter1d
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Import MEO theme
sys.path.insert(0, str(Path.home() / "aide" / "visualization_guides" / "meo"))
from meo_theme import set_theme_light, MEO_COLORS, THEME_LIGHT

# Apply MEO theme
plt.rcParams.update(THEME_LIGHT)

# Directories
_script_dir = Path(__file__).parent
OUTPUT_DIR = _script_dir / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR = _script_dir / ".cache"
CACHE_FILE = CACHE_DIR / "canada_sentiment_data.json"

# Colors
COLORS = {
    'sage': MEO_COLORS[0],       # Green
    'danger': MEO_COLORS[1],     # Burgundy/red
}

# Major events for timeline annotations
MAJOR_EVENTS = [
    {"date": date(2025, 1, 6), "short": "Trudeau\nResigns"},
    {"date": date(2025, 2, 1), "short": "Trump Signs\n25% Tariffs"},
    {"date": date(2025, 3, 4), "short": "Tariffs\nIn Effect"},
    {"date": date(2025, 3, 14), "short": "Carney\nBecomes PM"},
    {"date": date(2025, 4, 28), "short": "Carney Wins\nElection"},
    {"date": date(2025, 6, 16), "short": "G7 Summit\nTensions"},
    {"date": date(2025, 10, 13), "short": "Gaza Peace\nSummit"},
    {"date": date(2026, 1, 21), "short": "Davos\nTrade Talks"},
]


def load_data() -> List[Dict]:
    """Load cached sentiment data."""
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Cache file not found: {CACHE_FILE}\nRun 01_extract_data.py first.")

    with open(CACHE_FILE, 'r') as f:
        data = json.load(f)

    # Convert day strings back to datetime
    for row in data:
        row['day'] = datetime.fromisoformat(row['day'])

    return data


def plot_timeline(data: List[Dict]) -> str:
    """Create timeline with volume bars at bottom and 7-day rolling sentiment."""
    set_theme_light()

    fig, ax = plt.subplots(figsize=(14, 8))

    # Parse data
    days = [row['day'] for row in data]
    positive = np.array([row['positive'] for row in data])
    negative = np.array([row['negative'] for row in data])
    pct_of_total = np.array([row.get('pct_of_total', 0) for row in data])

    # Calculate sentiment ratio per day
    with np.errstate(divide='ignore', invalid='ignore'):
        sentiment_ratio = np.where(
            (positive + negative) > 0,
            (positive - negative) / (positive + negative),
            0
        )

    # 7-day rolling average
    sentiment_smooth = uniform_filter1d(sentiment_ratio.astype(float), size=7)

    # Calculate average sentiment over the period
    avg_sentiment = np.mean(sentiment_smooth)

    # Center sentiment around the average
    sentiment_centered = sentiment_smooth - avg_sentiment

    # Scale percentage for volume bars
    max_pct = max(pct_of_total) if max(pct_of_total) > 0 else 1

    # Y-axis layout
    y_top = 0.45
    y_bottom_sentiment = -0.3
    volume_height = 0.12
    volume_bottom = y_bottom_sentiment - volume_height - 0.02
    volume_scaled = (pct_of_total / max_pct) * volume_height

    # Volume bars at bottom
    ax.bar(days, volume_scaled, bottom=volume_bottom, width=1.0,
           color='#666666', alpha=0.4, zorder=1)

    # Sentiment line and fill
    ax.plot(days, sentiment_centered, color=COLORS['sage'], linewidth=2.5, zorder=5)

    ax.fill_between(days, 0, sentiment_centered,
                    where=(sentiment_centered >= 0),
                    alpha=0.5, color=COLORS['sage'], interpolate=True, zorder=4)
    ax.fill_between(days, 0, sentiment_centered,
                    where=(sentiment_centered < 0),
                    alpha=0.5, color=COLORS['danger'], interpolate=True, zorder=4)

    # Average line at zero
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.4, zorder=3)

    # Set y-axis
    ax.set_ylim(volume_bottom - 0.02, y_top)
    ax.set_ylabel('Sentiment (above/below average)', fontsize=10)

    # Y-axis ticks
    ticks = np.array([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:+.1f}' if t != 0 else 'avg' for t in ticks])

    # Event annotations
    for i, event in enumerate(MAJOR_EVENTS):
        event_datetime = datetime.combine(event['date'], datetime.min.time())
        y_label = y_top * 0.85 if i % 2 == 0 else y_top * 0.70

        ax.plot([event_datetime, event_datetime], [0, y_label],
                color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=6)

        ax.annotate(
            event['short'],
            xy=(event_datetime, y_label),
            fontsize=8,
            color='black',
            fontweight='normal',
            ha='center',
            va='bottom',
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='white',
                edgecolor='black',
                linewidth=0.5,
                alpha=0.95
            ),
            zorder=10
        )

    # X-axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlabel('')

    # Title
    ax.set_title('Sentiment of top US News/Politics/Culture podcasts towards Canada',
                 fontsize=12, fontweight='bold')

    # Grid
    ax.yaxis.grid(True, alpha=0.2, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=COLORS['sage'], linewidth=2.5, label='Sentiment (7-day rolling avg)'),
        Patch(facecolor=COLORS['sage'], alpha=0.5, label='Above average'),
        Patch(facecolor=COLORS['danger'], alpha=0.5, label='Below average'),
        Patch(facecolor='#666666', alpha=0.4, label=f'% of conversation (max={max_pct:.1f}%)'),
    ]

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=4, frameon=False, fontsize=10)

    # Caption
    caption_line1 = "Based on 318 podcasts, 75,819 episodes, and 53,145 hours of content (Jan 2025 – Jan 2026)."
    caption_line2 = "VADER sentiment on segments mentioning Canada/Canadian/Carney/Trudeau. Values show deviation from average."
    fig.text(0.98, 0.035, caption_line1, ha='right', va='bottom', fontsize=8, color='black')
    fig.text(0.98, 0.012, caption_line2, ha='right', va='bottom', fontsize=8, color='black')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)

    output_path = OUTPUT_DIR / "03_timeline.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return str(output_path)


def plot_timeline_tv(data: List[Dict]) -> str:
    """Create timeline with larger labels for television display."""
    set_theme_light()

    fig, ax = plt.subplots(figsize=(16, 9))

    # Parse data
    days = [row['day'] for row in data]
    positive = np.array([row['positive'] for row in data])
    negative = np.array([row['negative'] for row in data])
    pct_of_total = np.array([row.get('pct_of_total', 0) for row in data])

    # Calculate sentiment ratio per day
    with np.errstate(divide='ignore', invalid='ignore'):
        sentiment_ratio = np.where(
            (positive + negative) > 0,
            (positive - negative) / (positive + negative),
            0
        )

    # 7-day rolling average
    sentiment_smooth = uniform_filter1d(sentiment_ratio.astype(float), size=7)
    avg_sentiment = np.mean(sentiment_smooth)
    sentiment_centered = sentiment_smooth - avg_sentiment

    # Scale percentage for volume bars
    max_pct = max(pct_of_total) if max(pct_of_total) > 0 else 1

    # Y-axis layout
    y_top = 0.45
    y_bottom_sentiment = -0.3
    volume_height = 0.12
    volume_bottom = y_bottom_sentiment - volume_height - 0.02
    volume_scaled = (pct_of_total / max_pct) * volume_height

    # Volume bars at bottom
    ax.bar(days, volume_scaled, bottom=volume_bottom, width=1.0,
           color='#666666', alpha=0.4, zorder=1)

    # Sentiment line and fill - thicker for TV
    ax.plot(days, sentiment_centered, color=COLORS['sage'], linewidth=4, zorder=5)

    ax.fill_between(days, 0, sentiment_centered,
                    where=(sentiment_centered >= 0),
                    alpha=0.5, color=COLORS['sage'], interpolate=True, zorder=4)
    ax.fill_between(days, 0, sentiment_centered,
                    where=(sentiment_centered < 0),
                    alpha=0.5, color=COLORS['danger'], interpolate=True, zorder=4)

    # Average line at zero
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle='-', alpha=0.4, zorder=3)

    # Set y-axis
    ax.set_ylim(volume_bottom - 0.02, y_top)
    ax.set_ylabel('Sentiment (above/below average)', fontsize=18, fontweight='bold')

    # Y-axis ticks
    ticks = np.array([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'{t:+.1f}' if t != 0 else 'avg' for t in ticks], fontsize=15)

    # Event annotations - larger for TV
    for i, event in enumerate(MAJOR_EVENTS):
        event_datetime = datetime.combine(event['date'], datetime.min.time())
        y_label = y_top * 0.82 if i % 2 == 0 else y_top * 0.62

        ax.plot([event_datetime, event_datetime], [0, y_label],
                color='black', linestyle='-', linewidth=1.5, alpha=0.5, zorder=6)

        ax.annotate(
            event['short'],
            xy=(event_datetime, y_label),
            fontsize=12,
            color='black',
            fontweight='bold',
            ha='center',
            va='bottom',
            bbox=dict(
                boxstyle='round,pad=0.35',
                facecolor='white',
                edgecolor='black',
                linewidth=1.2,
                alpha=0.95
            ),
            zorder=10
        )

    # X-axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis='x', labelsize=12)
    ax.set_xlabel('')

    # Title
    ax.set_title('Sentiment of top US News/Politics/Culture podcasts towards Canada',
                 fontsize=22, fontweight='bold', pad=20)

    # Grid
    ax.yaxis.grid(True, alpha=0.2, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Legend
    legend_elements = [
        Line2D([0], [0], color=COLORS['sage'], linewidth=4, label='Sentiment (7-day rolling avg)'),
        Patch(facecolor=COLORS['sage'], alpha=0.5, label='Above average'),
        Patch(facecolor=COLORS['danger'], alpha=0.5, label='Below average'),
        Patch(facecolor='#666666', alpha=0.4, label=f'% of conversation (max={max_pct:.1f}%)'),
    ]

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.06),
              ncol=4, frameon=False, fontsize=15)

    # Caption
    caption_line1 = "Based on 318 podcasts, 75,819 episodes, and 53,145 hours of content (Jan 2025 – Jan 2026)."
    caption_line2 = "VADER sentiment on segments mentioning Canada/Canadian/Carney/Trudeau. Values show deviation from average."
    fig.text(0.98, 0.04, caption_line1, ha='right', va='bottom', fontsize=11, color='black')
    fig.text(0.98, 0.012, caption_line2, ha='right', va='bottom', fontsize=11, color='black')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)

    output_path = OUTPUT_DIR / "03_timeline_tv.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return str(output_path)


def main():
    print("=" * 60)
    print("US-Canada MEO Analysis - Timeline Plot")
    print("=" * 60)

    print("\nLoading data...")
    data = load_data()
    print(f"  Loaded {len(data)} days")

    total_hours = sum(row['total'] for row in data)
    pos_hours = sum(row['positive'] for row in data)
    neg_hours = sum(row['negative'] for row in data)
    print(f"  Total hours: {total_hours:.1f}")
    print(f"  Positive: {pos_hours:.1f} ({100*pos_hours/total_hours:.1f}%)")
    print(f"  Negative: {neg_hours:.1f} ({100*neg_hours/total_hours:.1f}%)")

    print("\nGenerating plots...")

    output_path = plot_timeline(data)
    print(f"  Standard: {output_path}")

    output_path_tv = plot_timeline_tv(data)
    print(f"  TV version: {output_path_tv}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
