#!/usr/bin/env python3
"""
Streamlit dashboard for reviewing tone analysis results.

Usage:
    uv run streamlit run tone_dashboard.py -- --results tone_<content_id>.json

Features:
- Audio playback for each speaker turn
- Tone analysis visualization
- Filter by speaker, emotion, etc.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
import boto3
from dotenv import load_dotenv

# Load environment
CORE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(CORE_DIR / ".env", override=True)


def get_s3_client():
    """Get S3/MinIO client."""
    return boto3.client(
        's3',
        endpoint_url="http://10.0.0.251:9000",
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name="us-east-1"
    )


@st.cache_data
def load_results(results_file: str):
    """Load tone analysis results."""
    with open(results_file) as f:
        return json.load(f)


@st.cache_data
def get_audio_url(content_id: str):
    """Get presigned URL for content audio."""
    s3 = get_s3_client()
    bucket = "av-content"

    for filename in ["audio.opus", "audio.mp3", "audio.wav"]:
        key = f"content/{content_id}/{filename}"
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=3600
            )
        except Exception:
            continue

    return None


def extract_audio_segment(presigned_url: str, start_time: float, end_time: float) -> bytes:
    """Extract audio segment and return as bytes."""
    duration = end_time - start_time

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        output_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", presigned_url,
            "-t", str(duration),
            "-ar", "44100",
            "-ac", "2",
            "-b:a", "128k",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            return None

        with open(output_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(output_path):
            os.unlink(output_path)


def format_tone_value(label: str, value: str):
    """Format a tone label/value pair."""
    return f"**{label}:** {value}"


def main():
    st.set_page_config(
        page_title="Tone Analysis Dashboard",
        page_icon="🎙️",
        layout="wide"
    )

    st.title("🎙️ Speaker Turn Tone Analysis")

    # File selector
    results_files = list(Path(".").glob("tone_*.json"))

    if not results_files:
        st.error("No tone analysis files found. Run analyze_speaker_turns.py first.")
        return

    selected_file = st.sidebar.selectbox(
        "Results File",
        options=[str(f) for f in results_files],
        format_func=lambda x: x
    )

    # Load data
    data = load_results(selected_file)

    # Header info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Title:** {data['title']}")
    st.sidebar.markdown(f"**Channel:** {data['channel']}")
    st.sidebar.markdown(f"**Turns:** {data['analyzed_turns']}")
    st.sidebar.markdown(f"**Avg Analysis:** {data['avg_analysis_time_s']}s")

    # Get audio URL
    audio_url = get_audio_url(data['content_id'])

    # Filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    # Get unique speakers
    speakers = sorted(set(t['speaker_name'] for t in data['turns']))
    selected_speakers = st.sidebar.multiselect("Speakers", speakers, default=speakers)

    # Get unique emotions
    emotions = sorted(set(
        t['tone'].get('primary_emotion', 'unknown')
        for t in data['turns']
        if t.get('tone')
    ))
    selected_emotions = st.sidebar.multiselect("Emotions", emotions, default=emotions)

    # Get unique sarcasm levels
    sarcasm_levels = sorted(set(
        t['tone'].get('sarcasm', 'none')
        for t in data['turns']
        if t.get('tone')
    ))
    selected_sarcasm = st.sidebar.multiselect("Sarcasm", sarcasm_levels, default=sarcasm_levels)

    # Filter for non-neutral tones
    show_only_notable = st.sidebar.checkbox("Show only notable tones", value=False)

    # Filter turns
    filtered_turns = []
    for turn in data['turns']:
        if turn['speaker_name'] not in selected_speakers:
            continue
        if turn.get('tone') and turn['tone'].get('primary_emotion') not in selected_emotions:
            continue
        if turn.get('tone') and turn['tone'].get('sarcasm', 'none') not in selected_sarcasm:
            continue
        if show_only_notable and turn.get('tone'):
            neutral_values = {"neutral", "none", "relaxed", "male", "female"}
            has_notable = any(
                v.lower() not in neutral_values
                for k, v in turn['tone'].items()
                if isinstance(v, str) and k != 'gender'
            )
            if not has_notable:
                continue
        filtered_turns.append(turn)

    st.markdown(f"Showing **{len(filtered_turns)}** of {len(data['turns'])} turns")

    # Display turns
    for i, turn in enumerate(filtered_turns):
        with st.expander(
            f"Turn {turn['turn_index']} | {turn['speaker_name']} | "
            f"{turn['start_time']:.1f}s - {turn['end_time']:.1f}s "
            f"({turn['end_time'] - turn['start_time']:.1f}s)",
            expanded=(i < 3)  # Expand first 3
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                # Text
                st.markdown(f"**Transcript:**")
                st.text(turn['text'][:500] + ("..." if len(turn['text']) > 500 else ""))

                # Audio player
                if audio_url:
                    if st.button(f"🔊 Play Audio", key=f"play_{turn['turn_id']}"):
                        with st.spinner("Loading audio..."):
                            audio_bytes = extract_audio_segment(
                                audio_url,
                                turn['start_time'],
                                turn['end_time']
                            )
                            if audio_bytes:
                                st.audio(audio_bytes, format="audio/mp3")
                            else:
                                st.error("Failed to load audio")

            with col2:
                # Tone analysis
                if turn.get('error'):
                    st.error(f"Error: {turn['error']}")
                elif turn.get('tone'):
                    tone = turn['tone']
                    # Show all tone values with no spacing between lines
                    tone_lines = [f"**{k}:** {v}  " for k, v in tone.items()]  # Two spaces = line break
                    st.markdown("\n".join(tone_lines))
                else:
                    st.warning("Parse failed")
                    if turn.get('raw_response'):
                        st.code(turn['raw_response'][:200])

    # Summary stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Summary Stats")

    # Count emotions
    emotion_counts = {}
    for turn in data['turns']:
        if turn.get('tone'):
            emotion = turn['tone'].get('primary_emotion', 'unknown')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    if emotion_counts:
        st.sidebar.markdown("**Emotions:**")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
            st.sidebar.markdown(f"- {emotion}: {count}")


if __name__ == "__main__":
    main()
