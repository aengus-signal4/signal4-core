#!/usr/bin/env python3
"""
Streamlit dashboard for reviewing sentence tone analysis from database.

Usage:
    uv run streamlit run sentence_tone_dashboard.py
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
import boto3
import psycopg2
from dotenv import load_dotenv

# Load environment
CORE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(CORE_DIR / ".env", override=True)


def get_db_connection():
    return psycopg2.connect(
        host="10.0.0.4",
        port=5432,
        database="av_content",
        user="signal4",
        password=os.environ["POSTGRES_PASSWORD"]
    )


def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url="http://10.0.0.251:9000",
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name="us-east-1"
    )


@st.cache_data(ttl=300)
def get_content_with_tone():
    """Get content that has sentence tone data."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT c.content_id, c.title, c.channel_name,
                       COUNT(s.id) as sentences_with_tone,
                       COUNT(DISTINCT s.emotion) as unique_emotions
                FROM content c
                JOIN sentences s ON s.content_id = c.id
                WHERE s.emotion IS NOT NULL
                GROUP BY c.content_id, c.title, c.channel_name
                HAVING COUNT(s.id) > 0
                ORDER BY c.publish_date DESC NULLS LAST
                LIMIT 100
            """)
            return [
                {"content_id": r[0], "title": r[1], "channel": r[2],
                 "sentences": r[3], "emotions": r[4]}
                for r in cur.fetchall()
            ]
    finally:
        conn.close()


@st.cache_data(ttl=60)
def get_sentences_for_content(content_id: str):
    """Get sentences with tone data for a content item."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    s.id, s.sentence_index, s.turn_index, s.text,
                    s.start_time, s.end_time, s.word_count,
                    s.emotion, s.emotion_confidence, s.emotion_scores,
                    s.arousal, s.valence, s.dominance,
                    sp.display_name, si.primary_name
                FROM sentences s
                JOIN content c ON c.id = s.content_id
                JOIN speakers sp ON sp.id = s.speaker_id
                LEFT JOIN speaker_identities si ON si.id = sp.speaker_identity_id
                WHERE c.content_id = %s
                  AND s.emotion IS NOT NULL
                ORDER BY s.sentence_index
            """, (content_id,))
            return [
                {
                    "id": r[0],
                    "sentence_index": r[1],
                    "turn_index": r[2],
                    "text": r[3],
                    "start_time": r[4],
                    "end_time": r[5],
                    "word_count": r[6],
                    "emotion": r[7],
                    "emotion_confidence": r[8],
                    "emotion_scores": r[9] if isinstance(r[9], dict) else json.loads(r[9]) if r[9] else {},
                    "arousal": r[10],
                    "valence": r[11],
                    "dominance": r[12],
                    "speaker_name": r[14] or r[13] or "Unknown"
                }
                for r in cur.fetchall()
            ]
    finally:
        conn.close()


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
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        output_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-i", presigned_url,
            "-t", str(end_time - start_time),
            "-ar", "44100", "-ac", "2", "-b:a", "128k",
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


def main():
    st.set_page_config(page_title="Sentence Tone Dashboard", page_icon="🎭", layout="wide")
    st.title("🎭 Sentence Tone Analysis")

    # Get content with tone data
    content_list = get_content_with_tone()

    if not content_list:
        st.warning("No content with tone analysis found. Run hydrate_sentence_tones.py first.")
        return

    # Content selector
    content_options = {f"{c['title'][:50]}... ({c['sentences']} sentences)": c['content_id']
                       for c in content_list}
    selected_label = st.sidebar.selectbox("Content", options=list(content_options.keys()))
    content_id = content_options[selected_label]

    # Get sentences
    sentences = get_sentences_for_content(content_id)
    audio_url = get_audio_url(content_id)

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Sentences:** {len(sentences)}")

    # Filters
    st.sidebar.markdown("### Filters")

    # Speakers
    speakers = sorted(set(s['speaker_name'] for s in sentences))
    selected_speakers = st.sidebar.multiselect("Speakers", speakers, default=speakers)

    # Emotions
    emotions = sorted(set(s['emotion'] for s in sentences if s['emotion']))
    selected_emotions = st.sidebar.multiselect("Emotions", emotions, default=emotions)

    # Tone filters from emotion_scores
    sarcasm_levels = sorted(set(s['emotion_scores'].get('sarcasm', 'none') for s in sentences if s['emotion_scores']))
    selected_sarcasm = st.sidebar.multiselect("Sarcasm", sarcasm_levels, default=sarcasm_levels)

    aggression_levels = sorted(set(s['emotion_scores'].get('aggression', 'none') for s in sentences if s['emotion_scores']))
    selected_aggression = st.sidebar.multiselect("Aggression", aggression_levels, default=aggression_levels)

    # Notable only
    show_notable = st.sidebar.checkbox("Show only notable tones", value=False)

    # Filter sentences
    filtered = []
    for s in sentences:
        if s['speaker_name'] not in selected_speakers:
            continue
        if s['emotion'] not in selected_emotions:
            continue
        scores = s['emotion_scores'] or {}
        if scores.get('sarcasm', 'none') not in selected_sarcasm:
            continue
        if scores.get('aggression', 'none') not in selected_aggression:
            continue
        if show_notable:
            notable_vals = {'neutral', 'none'}
            has_notable = any(
                v.lower() not in notable_vals
                for k, v in scores.items()
                if isinstance(v, str) and k not in {'gender'}
            )
            if not has_notable and s['emotion'] == 'neutral':
                continue
        filtered.append(s)

    st.markdown(f"Showing **{len(filtered)}** of {len(sentences)} sentences")

    # Display sentences
    for i, sent in enumerate(filtered):
        duration = sent['end_time'] - sent['start_time']
        scores = sent['emotion_scores'] or {}

        with st.expander(
            f"[{sent['sentence_index']}] {sent['speaker_name']} | "
            f"{sent['start_time']:.1f}s ({duration:.1f}s) | "
            f"{sent['emotion']}",
            expanded=(i < 5)
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Text:** {sent['text']}")

                if audio_url:
                    if st.button(f"Play", key=f"play_{sent['id']}"):
                        with st.spinner("Loading..."):
                            audio_bytes = extract_audio_segment(
                                audio_url, sent['start_time'], sent['end_time']
                            )
                            if audio_bytes:
                                st.audio(audio_bytes, format="audio/mp3")

            with col2:
                # Compact tone display
                tone_lines = []
                tone_lines.append(f"**emotion:** {sent['emotion']}")
                tone_lines.append(f"**gender:** {scores.get('gender', 'unknown')}")

                for key in ['dominance_cat', 'confidence_cat', 'aggression', 'sarcasm',
                            'condescension', 'empathy', 'victimhood', 'sincerity']:
                    val = scores.get(key, 'none')
                    if val and val.lower() not in {'none', 'neutral'}:
                        tone_lines.append(f"**{key.replace('_cat', '')}:** {val}")

                st.markdown("  \n".join(tone_lines))

                # Dimensional scores
                if sent['arousal'] is not None:
                    st.caption(f"A:{sent['arousal']:.2f} V:{sent['valence']:.2f} D:{sent['dominance']:.2f}")

    # Summary stats
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Emotion Distribution")
    emotion_counts = {}
    for s in sentences:
        e = s['emotion']
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    for e, count in sorted(emotion_counts.items(), key=lambda x: -x[1]):
        st.sidebar.markdown(f"- {e}: {count}")


if __name__ == "__main__":
    main()
