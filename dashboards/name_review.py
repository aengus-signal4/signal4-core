#!/usr/bin/env python3
"""
Name Alias Review Dashboard
===========================

Streamlit dashboard for reviewing speaker name merge suggestions.
Allows listening to audio samples and approving/rejecting merges.

Usage:
    streamlit run dashboards/name_review.py --server.port 8502
"""
import sys
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import yaml

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Configuration paths
CONFIG_DIR = Path(__file__).parent.parent / 'config'
SUGGESTIONS_JSON = CONFIG_DIR / 'name_suggestions.json'
ALIASES_YAML = CONFIG_DIR / 'name_aliases.yaml'

# Backend API URL for audio
BACKEND_API_URL = "http://localhost:7999"

st.set_page_config(
    page_title="Name Alias Review",
    page_icon="🎧",
    layout="wide"
)


def load_suggestions():
    """Load suggestions from JSON file."""
    if not SUGGESTIONS_JSON.exists():
        return None
    with open(SUGGESTIONS_JSON) as f:
        return json.load(f)


def load_aliases():
    """Load current aliases from YAML file."""
    if not ALIASES_YAML.exists():
        return {'aliases': {}, 'unresolved_handles': [], 'do_not_merge': []}
    with open(ALIASES_YAML) as f:
        return yaml.safe_load(f) or {'aliases': {}, 'unresolved_handles': [], 'do_not_merge': []}


def save_aliases(aliases):
    """Save aliases to YAML file."""
    with open(ALIASES_YAML, 'w') as f:
        yaml.dump(aliases, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_audio_url(content_id: str, start_time: float, end_time: float) -> str:
    """Generate audio URL for a segment."""
    return f"{BACKEND_API_URL}/api/media/content/{content_id}?start={start_time}&end={end_time}&media_type=audio"


def render_audio_player(sample: dict, label: str):
    """Render an audio player for a sample segment."""
    col1, col2 = st.columns([1, 2])

    with col1:
        audio_url = get_audio_url(
            sample['content_id'],
            sample['start_time'],
            sample['end_time']
        )
        st.audio(audio_url, format='audio/wav')
        st.caption(f"⏱️ {sample['duration']}s")

    with col2:
        st.markdown(f"**{label}**")
        st.caption(f"📺 {sample.get('episode_title', 'Unknown')}")
        st.text(sample.get('text', '')[:300])


def render_merge_suggestion(merge: dict, idx: int, aliases: dict):
    """Render a single merge suggestion with audio players."""
    canonical = merge['canonical']
    variant = merge['variant']
    similarity = merge['similarity']
    evidence = merge['evidence']

    # Check if already processed
    is_already_merged = variant.lower() in [a.lower() for al in aliases.get('aliases', {}).values() for a in al]
    do_not_merge_pairs = [frozenset(p) for p in aliases.get('do_not_merge', [])]
    is_already_blocked = frozenset([canonical.lower(), variant.lower()]) in [frozenset(p[0].lower(), p[1].lower()) if len(p) >= 2 else frozenset() for p in aliases.get('do_not_merge', [])]

    status_emoji = "✅" if is_already_merged else "❌" if is_already_blocked else "⏳"

    with st.expander(f"{status_emoji} {canonical} ← {variant} (sim: {similarity:.3f})", expanded=not is_already_merged):
        st.markdown(f"""
        | | **{canonical}** | **{variant}** |
        |---|---|---|
        | Hours | {evidence['canonical_hours']}h | {evidence['variant_hours']}h |
        | Episodes | {evidence['canonical_episodes']} | {evidence['variant_episodes']} |
        | Quality Score | {evidence['canonical_quality_score']} | {evidence['variant_quality_score']} |
        """)

        st.markdown("---")
        st.markdown("### 🎧 Audio Samples")

        # Canonical samples
        st.markdown(f"#### {canonical}")
        for i, sample in enumerate(merge.get('samples', {}).get('canonical', [])):
            render_audio_player(sample, f"Sample {i+1}")

        st.markdown(f"#### {variant}")
        for i, sample in enumerate(merge.get('samples', {}).get('variant', [])):
            render_audio_player(sample, f"Sample {i+1}")

        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(f"✅ Approve Merge", key=f"approve_{idx}", disabled=is_already_merged):
                # Add to aliases
                if canonical not in aliases['aliases']:
                    aliases['aliases'][canonical] = []
                if variant not in aliases['aliases'][canonical]:
                    aliases['aliases'][canonical].append(variant)
                save_aliases(aliases)
                st.success(f"Added: {variant} → {canonical}")
                st.rerun()

        with col2:
            if st.button(f"❌ Reject (Do Not Merge)", key=f"reject_{idx}", disabled=is_already_blocked):
                # Add to do_not_merge
                if 'do_not_merge' not in aliases:
                    aliases['do_not_merge'] = []
                aliases['do_not_merge'].append([canonical, variant])
                save_aliases(aliases)
                st.warning(f"Added to do-not-merge: [{canonical}, {variant}]")
                st.rerun()

        with col3:
            if st.button(f"⏭️ Skip", key=f"skip_{idx}"):
                st.info("Skipped")


def render_handle_suggestion(handle: dict, idx: int, aliases: dict):
    """Render a flagged handle with audio samples."""
    name = handle['handle']
    hours = handle['hours']
    episodes = handle['episodes']

    # Check if already in unresolved
    is_already_flagged = name.lower() in [h.lower() for h in aliases.get('unresolved_handles', [])]

    status_emoji = "🔖" if is_already_flagged else "⏳"

    with st.expander(f"{status_emoji} {name} ({hours}h, {episodes} episodes)", expanded=not is_already_flagged):
        st.markdown("### 🎧 Audio Samples")

        for i, sample in enumerate(handle.get('samples', [])):
            render_audio_player(sample, f"Sample {i+1}")

        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            real_name = st.text_input("Real name:", key=f"real_name_{idx}")
            if st.button(f"✅ Map to Real Name", key=f"map_{idx}", disabled=not real_name):
                if real_name not in aliases['aliases']:
                    aliases['aliases'][real_name] = []
                if name not in aliases['aliases'][real_name]:
                    aliases['aliases'][real_name].append(name)
                # Remove from unresolved if present
                if name.lower() in [h.lower() for h in aliases.get('unresolved_handles', [])]:
                    aliases['unresolved_handles'] = [h for h in aliases['unresolved_handles'] if h.lower() != name.lower()]
                save_aliases(aliases)
                st.success(f"Mapped: {name} → {real_name}")
                st.rerun()

        with col2:
            if st.button(f"❓ Mark Unresolved", key=f"unresolved_{idx}", disabled=is_already_flagged):
                if 'unresolved_handles' not in aliases:
                    aliases['unresolved_handles'] = []
                aliases['unresolved_handles'].append(name)
                save_aliases(aliases)
                st.info(f"Marked as unresolved: {name}")
                st.rerun()

        with col3:
            if st.button(f"⏭️ Skip", key=f"skip_handle_{idx}"):
                st.info("Skipped")


def render_do_not_merge_suggestion(pair: dict, idx: int, aliases: dict):
    """Render a potential do-not-merge pair with audio samples."""
    names = pair['names']
    similarity = pair['similarity']
    shared_episodes = pair['shared_episodes']

    # Check if already in do_not_merge
    do_not_merge_lower = [frozenset(n.lower() for n in p) for p in aliases.get('do_not_merge', [])]
    is_already_blocked = frozenset(n.lower() for n in names) in do_not_merge_lower

    status_emoji = "🚫" if is_already_blocked else "⚠️"

    with st.expander(f"{status_emoji} {names[0]} vs {names[1]} (sim: {similarity:.3f}, {shared_episodes} shared)", expanded=not is_already_blocked):
        st.markdown(f"**Reason:** {pair.get('reason', 'N/A')}")

        st.markdown("### 🎧 Audio Samples")

        samples = pair.get('samples', {})
        for name in names:
            st.markdown(f"#### {name}")
            for i, sample in enumerate(samples.get(name, [])):
                render_audio_player(sample, f"Sample {i+1}")

        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(f"🚫 Confirm Do-Not-Merge", key=f"confirm_dnm_{idx}", disabled=is_already_blocked):
                if 'do_not_merge' not in aliases:
                    aliases['do_not_merge'] = []
                aliases['do_not_merge'].append(names)
                save_aliases(aliases)
                st.success(f"Added to do-not-merge: {names}")
                st.rerun()

        with col2:
            if st.button(f"✅ Actually Same Person", key=f"same_person_{idx}"):
                # Add as alias (pick first as canonical based on hours if we had that data)
                canonical = names[0]
                variant = names[1]
                if canonical not in aliases['aliases']:
                    aliases['aliases'][canonical] = []
                if variant not in aliases['aliases'][canonical]:
                    aliases['aliases'][canonical].append(variant)
                save_aliases(aliases)
                st.success(f"Added: {variant} → {canonical}")
                st.rerun()

        with col3:
            if st.button(f"⏭️ Skip", key=f"skip_dnm_{idx}"):
                st.info("Skipped")


def main():
    st.title("🎧 Name Alias Review Dashboard")

    # Load data
    suggestions = load_suggestions()
    aliases = load_aliases()

    if suggestions is None:
        st.error(f"No suggestions file found at {SUGGESTIONS_JSON}")
        st.info("Run Phase 3 speaker identification to generate suggestions:")
        st.code("python -m src.speaker_identification.strategies.label_propagation_clustering --project CPRMV")
        return

    # Show generation info
    st.sidebar.markdown("### 📊 Suggestions Info")
    st.sidebar.markdown(f"**Generated:** {suggestions.get('generated_at', 'Unknown')}")
    st.sidebar.markdown(f"**Threshold:** {suggestions.get('threshold_used', 'Unknown')}")
    st.sidebar.markdown(f"**Merges:** {len(suggestions.get('suggested_merges', []))}")
    st.sidebar.markdown(f"**Handles:** {len(suggestions.get('flagged_handles', []))}")
    st.sidebar.markdown(f"**Do-Not-Merge:** {len(suggestions.get('potential_do_not_merge', []))}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📁 Current Aliases")
    st.sidebar.markdown(f"**Aliases:** {sum(len(v) for v in aliases.get('aliases', {}).values())}")
    st.sidebar.markdown(f"**Unresolved:** {len(aliases.get('unresolved_handles', []))}")
    st.sidebar.markdown(f"**Do-Not-Merge:** {len(aliases.get('do_not_merge', []))}")

    # Filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Filters")
    min_hours = st.sidebar.slider("Min hours (either side)", 0.0, 10.0, 0.5, 0.5)
    min_similarity = st.sidebar.slider("Min similarity", 0.70, 1.0, 0.90, 0.01)
    hide_processed = st.sidebar.checkbox("Hide already processed", value=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        f"🔀 Merge Suggestions ({len(suggestions.get('suggested_merges', []))})",
        f"🏷️ Flagged Handles ({len(suggestions.get('flagged_handles', []))})",
        f"🚫 Do-Not-Merge ({len(suggestions.get('potential_do_not_merge', []))})",
        "📝 Current Config"
    ])

    with tab1:
        st.markdown("### Suggested Name Merges")
        st.markdown("These pairs have similar voice embeddings and may be the same person.")

        merges = suggestions.get('suggested_merges', [])

        # Apply filters
        filtered_merges = []
        for m in merges:
            ev = m['evidence']
            # Hours filter
            if ev['canonical_hours'] < min_hours and ev['variant_hours'] < min_hours:
                continue
            # Similarity filter
            if m['similarity'] < min_similarity:
                continue
            # Hide processed filter
            if hide_processed:
                variant_lower = m['variant'].lower()
                canonical_lower = m['canonical'].lower()
                # Check if variant is already an alias
                all_aliases = set()
                for c, variants in aliases.get('aliases', {}).items():
                    all_aliases.add(c.lower())
                    for v in variants:
                        all_aliases.add(v.lower())
                if variant_lower in all_aliases and canonical_lower in all_aliases:
                    continue
                # Check if in do_not_merge
                pair_key = frozenset([variant_lower, canonical_lower])
                dnm_keys = [frozenset(n.lower() for n in p) for p in aliases.get('do_not_merge', [])]
                if pair_key in dnm_keys:
                    continue
            filtered_merges.append(m)

        st.info(f"Showing {len(filtered_merges)} of {len(merges)} suggestions (filtered by hours≥{min_hours}, similarity≥{min_similarity})")

        if not filtered_merges:
            st.success("All merge suggestions have been processed or filtered out!")
        else:
            # Bulk approve section
            st.markdown("---")
            st.markdown("### ⚡ Bulk Approve")

            # Find auto-approvable patterns
            auto_approve_candidates = []
            for m in filtered_merges:
                canonical = m['canonical']
                variant = m['variant']
                sim = m['similarity']

                # Pattern 1: "Dr. X" vs "X" (title prefix)
                title_prefixes = ['Dr. ', 'Prof. ', 'The Honourable ', 'Hon. ', 'Rev. ', 'Pastor ']
                is_title_variant = False
                for prefix in title_prefixes:
                    if variant.startswith(prefix) and variant[len(prefix):] == canonical:
                        is_title_variant = True
                        break
                    if canonical.startswith(prefix) and canonical[len(prefix):] == variant:
                        is_title_variant = True
                        break

                # Pattern 2: Very high similarity (>=0.995) - almost certainly same person
                is_very_high_sim = sim >= 0.995

                # Pattern 3: One is substring of the other (e.g., "John Smith" vs "John Smith Jr.")
                is_substring = (canonical.lower() in variant.lower() or variant.lower() in canonical.lower()) and sim >= 0.98

                if is_title_variant or is_very_high_sim or is_substring:
                    auto_approve_candidates.append({
                        'merge': m,
                        'reason': 'title_prefix' if is_title_variant else ('very_high_sim' if is_very_high_sim else 'substring')
                    })

            if auto_approve_candidates:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{len(auto_approve_candidates)} high-confidence matches** (≥0.995 similarity, title prefixes, or name substrings)")

                    # Show preview
                    with st.expander("Preview candidates", expanded=False):
                        for item in auto_approve_candidates[:20]:
                            m = item['merge']
                            reason_emoji = "🎓" if item['reason'] == 'title_prefix' else ("🎯" if item['reason'] == 'very_high_sim' else "📝")
                            st.write(f"{reason_emoji} `{m['canonical']}` ← `{m['variant']}` (sim={m['similarity']:.3f})")
                        if len(auto_approve_candidates) > 20:
                            st.write(f"... and {len(auto_approve_candidates) - 20} more")

                with col2:
                    if st.button(f"✅ Bulk Approve All ({len(auto_approve_candidates)})", type="primary"):
                        approved_count = 0
                        for item in auto_approve_candidates:
                            m = item['merge']
                            canonical = m['canonical']
                            variant = m['variant']
                            if canonical not in aliases['aliases']:
                                aliases['aliases'][canonical] = []
                            if variant not in aliases['aliases'][canonical]:
                                aliases['aliases'][canonical].append(variant)
                                approved_count += 1
                        save_aliases(aliases)
                        st.success(f"Approved {approved_count} merges!")
                        st.rerun()
            else:
                st.write("No high-confidence auto-approve candidates in current filter.")

            st.markdown("---")
            st.markdown("### 📋 Individual Review")

            for idx, merge in enumerate(sorted(filtered_merges, key=lambda x: -x['similarity'])):
                render_merge_suggestion(merge, idx, aliases)

    with tab2:
        st.markdown("### Flagged Handles")
        st.markdown("These appear to be pseudonyms/handles without known real names.")

        handles = suggestions.get('flagged_handles', [])
        if not handles:
            st.info("No flagged handles")
        else:
            for idx, handle in enumerate(sorted(handles, key=lambda x: -x['hours'])):
                render_handle_suggestion(handle, idx, aliases)

    with tab3:
        st.markdown("### Potential Do-Not-Merge Pairs")
        st.markdown("These pairs have similar embeddings BUT appear in same episodes (likely co-hosts).")

        dnm_pairs = suggestions.get('potential_do_not_merge', [])
        if not dnm_pairs:
            st.info("No potential do-not-merge pairs")
        else:
            for idx, pair in enumerate(sorted(dnm_pairs, key=lambda x: -x['similarity'])):
                render_do_not_merge_suggestion(pair, idx, aliases)

    with tab4:
        st.markdown("### Current name_aliases.yaml")
        st.code(yaml.dump(aliases, default_flow_style=False, allow_unicode=True), language='yaml')

        if st.button("🔄 Reload"):
            st.rerun()


if __name__ == "__main__":
    main()
