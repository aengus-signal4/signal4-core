#!/usr/bin/env python
"""
US-Canada MEO Analysis - Segment Classification

Classifies Davos reaction segments by stance toward Canada and Trump using LLM.

Input: .cache/davos_reaction_segments.json
Output: .cache/davos_classifications.json
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add paths for imports
_script_dir = Path(__file__).parent
_core_root = _script_dir.parent.parent
if str(_core_root) not in sys.path:
    sys.path.insert(0, str(_core_root))

from anthropic import Anthropic

from src.utils.config import get_credential

# Directories
CACHE_DIR = _script_dir / ".cache"
INPUT_FILE = CACHE_DIR / "davos_reaction_segments.json"
OUTPUT_FILE = CACHE_DIR / "davos_classifications.json"

# Classification prompt
CLASSIFICATION_PROMPT = """Analyze this podcast segment discussing Canada/Carney in the context of Davos 2026 and US-Canada relations.

Segment text:
"{text}"

Classify with these fields:
1. IS_RELEVANT: Is this segment actually discussing Canada/Carney in relation to Davos, tariffs, or US-Canada relations? (YES/NO)
2. STANCE_CANADA: How does the speaker portray Canada/Carney? (SUPPORTIVE/CRITICAL/NEUTRAL/NA)
3. STANCE_TRUMP: How does the speaker portray Trump's approach? (SUPPORTIVE/CRITICAL/NEUTRAL/NA)
4. FRAMING: Brief description of how Canada is framed (e.g., "weak ally", "principled statesman", "trading partner")
5. CONFIDENCE: How confident are you in this classification? (HIGH/MEDIUM/LOW)

Respond in this exact format:
IS_RELEVANT: [YES/NO]
STANCE_CANADA: [SUPPORTIVE/CRITICAL/NEUTRAL/NA]
STANCE_TRUMP: [SUPPORTIVE/CRITICAL/NEUTRAL/NA]
FRAMING: [brief description]
CONFIDENCE: [HIGH/MEDIUM/LOW]"""


def get_anthropic_client() -> Anthropic:
    """Get Anthropic client."""
    api_key = get_credential("ANTHROPIC_API_KEY")
    return Anthropic(api_key=api_key)


def parse_classification(response_text: str) -> Dict:
    """Parse LLM response into structured classification."""
    lines = response_text.strip().split('\n')
    result = {
        'is_relevant': False,
        'stance_canada': None,
        'stance_trump': None,
        'framing': None,
        'confidence': None,
        'raw_response': response_text
    }

    for line in lines:
        line = line.strip()
        if line.startswith('IS_RELEVANT:'):
            value = line.split(':', 1)[1].strip().upper()
            result['is_relevant'] = value == 'YES'
        elif line.startswith('STANCE_CANADA:'):
            value = line.split(':', 1)[1].strip().upper()
            if value in ('SUPPORTIVE', 'CRITICAL', 'NEUTRAL'):
                result['stance_canada'] = value
        elif line.startswith('STANCE_TRUMP:'):
            value = line.split(':', 1)[1].strip().upper()
            if value in ('SUPPORTIVE', 'CRITICAL', 'NEUTRAL'):
                result['stance_trump'] = value
        elif line.startswith('FRAMING:'):
            result['framing'] = line.split(':', 1)[1].strip()
        elif line.startswith('CONFIDENCE:'):
            result['confidence'] = line.split(':', 1)[1].strip().upper()

    return result


def classify_segment(client: Anthropic, text: str) -> Dict:
    """Classify a single segment using Claude."""
    prompt = CLASSIFICATION_PROMPT.format(text=text[:2000])  # Truncate very long segments

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    return parse_classification(response.content[0].text)


def load_existing_classifications() -> Dict[int, Dict]:
    """Load existing classifications to allow resume."""
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'r') as f:
            existing = json.load(f)
        return {c['segment_id']: c for c in existing}
    return {}


def main():
    print("=" * 60)
    print("US-Canada MEO Analysis - Segment Classification")
    print("=" * 60)

    # Load segments
    print("\nLoading segments...")
    with open(INPUT_FILE, 'r') as f:
        segments = json.load(f)
    print(f"  Found {len(segments)} segments")

    # Load existing classifications (for resume)
    existing = load_existing_classifications()
    print(f"  Already classified: {len(existing)}")

    # Initialize client
    client = get_anthropic_client()

    # Classify each segment
    classifications = list(existing.values())
    to_classify = [s for s in segments if s['segment_id'] not in existing]

    print(f"\nClassifying {len(to_classify)} new segments...")

    for i, segment in enumerate(to_classify):
        try:
            result = classify_segment(client, segment['text'])
            result['segment_id'] = segment['segment_id']
            classifications.append(result)

            # Progress
            if (i + 1) % 10 == 0:
                print(f"  Classified {i + 1}/{len(to_classify)}")
                # Save checkpoint
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(classifications, f, indent=2)

            # Rate limiting
            time.sleep(0.1)

        except Exception as e:
            print(f"  Error classifying segment {segment['segment_id']}: {e}")
            continue

    # Final save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(classifications, f, indent=2)
    print(f"\nSaved: {OUTPUT_FILE}")

    # Summary
    relevant = [c for c in classifications if c.get('is_relevant')]
    print(f"\nSummary:")
    print(f"  Total classified: {len(classifications)}")
    print(f"  Relevant: {len(relevant)}")

    # Stance breakdown
    from collections import Counter
    canada_stances = Counter(c.get('stance_canada') for c in relevant)
    trump_stances = Counter(c.get('stance_trump') for c in relevant)
    print(f"  Canada stances: {dict(canada_stances)}")
    print(f"  Trump stances: {dict(trump_stances)}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
