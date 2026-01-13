"""
Per-Phase Rejection Tracking for Speaker Identification
========================================================

Handles phase-specific status tracking in the `llm_identification` JSONB column.

Each phase (2=host, 3=guest, 5=propagation) records its result independently,
enabling:
- Idempotency: Each phase skips speakers it has already processed
- Phase independence: Rejection in Phase 2 doesn't block Phase 3

JSONB Schema:
{
    "phase_2_host": {
        "status": "rejected",
        "candidate_name": "Jordan B. Peterson",
        "similarity": 0.606,
        "confidence": "unlikely",
        "reasoning": "No name evidence in transcript...",
        "method": "phase_2c_llm_verification",
        "timestamp": "2025-12-07T07:52:45Z"
    },
    "phase_3_guest": null,
    "phase_5_propagation": {
        "status": "assigned",
        "identity_id": 456,
        ...
    }
}

Status Values:
- null / missing key: Not yet processed by this phase
- "rejected": Rejected by this phase (with reasoning)
- "assigned": Successfully assigned by this phase
- "retry_eligible": "probably" confidence, worth retrying
- "skipped": Prerequisites not met (e.g., no embedding)
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from src.database.session import get_session

logger = logging.getLogger(__name__)


class PhaseTracker:
    """Track speaker processing status per phase in llm_identification JSONB."""

    PHASE_KEYS = {
        2: "phase_2_host",
        3: "phase_3_guest",
        5: "phase_5_propagation"
    }

    # Status constants
    STATUS_REJECTED = "rejected"
    STATUS_ASSIGNED = "assigned"
    STATUS_RETRY_ELIGIBLE = "retry_eligible"
    STATUS_SKIPPED = "skipped"

    @classmethod
    def get_phase_key(cls, phase: int) -> str:
        """Get the JSONB key for a phase number."""
        return cls.PHASE_KEYS.get(phase, f"phase_{phase}")

    @classmethod
    def should_process(cls, llm_identification: Optional[Dict], phase: int) -> bool:
        """
        Check if speaker should be processed by this phase.

        Returns True if:
        - Speaker has no llm_identification data
        - Speaker has not been processed by this specific phase
        - Speaker was marked retry_eligible by this phase

        Args:
            llm_identification: The speaker's llm_identification JSONB (can be None)
            phase: Phase number (2, 3, or 5)

        Returns:
            True if speaker should be processed by this phase
        """
        if llm_identification is None:
            return True

        phase_key = cls.get_phase_key(phase)
        phase_data = llm_identification.get(phase_key)

        # Not processed by this phase yet
        if phase_data is None:
            return True

        # Retry eligible - allow reprocessing
        if phase_data.get('status') == cls.STATUS_RETRY_ELIGIBLE:
            return True

        return False

    @classmethod
    def get_phase_filter_sql(cls, phase: int, table_alias: str = "s") -> str:
        """
        Get SQL WHERE clause fragment to filter speakers needing processing by this phase.

        Args:
            phase: Phase number (2, 3, or 5)
            table_alias: Table alias for speakers table (default "s")

        Returns:
            SQL fragment like:
            "(s.llm_identification IS NULL
              OR NOT (s.llm_identification ? 'phase_2_host')
              OR s.llm_identification->'phase_2_host'->>'status' = 'retry_eligible')"
        """
        phase_key = cls.get_phase_key(phase)
        return f"""(
            {table_alias}.llm_identification IS NULL
            OR NOT ({table_alias}.llm_identification ? '{phase_key}')
            OR {table_alias}.llm_identification->'{phase_key}'->>'status' = 'retry_eligible'
        )"""

    @classmethod
    def record_result(
        cls,
        speaker_id: int,
        phase: int,
        status: str,
        result: Dict[str, Any],
        method: Optional[str] = None,
        identity_id: Optional[int] = None,
        dry_run: bool = False
    ) -> None:
        """
        Record the result of processing a speaker in a specific phase.

        Uses jsonb_set() to update only the phase key, preserving other phase data.

        Args:
            speaker_id: Speaker ID
            phase: Phase number (2, 3, or 5)
            status: 'rejected', 'assigned', 'retry_eligible', 'skipped'
            result: Result dict with confidence, reasoning, etc.
            method: Processing method name (e.g., 'phase_2c_llm_verification')
            identity_id: If assigned, the identity ID
            dry_run: If True, don't write to DB
        """
        if dry_run:
            logger.debug(f"[DRY RUN] Would record phase {phase} result for speaker {speaker_id}: {status}")
            return

        phase_key = cls.get_phase_key(phase)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Build phase entry
        phase_entry = {
            'status': status,
            'timestamp': timestamp,
            **result
        }

        if method:
            phase_entry['method'] = method

        if identity_id is not None:
            phase_entry['identity_id'] = identity_id

        with get_session() as session:
            # Use jsonb_set to update only this phase's key
            # First ensure llm_identification exists (default to empty object)
            # Note: We embed phase_key directly since it's a controlled value from PHASE_KEYS
            # Use CAST() instead of ::jsonb to avoid psycopg2 parameter parsing issues
            query = text(f"""
                UPDATE speakers
                SET llm_identification = jsonb_set(
                    COALESCE(llm_identification, '{{}}'::jsonb),
                    ARRAY['{phase_key}'],
                    CAST(:phase_entry AS jsonb)
                ),
                updated_at = NOW()
                WHERE id = :speaker_id
            """)

            session.execute(query, {
                'speaker_id': speaker_id,
                'phase_entry': json.dumps(phase_entry)
            })
            session.commit()

        logger.debug(f"Recorded phase {phase} ({phase_key}) result for speaker {speaker_id}: {status}")

    @classmethod
    def bulk_record_results(
        cls,
        results: List[Dict[str, Any]],
        phase: int,
        dry_run: bool = False
    ) -> int:
        """
        Bulk record phase results for multiple speakers.

        Args:
            results: List of dicts with keys: speaker_id, status, result, method (optional), identity_id (optional)
            phase: Phase number (2, 3, or 5)
            dry_run: If True, don't write to DB

        Returns:
            Number of records updated
        """
        if dry_run:
            logger.debug(f"[DRY RUN] Would bulk record {len(results)} phase {phase} results")
            return 0

        if not results:
            return 0

        phase_key = cls.get_phase_key(phase)
        timestamp = datetime.now(timezone.utc).isoformat()

        with get_session() as session:
            updated = 0
            for r in results:
                phase_entry = {
                    'status': r['status'],
                    'timestamp': timestamp,
                    **r.get('result', {})
                }

                if r.get('method'):
                    phase_entry['method'] = r['method']
                if r.get('identity_id') is not None:
                    phase_entry['identity_id'] = r['identity_id']

                query = text(f"""
                    UPDATE speakers
                    SET llm_identification = jsonb_set(
                        COALESCE(llm_identification, '{{}}'::jsonb),
                        ARRAY['{phase_key}'],
                        CAST(:phase_entry AS jsonb)
                    ),
                    updated_at = NOW()
                    WHERE id = :speaker_id
                """)

                session.execute(query, {
                    'speaker_id': r['speaker_id'],
                    'phase_entry': json.dumps(phase_entry)
                })
                updated += 1

            session.commit()

        logger.info(f"Bulk recorded {updated} phase {phase} results")
        return updated

    @classmethod
    def get_phase_status(cls, speaker_id: int, phase: int) -> Optional[str]:
        """
        Get the status of a speaker for a specific phase.

        Args:
            speaker_id: Speaker ID
            phase: Phase number (2, 3, or 5)

        Returns:
            Status string or None if not processed
        """
        phase_key = cls.get_phase_key(phase)

        with get_session() as session:
            query = text(f"""
                SELECT llm_identification->'{phase_key}'->>'status'
                FROM speakers
                WHERE id = :speaker_id
            """)
            result = session.execute(query, {
                'speaker_id': speaker_id
            }).scalar()

        return result

    @classmethod
    def get_all_phase_statuses(cls, speaker_id: int) -> Dict[str, Optional[str]]:
        """
        Get all phase statuses for a speaker.

        Args:
            speaker_id: Speaker ID

        Returns:
            Dict mapping phase key to status (or None if not processed)
        """
        with get_session() as session:
            query = text("""
                SELECT llm_identification
                FROM speakers
                WHERE id = :speaker_id
            """)
            result = session.execute(query, {'speaker_id': speaker_id}).scalar()

        if not result:
            return {key: None for key in cls.PHASE_KEYS.values()}

        return {
            key: result.get(key, {}).get('status') if result.get(key) else None
            for key in cls.PHASE_KEYS.values()
        }
