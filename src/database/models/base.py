"""
Base module for database models.

Contains the SQLAlchemy declarative base, enums, and common utility classes
shared across all model modules.
"""

from sqlalchemy.ext.declarative import declarative_base
import enum


Base = declarative_base()


class SpeakerProcessingStatus(enum.Enum):
    """Tracks speaker processing through the 3-phase pipeline"""
    PENDING = "PENDING"           # New from stitch, needs clustering (Phase 1)
    CLUSTERED = "CLUSTERED"       # Phase 1 complete - has speaker_identity_id
    IDENTIFIED = "IDENTIFIED"     # Phase 2 complete - identity has primary_name
    VALIDATED = "VALIDATED"       # Phase 3 complete - checked for merges (optional)


class IdentificationStatus:
    """
    Speaker identification pipeline status tracking.

    Used to prevent re-processing of speakers that have already been evaluated.
    Enables efficient filtering and future Phase 6 retry logic.
    """
    # Initial state
    UNPROCESSED = "unprocessed"                    # Never evaluated (DEFAULT)

    # Success state
    ASSIGNED = "assigned"                          # Linked to speaker_identity

    # Rejection states (detailed for Phase 6 retry logic)
    REJECTED_LOW_SIMILARITY = "rejected_low_similarity"      # Embedding match < 0.40
    REJECTED_NO_CONTEXT = "rejected_no_context"              # No transcript for LLM
    REJECTED_LLM_UNVERIFIED = "rejected_llm_unverified"     # LLM could not verify
    REJECTED_UNKNOWN = "rejected_unknown"                    # LLM returned "unknown"
    REJECTED_SHORT_DURATION = "rejected_short_duration"     # Below duration threshold
    REJECTED_POOR_EMBEDDING = "rejected_poor_embedding"     # Embedding quality too low

    # Review states (future Phase 6)
    PENDING_REVIEW = "pending_review"              # Flagged for human review
    RETRY_ELIGIBLE = "retry_eligible"              # "probably" confidence, worth retry

    @classmethod
    def all_rejected(cls) -> list:
        """Return all rejection status values."""
        return [
            cls.REJECTED_LOW_SIMILARITY,
            cls.REJECTED_NO_CONTEXT,
            cls.REJECTED_LLM_UNVERIFIED,
            cls.REJECTED_UNKNOWN,
            cls.REJECTED_SHORT_DURATION,
            cls.REJECTED_POOR_EMBEDDING,
        ]

    @classmethod
    def all_values(cls) -> list:
        """Return all possible status values."""
        return [
            cls.UNPROCESSED,
            cls.ASSIGNED,
            cls.REJECTED_LOW_SIMILARITY,
            cls.REJECTED_NO_CONTEXT,
            cls.REJECTED_LLM_UNVERIFIED,
            cls.REJECTED_UNKNOWN,
            cls.REJECTED_SHORT_DURATION,
            cls.REJECTED_POOR_EMBEDDING,
            cls.PENDING_REVIEW,
            cls.RETRY_ELIGIBLE,
        ]
