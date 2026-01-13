"""
Linear State Model - Simplified state tracking for the content pipeline.

The pipeline is fundamentally linear:

    NEW → DOWNLOADED → CONVERTED → DIARIZED → TRANSCRIBED → STITCHED → EMBEDDED → IDENTIFIED → COMPLETE
                                                                                      ↓
                                                                                   COMPRESSED

This can be tracked with a SINGLE column: `processing_state` (integer or enum)

Benefits:
1. Single source of truth (no conflicting boolean flags)
2. Easy to query ("give me all content in TRANSCRIBED state")
3. Clear progression (state N means all states < N are complete)
4. Simple failure handling (just track which state failed)

Current Schema (7 boolean flags):
    is_downloaded, is_converted, is_diarized, is_transcribed,
    is_stitched, is_embedded, is_compressed

Proposed Schema (1 integer + 1 for failure):
    processing_state: INTEGER  -- Current state (0-9)
    failed_at_state: INTEGER   -- NULL or the state where failure occurred
    failure_reason: TEXT       -- Error message if failed

State Values:
    0 = NEW           (nothing done)
    1 = DOWNLOADED    (source file exists)
    2 = CONVERTED     (audio.wav + chunks exist)
    3 = DIARIZED      (diarization.json exists)
    4 = TRANSCRIBED   (all chunks have transcripts)
    5 = STITCHED      (transcript_diarized.json + sentences exist)
    6 = EMBEDDED      (semantic_segments.json + embedding_segments exist)
    7 = IDENTIFIED    (speakers identified in database)
    8 = COMPLETE      (fully processed)
    9 = COMPRESSED    (cleanup done, intermediate files removed)

    -1 = BLOCKED      (permanent failure, won't process)

Migration:
    UPDATE content SET processing_state =
        CASE
            WHEN blocked_download THEN -1
            WHEN is_compressed THEN 9
            WHEN is_embedded THEN 6
            WHEN is_stitched THEN 5
            WHEN is_transcribed THEN 4
            WHEN is_diarized THEN 3
            WHEN is_converted THEN 2
            WHEN is_downloaded THEN 1
            ELSE 0
        END;
"""

from enum import IntEnum
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass


class ProcessingState(IntEnum):
    """
    Linear processing states for content.

    States are ordered - a higher state implies all lower states are complete.
    """
    BLOCKED = -1      # Permanent failure
    NEW = 0           # Just added
    DOWNLOADED = 1    # Source file exists
    CONVERTED = 2     # Audio extracted, chunks created
    DIARIZED = 3      # Speaker diarization complete
    TRANSCRIBED = 4   # All chunks transcribed
    STITCHED = 5      # Transcript assembled with speaker info
    EMBEDDED = 6      # Semantic segments created
    IDENTIFIED = 7    # Speakers identified/matched
    COMPLETE = 8      # Fully processed
    COMPRESSED = 9    # Intermediate files cleaned up

    @classmethod
    def from_flags(
        cls,
        is_downloaded: bool = False,
        is_converted: bool = False,
        is_diarized: bool = False,
        is_transcribed: bool = False,
        is_stitched: bool = False,
        is_embedded: bool = False,
        is_identified: bool = False,
        is_compressed: bool = False,
        blocked_download: bool = False
    ) -> 'ProcessingState':
        """Convert legacy boolean flags to a single state."""
        if blocked_download:
            return cls.BLOCKED
        if is_compressed:
            return cls.COMPRESSED
        if is_identified:
            return cls.IDENTIFIED
        if is_embedded:
            return cls.EMBEDDED
        if is_stitched:
            return cls.STITCHED
        if is_transcribed:
            return cls.TRANSCRIBED
        if is_diarized:
            return cls.DIARIZED
        if is_converted:
            return cls.CONVERTED
        if is_downloaded:
            return cls.DOWNLOADED
        return cls.NEW

    def to_flags(self) -> Dict[str, bool]:
        """Convert state back to boolean flags (for compatibility)."""
        return {
            'blocked_download': self == ProcessingState.BLOCKED,
            'is_downloaded': self >= ProcessingState.DOWNLOADED,
            'is_converted': self >= ProcessingState.CONVERTED,
            'is_diarized': self >= ProcessingState.DIARIZED,
            'is_transcribed': self >= ProcessingState.TRANSCRIBED,
            'is_stitched': self >= ProcessingState.STITCHED,
            'is_embedded': self >= ProcessingState.EMBEDDED,
            'is_identified': self >= ProcessingState.IDENTIFIED,
            'is_compressed': self >= ProcessingState.COMPRESSED,
        }

    @property
    def next_state(self) -> Optional['ProcessingState']:
        """Get the next state in the pipeline."""
        if self == ProcessingState.BLOCKED:
            return None
        if self == ProcessingState.COMPRESSED:
            return None  # Terminal
        try:
            return ProcessingState(self.value + 1)
        except ValueError:
            return None

    @property
    def required_task(self) -> Optional[str]:
        """Get the task type needed to advance from this state."""
        task_map = {
            ProcessingState.NEW: 'download',
            ProcessingState.DOWNLOADED: 'convert',
            ProcessingState.CONVERTED: 'diarize',  # Note: transcribe runs after diarize
            ProcessingState.DIARIZED: 'transcribe',
            ProcessingState.TRANSCRIBED: 'stitch',
            ProcessingState.STITCHED: 'embed',  # If you have embedding task
            ProcessingState.EMBEDDED: 'identify',  # Speaker identification
            ProcessingState.IDENTIFIED: 'cleanup',
            ProcessingState.COMPLETE: 'cleanup',
        }
        return task_map.get(self)

    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in (ProcessingState.BLOCKED, ProcessingState.COMPRESSED)

    @property
    def is_processable(self) -> bool:
        """Check if content in this state can be processed further."""
        return self not in (ProcessingState.BLOCKED, ProcessingState.COMPRESSED)


@dataclass
class StateTransitionResult:
    """Result of a state transition attempt."""
    success: bool
    previous_state: ProcessingState
    new_state: ProcessingState
    task_completed: Optional[str] = None
    error: Optional[str] = None


class LinearStateMachine:
    """
    Simple linear state machine for content processing.

    Much simpler than the full state machine because the pipeline is linear!
    """

    # Task type -> (required_state, result_state)
    TASK_TRANSITIONS: Dict[str, Tuple[ProcessingState, ProcessingState]] = {
        'download_youtube': (ProcessingState.NEW, ProcessingState.DOWNLOADED),
        'download_podcast': (ProcessingState.NEW, ProcessingState.DOWNLOADED),
        'download_rumble': (ProcessingState.NEW, ProcessingState.DOWNLOADED),
        'convert': (ProcessingState.DOWNLOADED, ProcessingState.CONVERTED),
        'diarize': (ProcessingState.CONVERTED, ProcessingState.DIARIZED),
        'transcribe': (ProcessingState.DIARIZED, ProcessingState.TRANSCRIBED),
        'stitch': (ProcessingState.TRANSCRIBED, ProcessingState.STITCHED),
        'embed': (ProcessingState.STITCHED, ProcessingState.EMBEDDED),
        'identify': (ProcessingState.EMBEDDED, ProcessingState.IDENTIFIED),
        'cleanup': (ProcessingState.IDENTIFIED, ProcessingState.COMPRESSED),
    }

    # Error patterns that cause permanent blocking
    BLOCKING_ERRORS = [
        'Video unavailable',
        'This video is private',
        'Sign in to confirm your age',
        'Join this channel to get access to members-only content',
        'Bad URL',
    ]

    def can_run_task(self, current_state: ProcessingState, task_type: str) -> Tuple[bool, str]:
        """
        Check if a task can run given the current state.

        Returns (can_run, reason).
        """
        transition = self.TASK_TRANSITIONS.get(task_type)
        if not transition:
            return False, f"Unknown task type: {task_type}"

        required_state, _ = transition

        if current_state == ProcessingState.BLOCKED:
            return False, "Content is blocked"

        if current_state < required_state:
            return False, f"Requires state {required_state.name}, currently at {current_state.name}"

        if current_state > required_state:
            return False, f"Already past state {required_state.name}, currently at {current_state.name}"

        return True, "OK"

    def apply_task_completion(
        self,
        current_state: ProcessingState,
        task_type: str,
        success: bool,
        error_message: Optional[str] = None
    ) -> StateTransitionResult:
        """
        Apply the result of a task completion.

        Returns the new state and any error info.
        """
        transition = self.TASK_TRANSITIONS.get(task_type)
        if not transition:
            return StateTransitionResult(
                success=False,
                previous_state=current_state,
                new_state=current_state,
                error=f"Unknown task type: {task_type}"
            )

        required_state, result_state = transition

        if not success:
            # Check if this is a blocking error
            if error_message and any(e in error_message for e in self.BLOCKING_ERRORS):
                return StateTransitionResult(
                    success=False,
                    previous_state=current_state,
                    new_state=ProcessingState.BLOCKED,
                    task_completed=task_type,
                    error=error_message
                )

            # Non-blocking failure - stay in current state for retry
            return StateTransitionResult(
                success=False,
                previous_state=current_state,
                new_state=current_state,
                task_completed=task_type,
                error=error_message
            )

        # Success - advance to next state
        return StateTransitionResult(
            success=True,
            previous_state=current_state,
            new_state=result_state,
            task_completed=task_type
        )

    def get_next_task(self, current_state: ProcessingState) -> Optional[str]:
        """Get the next task to run for content in this state."""
        if not current_state.is_processable:
            return None
        return current_state.required_task

    def get_progress_percentage(self, state: ProcessingState) -> int:
        """Get processing progress as a percentage."""
        if state == ProcessingState.BLOCKED:
            return 0
        # COMPRESSED is state 9, so 9/9 = 100%
        return int((state.value / ProcessingState.COMPRESSED.value) * 100)


# SQL Migration helpers
def generate_migration_sql() -> str:
    """Generate SQL to add the new column and migrate data."""
    return """
-- Add new column
ALTER TABLE content ADD COLUMN IF NOT EXISTS processing_state INTEGER DEFAULT 0;
ALTER TABLE content ADD COLUMN IF NOT EXISTS failed_at_state INTEGER DEFAULT NULL;
ALTER TABLE content ADD COLUMN IF NOT EXISTS failure_reason TEXT DEFAULT NULL;

-- Migrate existing data
UPDATE content SET processing_state =
    CASE
        WHEN blocked_download THEN -1
        WHEN is_compressed THEN 9
        WHEN is_embedded THEN 6
        WHEN is_stitched THEN 5
        WHEN is_transcribed THEN 4
        WHEN is_diarized THEN 3
        WHEN is_converted THEN 2
        WHEN is_downloaded THEN 1
        ELSE 0
    END
WHERE processing_state IS NULL OR processing_state = 0;

-- Create index for efficient queries
CREATE INDEX IF NOT EXISTS idx_content_processing_state ON content(processing_state);

-- Example queries with new model:
-- Get all content ready for transcription:
--   SELECT * FROM content WHERE processing_state = 3;  -- DIARIZED
--
-- Get processing progress:
--   SELECT processing_state, COUNT(*) FROM content GROUP BY processing_state ORDER BY processing_state;
--
-- Find stuck content:
--   SELECT * FROM content WHERE processing_state < 5 AND last_updated < NOW() - INTERVAL '24 hours';
"""


def generate_view_sql() -> str:
    """Generate SQL view that provides the old boolean flags from new state."""
    return """
-- View for backwards compatibility
CREATE OR REPLACE VIEW content_with_flags AS
SELECT
    *,
    (processing_state = -1) as blocked_download,
    (processing_state >= 1) as is_downloaded,
    (processing_state >= 2) as is_converted,
    (processing_state >= 3) as is_diarized,
    (processing_state >= 4) as is_transcribed,
    (processing_state >= 5) as is_stitched,
    (processing_state >= 6) as is_embedded,
    (processing_state >= 9) as is_compressed,
    CASE processing_state
        WHEN -1 THEN 'BLOCKED'
        WHEN 0 THEN 'NEW'
        WHEN 1 THEN 'DOWNLOADED'
        WHEN 2 THEN 'CONVERTED'
        WHEN 3 THEN 'DIARIZED'
        WHEN 4 THEN 'TRANSCRIBED'
        WHEN 5 THEN 'STITCHED'
        WHEN 6 THEN 'EMBEDDED'
        WHEN 7 THEN 'IDENTIFIED'
        WHEN 8 THEN 'COMPLETE'
        WHEN 9 THEN 'COMPRESSED'
        ELSE 'UNKNOWN'
    END as state_name
FROM content;
"""


# Quick example usage
if __name__ == "__main__":
    # Demo the state machine
    sm = LinearStateMachine()

    # Simulate processing a piece of content
    state = ProcessingState.NEW
    print(f"Starting state: {state.name}")

    tasks = ['download_youtube', 'convert', 'diarize', 'transcribe', 'stitch']

    for task in tasks:
        can_run, reason = sm.can_run_task(state, task)
        if can_run:
            result = sm.apply_task_completion(state, task, success=True)
            print(f"  {task}: {state.name} → {result.new_state.name}")
            state = result.new_state
        else:
            print(f"  {task}: Cannot run - {reason}")

    print(f"\nFinal state: {state.name}")
    print(f"Progress: {sm.get_progress_percentage(state)}%")
    print(f"Next task: {sm.get_next_task(state)}")
