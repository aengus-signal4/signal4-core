"""
Content State Machine - Models content processing as explicit state transitions.

This provides a cleaner mental model for:
1. What state content is currently in
2. What transitions are valid from each state
3. How to recover from failures
4. What the next action should be

State Diagram:

    ┌──────────┐
    │  NEW     │
    └────┬─────┘
         │ download
         ▼
    ┌──────────┐     fail
    │DOWNLOADING├──────────┐
    └────┬─────┘           │
         │ success         ▼
         │           ┌──────────┐
         │           │ BLOCKED  │
         │           └──────────┘
         ▼
    ┌──────────┐
    │DOWNLOADED│
    └────┬─────┘
         │ convert
         ▼
    ┌──────────┐
    │CONVERTING├───► (retry on transient error)
    └────┬─────┘
         │ success
         ▼
    ┌──────────┐
    │CONVERTED │──┬─────────────┐
    └──────────┘  │             │
                  │ diarize     │ transcribe (parallel)
                  ▼             ▼
            ┌─────────┐   ┌──────────┐
            │DIARIZING│   │TRANSCRIBING│
            └────┬────┘   └────┬─────┘
                 │             │
                 ▼             ▼
            ┌─────────┐   ┌──────────┐
            │DIARIZED │   │TRANSCRIBED│
            └────┬────┘   └────┬─────┘
                 │             │
                 └──────┬──────┘
                        │ (both complete)
                        ▼
                  ┌──────────┐
                  │ STITCHING│
                  └────┬─────┘
                       │ success
                       ▼
                  ┌──────────┐
                  │ STITCHED │
                  └────┬─────┘
                       │ cleanup
                       ▼
                  ┌──────────┐
                  │ COMPLETE │
                  └──────────┘

Failure Handling:
- TRANSIENT errors → Retry same state (with backoff)
- PERMANENT errors → Move to BLOCKED state
- PREREQUISITE errors → Move back to prerequisite state
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List, Callable, Any
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class ContentState(Enum):
    """Possible states for content in the processing pipeline."""

    # Initial states
    NEW = auto()                  # Just added, nothing done
    BLOCKED = auto()              # Permanently blocked (members-only, age-restricted, etc.)

    # Download phase
    DOWNLOADING = auto()          # Download in progress
    DOWNLOADED = auto()           # Source file exists

    # Conversion phase
    CONVERTING = auto()           # Converting to audio chunks
    CONVERTED = auto()            # Audio + chunks exist

    # Parallel processing phase
    DIARIZING = auto()            # Diarization in progress
    DIARIZED = auto()             # Diarization complete
    TRANSCRIBING = auto()         # Transcription in progress (any chunk)
    TRANSCRIBED = auto()          # All chunks transcribed

    # Stitch phase
    READY_FOR_STITCH = auto()     # Both diarized and transcribed
    STITCHING = auto()            # Stitch in progress
    STITCHED = auto()             # Stitch complete

    # Final phase
    COMPRESSING = auto()          # Cleanup/compression in progress
    COMPLETE = auto()             # Fully processed and compressed

    # Error states
    FAILED_TRANSIENT = auto()     # Failed with retryable error
    FAILED_PERMANENT = auto()     # Failed with permanent error


class FailureType(Enum):
    """Types of failures that can occur."""
    TRANSIENT = auto()      # Network error, timeout - retry
    PERMANENT = auto()       # Content unavailable - block
    PREREQUISITE = auto()    # Missing dependency - fix prerequisite first
    RESOURCE = auto()        # Out of disk/memory - wait and retry


@dataclass
class StateTransition:
    """Represents a valid state transition."""
    from_state: ContentState
    to_state: ContentState
    trigger: str                          # What triggers this transition (task type or event)
    condition: Optional[Callable] = None  # Optional condition that must be true


@dataclass
class FailureRecovery:
    """Defines how to recover from a failure in a given state."""
    state: ContentState
    failure_type: FailureType
    recovery_state: ContentState
    max_retries: int = 3
    action: Optional[str] = None  # Additional action to take


class ContentStateMachine:
    """
    State machine for content processing.

    This class:
    1. Determines the current state based on DB flags and S3 files
    2. Validates state transitions
    3. Determines the correct recovery action for failures
    4. Provides the next action to take
    """

    # Valid state transitions
    TRANSITIONS: List[StateTransition] = [
        # Download phase
        StateTransition(ContentState.NEW, ContentState.DOWNLOADING, "download"),
        StateTransition(ContentState.DOWNLOADING, ContentState.DOWNLOADED, "download_complete"),
        StateTransition(ContentState.DOWNLOADING, ContentState.BLOCKED, "download_blocked"),
        StateTransition(ContentState.DOWNLOADING, ContentState.FAILED_TRANSIENT, "download_failed_transient"),

        # Convert phase
        StateTransition(ContentState.DOWNLOADED, ContentState.CONVERTING, "convert"),
        StateTransition(ContentState.CONVERTING, ContentState.CONVERTED, "convert_complete"),
        StateTransition(ContentState.CONVERTING, ContentState.FAILED_TRANSIENT, "convert_failed"),

        # Parallel phase - diarization
        StateTransition(ContentState.CONVERTED, ContentState.DIARIZING, "diarize"),
        StateTransition(ContentState.DIARIZING, ContentState.DIARIZED, "diarize_complete"),
        StateTransition(ContentState.DIARIZING, ContentState.FAILED_TRANSIENT, "diarize_failed"),

        # Parallel phase - transcription
        StateTransition(ContentState.CONVERTED, ContentState.TRANSCRIBING, "transcribe"),
        StateTransition(ContentState.DIARIZED, ContentState.TRANSCRIBING, "transcribe"),  # Can start after diarize too
        StateTransition(ContentState.TRANSCRIBING, ContentState.TRANSCRIBED, "all_chunks_transcribed"),
        StateTransition(ContentState.TRANSCRIBING, ContentState.FAILED_TRANSIENT, "transcribe_failed"),

        # Ready for stitch (both complete)
        StateTransition(ContentState.DIARIZED, ContentState.READY_FOR_STITCH, "transcription_complete",
                       condition=lambda c: c.is_transcribed),
        StateTransition(ContentState.TRANSCRIBED, ContentState.READY_FOR_STITCH, "diarization_complete",
                       condition=lambda c: c.is_diarized),

        # Stitch phase
        StateTransition(ContentState.READY_FOR_STITCH, ContentState.STITCHING, "stitch"),
        StateTransition(ContentState.STITCHING, ContentState.STITCHED, "stitch_complete"),
        StateTransition(ContentState.STITCHING, ContentState.FAILED_TRANSIENT, "stitch_failed"),

        # Cleanup phase
        StateTransition(ContentState.STITCHED, ContentState.COMPRESSING, "cleanup"),
        StateTransition(ContentState.COMPRESSING, ContentState.COMPLETE, "cleanup_complete"),

        # Recovery transitions
        StateTransition(ContentState.FAILED_TRANSIENT, ContentState.NEW, "retry_download"),
        StateTransition(ContentState.FAILED_TRANSIENT, ContentState.DOWNLOADED, "retry_convert"),
        StateTransition(ContentState.FAILED_TRANSIENT, ContentState.CONVERTED, "retry_diarize"),
        StateTransition(ContentState.FAILED_TRANSIENT, ContentState.CONVERTED, "retry_transcribe"),
        StateTransition(ContentState.FAILED_TRANSIENT, ContentState.READY_FOR_STITCH, "retry_stitch"),
    ]

    # Failure recovery policies
    RECOVERY_POLICIES: List[FailureRecovery] = [
        # Download failures
        FailureRecovery(ContentState.DOWNLOADING, FailureType.TRANSIENT,
                       ContentState.NEW, max_retries=3, action="wait_and_retry"),
        FailureRecovery(ContentState.DOWNLOADING, FailureType.PERMANENT,
                       ContentState.BLOCKED, max_retries=0, action="block_content"),

        # Convert failures
        FailureRecovery(ContentState.CONVERTING, FailureType.TRANSIENT,
                       ContentState.DOWNLOADED, max_retries=3, action="retry_convert"),
        FailureRecovery(ContentState.CONVERTING, FailureType.PREREQUISITE,
                       ContentState.NEW, max_retries=1, action="redownload"),

        # Diarize failures
        FailureRecovery(ContentState.DIARIZING, FailureType.TRANSIENT,
                       ContentState.CONVERTED, max_retries=3, action="retry_diarize"),

        # Transcribe failures
        FailureRecovery(ContentState.TRANSCRIBING, FailureType.TRANSIENT,
                       ContentState.CONVERTED, max_retries=3, action="retry_transcribe_chunk"),
        FailureRecovery(ContentState.TRANSCRIBING, FailureType.PREREQUISITE,
                       ContentState.CONVERTED, max_retries=1, action="reextract_chunk"),

        # Stitch failures
        FailureRecovery(ContentState.STITCHING, FailureType.TRANSIENT,
                       ContentState.READY_FOR_STITCH, max_retries=3, action="retry_stitch"),
        FailureRecovery(ContentState.STITCHING, FailureType.PREREQUISITE,
                       ContentState.CONVERTED, max_retries=1, action="check_prerequisites"),
    ]

    # Map task types to their associated states
    TASK_STATE_MAP = {
        'download_youtube': (ContentState.DOWNLOADING, ContentState.DOWNLOADED),
        'download_podcast': (ContentState.DOWNLOADING, ContentState.DOWNLOADED),
        'download_rumble': (ContentState.DOWNLOADING, ContentState.DOWNLOADED),
        'convert': (ContentState.CONVERTING, ContentState.CONVERTED),
        'diarize': (ContentState.DIARIZING, ContentState.DIARIZED),
        'transcribe': (ContentState.TRANSCRIBING, ContentState.TRANSCRIBED),
        'stitch': (ContentState.STITCHING, ContentState.STITCHED),
        'cleanup': (ContentState.COMPRESSING, ContentState.COMPLETE),
    }

    def __init__(self):
        # Build transition lookup for fast access
        self._transitions_from: Dict[ContentState, List[StateTransition]] = {}
        for t in self.TRANSITIONS:
            if t.from_state not in self._transitions_from:
                self._transitions_from[t.from_state] = []
            self._transitions_from[t.from_state].append(t)

        # Build recovery lookup
        self._recovery_lookup: Dict[tuple, FailureRecovery] = {}
        for r in self.RECOVERY_POLICIES:
            self._recovery_lookup[(r.state, r.failure_type)] = r

    def determine_state(
        self,
        is_downloaded: bool,
        is_converted: bool,
        is_diarized: bool,
        is_transcribed: bool,
        is_stitched: bool,
        is_compressed: bool,
        blocked_download: bool,
        has_pending_tasks: bool = False,
        pending_task_type: Optional[str] = None
    ) -> ContentState:
        """
        Determine the current state of content based on flags.

        This is the key function that maps database state to FSM state.
        """
        # Check blocked first
        if blocked_download:
            return ContentState.BLOCKED

        # Check for in-progress states (if we have pending task info)
        if has_pending_tasks and pending_task_type:
            task_states = self.TASK_STATE_MAP.get(pending_task_type)
            if task_states:
                return task_states[0]  # Return the "in progress" state

        # Determine state from completion flags (work backwards from most complete)
        if is_compressed:
            return ContentState.COMPLETE
        if is_stitched:
            return ContentState.STITCHED
        if is_diarized and is_transcribed:
            return ContentState.READY_FOR_STITCH
        if is_transcribed and not is_diarized:
            return ContentState.TRANSCRIBED
        if is_diarized and not is_transcribed:
            return ContentState.DIARIZED
        if is_converted:
            return ContentState.CONVERTED
        if is_downloaded:
            return ContentState.DOWNLOADED

        return ContentState.NEW

    def get_next_action(self, state: ContentState) -> Optional[str]:
        """
        Get the next action to take from a given state.

        Returns the task type to create, or None if no action needed.
        """
        action_map = {
            ContentState.NEW: 'download',
            ContentState.DOWNLOADED: 'convert',
            ContentState.CONVERTED: 'diarize',  # Also transcribe in parallel
            ContentState.DIARIZED: 'transcribe' if not self._is_transcribed else None,
            ContentState.TRANSCRIBED: None,  # Wait for diarization
            ContentState.READY_FOR_STITCH: 'stitch',
            ContentState.STITCHED: 'cleanup',
            ContentState.COMPLETE: None,
            ContentState.BLOCKED: None,
        }
        return action_map.get(state)

    def get_required_tasks(self, state: ContentState, content_flags: Dict[str, bool]) -> List[str]:
        """
        Get all tasks that should exist for the given state.

        Returns list of task types that should be created/running.
        """
        tasks = []

        if state == ContentState.NEW:
            tasks.append('download')
        elif state == ContentState.DOWNLOADED:
            tasks.append('convert')
        elif state == ContentState.CONVERTED:
            tasks.append('diarize')
            # Transcribe can run in parallel if diarization is done or ignored
            if content_flags.get('diarization_ignored') or content_flags.get('is_diarized'):
                tasks.append('transcribe')
        elif state == ContentState.DIARIZED:
            if not content_flags.get('is_transcribed'):
                tasks.append('transcribe')
        elif state == ContentState.READY_FOR_STITCH:
            tasks.append('stitch')
        elif state == ContentState.STITCHED:
            if not content_flags.get('is_compressed'):
                tasks.append('cleanup')

        return tasks

    def get_recovery_action(
        self,
        state: ContentState,
        failure_type: FailureType,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Get the recovery action for a failure.

        Returns dict with:
        - recovery_state: State to transition to
        - action: Action to take
        - should_retry: Whether to retry
        - block: Whether to block content
        """
        recovery = self._recovery_lookup.get((state, failure_type))

        if not recovery:
            # Default: stay in failed state
            return {
                'recovery_state': ContentState.FAILED_PERMANENT,
                'action': 'mark_failed',
                'should_retry': False,
                'block': False
            }

        should_retry = retry_count < recovery.max_retries

        return {
            'recovery_state': recovery.recovery_state if should_retry else ContentState.FAILED_PERMANENT,
            'action': recovery.action,
            'should_retry': should_retry,
            'block': failure_type == FailureType.PERMANENT
        }

    def classify_error(self, task_type: str, error_message: str) -> FailureType:
        """
        Classify an error message into a failure type.

        This centralizes error classification logic.
        """
        # Permanent errors
        permanent_patterns = [
            'Join this channel to get access to members-only content',
            'Sign in to confirm your age',
            'Video unavailable',
            'This video is private',
            'Private video',
            'Bad URL',
            'HTTP Error 400',
            'HTTP Error 403',
        ]
        if any(p in error_message for p in permanent_patterns):
            return FailureType.PERMANENT

        # Prerequisite errors
        prerequisite_patterns = [
            'No audio file found',
            'No diarization data',
            'No transcription data',
            'Missing prerequisite',
            'File not found',
        ]
        if any(p in error_message for p in prerequisite_patterns):
            return FailureType.PREREQUISITE

        # Resource errors
        resource_patterns = [
            'Out of memory',
            'No space left on device',
            'Resource temporarily unavailable',
        ]
        if any(p in error_message for p in resource_patterns):
            return FailureType.RESOURCE

        # Default to transient (network issues, timeouts, etc.)
        return FailureType.TRANSIENT

    def validate_transition(
        self,
        from_state: ContentState,
        to_state: ContentState,
        trigger: str
    ) -> bool:
        """
        Validate that a state transition is allowed.
        """
        valid_transitions = self._transitions_from.get(from_state, [])
        for t in valid_transitions:
            if t.to_state == to_state and t.trigger == trigger:
                return True
        return False

    def get_state_info(self, state: ContentState) -> Dict[str, Any]:
        """
        Get information about a state for debugging/logging.
        """
        valid_next = self._transitions_from.get(state, [])
        return {
            'state': state.name,
            'valid_transitions': [
                {'to': t.to_state.name, 'trigger': t.trigger}
                for t in valid_next
            ],
            'next_action': self.get_next_action(state),
            'is_terminal': state in [ContentState.COMPLETE, ContentState.BLOCKED],
            'is_error': state in [ContentState.FAILED_TRANSIENT, ContentState.FAILED_PERMANENT],
        }


# Convenience function for quick state determination
def get_content_state(content) -> ContentState:
    """
    Quick helper to get state from a Content object.
    """
    sm = ContentStateMachine()
    return sm.determine_state(
        is_downloaded=content.is_downloaded,
        is_converted=content.is_converted,
        is_diarized=content.is_diarized,
        is_transcribed=content.is_transcribed,
        is_stitched=content.is_stitched,
        is_compressed=content.is_compressed,
        blocked_download=content.blocked_download,
    )
