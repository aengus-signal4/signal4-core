# Pipeline Manager Refactoring Plan

**File**: `src/processing/pipeline_manager.py`
**Current Size**: 2088 lines
**Target Size**: ~200 lines (coordinator) + modular components
**Date**: 2026-01-17

---

## Executive Summary

The `PipelineManager` class is a monolith handling 6+ distinct responsibilities. This plan breaks it into focused, testable modules while maintaining backward compatibility with the orchestrator.

---

## Current Architecture Problems

### 1. Single Responsibility Violations

| Responsibility | Lines | Complexity |
|----------------|-------|------------|
| Error handling & task recreation | 284-590 | High - legacy + new paths |
| Content state evaluation | 1098-1896 | Very High - 800 lines |
| Task creation logic | 849-1069 | Medium |
| S3 file checking | 1163-1226, 1956-1999 | Medium - duplicated |
| Worker behavior calculation | 754-802 | Low |
| Chunk management | 1070-1096 | Low |

### 2. Code Duplication

- S3 file existence checks duplicated in `evaluate_content_state()` and `bulk_reconcile_content_states()`
- Error handling patterns repeated across legacy and new paths
- Task creation boilerplate repeated for each task type

### 3. Testing Challenges

- Cannot unit test state evaluation without mocking S3
- Error policies are embedded in method logic
- No clear seams for dependency injection

---

## Proposed Module Structure

```
src/processing/
├── pipeline_manager.py          # Coordinator only (~200 lines)
├── content_state/
│   ├── __init__.py
│   ├── file_checker.py          # S3 file existence checks (~150 lines)
│   ├── flag_updater.py          # Database flag updates (~200 lines)
│   └── reconciler.py            # State reconciliation (~400 lines)
├── task_creation/
│   ├── __init__.py
│   ├── factory.py               # Task creation base (~150 lines)
│   └── strategies.py            # Per-task-type strategies (~300 lines)
├── error_handling/
│   ├── __init__.py
│   ├── policies.py              # Error classification (~100 lines)
│   └── actions.py               # Error response actions (~150 lines)
└── error_handler.py             # Existing file (keep)
```

---

## Phase 1: Extract ContentFileChecker (Low Risk)

**Goal**: Centralize S3 file existence logic into a reusable, testable class.

### New File: `src/processing/content_state/file_checker.py`

```python
"""
Content file existence checker for S3 storage.

Provides a single source of truth for determining what files exist
for a given content item, handling both original and compressed formats.
"""

from dataclasses import dataclass
from typing import Set, Optional
import re


@dataclass
class ContentFiles:
    """Result of checking files for a content item."""
    content_id: str
    prefix: str

    # Core files
    audio_exists: bool
    source_exists: bool
    storage_manifest_exists: bool

    # Processing outputs
    diarization_exists: bool
    speaker_embeddings_exist: bool
    speaker_mapping_exists: bool
    stitched_exists: bool
    semantic_segments_exist: bool

    # Chunk information
    chunk_indices_with_audio: Set[int]
    chunk_indices_with_transcript: Set[int]

    # Raw file list for custom checks
    all_files: Set[str]

    @property
    def is_downloadable(self) -> bool:
        """Content has source/audio or was compressed (manifest exists)."""
        return self.source_exists or self.audio_exists or self.storage_manifest_exists

    @property
    def is_convertible(self) -> bool:
        """Content has audio and chunk files."""
        return self.audio_exists and len(self.chunk_indices_with_audio) > 0


class ContentFileChecker:
    """
    Checks S3 storage for content files.

    Handles both original and compressed file formats (.gz).
    Thread-safe and stateless - can be shared across calls.
    """

    # File extensions for source files
    SOURCE_EXTENSIONS = ['.mp4', '.mp3', '.wav', '.m4a']
    VIDEO_EXTENSIONS = ['.mp4', '.webm']
    AUDIO_EXTENSIONS = ['.wav', '.opus', '.mp3']

    def __init__(self):
        # Precompile regex for chunk detection
        self._chunk_audio_pattern = re.compile(r'chunks/(\d+)/audio\.wav')
        self._chunk_transcript_pattern = re.compile(r'chunks/(\d+)/transcript_words\.json(\.gz)?')

    def check(self, all_files: Set[str], content_id: str) -> ContentFiles:
        """
        Check what files exist for a content item.

        Args:
            all_files: Set of all file paths from S3 listing
            content_id: The content ID to check

        Returns:
            ContentFiles dataclass with existence flags
        """
        prefix = f"content/{content_id}/"

        # Filter to only this content's files for efficiency
        content_files = {f for f in all_files if f.startswith(prefix)}

        return ContentFiles(
            content_id=content_id,
            prefix=prefix,

            # Core files
            audio_exists=self._check_audio(content_files, prefix),
            source_exists=self._check_source(content_files, prefix),
            storage_manifest_exists=f"{prefix}storage_manifest.json" in content_files,

            # Processing outputs
            diarization_exists=self._check_json_file(content_files, prefix, "diarization.json"),
            speaker_embeddings_exist=self._check_json_file(content_files, prefix, "speaker_embeddings.json"),
            speaker_mapping_exists=self._check_json_file(content_files, prefix, "speaker_mapping.json"),
            stitched_exists=self._check_json_file(content_files, prefix, "transcript_diarized.json"),
            semantic_segments_exist=self._check_json_file(content_files, prefix, "semantic_segments.json"),

            # Chunks
            chunk_indices_with_audio=self._get_chunk_indices(content_files, self._chunk_audio_pattern),
            chunk_indices_with_transcript=self._get_chunk_indices(content_files, self._chunk_transcript_pattern),

            all_files=content_files
        )

    def _check_audio(self, files: Set[str], prefix: str) -> bool:
        """Check if any audio file exists (original or compressed)."""
        return any(
            f"{prefix}audio{ext}" in files
            for ext in self.AUDIO_EXTENSIONS
        )

    def _check_source(self, files: Set[str], prefix: str) -> bool:
        """Check if source or video file exists."""
        has_source = any(
            f"{prefix}source{ext}" in files
            for ext in self.SOURCE_EXTENSIONS
        )
        has_video = any(
            f"{prefix}video{ext}" in files
            for ext in self.VIDEO_EXTENSIONS
        )
        return has_source or has_video

    def _check_json_file(self, files: Set[str], prefix: str, filename: str) -> bool:
        """Check if JSON file exists (original or gzipped)."""
        return (
            f"{prefix}{filename}" in files or
            f"{prefix}{filename}.gz" in files
        )

    def _get_chunk_indices(self, files: Set[str], pattern: re.Pattern) -> Set[int]:
        """Extract chunk indices matching a pattern."""
        indices = set()
        for f in files:
            if match := pattern.search(f):
                indices.add(int(match.group(1)))
        return indices
```

### Migration Steps

1. Create `src/processing/content_state/__init__.py`
2. Create `src/processing/content_state/file_checker.py` with above code
3. Add `ContentFileChecker` instance to `PipelineManager.__init__()`
4. Replace inline file checks in `evaluate_content_state()` with `self.file_checker.check()`
5. Replace inline file checks in `bulk_reconcile_content_states()` with `self.file_checker.check()`
6. Run existing tests to verify no regression

### Lines Removed: ~120 (duplicated file checking logic)

---

## Phase 2: Remove Legacy Error Handling (Medium Risk)

**Goal**: Delete the legacy error handling paths and ensure all tasks use error codes.

### Current Legacy Paths (Lines 433-590)

These are all marked with `logger.warning("Legacy ... handling")`:

| Pattern | Error Code to Use | Task Type |
|---------|-------------------|-----------|
| `error_type == 'youtube_auth'` | `YOUTUBE_AUTH` | download_youtube |
| `corrupt_media_detected` | `CORRUPT_MEDIA` | convert |
| `"No audio file found"` | `MISSING_AUDIO` | stitch |
| `permanent` flag | Various | all |
| `"Diarization pipeline returned empty"` | `EMPTY_RESULT` | diarize |
| `"Bad URL"` | `BAD_URL` | download_podcast |
| `"members-only content"` | `MEMBERS_ONLY` | download_youtube |
| `"Sign in to confirm your age"` | `AGE_RESTRICTED` | download_youtube |

### Migration Steps

1. **Audit task processors** - Verify each task type returns proper `error_code` in result
2. **Update task processors** that still use legacy patterns:
   - `src/processing_steps/download_youtube.py`
   - `src/processing_steps/convert.py`
   - `src/processing_steps/stitch_steps/`
   - `src/processing_steps/diarize.py`
   - `src/processing_steps/download_podcast.py`
3. **Delete legacy handling** (lines 433-590 in pipeline_manager.py)
4. **Test each task type** to ensure errors are properly handled

### Lines Removed: ~160

---

## Phase 3: Extract StateReconciler (Medium Risk)

**Goal**: Move the 800-line `evaluate_content_state()` into a focused class.

### New File: `src/processing/content_state/reconciler.py`

```python
"""
Content state reconciliation - ensures database matches S3 reality.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime, timezone
import logging

from sqlalchemy.orm import Session

from src.database.models import Content, ContentChunk, EmbeddingSegment, Sentence
from src.processing.content_state.file_checker import ContentFileChecker, ContentFiles
from src.storage.s3_utils import S3Storage

if TYPE_CHECKING:
    from src.processing.task_creation.factory import TaskFactory

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    """Result of reconciling content state."""
    content_id: str
    flags_updated: List[str]
    tasks_created: List[str]
    tasks_blocked: List[str]
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'content_id': self.content_id,
            'flags_updated': self.flags_updated,
            'tasks_created': self.tasks_created,
            'tasks_blocked': self.tasks_blocked,
            'errors': self.errors
        }


class FlagUpdater:
    """Updates content database flags based on file existence."""

    def __init__(self, config: Dict[str, Any], s3_storage: S3Storage):
        self.config = config
        self.s3_storage = s3_storage
        self._safe_models_english = config.get('processing', {}).get(
            'transcription', {}
        ).get('safe_models_english', [])
        self._safe_models_other = config.get('processing', {}).get(
            'transcription', {}
        ).get('safe_models_other', [])

    def get_current_stitch_version(self) -> str:
        """Get current stitch version from config."""
        return self.config.get('processing', {}).get(
            'stitch', {}
        ).get('current_version', 'stitch_v1')

    def update_download_flag(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> Optional[str]:
        """Update is_downloaded flag. Returns flag name if updated."""
        should_be_downloaded = files.is_downloadable

        if should_be_downloaded != content.is_downloaded:
            logger.info(
                f"State change for {content.content_id}: "
                f"is_downloaded {content.is_downloaded} -> {should_be_downloaded}"
            )
            content.is_downloaded = should_be_downloaded
            content.last_updated = datetime.now(timezone.utc)
            session.add(content)
            session.commit()
            session.refresh(content)
            return 'is_downloaded'
        return None

    def update_convert_flag(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> Optional[str]:
        """Update is_converted flag. Returns flag name if updated."""
        # Complex logic for determining conversion state...
        # (Extract from current evaluate_content_state lines 1253-1299)
        pass

    def update_transcription_flag(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> Optional[str]:
        """Update is_transcribed flag based on chunk status."""
        pass

    def update_diarization_flag(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> Optional[str]:
        """Update is_diarized flag."""
        if files.diarization_exists != content.is_diarized:
            content.is_diarized = files.diarization_exists
            return 'is_diarized'
        return None

    def update_stitch_flag(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> Optional[str]:
        """Update is_stitched flag with version check."""
        pass

    def update_embedded_flag(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> Optional[str]:
        """Update is_embedded flag."""
        pass

    def update_compressed_flag(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> Optional[str]:
        """Update is_compressed flag."""
        if files.storage_manifest_exists != content.is_compressed:
            content.is_compressed = files.storage_manifest_exists
            return 'is_compressed'
        return None


class StateReconciler:
    """
    Reconciles database state with S3 storage reality.

    This class:
    1. Checks what files actually exist in S3
    2. Updates database flags to match reality
    3. Creates any tasks needed based on actual state
    """

    def __init__(
        self,
        config: Dict[str, Any],
        file_checker: ContentFileChecker,
        flag_updater: FlagUpdater,
        task_factory: 'TaskFactory',
        s3_storage: S3Storage
    ):
        self.config = config
        self.file_checker = file_checker
        self.flag_updater = flag_updater
        self.task_factory = task_factory
        self.s3_storage = s3_storage

    async def reconcile(
        self,
        session: Session,
        content: Content
    ) -> ReconciliationResult:
        """
        Reconcile a single content item's state.

        Args:
            session: Active database session
            content: Content object to reconcile

        Returns:
            ReconciliationResult with actions taken
        """
        result = ReconciliationResult(
            content_id=content.content_id,
            flags_updated=[],
            tasks_created=[],
            tasks_blocked=[],
            errors=[]
        )

        try:
            # Step 1: Get file listing from S3
            content_prefix = f"content/{content.content_id}/"
            all_files = set(self.s3_storage.list_files(content_prefix))
            files = self.file_checker.check(all_files, content.content_id)

            # Step 2: Update database flags
            flag_updates = [
                self.flag_updater.update_download_flag(session, content, files),
                self.flag_updater.update_convert_flag(session, content, files),
                self.flag_updater.update_transcription_flag(session, content, files),
                self.flag_updater.update_diarization_flag(session, content, files),
                self.flag_updater.update_stitch_flag(session, content, files),
                self.flag_updater.update_embedded_flag(session, content, files),
                self.flag_updater.update_compressed_flag(session, content, files),
            ]
            result.flags_updated = [f for f in flag_updates if f]

            # Step 3: Create missing tasks
            await self._create_missing_tasks(session, content, files, result)

            # Commit changes
            if result.flags_updated or result.tasks_created:
                content.last_updated = datetime.now(timezone.utc)
                session.add(content)
                session.commit()

        except Exception as e:
            logger.error(f"Error reconciling {content.content_id}: {e}", exc_info=True)
            result.errors.append(str(e))

        return result

    async def _create_missing_tasks(
        self,
        session: Session,
        content: Content,
        files: ContentFiles,
        result: ReconciliationResult
    ) -> None:
        """Create any tasks needed based on current state."""
        project = content.projects[0] if content.projects else 'unknown'

        # Download task
        if not files.source_exists and not files.audio_exists and not content.blocked_download:
            task_id, reason = await self.task_factory.create_download_task(
                session, content, project
            )
            if task_id:
                result.tasks_created.append(f"download_{content.platform}")
            elif reason:
                result.tasks_blocked.append(f"download_{content.platform}: {reason}")

        # Convert task
        if files.source_exists and not files.audio_exists:
            task_id, reason = await self.task_factory.create_convert_task(
                session, content, project
            )
            if task_id:
                result.tasks_created.append('convert')
            elif reason:
                result.tasks_blocked.append(f"convert: {reason}")

        # ... similar patterns for diarize, transcribe, stitch, cleanup
```

### Migration Steps

1. Create `FlagUpdater` class with methods extracted from `evaluate_content_state()`
2. Create `StateReconciler` class that orchestrates the reconciliation
3. Update `PipelineManager` to use `StateReconciler`
4. Keep `evaluate_content_state()` as a thin wrapper that calls `StateReconciler.reconcile()`
5. Migrate `bulk_reconcile_content_states()` to use `StateReconciler`

### Lines Moved: ~800 (into focused classes)

---

## Phase 4: Extract TaskFactory (Low Risk)

**Goal**: Centralize task creation logic with per-task-type strategies.

### New File: `src/processing/task_creation/factory.py`

```python
"""
Task creation factory with strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from src.database.models import Content, TaskQueue
from src.utils.priority import calculate_priority_by_date

TASK_STATUS_PENDING = 'pending'
TASK_STATUS_PROCESSING = 'processing'
TASK_STATUS_FAILED = 'failed'


class TaskCreationStrategy(ABC):
    """Base class for task creation strategies."""

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return the task type this strategy handles."""
        pass

    @abstractmethod
    def should_create(
        self,
        session: Session,
        content: Content,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if task should be created.

        Returns:
            Tuple of (should_create, reason_if_not)
        """
        pass

    def get_input_data(self, content: Content, project: str, **kwargs) -> Dict[str, Any]:
        """Get input data for the task."""
        return {'project': project}


class DownloadTaskStrategy(TaskCreationStrategy):
    """Strategy for creating download tasks."""

    def __init__(self, platform: str):
        self._platform = platform

    @property
    def task_type(self) -> str:
        return f"download_{self._platform}"

    def should_create(
        self,
        session: Session,
        content: Content,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        if content.blocked_download:
            return False, "Content is blocked for download"

        # Check for too many failed attempts
        failed_count = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.task_type == self.task_type,
            TaskQueue.status == TASK_STATUS_FAILED
        ).count()

        if failed_count >= 3:
            return False, f"Already have {failed_count} failed attempts"

        return True, None


class StitchTaskStrategy(TaskCreationStrategy):
    """Strategy for creating stitch tasks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @property
    def task_type(self) -> str:
        return "stitch"

    def should_create(
        self,
        session: Session,
        content: Content,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        from src.utils.version_utils import should_recreate_stitch_task

        current_version = self.config.get('processing', {}).get(
            'stitch', {}
        ).get('current_version', 'stitch_v1')

        if content.is_stitched and not should_recreate_stitch_task(
            current_version, content.stitch_version
        ):
            return False, f"Already stitched with compatible version {content.stitch_version}"

        if not content.is_transcribed:
            return False, "Not fully transcribed"

        if not content.is_diarized:
            return False, "Not diarized"

        return True, None


class TaskFactory:
    """
    Factory for creating pipeline tasks.

    Uses strategy pattern to encapsulate per-task-type logic.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._strategies: Dict[str, TaskCreationStrategy] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register built-in task strategies."""
        self.register(DownloadTaskStrategy('youtube'))
        self.register(DownloadTaskStrategy('rumble'))
        self.register(DownloadTaskStrategy('podcast'))
        self.register(StitchTaskStrategy(self.config))
        # ... register other strategies

    def register(self, strategy: TaskCreationStrategy):
        """Register a task creation strategy."""
        self._strategies[strategy.task_type] = strategy

    async def create_task(
        self,
        session: Session,
        content: Content,
        task_type: str,
        project: str,
        **kwargs
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Create a task if appropriate.

        Returns:
            Tuple of (task_id, block_reason)
        """
        strategy = self._strategies.get(task_type)
        if not strategy:
            return None, f"Unknown task type: {task_type}"

        # Check if should create
        should_create, reason = strategy.should_create(session, content, **kwargs)
        if not should_create:
            return None, reason

        # Check for existing task
        existing = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.task_type == task_type,
            TaskQueue.status.in_([TASK_STATUS_PENDING, TASK_STATUS_PROCESSING])
        ).first()

        if existing:
            return None, "Task already exists"

        # Create the task
        input_data = strategy.get_input_data(content, project, **kwargs)
        priority = calculate_priority_by_date(content.publish_date)

        task = TaskQueue(
            content_id=content.content_id,
            task_type=task_type,
            status=TASK_STATUS_PENDING,
            priority=priority,
            input_data=input_data,
            created_at=datetime.now(timezone.utc)
        )

        try:
            session.add(task)
            session.flush()
            return task.id, None
        except IntegrityError:
            session.rollback()
            return None, "Task already exists (race condition)"

    # Convenience methods
    async def create_download_task(
        self, session: Session, content: Content, project: str
    ) -> Tuple[Optional[int], Optional[str]]:
        return await self.create_task(
            session, content, f"download_{content.platform}", project
        )

    async def create_stitch_task(
        self, session: Session, content: Content, project: str
    ) -> Tuple[Optional[int], Optional[str]]:
        return await self.create_task(session, content, "stitch", project)
```

### Migration Steps

1. Create `TaskCreationStrategy` base class
2. Implement strategies for each task type
3. Create `TaskFactory` that uses strategies
4. Update `StateReconciler` to use `TaskFactory`
5. Remove `_create_task_if_not_exists()` from `PipelineManager`

### Lines Moved: ~300

---

## Phase 5: Refactor PipelineManager (Final)

**Goal**: Reduce `PipelineManager` to a thin coordinator.

### Final `PipelineManager` Structure

```python
"""
Pipeline Manager - Coordinates content processing workflow.

This is a thin coordinator that delegates to specialized modules:
- StateReconciler: Ensures database matches S3 reality
- TaskFactory: Creates pipeline tasks
- ErrorHandler: Handles task failures
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.database.models import TaskQueue, Content
from src.processing.content_state.reconciler import StateReconciler
from src.processing.content_state.file_checker import ContentFileChecker
from src.processing.content_state.flag_updater import FlagUpdater
from src.processing.task_creation.factory import TaskFactory
from src.processing.error_handler import ErrorHandler
from src.utils.human_behavior import HumanBehaviorManager
from src.storage.s3_utils import S3Storage, S3StorageConfig


class PipelineManager:
    """
    Coordinates content processing workflow after task completion.

    Responsibilities:
    - Route task results to appropriate handlers
    - Coordinate state reconciliation
    - Calculate worker availability times
    """

    VERSION = "3.0.0"

    def __init__(
        self,
        behavior_manager: HumanBehaviorManager,
        config: Dict[str, Any],
        worker_task_failures: Dict[str, Any]
    ):
        self.behavior_manager = behavior_manager
        self.config = config

        # Initialize S3 storage
        s3_config = S3StorageConfig(
            endpoint_url=config['storage']['s3']['endpoint_url'],
            access_key=config['storage']['s3']['access_key'],
            secret_key=config['storage']['s3']['secret_key'],
            bucket_name=config['storage']['s3']['bucket_name'],
            use_ssl=config['storage']['s3']['use_ssl']
        )
        self.s3_storage = S3Storage(s3_config)

        # Initialize components
        self.file_checker = ContentFileChecker()
        self.flag_updater = FlagUpdater(config, self.s3_storage)
        self.task_factory = TaskFactory(config)
        self.error_handler = ErrorHandler(config, worker_task_failures)

        self.state_reconciler = StateReconciler(
            config=config,
            file_checker=self.file_checker,
            flag_updater=self.flag_updater,
            task_factory=self.task_factory,
            s3_storage=self.s3_storage
        )

    async def handle_task_result(
        self,
        session: Session,
        content_id: str,
        task_type: str,
        status: str,
        result: Dict[str, Any],
        db_task: TaskQueue,
        content: Content,
        worker_id: str,
        chunk_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Handle task completion and determine next steps.

        This method:
        1. Handles errors via ErrorHandler
        2. Reconciles state via StateReconciler
        3. Calculates worker availability
        """
        outcome = {
            'available_after': None,
            'outcome_message': "No outcome message set"
        }

        # Early exit for blocked content
        if content.blocked_download:
            outcome['outcome_message'] = "Content blocked - no further processing"
            return outcome

        # Handle failures
        if status in ('failed', 'failed_permanent'):
            error_result = self._handle_failure(
                session, content, task_type, result, db_task, worker_id
            )
            if error_result.get('handled'):
                return error_result

        # Handle successful completion - reconcile state
        reconcile_result = await self.state_reconciler.reconcile(session, content)

        # Calculate worker availability
        outcome['available_after'] = self._calculate_worker_available_after(
            worker_id, task_type, result
        )

        # Build outcome message
        outcome['outcome_message'] = self._build_outcome_message(reconcile_result)

        return outcome

    def _handle_failure(
        self,
        session: Session,
        content: Content,
        task_type: str,
        result: Dict[str, Any],
        db_task: TaskQueue,
        worker_id: str
    ) -> Dict[str, Any]:
        """Handle task failure via ErrorHandler."""
        error_code = result.get('error_code')
        error_message = str(result.get('error', ''))
        error_details = result.get('error_details', {})

        if error_code:
            return self.error_handler.handle_error(
                task_type=task_type,
                error_code=error_code,
                error_message=error_message,
                error_details=error_details,
                content=content,
                session=session,
                worker_id=worker_id
            )

        return {'handled': False}

    def _calculate_worker_available_after(
        self,
        worker_id: str,
        task_type: str,
        result: Dict[str, Any]
    ) -> Optional[datetime]:
        """Calculate when worker should be available."""
        if task_type not in ('download_youtube', 'download_rumble'):
            return None

        if result.get('skip_wait_time') or result.get('skipped_existing'):
            return None

        if not self.behavior_manager:
            return None

        return self.behavior_manager.calculate_next_available_time(worker_id, task_type)

    def _build_outcome_message(self, reconcile_result) -> str:
        """Build human-readable outcome message."""
        parts = []
        if reconcile_result.flags_updated:
            parts.append(f"Updated {', '.join(reconcile_result.flags_updated)}")
        if reconcile_result.tasks_created:
            parts.append(f"Created {', '.join(reconcile_result.tasks_created)}")
        if reconcile_result.tasks_blocked:
            parts.append(f"Blocked {', '.join(reconcile_result.tasks_blocked)}")
        return '. '.join(parts) if parts else "No changes needed"

    async def evaluate_content_state(
        self,
        session: Session,
        content: Content,
        db_task: Optional[TaskQueue] = None
    ) -> Dict[str, Any]:
        """
        Backward-compatible wrapper for state reconciliation.

        Delegates to StateReconciler.reconcile().
        """
        result = await self.state_reconciler.reconcile(session, content)
        return result.to_dict()

    async def bulk_reconcile_content_states(
        self,
        session: Session
    ) -> Dict[str, Any]:
        """
        Bulk reconcile all content states.

        Delegates to StateReconciler with batch processing.
        """
        return await self.state_reconciler.bulk_reconcile(session)
```

---

## Implementation Timeline

| Phase | Description | Risk | Lines Changed | Dependencies |
|-------|-------------|------|---------------|--------------|
| 1 | Extract ContentFileChecker | Low | ~120 removed, ~150 new | None |
| 2 | Remove legacy error handling | Medium | ~160 removed | Task processors updated |
| 3 | Extract StateReconciler | Medium | ~800 moved | Phase 1 |
| 4 | Extract TaskFactory | Low | ~300 moved | None |
| 5 | Refactor PipelineManager | Low | Coordinator only | Phases 1-4 |

**Recommended Order**: 1 → 4 → 2 → 3 → 5

Phases 1 and 4 can be done independently with minimal risk. Phase 2 requires updating task processors first. Phase 3 is the largest change but has clear boundaries. Phase 5 is cleanup after other phases.

---

## Testing Strategy

### Unit Tests (New)

```
tests/processing/
├── content_state/
│   ├── test_file_checker.py      # Test file pattern matching
│   ├── test_flag_updater.py      # Test flag update logic
│   └── test_reconciler.py        # Test reconciliation flow
├── task_creation/
│   ├── test_strategies.py        # Test per-task strategies
│   └── test_factory.py           # Test factory behavior
└── test_pipeline_manager.py      # Integration tests
```

### Key Test Cases

1. **ContentFileChecker**
   - Handles empty file sets
   - Detects compressed files (.gz)
   - Correctly identifies chunk indices
   - Performance with large file sets

2. **FlagUpdater**
   - Each flag updated correctly
   - No changes when state matches
   - Handles concurrent updates

3. **TaskFactory**
   - Respects task limits (3+ failures)
   - Handles race conditions
   - Creates with correct priority

4. **StateReconciler**
   - Full reconciliation flow
   - Handles S3 errors gracefully
   - Creates appropriate tasks

---

## Rollback Plan

Each phase is independently revertible:

1. **Phase 1**: Delete `content_state/file_checker.py`, revert imports
2. **Phase 2**: Re-add legacy handling code block
3. **Phase 3**: Delete `content_state/reconciler.py`, restore inline code
4. **Phase 4**: Delete `task_creation/`, restore `_create_task_if_not_exists()`
5. **Phase 5**: Restore original `PipelineManager` class

Keep the original `pipeline_manager.py` as `pipeline_manager_legacy.py` until all phases complete and are verified in production.

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| `pipeline_manager.py` lines | 2088 | ~200 |
| Cyclomatic complexity (max method) | ~45 | <15 |
| Test coverage | ~0% | >80% |
| Methods > 100 lines | 2 | 0 |
| Duplicated code blocks | 4 | 0 |

---

## Appendix: File-by-File Line Counts

### Before Refactoring

```
src/processing/
└── pipeline_manager.py          2088 lines
```

### After Refactoring

```
src/processing/
├── pipeline_manager.py           200 lines
├── content_state/
│   ├── __init__.py                10 lines
│   ├── file_checker.py           150 lines
│   ├── flag_updater.py           200 lines
│   └── reconciler.py             400 lines
├── task_creation/
│   ├── __init__.py                10 lines
│   ├── factory.py                150 lines
│   └── strategies.py             300 lines
└── error_handler.py              (existing, unchanged)

Total: ~1420 lines (vs 2088 original)
Reduction: ~670 lines (32% smaller)
```

The reduction comes from:
- Removing duplicated S3 file checking (~120 lines)
- Removing legacy error handling (~160 lines)
- Removing redundant comments and debug logging (~200 lines)
- Simplifying control flow (~190 lines)
