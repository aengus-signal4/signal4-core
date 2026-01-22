"""
Embedding Hydrator
==================

Core class for batch generating embeddings for segments.
Called via scheduled tasks (see config/config.yaml under 'scheduled_tasks').

Uses local model loading for all embeddings:
- Primary (0.6B): Qwen/Qwen3-Embedding-0.6B
- Alternative (4B): Qwen/Qwen3-Embedding-4B (for configured projects)
"""

import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
import argparse
import asyncio
import logging
import os
import yaml
from typing import List, Optional
from datetime import datetime, timezone
import numpy as np
from tqdm import tqdm
import gc

from src.utils.logger import setup_worker_logger
from src.database.session import get_session
from src.database.models import Content, EmbeddingSegment
from sentence_transformers import SentenceTransformer
import torch
from collections import namedtuple

logger = setup_worker_logger('embedding_hydrator')

# Define namedtuple ONCE at module level (not inside loop)
SegmentData = namedtuple('SegmentData', ['id', 'text', 'embedding_version', 'embedding_alt_model', 'meta_data'])


class EmbeddingHydrator:
    """Batch generates embeddings for segments."""

    def __init__(self, primary_model_override: Optional[str] = None,
                 alternative_model_override: Optional[str] = None,
                 alternative_dim: Optional[int] = None):
        """
        Initialize the embedding hydrator.

        Args:
            primary_model_override: Override primary model name
            alternative_model_override: Override alternative model name
            alternative_dim: Override alternative model dimension
        """
        # Load config - navigate from src/utils/ up to repo root
        config_path = get_config_path()
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        seg_config = self.config.get('embedding', {}).get('embedding_segmentation', {})

        # Primary embedding model - allow override
        if primary_model_override:
            self.embedding_model = primary_model_override
            logger.info(f"Using primary model override: {self.embedding_model}")
        else:
            self.embedding_model = seg_config.get('embedding_model', 'Qwen/Qwen3-Embedding-0.6B')

        # Alternative embedding model - allow override
        if alternative_model_override:
            self.alternative_model = alternative_model_override
            self.alternative_model_dim = alternative_dim or 2000
            logger.info(f"Using alternative model override: {self.alternative_model} ({self.alternative_model_dim}d)")
        else:
            self.alternative_model = seg_config.get('alternative_model', 'Qwen/Qwen3-Embedding-4B')
            self.alternative_model_dim = seg_config.get('alternative_model_dim', 2000)

        # Determine embedding versions for tracking
        self.primary_embedding_version = self.embedding_model
        if '4B' in self.alternative_model and self.alternative_model_dim:
            self.alternative_embedding_version = f"{self.alternative_model}-{self.alternative_model_dim}d"
        else:
            self.alternative_embedding_version = self.alternative_model

        # Initialize models (lazy loading)
        self.primary_model = None
        self.alternative_model_obj = None

        logger.info(f"Embedding hydrator initialized")
        logger.info(f"Primary model (will load on first use): {self.embedding_model}")
        if self.alternative_model:
            logger.info(f"Alternative model (will load on first use): {self.alternative_model}")

    def cleanup(self):
        """Release all resources - models and clear caches."""
        logger.info("Cleaning up embedding hydrator resources...")

        # Unload primary model
        if self.primary_model is not None:
            del self.primary_model
            self.primary_model = None
            logger.info("Primary model unloaded")

        # Unload alternative model
        if self.alternative_model_obj is not None:
            del self.alternative_model_obj
            self.alternative_model_obj = None
            logger.info("Alternative model unloaded")

        # Force garbage collection and clear GPU caches
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        logger.info("Cleanup complete")

    def _init_primary_model(self):
        """Lazy load primary embedding model on first use."""
        if self.primary_model is not None:
            logger.debug("Primary model already loaded")
            return

        logger.info(f"Lazy loading primary model: {self.embedding_model}...")

        # Support truncate_dim for models that need it
        if 'Qwen3-Embedding-4B' in self.embedding_model:
            # Infer truncate_dim from version string
            if '-' in self.primary_embedding_version and 'd' in self.primary_embedding_version:
                try:
                    dim_str = self.primary_embedding_version.split('-')[-1].replace('d', '')
                    truncate_dim = int(dim_str)
                    self.primary_model = SentenceTransformer(self.embedding_model, trust_remote_code=True, truncate_dim=truncate_dim, local_files_only=True)
                    logger.info(f"Using truncate_dim={truncate_dim} for 4B model")
                except:
                    self.primary_model = SentenceTransformer(self.embedding_model, trust_remote_code=True, local_files_only=True)
            else:
                self.primary_model = SentenceTransformer(self.embedding_model, trust_remote_code=True, local_files_only=True)
        else:
            self.primary_model = SentenceTransformer(self.embedding_model, trust_remote_code=True, local_files_only=True)

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.primary_model = self.primary_model.to(device)

        test_embedding = self.primary_model.encode(["test"], convert_to_numpy=True)
        self.primary_embedding_dim = test_embedding.shape[1]

        logger.info(f"Primary model loaded on {device}, embedding dimension: {self.primary_embedding_dim}")

    def _init_alternative_model(self):
        """Lazy load alternative embedding model on first use (typically 4B)."""
        if self.alternative_model_obj is not None:
            logger.debug("Alternative model already loaded")
            return

        if not self.alternative_model:
            logger.warning("No alternative model configured")
            return

        logger.info(f"Lazy loading alternative model (4B): {self.alternative_model}...")
        logger.info("This may take a moment as the 4B model is large...")

        if 'Qwen3-Embedding-4B' in self.alternative_model:
            self.alternative_model_obj = SentenceTransformer(
                self.alternative_model,
                trust_remote_code=True,
                truncate_dim=self.alternative_model_dim,
                local_files_only=True
            )
        else:
            self.alternative_model_obj = SentenceTransformer(
                self.alternative_model,
                trust_remote_code=True,
                local_files_only=True
            )

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.alternative_model_obj = self.alternative_model_obj.to(device)

        test_embedding = self.alternative_model_obj.encode(["test"], convert_to_numpy=True)
        embedding_dim = test_embedding.shape[1]

        logger.info(f"Alternative model loaded on {device}, embedding dimension: {embedding_dim}")

    def _should_use_alternative_for_project(self, project_name: str) -> bool:
        """Check if a project requires alternative embeddings."""
        active_projects = self.config.get('active_projects', {})
        project_config = active_projects.get(project_name, {})
        return project_config.get('use_alternative_embeddings', False)

    def _should_use_alternative_for_content(self, content) -> bool:
        """Check if content needs alternative embeddings (project config OR recent content).

        Returns True if:
        1. Content's project has use_alternative_embeddings=true, OR
        2. Content was published in the last 30 days (regardless of project)
        """
        from datetime import datetime, timedelta, timezone

        # Check project config first
        if hasattr(content, 'projects') and content.projects:
            # Handle both string and array project fields
            if isinstance(content.projects, str):
                project_list = [content.projects] if content.projects else []
            else:
                project_list = content.projects if content.projects else []

            for project in project_list:
                if self._should_use_alternative_for_project(project):
                    return True

        # Check if content is recent (published in last 30 days)
        if hasattr(content, 'publish_date') and content.publish_date:
            try:
                # Handle both datetime and string publish_date
                if isinstance(content.publish_date, str):
                    from dateutil import parser
                    publish_date = parser.parse(content.publish_date)
                else:
                    publish_date = content.publish_date

                # Make timezone-aware if needed
                if publish_date.tzinfo is None:
                    publish_date = publish_date.replace(tzinfo=timezone.utc)

                cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)

                if publish_date >= cutoff_date:
                    return True
            except Exception as e:
                logger.warning(f"Error checking publish_date for content: {e}")

        return False

    def _count_missing_embeddings(self, project: Optional[str] = None,
                                  primary: bool = True,
                                  alternative: bool = False,
                                  stitch_versions: Optional[dict] = None,
                                  recent_content_days: Optional[int] = None) -> int:
        """Count how many segments need embeddings using version columns (fast).

        Args:
            recent_content_days: If set, include content published within this many days (for alternative embeddings)
        """
        from datetime import datetime, timedelta, timezone

        with get_session() as session:
            query = session.query(EmbeddingSegment)

            if project:
                # Single project filter - use PostgreSQL array operator
                from sqlalchemy import any_
                content_query = session.query(Content.id).filter(
                    project == any_(Content.projects)
                )
                if stitch_versions:
                    from sqlalchemy import or_
                    version_filters = []
                    # Add exact matches
                    if stitch_versions.get('exact'):
                        version_filters.extend([Content.stitch_version == v for v in stitch_versions['exact']])
                    # Add prefix matches
                    if stitch_versions.get('prefix'):
                        version_filters.extend([Content.stitch_version.like(f"{v}%") for v in stitch_versions['prefix']])
                    if version_filters:
                        content_query = content_query.filter(or_(*version_filters))

                content_ids = content_query.all()
                content_ids = [c.id for c in content_ids]
                query = query.filter(EmbeddingSegment.content_id.in_(content_ids))
            elif alternative and not primary:
                # Alternative embeddings: filter to projects that need them OR recent content
                active_projects = self.config.get('active_projects', {})
                projects_needing_alt = [
                    proj_name for proj_name, proj_config in active_projects.items()
                    if proj_config.get('use_alternative_embeddings', False)
                ]

                # Build query for projects OR recent content
                from sqlalchemy import or_, any_
                conditions = []

                # Add project filters
                if projects_needing_alt:
                    project_filters = [
                        proj == any_(Content.projects) for proj in projects_needing_alt
                    ]
                    conditions.append(or_(*project_filters))

                # Add recent content filter
                if recent_content_days:
                    cutoff_date = datetime.now(timezone.utc) - timedelta(days=recent_content_days)
                    conditions.append(Content.publish_date >= cutoff_date)

                if conditions:
                    content_query = session.query(Content.id).filter(or_(*conditions))

                    if stitch_versions:
                        version_filters = []
                        # Add exact matches
                        if stitch_versions.get('exact'):
                            version_filters.extend([Content.stitch_version == v for v in stitch_versions['exact']])
                        # Add prefix matches
                        if stitch_versions.get('prefix'):
                            version_filters.extend([Content.stitch_version.like(f"{v}%") for v in stitch_versions['prefix']])
                        if version_filters:
                            content_query = content_query.filter(or_(*version_filters))

                    content_ids = content_query.all()
                    content_ids = [c.id for c in content_ids]
                    query = query.filter(EmbeddingSegment.content_id.in_(content_ids))
                else:
                    # No projects need alternative embeddings and no recent content filter
                    return 0
            elif stitch_versions:
                # No project specified but stitch versions provided - filter all content
                from sqlalchemy import or_
                version_filters = []
                # Add exact matches
                if stitch_versions.get('exact'):
                    version_filters.extend([Content.stitch_version == v for v in stitch_versions['exact']])
                # Add prefix matches
                if stitch_versions.get('prefix'):
                    version_filters.extend([Content.stitch_version.like(f"{v}%") for v in stitch_versions['prefix']])
                if version_filters:
                    content_ids = session.query(Content.id).filter(or_(*version_filters)).all()
                    content_ids = [c.id for c in content_ids]
                    query = query.filter(EmbeddingSegment.content_id.in_(content_ids))

            # Exclude segments that are too long or already marked as skipped
            from sqlalchemy import func
            query = query.filter(
                func.length(EmbeddingSegment.text) <= 3000,
                ~EmbeddingSegment.embedding_version.like('SKIPPED%') | EmbeddingSegment.embedding_version.is_(None)
            )

            if primary and not alternative:
                # Count segments with wrong/missing version (uses btree index)
                query = query.filter(
                    (EmbeddingSegment.embedding_version.is_(None)) |
                    (EmbeddingSegment.embedding_version != self.primary_embedding_version)
                )
            elif alternative and not primary:
                # Count segments missing alternative embeddings (but have primary)
                query = query.filter(
                    EmbeddingSegment.embedding_version.isnot(None),
                    (EmbeddingSegment.embedding_alt_model.is_(None)) |
                    (EmbeddingSegment.embedding_alt_model != self.alternative_embedding_version)
                )

            return query.count()

    def _get_segment_ids_needing_embeddings(self, content_id: Optional[str] = None,
                                            project: Optional[str] = None,
                                            primary_only: bool = False,
                                            alternative_only: bool = False,
                                            force_reembed: bool = False,
                                            stitch_versions: Optional[dict] = None) -> List[int]:
        """Get IDs of segments that need embeddings generated or re-generated."""
        with get_session() as session:
            query = session.query(EmbeddingSegment.id)

            if content_id:
                # Get specific content
                content = session.query(Content).filter_by(content_id=content_id).first()
                if content:
                    query = query.filter(EmbeddingSegment.content_id == content.id)
                else:
                    logger.warning(f"Content {content_id} not found")
                    return []

            if project:
                # Get all content for this project - use PostgreSQL array operator
                from sqlalchemy import any_
                content_query = session.query(Content.id).filter(
                    project == any_(Content.projects)
                )
                # Filter by stitch version if provided
                if stitch_versions:
                    from sqlalchemy import or_
                    version_filters = []
                    # Add exact matches
                    if stitch_versions.get('exact'):
                        version_filters.extend([Content.stitch_version == v for v in stitch_versions['exact']])
                    # Add prefix matches
                    if stitch_versions.get('prefix'):
                        version_filters.extend([Content.stitch_version.like(f"{v}%") for v in stitch_versions['prefix']])
                    if version_filters:
                        content_query = content_query.filter(or_(*version_filters))

                content_ids = content_query.all()
                content_ids = [c.id for c in content_ids]
                query = query.filter(EmbeddingSegment.content_id.in_(content_ids))
            elif alternative_only:
                # For alternative embeddings: include projects with config OR content from last 30 days
                from datetime import datetime, timedelta, timezone
                from sqlalchemy import or_, any_

                # Get projects needing alt embeddings from config
                active_projects = self.config.get('active_projects', {})
                projects_needing_alt = [
                    proj_name for proj_name, proj_config in active_projects.items()
                    if proj_config.get('use_alternative_embeddings', False)
                ]

                conditions = []

                # Add project filters
                if projects_needing_alt:
                    project_filters = [
                        proj == any_(Content.projects) for proj in projects_needing_alt
                    ]
                    conditions.append(or_(*project_filters))

                # Add recent content filter (last 30 days)
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
                conditions.append(Content.publish_date >= cutoff_date)

                # Combine with OR
                if conditions:
                    content_query = session.query(Content.id).filter(or_(*conditions))

                    # Filter by stitch version if provided
                    if stitch_versions:
                        version_filters = []
                        # Add exact matches
                        if stitch_versions.get('exact'):
                            version_filters.extend([Content.stitch_version == v for v in stitch_versions['exact']])
                        # Add prefix matches
                        if stitch_versions.get('prefix'):
                            version_filters.extend([Content.stitch_version.like(f"{v}%") for v in stitch_versions['prefix']])
                        if version_filters:
                            content_query = content_query.filter(or_(*version_filters))

                    content_ids = content_query.all()
                    content_ids = [c.id for c in content_ids]
                    query = query.filter(EmbeddingSegment.content_id.in_(content_ids))
            elif stitch_versions:
                # No project specified but stitch versions provided - filter all content
                from sqlalchemy import or_
                version_filters = []
                # Add exact matches
                if stitch_versions.get('exact'):
                    version_filters.extend([Content.stitch_version == v for v in stitch_versions['exact']])
                # Add prefix matches
                if stitch_versions.get('prefix'):
                    version_filters.extend([Content.stitch_version.like(f"{v}%") for v in stitch_versions['prefix']])
                if version_filters:
                    content_ids = session.query(Content.id).filter(or_(*version_filters)).all()
                    content_ids = [c.id for c in content_ids]
                    query = query.filter(EmbeddingSegment.content_id.in_(content_ids))

            # Filter based on what needs to be generated
            if force_reembed:
                # Force re-embed: check if embedding_version doesn't match target
                if primary_only:
                    query = query.filter(
                        (EmbeddingSegment.embedding.is_(None)) |
                        (EmbeddingSegment.embedding_version != self.primary_embedding_version)
                    )
                elif alternative_only:
                    query = query.filter(
                        (EmbeddingSegment.embedding_alt.is_(None)) |
                        (EmbeddingSegment.embedding_alt_model != self.alternative_embedding_version)
                    )
                else:
                    # Both: check versions for both columns
                    query = query.filter(
                        (EmbeddingSegment.embedding.is_(None)) |
                        (EmbeddingSegment.embedding_version != self.primary_embedding_version) |
                        (EmbeddingSegment.embedding_alt.is_(None)) |
                        (EmbeddingSegment.embedding_alt_model != self.alternative_embedding_version)
                    )
            else:
                # Normal mode: generate missing embeddings OR update wrong versions
                if primary_only:
                    query = query.filter(
                        (EmbeddingSegment.embedding.is_(None)) |
                        (EmbeddingSegment.embedding_version != self.primary_embedding_version)
                    )
                elif alternative_only:
                    query = query.filter(
                        EmbeddingSegment.embedding.isnot(None),
                        (EmbeddingSegment.embedding_alt.is_(None)) |
                        (EmbeddingSegment.embedding_alt_model != self.alternative_embedding_version)
                    )
                else:
                    # Default: anything missing either primary or alt, or with wrong version
                    query = query.filter(
                        (EmbeddingSegment.embedding.is_(None)) |
                        (EmbeddingSegment.embedding_version != self.primary_embedding_version) |
                        (EmbeddingSegment.embedding_alt.is_(None)) |
                        (EmbeddingSegment.embedding_alt_model != self.alternative_embedding_version)
                    )

            # Order by ID for efficient pagination
            query = query.order_by(EmbeddingSegment.id)

            segment_ids = [row[0] for row in query.all()]
            logger.info(f"Found {len(segment_ids)} segment IDs needing embeddings")
            return segment_ids

    def _get_segments_by_ids(self, segment_ids: List[int], load_minimal: bool = True) -> List[EmbeddingSegment]:
        """Load segments by IDs, optionally deferring large embedding columns.

        Args:
            segment_ids: List of segment IDs to load
            load_minimal: If True, only load id, text, and version fields (not existing embeddings)
        """
        from sqlalchemy.orm import defer, load_only

        with get_session() as session:
            if load_minimal:
                # Only load what we need for generating embeddings
                segments = session.query(EmbeddingSegment).options(
                    load_only(
                        EmbeddingSegment.id,
                        EmbeddingSegment.text,
                        EmbeddingSegment.embedding_version,
                        EmbeddingSegment.embedding_alt_model,
                        EmbeddingSegment.meta_data
                    )
                ).filter(
                    EmbeddingSegment.id.in_(segment_ids)
                ).order_by(EmbeddingSegment.id).all()
            else:
                segments = session.query(EmbeddingSegment).filter(
                    EmbeddingSegment.id.in_(segment_ids)
                ).order_by(EmbeddingSegment.id).all()

            return segments

    async def hydrate_segments(self, segments: List[EmbeddingSegment],
                              generate_primary: bool = True,
                              generate_alternative: bool = True,
                              batch_size: int = 100) -> dict:
        """Generate embeddings for a list of segments.

        Args:
            segments: List of segments to process
            generate_primary: Generate primary embeddings
            generate_alternative: Generate alternative embeddings
            batch_size: DB batch size (how many segments to process at once)
        """

        if not segments:
            return {
                'status': 'success',
                'primary_generated': 0,
                'alternative_generated': 0,
                'total_segments': 0
            }

        # Initialize models as needed
        if generate_primary:
            self._init_primary_model()

        if generate_alternative:
            self._init_alternative_model()

        primary_count = 0
        alternative_count = 0

        # Process in batches with progress bar
        embedding_type = "alternative" if (generate_alternative and not generate_primary) else "primary"
        with tqdm(total=len(segments), desc=f"Generating {embedding_type} embeddings", unit="segment") as pbar:
            for i in range(0, len(segments), batch_size):
                batch = segments[i:i+batch_size]
                batch_texts = [seg.text for seg in batch]

                logger.info(f"Processing batch {i//batch_size + 1}/{(len(segments)-1)//batch_size + 1} ({len(batch)} segments)")

                # Clear MPS cache BEFORE embedding generation to ensure clean memory
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Generate primary embeddings with internal model batching
                primary_embeddings = None
                if generate_primary and self.primary_model:
                    # Batch size for 0.6B model
                    model_batch_size = 32
                    logger.info(f"Generating primary embeddings (model_batch_size={model_batch_size})")
                    primary_embeddings = self.primary_model.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=model_batch_size
                    )

                # Generate alternative embeddings with internal model batching
                alternative_embeddings = None
                if generate_alternative and self.alternative_model_obj:
                    # Smaller batch size for 4B model (it's much larger)
                    model_batch_size = 16
                    logger.info(f"Generating alternative embeddings (model_batch_size={model_batch_size})")
                    alternative_embeddings = self.alternative_model_obj.encode(
                        batch_texts,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=model_batch_size
                    )

                # Save to database using bulk update
                with get_session() as session:
                    updates = []
                    timestamp = datetime.now(timezone.utc).isoformat()

                    for idx, seg in enumerate(batch):
                        update_dict = {'id': seg.id}

                        # Update primary embedding if generated
                        if primary_embeddings is not None:
                            update_dict['embedding'] = primary_embeddings[idx].tolist()
                            update_dict['embedding_version'] = self.primary_embedding_version
                            primary_count += 1

                        # Update alternative embedding if generated
                        if alternative_embeddings is not None:
                            update_dict['embedding_alt'] = alternative_embeddings[idx].tolist()
                            update_dict['embedding_alt_model'] = self.alternative_embedding_version
                            alternative_count += 1

                        # Update metadata
                        if seg.meta_data:
                            meta = dict(seg.meta_data)
                        else:
                            meta = {}
                        meta['embeddings_pending'] = False
                        meta['embeddings_generated_at'] = timestamp
                        if primary_embeddings is not None:
                            meta['primary_embedding_model'] = self.embedding_model
                        if alternative_embeddings is not None:
                            meta['alternative_embedding_model'] = self.alternative_model
                        update_dict['meta_data'] = meta

                        updates.append(update_dict)

                    # Bulk update all segments at once
                    session.bulk_update_mappings(EmbeddingSegment, updates)
                    session.commit()
                    logger.info(f"Saved batch {i//batch_size + 1}: {primary_count} primary, {alternative_count} alternative")

                # Update progress bar
                pbar.update(len(batch))

                # Explicit memory cleanup after EVERY batch to avoid accumulation
                del primary_embeddings, alternative_embeddings, batch_texts, batch, updates
                gc.collect()

                # Clear PyTorch/MPS cache after EVERY batch (important for large batch sizes)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info(f"Hydration complete: {primary_count} primary embeddings, {alternative_count} alternative embeddings")

        return {
            'status': 'success',
            'primary_generated': primary_count,
            'alternative_generated': alternative_count,
            'total_segments': len(segments)
        }

    async def hydrate_content(self, content_id: str, rewrite: bool = False) -> dict:
        """Hydrate embeddings for a specific content."""
        logger.info(f"[{content_id}] Hydrating embeddings (rewrite={rewrite})")

        with get_session() as session:
            content = session.query(Content).filter_by(content_id=content_id).first()
            if not content:
                return {'status': 'error', 'error': f'Content {content_id} not found'}

            # Check if content's projects need alternative embeddings
            need_alternative = False
            if hasattr(content, 'projects') and content.projects:
                if isinstance(content.projects, str):
                    project_list = content.projects if content.projects else []
                else:
                    project_list = [str(content.projects)]

                for project in project_list:
                    if self._should_use_alternative_for_project(project):
                        need_alternative = True
                        logger.info(f"[{content_id}] Project '{project}' requires alternative embeddings")
                        break

            # Get segments
            if rewrite:
                segments = session.query(EmbeddingSegment).filter_by(
                    content_id=content.id
                ).all()
            else:
                # Only get segments missing embeddings
                query = session.query(EmbeddingSegment).filter_by(content_id=content.id)
                if need_alternative:
                    query = query.filter(
                        (EmbeddingSegment.embedding == None) |
                        (EmbeddingSegment.embedding_alt == None)
                    )
                else:
                    query = query.filter(EmbeddingSegment.embedding == None)
                segments = query.all()

            if not segments:
                logger.info(f"[{content_id}] No segments need embedding hydration")
                return {
                    'status': 'skipped',
                    'message': 'No segments need embeddings'
                }

        # Hydrate embeddings
        result = await self.hydrate_segments(
            segments,
            generate_primary=True,
            generate_alternative=need_alternative
        )

        # Update content status
        if result['status'] == 'success' and result['primary_generated'] > 0:
            with get_session() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if content:
                    content.is_embedded = True
                    session.commit()

        return result


    def _build_stitch_version_sql(self, stitch_versions: Optional[dict]) -> tuple:
        """Build SQL WHERE clause and parameters for stitch version filtering.

        Returns:
            tuple: (where_clause, params_dict) - where_clause is a SQL string, params_dict contains bind parameters
        """
        if not stitch_versions or (not stitch_versions.get('exact') and not stitch_versions.get('prefix')):
            return ("", {})

        conditions = []
        params = {}
        idx = 0

        # Add exact matches
        if stitch_versions.get('exact'):
            for ver in stitch_versions['exact']:
                param_name = f'stitch_version_{idx}'
                conditions.append(f"c.stitch_version = :{param_name}")
                params[param_name] = ver
                idx += 1

        # Add prefix matches using LIKE
        if stitch_versions.get('prefix'):
            for ver in stitch_versions['prefix']:
                param_name = f'stitch_version_{idx}'
                conditions.append(f"c.stitch_version LIKE :{param_name}")
                params[param_name] = f"{ver}%"
                idx += 1

        where_clause = f"AND ({' OR '.join(conditions)})" if conditions else ""
        return (where_clause, params)

    async def hydrate_batch(self, batch_size: int = 128,
                          project: Optional[str] = None,
                          primary_only: bool = False,
                          alternative_only: bool = False,
                          alt_first: bool = False,
                          force_reembed: bool = False,
                          stitch_versions: Optional[dict] = None,
                          max_per_run: Optional[int] = None) -> dict:
        """Hydrate embeddings for a batch of segments.

        Args:
            batch_size: Number of segments to process
            project: Filter to specific project
            primary_only: Only generate primary embeddings (0.6B)
            alternative_only: Only generate alternative embeddings (4B)
            alt_first: Prioritize alternative embeddings for projects that need them
            force_reembed: Re-generate embeddings even if version exists
            stitch_versions: Dict with 'exact' and 'prefix' lists for version filtering
            max_per_run: Maximum embeddings to generate in this run (None = unlimited)
        """
        logger.info(f"Starting batch hydration (batch_size={batch_size}, project={project}, "
                   f"alt_first={alt_first}, force_reembed={force_reembed}, stitch_versions={stitch_versions}, max_per_run={max_per_run})")

        # Determine what to generate based on flags and mode
        if alt_first and not primary_only:
            # Alternative-first mode: process segments needing alt embeddings for projects with config enabled OR recent content
            logger.info("Alternative-first mode: prioritizing projects with use_alternative_embeddings=true AND content from last 30 days")

            # Get projects that need alternative embeddings
            active_projects = self.config.get('active_projects', {})
            projects_needing_alt = [
                proj_name for proj_name, proj_config in active_projects.items()
                if proj_config.get('use_alternative_embeddings', False)
            ]

            logger.info(f"Projects needing alternative embeddings: {projects_needing_alt}")
            logger.info("Additionally generating 4B embeddings for all content published in last 30 days")

            # Process alternative embeddings for these projects
            generate_primary = False
            generate_alternative = True

        else:
            # Normal mode: primary embeddings ONLY by default (don't load 4B unless explicitly requested)
            # Default behavior: only generate primary embeddings (0.6B)
            # User must use --alt-first or --alternative-only to generate 4B embeddings

            if alternative_only:
                # Explicitly requested alternative only
                generate_primary = False
                generate_alternative = True
            else:
                # Default: primary only (unless alt_first was set, which is handled above)
                generate_primary = True
                generate_alternative = False

        # Initialize models as needed
        if generate_primary:
            self._init_primary_model()
        if generate_alternative:
            self._init_alternative_model()

        # Use streaming pagination instead of loading all IDs at once
        # This dramatically reduces memory usage for large datasets
        embedding_type = "alternative (4B)" if (generate_alternative and not generate_alternative) else "primary (0.6B)"

        # First, get a count (fast query using index)
        total_needing = self._count_missing_embeddings(
            project=project,
            primary=(generate_primary and not generate_alternative),
            alternative=(generate_alternative and not generate_primary),
            stitch_versions=stitch_versions,
            recent_content_days=30 if (generate_alternative and not generate_primary) else None
        )

        if total_needing == 0:
            embedding_type = "alternative (4B)" if (generate_alternative and not generate_primary) else "primary (0.6B)"
            logger.info(f"No {embedding_type} embeddings needed")
            return {
                'status': 'success',
                'primary_generated': 0,
                'alternative_generated': 0,
                'total_segments': 0
            }

        # Determine how many to process this run
        segments_to_process = total_needing
        if max_per_run and total_needing > max_per_run:
            logger.info(f"Found {total_needing:,} segments needing {embedding_type} embeddings, limiting to {max_per_run:,}")
            segments_to_process = max_per_run
        else:
            logger.info(f"Found {total_needing:,} segments needing {embedding_type} embeddings")

        logger.info(f"Processing up to {segments_to_process:,} segments in batches of {batch_size}")

        # Process segments in batches using streaming pagination
        # Fetch batches directly from DB instead of pre-loading all IDs
        total_primary_generated = 0
        total_alternative_generated = 0
        last_processed_id = 0  # For keyset pagination
        loop_iteration = 0

        with tqdm(total=segments_to_process, desc=f"Generating {embedding_type} embeddings", unit="segment") as pbar:
            while total_primary_generated + total_alternative_generated < segments_to_process:
                loop_iteration += 1

                try:
                    # Fetch next batch using keyset pagination (much more efficient than OFFSET)
                    from sqlalchemy import text
                    with get_session() as session:
                        # Build WHERE clause for filtering
                        if generate_primary and not generate_alternative:
                            version_filter = """
                                AND (es.embedding_version IS NULL
                                     OR es.embedding_version != :target_version)
                            """
                            params = {
                                'last_id': last_processed_id,
                                'batch_size': batch_size,
                                'target_version': self.primary_embedding_version
                            }
                        else:
                            # Alternative embeddings
                            version_filter = """
                                AND es.embedding_version IS NOT NULL
                                AND (es.embedding_alt_model IS NULL
                                     OR es.embedding_alt_model != :target_version)
                            """
                            params = {
                                'last_id': last_processed_id,
                                'batch_size': batch_size,
                                'target_version': self.alternative_embedding_version
                            }

                        # Add project filter if specified
                        project_filter = ""
                        if project:
                            project_filter = "AND :project = ANY(c.projects)"
                            params['project'] = project

                        # Filter out segments that are too long (>8000 chars) or already marked as skipped
                        sql = text(f"""
                            SELECT es.id, es.text, es.embedding_version, es.embedding_alt_model, es.meta_data
                            FROM public.embedding_segments es
                            JOIN public.content c ON c.id = es.content_id
                            WHERE es.id > :last_id
                            AND LENGTH(es.text) <= 3000
                            AND (es.embedding_version IS NULL OR es.embedding_version NOT LIKE 'SKIPPED%')
                            {version_filter}
                            {project_filter}
                            ORDER BY es.id
                            LIMIT :batch_size
                        """)
                        result = session.execute(sql, params)
                        rows = result.fetchall()

                    if not rows:
                        logger.info(f"No more segments to process after batch {loop_iteration}")
                        break

                    # Convert rows to simple objects using module-level namedtuple
                    segments = [SegmentData(*row) for row in rows]

                    # Update last_processed_id for next iteration
                    last_processed_id = segments[-1].id

                    # Clean up database result objects
                    del rows
                    gc.collect()

                except Exception as e:
                    logger.error(f"Error fetching batch {loop_iteration}: {e}", exc_info=True)
                    # Try to continue to next batch
                    continue

                # Note: Long segments (>8000 chars) are now filtered out in the SQL query
                # Process this batch
                batch_texts = [seg.text for seg in segments]

                logger.info(f"Processing batch {loop_iteration} with {len(segments)} segments (IDs {segments[0].id} to {segments[-1].id})")

                # Clear MPS cache BEFORE embedding generation to ensure clean memory
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                    mem_allocated = torch.mps.driver_allocated_memory() / 1024**3
                    logger.info(f"MPS memory before encoding: {mem_allocated:.2f} GB")
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Generate primary embeddings with chunked processing to avoid MPS OOM
                try:
                    logger.info(f"About to generate embeddings for {len(batch_texts)} texts")
                    import time
                    primary_embeddings = None

                    if generate_primary and self.primary_model:
                        # Local model path (original code)
                        # Check actual device
                        logger.info(f"Model device: {self.primary_model.device}")

                        # Process in chunks to avoid MPS memory issues with large batches
                        # Use VERY small chunks to prevent MPS accumulation (MPS has memory leak issues)
                        max_chunk_size = 32  # Much smaller to force frequent cache clears
                        model_batch_size = 8  # Also reduce internal batch size
                        start_time = time.time()

                        all_embeddings = []
                        for chunk_start in range(0, len(batch_texts), max_chunk_size):
                            chunk_end = min(chunk_start + max_chunk_size, len(batch_texts))
                            chunk_texts = batch_texts[chunk_start:chunk_end]

                            # Use no_grad to prevent gradient tracking memory
                            with torch.no_grad():
                                chunk_embeddings = self.primary_model.encode(
                                    chunk_texts,
                                    convert_to_numpy=True,
                                    show_progress_bar=False,
                                    batch_size=model_batch_size
                                )

                            all_embeddings.append(chunk_embeddings)

                            # Aggressive cleanup after each chunk to prevent accumulation
                            del chunk_texts
                            del chunk_embeddings
                            gc.collect()

                            # Synchronize and clear MPS cache
                            if torch.backends.mps.is_available():
                                torch.mps.synchronize()
                                torch.mps.empty_cache()
                            gc.collect()

                        primary_embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]

                        # Move to CPU immediately to free MPS memory
                        primary_embeddings = np.asarray(primary_embeddings, dtype=np.float32)

                        # Clean up intermediate list
                        del all_embeddings

                        # Force MPS cleanup immediately after embedding generation
                        if torch.backends.mps.is_available():
                            torch.mps.synchronize()
                            torch.mps.empty_cache()
                        gc.collect()

                        encode_time = time.time() - start_time
                        logger.info(f"Generated {len(primary_embeddings)} primary embeddings in {encode_time:.2f}s ({len(primary_embeddings)/encode_time:.1f} emb/s)")

                    # Generate alternative embeddings with chunked processing
                    alternative_embeddings = None
                    if generate_alternative and self.alternative_model_obj:
                        # Smaller chunks for 4B model (it's much larger)
                        max_chunk_size = 16  # Very small for 4B model
                        model_batch_size = 4  # Very small internal batches

                        all_embeddings = []
                        for chunk_start in range(0, len(batch_texts), max_chunk_size):
                            chunk_end = min(chunk_start + max_chunk_size, len(batch_texts))
                            chunk_texts = batch_texts[chunk_start:chunk_end]

                            # Use no_grad to prevent gradient tracking memory
                            with torch.no_grad():
                                chunk_embeddings = self.alternative_model_obj.encode(
                                    chunk_texts,
                                    convert_to_numpy=True,
                                    show_progress_bar=False,
                                    batch_size=model_batch_size
                                )

                            all_embeddings.append(chunk_embeddings)

                            # Aggressive cleanup after each chunk to prevent accumulation
                            del chunk_texts
                            del chunk_embeddings
                            gc.collect()

                            # Synchronize and clear MPS cache
                            if torch.backends.mps.is_available():
                                torch.mps.synchronize()
                                torch.mps.empty_cache()
                            gc.collect()

                        alternative_embeddings = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]

                        # Move to CPU immediately to free MPS memory
                        alternative_embeddings = np.asarray(alternative_embeddings, dtype=np.float32)

                        # Clean up intermediate list
                        del all_embeddings

                        # Force MPS cleanup immediately after embedding generation
                        if torch.backends.mps.is_available():
                            torch.mps.synchronize()
                            torch.mps.empty_cache()
                        gc.collect()

                except Exception as e:
                    logger.error(f"Error generating embeddings at iteration {loop_iteration}: {e}", exc_info=True)
                    # Clean up and continue to next batch
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                    gc.collect()
                    continue

                # Save to database using raw SQL bulk update with psycopg2
                # MEMORY OPTIMIZATION: Use generator to avoid building intermediate lists
                try:
                    logger.info(f"About to save {len(segments)} segments to database")
                    import time
                    import json as json_module
                    from psycopg2.extras import execute_values

                    # Store flags before processing
                    has_primary = primary_embeddings is not None
                    has_alternative = alternative_embeddings is not None
                    primary_count = len(segments) if has_primary else 0
                    alternative_count = len(segments) if has_alternative else 0
                    timestamp = datetime.now(timezone.utc).isoformat()

                    bulk_start = time.time()

                    # Generator function to yield tuples directly without building list
                    def generate_update_tuples():
                        for idx, seg in enumerate(segments):
                            # Build metadata
                            if seg.meta_data:
                                meta = dict(seg.meta_data)
                            else:
                                meta = {}
                            meta['embeddings_pending'] = False
                            meta['embeddings_generated_at'] = timestamp
                            if has_primary:
                                meta['primary_embedding_model'] = self.embedding_model
                            if has_alternative:
                                meta['alternative_embedding_model'] = self.alternative_model
                            meta_json = json_module.dumps(meta)

                            # Convert embeddings to pgvector format - yield immediately
                            if has_primary and has_alternative:
                                emb_str = '[' + ','.join(f'{x:.6f}' for x in primary_embeddings[idx]) + ']'
                                emb_alt_str = '[' + ','.join(f'{x:.6f}' for x in alternative_embeddings[idx]) + ']'
                                yield (seg.id, emb_str, self.primary_embedding_version,
                                       emb_alt_str, self.alternative_embedding_version, meta_json)
                            elif has_primary:
                                emb_str = '[' + ','.join(f'{x:.6f}' for x in primary_embeddings[idx]) + ']'
                                yield (seg.id, emb_str, self.primary_embedding_version, meta_json)
                            elif has_alternative:
                                emb_alt_str = '[' + ','.join(f'{x:.6f}' for x in alternative_embeddings[idx]) + ']'
                                yield (seg.id, emb_alt_str, self.alternative_embedding_version, meta_json)

                    with get_session() as session:
                        # Use raw connection for execute_values
                        connection = session.connection().connection
                        cursor = connection.cursor()

                        # Execute with generator (psycopg2 will iterate it)
                        if has_primary and has_alternative:
                            execute_values(
                                cursor,
                                """
                                UPDATE public.embedding_segments
                                SET embedding = data.emb::vector(1024),
                                    embedding_version = data.emb_ver,
                                    embedding_alt = data.emb_alt::vector(2000),
                                    embedding_alt_model = data.emb_alt_model,
                                    meta_data = data.meta::json
                                FROM (VALUES %s) AS data(id, emb, emb_ver, emb_alt, emb_alt_model, meta)
                                WHERE embedding_segments.id = data.id
                                """,
                                generate_update_tuples()
                            )
                        elif has_primary:
                            execute_values(
                                cursor,
                                """
                                UPDATE public.embedding_segments
                                SET embedding = data.emb::vector(1024),
                                    embedding_version = data.emb_ver,
                                    meta_data = data.meta::json
                                FROM (VALUES %s) AS data(id, emb, emb_ver, meta)
                                WHERE embedding_segments.id = data.id
                                """,
                                generate_update_tuples()
                            )
                        elif has_alternative:
                            execute_values(
                                cursor,
                                """
                                UPDATE public.embedding_segments
                                SET embedding_alt = data.emb_alt::vector(2000),
                                    embedding_alt_model = data.emb_alt_model,
                                    meta_data = data.meta::json
                                FROM (VALUES %s) AS data(id, emb_alt, emb_alt_model, meta)
                                WHERE embedding_segments.id = data.id
                                """,
                                generate_update_tuples()
                            )

                        bulk_time = time.time() - bulk_start
                        logger.info(f"execute_values bulk update took {bulk_time:.2f}s")

                        commit_start = time.time()
                        session.commit()
                        commit_time = time.time() - commit_start
                        logger.info(f"Commit took {commit_time:.2f}s")

                        # Clean up database resources
                        cursor.close()
                        del cursor, connection

                    # Free embeddings AFTER database operations complete
                    del primary_embeddings, alternative_embeddings
                    primary_embeddings = None
                    alternative_embeddings = None
                    gc.collect()

                    # Update progress bar
                    pbar.update(len(segments))

                    # Update totals
                    total_primary_generated += primary_count
                    total_alternative_generated += alternative_count

                except Exception as e:
                    logger.error(f"Error saving to database at iteration {loop_iteration}: {e}", exc_info=True)
                    # Continue to next batch
                    continue

                # Update is_embedded flags periodically (every ~10k embeddings)
                if primary_count > 0 and total_primary_generated % 10000 < batch_size:
                    logger.info(f"Updating is_embedded flags at {total_primary_generated} embeddings...")
                    with get_session() as flag_session:
                        from sqlalchemy import text
                        update_sql = text("""
                            UPDATE content
                            SET is_embedded = true
                            WHERE id IN (
                                SELECT c.id
                                FROM content c
                                WHERE c.is_embedded = false
                                AND EXISTS (
                                    SELECT 1 FROM embedding_segments es WHERE es.content_id = c.id
                                )
                                AND NOT EXISTS (
                                    SELECT 1 FROM embedding_segments es
                                    WHERE es.content_id = c.id AND es.embedding IS NULL
                                )
                                LIMIT 1000
                            )
                        """)
                        result = flag_session.execute(update_sql)
                        updated_count = result.rowcount
                        flag_session.commit()
                        if updated_count > 0:
                            logger.info(f"Updated is_embedded flag for {updated_count} content items")

                # Explicit memory cleanup after EVERY batch to avoid accumulation
                cleanup_start = time.time()
                del batch_texts, segments
                # Embeddings already deleted before database operations
                # Note: values, cursor, connection already deleted in database block
                gc.collect()
                cleanup_time = time.time() - cleanup_start
                logger.info(f"Memory cleanup (gc.collect) took {cleanup_time:.2f}s")

                # Clear PyTorch/MPS cache after EVERY batch (important for large batch sizes)
                cache_start = time.time()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                cache_time = time.time() - cache_start
                logger.info(f"Cache clear took {cache_time:.2f}s")

                # AGGRESSIVE MEMORY MANAGEMENT: Reload model every N batches to force complete memory reset
                # MPS has memory fragmentation issues that empty_cache() doesn't fully resolve
                batches_processed = (total_primary_generated + total_alternative_generated) // batch_size
                if batches_processed > 0 and batches_processed % 20 == 0:  # Every 20 batches
                    logger.info(f"Reloading model after {batches_processed} batches to reset MPS memory...")
                    if generate_primary and self.primary_model:
                        del self.primary_model
                        self.primary_model = None
                        gc.collect()
                        if torch.backends.mps.is_available():
                            torch.mps.synchronize()
                            torch.mps.empty_cache()
                        gc.collect()
                        # CRITICAL: Wait for MPS to fully release memory before reloading
                        logger.info("Waiting 5s for MPS memory release...")
                        time.sleep(5)
                        self._init_primary_model()
                        logger.info("Primary model reloaded")
                    if generate_alternative and self.alternative_model_obj:
                        del self.alternative_model_obj
                        self.alternative_model_obj = None
                        gc.collect()
                        if torch.backends.mps.is_available():
                            torch.mps.synchronize()
                            torch.mps.empty_cache()
                        gc.collect()
                        # CRITICAL: Wait for MPS to fully release memory before reloading
                        logger.info("Waiting 5s for MPS memory release...")
                        time.sleep(5)
                        self._init_alternative_model()
                        logger.info("Alternative model reloaded")

        logger.info(f"Hydration complete: {total_primary_generated} primary embeddings, {total_alternative_generated} alternative embeddings")

        # Update is_embedded flag for content items that now have all segments embedded
        if total_primary_generated > 0:
            logger.info("Updating is_embedded flags for content with complete embeddings...")
            with get_session() as session:
                # Find content where all segments have primary embeddings but is_embedded = false
                from sqlalchemy import text
                update_sql = text("""
                    UPDATE content
                    SET is_embedded = true
                    WHERE id IN (
                        SELECT c.id
                        FROM content c
                        WHERE c.is_embedded = false
                        AND EXISTS (
                            SELECT 1 FROM embedding_segments es WHERE es.content_id = c.id
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM embedding_segments es
                            WHERE es.content_id = c.id AND es.embedding IS NULL
                        )
                    )
                """)
                result = session.execute(update_sql)
                updated_count = result.rowcount
                session.commit()

                if updated_count > 0:
                    logger.info(f"Updated is_embedded flag for {updated_count} content items")

        return {
            'status': 'success',
            'primary_generated': total_primary_generated,
            'alternative_generated': total_alternative_generated,
            'total_segments': total_primary_generated + total_alternative_generated
        }


    async def embed_descriptions(self, batch_size: int = 32, limit: Optional[int] = None, max_per_run: Optional[int] = 1000) -> dict:
        """
        Embed descriptions for all content that has is_embedded=True but no description_embedding.
        Content is processed in descending order by publish_date (newest first).

        Args:
            batch_size: Number of content items to process per batch
            limit: Optional limit on total items to process (for testing)
            max_per_run: Maximum embeddings to generate in this run (None = unlimited)

        Returns:
            Dictionary with status and counts
        """
        # Use max_per_run as effective limit if provided
        effective_limit = limit
        if max_per_run:
            if effective_limit:
                effective_limit = min(effective_limit, max_per_run)
            else:
                effective_limit = max_per_run

        logger.info(f"Starting description embedding (batch_size={batch_size}, limit={effective_limit})")

        # Initialize primary model (we use the same model as segment embeddings)
        self._init_primary_model()

        total_embedded = 0
        total_skipped = 0

        with get_session() as session:
            # Build query for content needing description embeddings
            # Order by publish_date DESC (newest first) to prioritize recent content
            query = session.query(Content).filter(
                Content.is_embedded == True,
                Content.description.isnot(None),
                Content.description != '',
                Content.description_embedding.is_(None)
            ).order_by(Content.publish_date.desc().nulls_last())

            if effective_limit:
                query = query.limit(effective_limit)

            # Get total count
            count_query = session.query(Content).filter(
                Content.is_embedded == True,
                Content.description.isnot(None),
                Content.description != '',
                Content.description_embedding.is_(None)
            )
            total_count = count_query.count()

            if effective_limit:
                total_count = min(total_count, effective_limit)

            if total_count == 0:
                logger.info("No content needs description embeddings")
                return {
                    'status': 'success',
                    'embedded': 0,
                    'skipped': 0,
                    'total': 0
                }

            logger.info(f"Found {total_count:,} content items needing description embeddings")

        # Process in batches
        offset = 0
        with tqdm(total=total_count, desc="Embedding descriptions", unit="item") as pbar:
            while True:
                with get_session() as session:
                    # Fetch batch - order by publish_date DESC (newest first)
                    query = session.query(Content).filter(
                        Content.is_embedded == True,
                        Content.description.isnot(None),
                        Content.description != '',
                        Content.description_embedding.is_(None)
                    ).order_by(Content.publish_date.desc().nulls_last()).limit(batch_size)

                    content_items = query.all()

                    if not content_items:
                        break

                    # Prepare texts for embedding
                    descriptions = []
                    content_ids = []
                    for content in content_items:
                        desc = content.description.strip() if content.description else ''
                        if desc:
                            descriptions.append(desc)
                            content_ids.append(content.id)
                        else:
                            total_skipped += 1

                    if not descriptions:
                        pbar.update(len(content_items))
                        continue

                    # Generate embeddings
                    try:
                        # Clear MPS cache before embedding
                        if torch.backends.mps.is_available():
                            torch.mps.synchronize()
                            torch.mps.empty_cache()
                        gc.collect()

                        # Use local model with small batch size to avoid MPS OOM
                        with torch.no_grad():
                            embeddings = self.primary_model.encode(
                                descriptions,
                                convert_to_numpy=True,
                                show_progress_bar=False,
                                batch_size=8
                            )

                        embeddings = np.asarray(embeddings, dtype=np.float32)

                        # Update database
                        for idx, content_id in enumerate(content_ids):
                            content = session.query(Content).filter(Content.id == content_id).first()
                            if content:
                                content.description_embedding = embeddings[idx].tolist()
                                total_embedded += 1

                        session.commit()

                        # Clear memory
                        del embeddings, descriptions
                        gc.collect()
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()

                    except Exception as e:
                        logger.error(f"Error embedding batch: {e}", exc_info=True)
                        session.rollback()
                        # Continue to next batch
                        continue

                    pbar.update(len(content_items))

                # Check if we've hit the limit
                if effective_limit and total_embedded >= effective_limit:
                    break

                # Wait between batches to let MPS memory settle
                import time
                time.sleep(1)

        logger.info(f"Description embedding complete: {total_embedded} embedded, {total_skipped} skipped")

        return {
            'status': 'success',
            'embedded': total_embedded,
            'skipped': total_skipped,
            'total': total_embedded + total_skipped
        }


# CLI entry point for scheduled task execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch generate embeddings for segments',
        epilog='''
Examples:
  # Default: Generate up to 10,000 embeddings total
  # Order: segment embeddings first, then description embeddings (newest content first)
  uv run python -m src.utils.embedding_hydrator

  # Only segment embeddings (primary 0.6B model, no descriptions)
  uv run python -m src.utils.embedding_hydrator --primary-only

  # Only segment embeddings (alternative 4B model for configured projects)
  uv run python -m src.utils.embedding_hydrator --alternative-only

  # Only description embeddings (skip segments)
  uv run python -m src.utils.embedding_hydrator --embed-descriptions

  # Process more embeddings per run
  uv run python -m src.utils.embedding_hydrator --max-per-run 20000

  # Process unlimited embeddings (no cap)
  uv run python -m src.utils.embedding_hydrator --max-per-run 0

IMPORTANT:
  - Default max-per-run is 10,000 embeddings (use --max-per-run 0 for unlimited)
  - Default order: segment embeddings FIRST, then description embeddings (by publish_date DESC)
  - The 10k budget is shared across both phases
  - --primary-only or --alternative-only: Only segment embeddings (skips descriptions)
  - --embed-descriptions: Only description embeddings (skips segments)
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation (default: 32)")
    parser.add_argument("--project", type=str, default=None, help="Filter to specific project")
    parser.add_argument("--primary-only", action="store_true", help="Only generate primary (0.6B) embeddings")
    parser.add_argument("--alternative-only", action="store_true", help="Only generate alternative (4B) embeddings")
    parser.add_argument("--alt-first", action="store_true", help="Generate alternative (4B) embeddings first, then primary (0.6B)")
    parser.add_argument("--force-reembed", action="store_true", help="Re-generate embeddings even if version exists")
    parser.add_argument("--stitch-version", type=str, default=None, help="Filter to specific stitch version prefix (e.g., 'stitch_v14')")
    parser.add_argument("--embed-descriptions", action="store_true", help="Embed content descriptions (for semantic related episodes)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to process (for testing)")
    parser.add_argument("--max-per-run", type=int, default=10000, help="Maximum embeddings to generate per run cycle (default: 10000)")

    args = parser.parse_args()

    # Parse stitch version filter
    stitch_versions = None
    if args.stitch_version:
        stitch_versions = {'prefix': [args.stitch_version]}

    # Handle max_per_run: 0 means unlimited (None)
    max_per_run = args.max_per_run if args.max_per_run > 0 else None

    # Import time for cleanup delays
    import time

    # Run hydration with local model
    hydrator = EmbeddingHydrator()

    # Track total embeddings generated across all phases
    total_generated = 0
    remaining_budget = max_per_run

    # Handle --embed-descriptions ONLY mode (skip segments)
    if args.embed_descriptions:
        logger.info("Running description embedding mode (only)...")
        result = asyncio.run(hydrator.embed_descriptions(
            batch_size=args.batch_size,
            limit=args.limit,
            max_per_run=max_per_run
        ))
        logger.info(f"Description embedding result: {result}")

        # Final cleanup
        hydrator.cleanup()

        # Print machine-readable summary for orchestrator
        import json
        summary = {
            'description_embeddings_generated': result.get('embedded', 0),
            'skipped': result.get('skipped', 0),
            'total_processed': result.get('total', 0)
        }
        print(f"TASK_SUMMARY: {json.dumps(summary)}")

        # Exit with appropriate code
        sys.exit(0 if result.get('status') == 'success' else 1)

    # Default behavior: segments first, then descriptions (all within max_per_run budget)
    # --primary-only: only do primary segment embeddings
    # --alternative-only: only do alternative segment embeddings

    result_primary = {'primary_generated': 0, 'total_segments': 0}
    result_alt = {'alternative_generated': 0, 'total_segments': 0}
    result_desc = {'embedded': 0}

    if args.primary_only:
        # Only primary segment embeddings
        logger.info("Generating primary (0.6B) segment embeddings...")
        result_primary = asyncio.run(hydrator.hydrate_batch(
            batch_size=args.batch_size,
            project=args.project,
            primary_only=True,
            alternative_only=False,
            alt_first=False,
            force_reembed=args.force_reembed,
            stitch_versions=stitch_versions,
            max_per_run=remaining_budget
        ))
        total_generated += result_primary.get('primary_generated', 0)

    elif args.alternative_only:
        # Only alternative segment embeddings
        logger.info("Generating alternative (4B) segment embeddings...")
        result_alt = asyncio.run(hydrator.hydrate_batch(
            batch_size=args.batch_size,
            project=args.project,
            primary_only=False,
            alternative_only=True,
            alt_first=True,
            force_reembed=args.force_reembed,
            stitch_versions=stitch_versions,
            max_per_run=remaining_budget
        ))
        total_generated += result_alt.get('alternative_generated', 0)

    else:
        # Default: segments first, then descriptions
        # Phase 1: Primary segment embeddings
        logger.info("Phase 1: Generating primary (0.6B) segment embeddings...")
        result_primary = asyncio.run(hydrator.hydrate_batch(
            batch_size=args.batch_size,
            project=args.project,
            primary_only=True,
            alternative_only=False,
            alt_first=False,
            force_reembed=args.force_reembed,
            stitch_versions=stitch_versions,
            max_per_run=remaining_budget
        ))
        total_generated += result_primary.get('primary_generated', 0)

        # Update remaining budget
        if remaining_budget:
            remaining_budget = max(0, remaining_budget - result_primary.get('primary_generated', 0))
            if remaining_budget == 0:
                logger.info(f"Budget exhausted after segment embeddings ({total_generated:,} generated)")

        # MEMORY CLEANUP: Release resources between phases
        if remaining_budget is None or remaining_budget > 0:
            logger.info("Cleaning up between phases...")
            hydrator.cleanup()
            # Wait for memory to settle before next phase
            time.sleep(2)

        # Phase 2: Description embeddings (if budget remains)
        if remaining_budget is None or remaining_budget > 0:
            logger.info("Phase 2: Generating content description embeddings (newest first)...")
            result_desc = asyncio.run(hydrator.embed_descriptions(
                batch_size=args.batch_size,
                max_per_run=remaining_budget
            ))
            total_generated += result_desc.get('embedded', 0)

    # Final cleanup
    hydrator.cleanup()

    # Combine results
    result = {
        'status': 'success',
        'primary_generated': result_primary.get('primary_generated', 0),
        'alternative_generated': result_alt.get('alternative_generated', 0),
        'description_embeddings': result_desc.get('embedded', 0),
        'total_embeddings': total_generated
    }

    logger.info(f"Hydration result: {result}")

    # Print machine-readable summary for orchestrator
    import json
    summary = {
        'primary_embeddings_generated': result.get('primary_generated', 0),
        'alternative_embeddings_generated': result.get('alternative_generated', 0),
        'description_embeddings_generated': result.get('description_embeddings', 0),
        'total_embeddings_generated': result.get('total_embeddings', 0)
    }
    print(f"TASK_SUMMARY: {json.dumps(summary)}")

    # Exit with appropriate code
    sys.exit(0 if result.get('status') == 'success' else 1)
