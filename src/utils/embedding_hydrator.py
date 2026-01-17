"""
Embedding Hydrator
==================

Core class for batch generating embeddings for segments.
Called via scheduled tasks (see config/config.yaml under 'scheduled_tasks').
"""

import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
import argparse
import asyncio
import logging
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

logger = setup_worker_logger('embedding_hydrator')


class EmbeddingHydrator:
    """Batch generates embeddings for segments."""

    def __init__(self, primary_model_override: Optional[str] = None,
                 alternative_model_override: Optional[str] = None,
                 alternative_dim: Optional[int] = None):
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
                          stitch_versions: Optional[dict] = None) -> dict:
        """Hydrate embeddings for a batch of segments.

        Args:
            batch_size: Number of segments to process
            project: Filter to specific project
            primary_only: Only generate primary embeddings (0.6B)
            alternative_only: Only generate alternative embeddings (4B)
            alt_first: Prioritize alternative embeddings for projects that need them
            force_reembed: Re-generate embeddings even if version exists
            stitch_versions: Dict with 'exact' and 'prefix' lists for version filtering
        """
        logger.info(f"Starting batch hydration (batch_size={batch_size}, project={project}, "
                   f"alt_first={alt_first}, force_reembed={force_reembed}, stitch_versions={stitch_versions})")

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

        # Load all segment IDs into memory first (integers are small, ~8 bytes each)
        # This replaces the separate count + fetch, doing just one query
        logger.info("Loading segment IDs needing embeddings...")
        all_segment_ids = self._get_segment_ids_needing_embeddings(
            project=project,
            primary_only=(generate_primary and not generate_alternative),
            alternative_only=(generate_alternative and not generate_primary),
            force_reembed=force_reembed,
            stitch_versions=stitch_versions
        )

        if not all_segment_ids:
            embedding_type = "alternative (4B)" if (generate_alternative and not generate_primary) else "primary (0.6B)"
            logger.info(f"✓ No {embedding_type} embeddings needed")
            return {
                'status': 'success',
                'primary_generated': 0,
                'alternative_generated': 0,
                'total_segments': 0
            }

        embedding_type = "alternative (4B)" if (generate_alternative and not generate_primary) else "primary (0.6B)"
        logger.info(f"📊 Found {len(all_segment_ids):,} segments needing {embedding_type} embeddings")
        logger.info(f"📦 Processing in batches of {batch_size}")

        # Process segments in batches using the ID list
        total_primary_generated = 0
        total_alternative_generated = 0

        with tqdm(total=len(all_segment_ids), desc=f"Generating {embedding_type} embeddings", unit="segment") as pbar:
            for batch_start in range(0, len(all_segment_ids), batch_size):
                batch_end = min(batch_start + batch_size, len(all_segment_ids))
                batch_ids = all_segment_ids[batch_start:batch_end]
                loop_iteration = (batch_start // batch_size) + 1

                try:
                    # Fetch segments by IDs
                    from sqlalchemy import text
                    with get_session() as session:
                        sql = text("""
                            SELECT id, text, embedding_version, embedding_alt_model, meta_data
                            FROM public.embedding_segments
                            WHERE id = ANY(:ids)
                            ORDER BY id
                        """)
                        result = session.execute(sql, {'ids': batch_ids})
                        rows = result.fetchall()

                    if not rows:
                        logger.warning(f"No segments found for batch {loop_iteration} (IDs {batch_ids[0]} to {batch_ids[-1]})")
                        continue

                    # Convert rows to simple objects
                    from collections import namedtuple
                    Segment = namedtuple('Segment', ['id', 'text', 'embedding_version', 'embedding_alt_model', 'meta_data'])
                    segments = [Segment(*row) for row in rows]

                    # Clean up database result objects
                    del rows
                    gc.collect()

                except Exception as e:
                    logger.error(f"Error fetching batch {loop_iteration}: {e}", exc_info=True)
                    # Try to continue to next batch
                    continue

                # Filter out segments with text longer than 8000 characters (prevents OOM with pathological cases)
                # Note: The model will automatically truncate to its max token limit (~512 tokens = ~2000-2500 chars)
                # We only skip truly pathological cases that might cause memory issues
                MAX_TEXT_LENGTH = 8000
                original_segments = segments
                segments = [seg for seg in original_segments if len(seg.text) <= MAX_TEXT_LENGTH]
                skipped_segments = [seg for seg in original_segments if len(seg.text) > MAX_TEXT_LENGTH]

                if skipped_segments:
                    logger.warning(f"Skipped {len(skipped_segments)} segments with text > {MAX_TEXT_LENGTH} chars")

                    # Mark skipped segments in database so they don't get queried again
                    with get_session() as skip_session:
                        skipped_ids = [seg.id for seg in skipped_segments]
                        # Update each individually since json type doesn't support || operator
                        for skip_id in skipped_ids:
                            existing_meta = skip_session.execute(text(
                                "SELECT meta_data FROM public.embedding_segments WHERE id = :id"
                            ), {'id': skip_id}).scalar()

                            if existing_meta:
                                import json as json_module
                                meta_dict = json_module.loads(existing_meta) if isinstance(existing_meta, str) else dict(existing_meta)
                                meta_dict['skipped_reason'] = 'text_too_long'
                                new_meta = json_module.dumps(meta_dict)
                            else:
                                new_meta = '{"skipped_reason": "text_too_long"}'

                            skip_session.execute(text("""
                                UPDATE public.embedding_segments
                                SET embedding_version = 'SKIPPED_TOO_LONG',
                                    meta_data = :meta_data
                                WHERE id = :id
                            """), {'id': skip_id, 'meta_data': new_meta})

                        skip_session.commit()
                        logger.info(f"Marked {len(skipped_ids)} segments as SKIPPED_TOO_LONG")

                if not segments:
                    logger.info("All segments in batch were too long, continuing...")
                    # Count as progress even though we skipped them
                    pbar.update(len(skipped_segments))
                    continue

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
                try:
                    logger.info(f"About to save {len(segments)} segments to database")
                    import time
                    import json as json_module
                    from sqlalchemy import text
                    from datetime import datetime, timezone

                    primary_count = 0
                    alternative_count = 0
                    prep_start = time.time()

                    # Prepare data for bulk update - convert embeddings in streaming fashion to avoid memory copies
                    update_data = []
                    timestamp = datetime.now(timezone.utc).isoformat()

                    for idx, seg in enumerate(segments):
                        # Update metadata
                        if seg.meta_data:
                            meta = dict(seg.meta_data)
                        else:
                            meta = {}
                        meta['embeddings_pending'] = False
                        meta['embeddings_generated_at'] = timestamp
                        if primary_embeddings is not None:
                            meta['primary_embedding_model'] = self.embedding_model
                            primary_count += 1
                        if alternative_embeddings is not None:
                            meta['alternative_embedding_model'] = self.alternative_model
                            alternative_count += 1

                        # Convert embeddings to pgvector format string (minimize memory copies)
                        emb_str = None
                        if primary_embeddings is not None:
                            # Use direct numpy array access to avoid creating intermediate lists
                            emb_str = '[' + ','.join(f'{x:.6f}' for x in primary_embeddings[idx]) + ']'

                        emb_alt_str = None
                        if alternative_embeddings is not None:
                            # Use direct numpy array access to avoid creating intermediate lists
                            emb_alt_str = '[' + ','.join(f'{x:.6f}' for x in alternative_embeddings[idx]) + ']'

                        update_data.append({
                            'id': seg.id,
                            'embedding': emb_str,
                            'embedding_version': self.primary_embedding_version if primary_embeddings is not None else None,
                            'embedding_alt': emb_alt_str,
                            'embedding_alt_model': self.alternative_embedding_version if alternative_embeddings is not None else None,
                            'meta_data': json_module.dumps(meta)
                        })

                        # Periodically collect garbage during large loops to prevent buildup
                        if idx > 0 and idx % 50 == 0:
                            gc.collect()

                    prep_time = time.time() - prep_start
                    logger.info(f"Prepared {len(update_data)} updates in {prep_time:.2f}s")

                    # Store flags before freeing embeddings
                    has_primary = primary_embeddings is not None
                    has_alternative = alternative_embeddings is not None

                    # Free embeddings BEFORE database operations to reduce peak memory
                    del primary_embeddings, alternative_embeddings
                    primary_embeddings = None
                    alternative_embeddings = None
                    gc.collect()

                    # Execute bulk update using raw SQL
                    with get_session() as session:
                        bulk_start = time.time()

                        # Build SQL for bulk update
                        if has_primary and has_alternative:
                            # Update both embeddings
                            sql = text("""
                                UPDATE public.embedding_segments
                                SET embedding = data.emb::vector(1024),
                                    embedding_version = data.emb_ver,
                                    embedding_alt = data.emb_alt::vector(2000),
                                    embedding_alt_model = data.emb_alt_model,
                                    meta_data = data.meta::json
                                FROM (VALUES :values) AS data(id, emb, emb_ver, emb_alt, emb_alt_model, meta)
                                WHERE embedding_segments.id = data.id::integer
                            """)
                            values = [(d['id'], d['embedding'], d['embedding_version'], d['embedding_alt'], d['embedding_alt_model'], d['meta_data']) for d in update_data]
                        elif has_primary:
                            # Update only primary embedding
                            sql = text("""
                                UPDATE public.embedding_segments
                                SET embedding = data.emb::vector(1024),
                                    embedding_version = data.emb_ver,
                                    meta_data = data.meta::json
                                FROM (VALUES :values) AS data(id, emb, emb_ver, meta)
                                WHERE embedding_segments.id = data.id::integer
                            """)
                            values = [(d['id'], d['embedding'], d['embedding_version'], d['meta_data']) for d in update_data]
                        elif has_alternative:
                            # Update only alternative embedding
                            sql = text("""
                                UPDATE public.embedding_segments
                                SET embedding_alt = data.emb_alt::vector(2000),
                                    embedding_alt_model = data.emb_alt_model,
                                    meta_data = data.meta::json
                                FROM (VALUES :values) AS data(id, emb_alt, emb_alt_model, meta)
                                WHERE embedding_segments.id = data.id::integer
                            """)
                            values = [(d['id'], d['embedding_alt'], d['embedding_alt_model'], d['meta_data']) for d in update_data]

                        # Use raw connection for execute_values
                        connection = session.connection().connection
                        cursor = connection.cursor()

                        # Build VALUES string manually for psycopg2
                        from psycopg2.extras import execute_values

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
                                values
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
                                values
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
                                values
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
                del batch_texts, segments, update_data
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
                            torch.mps.empty_cache()
                        self._init_primary_model()
                        logger.info("Primary model reloaded")
                    if generate_alternative and self.alternative_model_obj:
                        del self.alternative_model_obj
                        self.alternative_model_obj = None
                        gc.collect()
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()
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


# CLI entry point for scheduled task execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Hydrator CLI")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding generation")
    parser.add_argument("--project", type=str, default=None, help="Filter to specific project")
    parser.add_argument("--primary-only", action="store_true", help="Only generate primary (0.6B) embeddings")
    parser.add_argument("--alternative-only", action="store_true", help="Only generate alternative (4B) embeddings")
    parser.add_argument("--alt-first", action="store_true", help="Prioritize alternative embeddings for configured projects")
    parser.add_argument("--force-reembed", action="store_true", help="Re-generate embeddings even if version exists")
    parser.add_argument("--stitch-version", type=str, default=None, help="Filter to specific stitch version prefix (e.g., 'stitch_v14')")

    args = parser.parse_args()

    # Parse stitch version filter
    stitch_versions = None
    if args.stitch_version:
        stitch_versions = {'prefix': [args.stitch_version]}

    # Run hydration
    hydrator = EmbeddingHydrator()
    result = asyncio.run(hydrator.hydrate_batch(
        batch_size=args.batch_size,
        project=args.project,
        primary_only=args.primary_only,
        alternative_only=args.alternative_only,
        alt_first=args.alt_first,
        force_reembed=args.force_reembed,
        stitch_versions=stitch_versions
    ))

    logger.info(f"Hydration result: {result}")

    # Exit with appropriate code
    sys.exit(0 if result.get('status') == 'success' else 1)
