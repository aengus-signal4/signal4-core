"""
Database writer for theme classification results.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional, Set
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

from src.database.session import get_session
from src.database.models import ClassificationSchema, ThemeClassification, EmbeddingSegment
from .data_structures import SearchCandidate

logger = logging.getLogger(__name__)


class DatabaseWriter:
    """
    Handles database operations for theme classification.

    Features:
    - Schema loading and caching
    - Incremental batch inserts with upserts
    - Progress tracking via database queries
    - Efficient batch commits
    """

    def __init__(self, schema_name: str, schema_version: str, batch_size: int = 100):
        """
        Initialize database writer.

        Args:
            schema_name: Name of classification schema (e.g., 'CPRMV')
            schema_version: Version of schema (e.g., 'v1.0', '2025-01-15')
            batch_size: Number of records to batch before committing
        """
        self.schema_name = schema_name
        self.schema_version = schema_version
        self.batch_size = batch_size
        self.schema_id = None
        self.schema_data = None

    def load_or_create_schema(self, csv_path: str, description: Optional[str] = None,
                               query_embeddings: Optional[Dict] = None) -> int:
        """
        Load schema from database or create it from CSV if it doesn't exist.

        Args:
            csv_path: Path to CSV file with theme/subtheme definitions
            description: Optional schema description
            query_embeddings: Optional pre-computed query embeddings to cache

        Returns:
            schema_id: Database ID of the schema
        """
        with get_session() as session:
            # Try to load existing schema
            schema = session.query(ClassificationSchema).filter_by(
                name=self.schema_name,
                version=self.schema_version
            ).first()

            if schema:
                logger.info(f"Found existing schema: {self.schema_name} v{self.schema_version} (id={schema.id})")

                # Update query embeddings if provided and not already cached
                if query_embeddings and not schema.query_embeddings:
                    logger.info("Caching query embeddings to schema...")
                    schema.query_embeddings = query_embeddings
                    session.commit()
                    logger.info(f"Cached {len(query_embeddings)} subtheme embeddings")

                self.schema_id = schema.id
                self.schema_data = {
                    'themes': schema.themes_json,
                    'subthemes': schema.subthemes_json,
                    'query_embeddings': schema.query_embeddings
                }
                return schema.id

            # Create new schema from CSV
            logger.info(f"Creating new schema: {self.schema_name} v{self.schema_version}")
            themes, subthemes = self._load_schema_from_csv(csv_path)

            schema = ClassificationSchema(
                name=self.schema_name,
                version=self.schema_version,
                description=description,
                themes_json=themes,
                subthemes_json=subthemes,
                query_embeddings=query_embeddings,  # Cache embeddings on creation
                source_file=csv_path,
                created_at=datetime.now()
            )

            session.add(schema)
            session.commit()
            session.refresh(schema)

            logger.info(f"Created schema: {self.schema_name} v{self.schema_version} (id={schema.id})")
            logger.info(f"  Themes: {len(themes)}")
            logger.info(f"  Subthemes: {len(subthemes)}")
            if query_embeddings:
                logger.info(f"  Cached query embeddings: {len(query_embeddings)} subthemes")

            self.schema_id = schema.id
            self.schema_data = {
                'themes': themes,
                'subthemes': subthemes,
                'query_embeddings': query_embeddings
            }
            return schema.id

    def _load_schema_from_csv(self, csv_path: str) -> tuple:
        """Load themes and subthemes from CSV file"""
        df = pd.read_csv(csv_path)

        themes = {}
        subthemes = {}

        for theme_id in df['theme_id'].unique():
            theme_df = df[df['theme_id'] == theme_id]
            theme_info = theme_df.iloc[0]

            themes[str(theme_id)] = {
                'id': str(theme_id),
                'name': theme_info['theme_name'],
                'description_en': theme_info['theme_description'],
                'description_fr': theme_info.get('theme_description_fr', theme_info['theme_description'])
            }

            # Collect subthemes
            for _, row in theme_df.iterrows():
                if pd.notna(row.get('subtheme_id')):
                    subtheme_id = row['subtheme_id']
                    subthemes[str(subtheme_id)] = {
                        'id': str(subtheme_id),
                        'theme_id': str(theme_id),
                        'name': row['subtheme_name'],
                        'description_en': row.get('subtheme_description_short', row['subtheme_description']),
                        'description_fr': row.get('subtheme_description_short_fr',
                                                  row.get('subtheme_description_short', row['subtheme_description']))
                    }

        return themes, subthemes

    def get_processed_segments(self) -> Set[int]:
        """
        Get set of segment IDs that have already been processed for this schema.

        Returns:
            Set of segment IDs
        """
        if not self.schema_id:
            raise ValueError("Schema not loaded. Call load_or_create_schema() first.")

        with get_session() as session:
            result = session.execute(
                text("SELECT segment_id FROM theme_classifications WHERE schema_id = :schema_id"),
                {'schema_id': self.schema_id}
            )
            segment_ids = {row[0] for row in result}
            logger.info(f"Found {len(segment_ids)} already processed segments")
            return segment_ids

    def write_candidates_batch(self, candidates: List[SearchCandidate], commit: bool = True):
        """
        Write a batch of candidates to database using upsert (INSERT ... ON CONFLICT UPDATE).

        Args:
            candidates: List of SearchCandidate objects with classification results
            commit: Whether to commit immediately (default True)
        """
        if not self.schema_id:
            raise ValueError("Schema not loaded. Call load_or_create_schema() first.")

        if not candidates:
            return

        with get_session() as session:
            # Batch fetch embeddings for all candidates
            segment_ids = [c.segment_id for c in candidates]
            embeddings_map = {}

            segments = session.query(EmbeddingSegment).filter(
                EmbeddingSegment.id.in_(segment_ids)
            ).all()

            for seg in segments:
                if seg.embedding_alt is not None:
                    embeddings_map[seg.id] = seg.embedding_alt

            # Prepare records for upsert
            records = []
            for candidate in candidates:
                # Calculate final confidence scores
                final_conf_scores = self._calculate_final_confidence(candidate)

                # Convert theme/subtheme IDs to strings
                theme_ids = [str(tid) for tid in candidate.theme_ids] if candidate.theme_ids else []

                # Collect all subtheme IDs from subtheme_results
                subtheme_ids = []
                if candidate.subtheme_results:
                    for theme_id, result in candidate.subtheme_results.items():
                        sub_ids = result.get('subtheme_ids', [])
                        subtheme_ids.extend([str(sid) for sid in sub_ids])

                # high_confidence_themes = subtheme_ids from themes with confidence >= 0.75
                high_conf_themes = []
                if candidate.subtheme_results:
                    for theme_id, result in candidate.subtheme_results.items():
                        theme_conf = final_conf_scores.get(str(theme_id), 0.0)
                        if theme_conf >= 0.75:
                            sub_ids = result.get('subtheme_ids', [])
                            high_conf_themes.extend([str(sid) for sid in sub_ids])

                # Prepare stage similarities
                stage1_sims = {}
                if candidate.theme_similarities:
                    for tid, score in candidate.theme_similarities.items():
                        stage1_sims[f"theme_{tid}"] = float(score)
                if candidate.subtheme_similarities:
                    for sid, score in candidate.subtheme_similarities.items():
                        stage1_sims[f"subtheme_{sid}"] = float(score)

                # Convert numpy types to native Python types for JSONB fields
                def sanitize_for_jsonb(obj):
                    """Recursively convert numpy types to native Python types"""
                    if obj is None:
                        return None
                    elif isinstance(obj, dict):
                        return {str(k): sanitize_for_jsonb(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [sanitize_for_jsonb(item) for item in obj]
                    elif hasattr(obj, 'item'):  # numpy scalar
                        return obj.item()
                    elif isinstance(obj, (int, float, str, bool)):
                        return obj
                    else:
                        return str(obj)

                record = {
                    'segment_id': int(candidate.segment_id),
                    'schema_id': self.schema_id,
                    'theme_ids': theme_ids,
                    'subtheme_ids': subtheme_ids,
                    'high_confidence_themes': high_conf_themes,
                    'stage1_similarities': stage1_sims if stage1_sims else None,
                    'stage3_results': sanitize_for_jsonb(candidate.subtheme_results),
                    'stage4_validations': sanitize_for_jsonb(candidate.validation_results),
                    'final_confidence_scores': sanitize_for_jsonb(final_conf_scores),
                    'matched_via': candidate.matched_via,
                    'max_similarity_score': float(candidate.similarity_score) if candidate.similarity_score else None,
                    'embedding': embeddings_map.get(candidate.segment_id),
                    'updated_at': datetime.now()
                }

                records.append(record)

            # Upsert using INSERT ... ON CONFLICT UPDATE
            stmt = insert(ThemeClassification).values(records)
            stmt = stmt.on_conflict_do_update(
                constraint='uq_segment_schema',
                set_={
                    'theme_ids': stmt.excluded.theme_ids,
                    'subtheme_ids': stmt.excluded.subtheme_ids,
                    'high_confidence_themes': stmt.excluded.high_confidence_themes,
                    'stage1_similarities': stmt.excluded.stage1_similarities,
                    'stage3_results': stmt.excluded.stage3_results,
                    'stage4_validations': stmt.excluded.stage4_validations,
                    'final_confidence_scores': stmt.excluded.final_confidence_scores,
                    'matched_via': stmt.excluded.matched_via,
                    'max_similarity_score': stmt.excluded.max_similarity_score,
                    'embedding': stmt.excluded.embedding,
                    'updated_at': stmt.excluded.updated_at
                }
            )

            session.execute(stmt)

            if commit:
                session.commit()
                logger.info(f"Wrote batch of {len(records)} records to database")

    def _calculate_final_confidence(self, candidate: SearchCandidate) -> Dict[str, float]:
        """
        Calculate final confidence scores for themes/subthemes.

        Uses validation results if available, otherwise uses stage 3 confidence.

        Handles two validation formats:
        1. Likert format: {'likert_score': 4, 'confidence': 0.8}
        2. Subthemes format: {'subthemes': {'sub_id': {'confidence': 0.8}}}

        Returns:
            Dict mapping theme_id to confidence score (0.0 - 1.0)
        """
        confidence_scores = {}

        if not candidate.validation_results:
            # No validation, use stage 3 confidence
            if candidate.subtheme_results:
                for theme_id, result in candidate.subtheme_results.items():
                    confidence_scores[str(theme_id)] = result.get('confidence', 0.0)
            return confidence_scores

        # Use validation results
        for theme_id, validation in candidate.validation_results.items():
            # Format 1: Likert scale (from _validate_subthemes_async)
            # {'likert_score': 4, 'confidence': 0.8}
            if 'likert_score' in validation:
                likert = validation.get('likert_score', 3)
                # Convert Likert 1-5 to confidence 0.0-1.0
                # 5 = 1.0, 4 = 0.8, 3 = 0.5, 2 = 0.25, 1 = 0.0
                likert_to_conf = {5: 1.0, 4: 0.8, 3: 0.5, 2: 0.25, 1: 0.0}
                confidence_scores[str(theme_id)] = likert_to_conf.get(likert, 0.5)

            # Format 2: Subthemes structure (from old stage4_validate)
            # {'subthemes': {'sub_id': {'confidence': 0.8}}}
            elif 'subthemes' in validation:
                subtheme_confs = []
                for sub_id, sub_val in validation['subthemes'].items():
                    conf = sub_val.get('confidence', 0.0)
                    confidence_scores[str(sub_id)] = conf
                    subtheme_confs.append(conf)

                if subtheme_confs:
                    confidence_scores[str(theme_id)] = sum(subtheme_confs) / len(subtheme_confs)

            # Format 3: Direct confidence (fallback)
            elif 'confidence' in validation:
                confidence_scores[str(theme_id)] = validation.get('confidence', 0.5)

        return confidence_scores

    def write_candidates_incremental(self, candidates: List[SearchCandidate]):
        """
        Write candidates incrementally in batches with periodic commits.

        Args:
            candidates: List of SearchCandidate objects
        """
        if not candidates:
            return

        logger.info(f"Writing {len(candidates)} candidates to database in batches of {self.batch_size}")

        total_written = 0
        for i in range(0, len(candidates), self.batch_size):
            batch = candidates[i:i + self.batch_size]
            self.write_candidates_batch(batch, commit=True)
            total_written += len(batch)

            if total_written % (self.batch_size * 5) == 0:
                logger.info(f"  Progress: {total_written}/{len(candidates)} records written")

        logger.info(f"Completed writing {total_written} records to database")

    def get_pending_count(self) -> int:
        """
        Get count of pending candidates (stage3_results IS NULL) for this schema.

        Returns:
            Number of pending candidates
        """
        if not self.schema_id:
            raise ValueError("Schema not loaded. Call load_or_create_schema() first.")

        with get_session() as session:
            result = session.execute(
                text("""
                    SELECT COUNT(*)
                    FROM theme_classifications
                    WHERE schema_id = :schema_id
                      AND (stage3_results IS NULL OR stage3_results = 'null'::jsonb)
                """),
                {'schema_id': self.schema_id}
            )
            return result.scalar() or 0

    def bulk_insert_stage1_candidates(self, candidates: List[SearchCandidate]) -> int:
        """
        Bulk insert Stage 1 results into theme_classifications.

        Only inserts segment_id, schema_id, stage1_similarities, theme_ids.
        Leaves stage3_results NULL to mark as pending.

        Args:
            candidates: List of SearchCandidate objects from Stage 1

        Returns:
            Number of rows inserted (excludes conflicts)
        """
        if not self.schema_id:
            raise ValueError("Schema not loaded. Call load_or_create_schema() first.")

        if not candidates:
            return 0

        logger.info(f"Bulk inserting {len(candidates)} Stage 1 candidates...")

        # Process in batches to avoid memory issues
        batch_size = 5000
        total_inserted = 0

        for i in range(0, len(candidates), batch_size):
            batch = candidates[i:i + batch_size]

            records = []
            for c in batch:
                # Prepare stage1_similarities - convert all keys/values to native Python types
                stage1_sims = {}
                if c.subtheme_similarities:
                    for sid, score in c.subtheme_similarities.items():
                        # Convert numpy types to native Python
                        key = str(sid)
                        val = float(score.item()) if hasattr(score, 'item') else float(score)
                        stage1_sims[key] = val

                # Ensure theme_ids are native Python strings
                theme_ids = []
                if c.theme_ids:
                    for t in c.theme_ids:
                        if hasattr(t, 'item'):
                            theme_ids.append(str(t.item()))
                        else:
                            theme_ids.append(str(t))

                records.append({
                    'segment_id': int(c.segment_id.item()) if hasattr(c.segment_id, 'item') else int(c.segment_id),
                    'schema_id': self.schema_id,
                    'theme_ids': theme_ids,
                    'subtheme_ids': [],
                    'high_confidence_themes': [],
                    'stage1_similarities': stage1_sims if stage1_sims else None,
                    'stage3_results': None,  # Marks as pending
                    'stage4_validations': None,
                    'final_confidence_scores': {},
                    'matched_via': c.matched_via,
                    'max_similarity_score': float(c.similarity_score.item()) if hasattr(c.similarity_score, 'item') else (float(c.similarity_score) if c.similarity_score else None),
                    'embedding': None,  # Don't fetch embedding until processing
                    'updated_at': datetime.now()
                })

            with get_session() as session:
                stmt = insert(ThemeClassification).values(records)
                stmt = stmt.on_conflict_do_nothing(constraint='uq_segment_schema')
                result = session.execute(stmt)
                session.commit()
                total_inserted += result.rowcount

            logger.info(f"  Batch {i // batch_size + 1}: inserted {result.rowcount} rows")

        logger.info(f"Bulk insert complete: {total_inserted} new candidates")
        return total_inserted

    def get_pending_candidates_with_metadata(self, batch_size: int = 100) -> List[Dict]:
        """
        Get pending candidates with metadata in a single query.

        Uses speaker_positions for speaker attribution (no speaker_transcriptions join).

        Args:
            batch_size: Number of candidates to fetch

        Returns:
            List of dicts with candidate data and metadata
        """
        if not self.schema_id:
            raise ValueError("Schema not loaded. Call load_or_create_schema() first.")

        with get_session() as session:
            query = text("""
                SELECT
                    tc.id as tc_id,
                    tc.segment_id,
                    tc.theme_ids,
                    tc.stage1_similarities,
                    tc.max_similarity_score,
                    tc.matched_via,
                    es.text as segment_text,
                    es.start_time,
                    es.end_time,
                    es.segment_index,
                    es.content_id,
                    es.speaker_positions,
                    es.embedding_alt,
                    c.title as episode_title,
                    c.channel_name as episode_channel,
                    c.content_id as content_id_string
                FROM theme_classifications tc
                JOIN embedding_segments es ON tc.segment_id = es.id
                JOIN content c ON es.content_id = c.id
                WHERE tc.schema_id = :schema_id
                  AND (tc.stage3_results IS NULL OR tc.stage3_results = 'null'::jsonb)
                ORDER BY tc.max_similarity_score DESC
                LIMIT :limit
            """)
            result = session.execute(query, {
                'schema_id': self.schema_id,
                'limit': batch_size
            })
            return [dict(row._mapping) for row in result.fetchall()]

    def update_classification_result(self, tc_id: int, candidate: SearchCandidate, embedding=None):
        """
        Update a theme_classification row with LLM results.

        Args:
            tc_id: ID of the theme_classifications row
            candidate: SearchCandidate with classification results
            embedding: Optional embedding vector to store
        """
        # Calculate final confidence scores
        final_conf_scores = self._calculate_final_confidence(candidate)

        # Collect all subtheme IDs
        subtheme_ids = []
        if candidate.subtheme_results:
            for theme_id, result in candidate.subtheme_results.items():
                sub_ids = result.get('subtheme_ids', [])
                subtheme_ids.extend([str(sid) for sid in sub_ids])

        # high_confidence_themes = subtheme_ids from themes with confidence >= 0.75
        high_conf_themes = []
        if candidate.subtheme_results:
            for theme_id, result in candidate.subtheme_results.items():
                theme_conf = final_conf_scores.get(str(theme_id), 0.0)
                if theme_conf >= 0.75:
                    sub_ids = result.get('subtheme_ids', [])
                    high_conf_themes.extend([str(sid) for sid in sub_ids])

        # Convert numpy types to native Python types for JSONB fields
        def sanitize_for_jsonb(obj):
            if obj is None:
                return None
            elif isinstance(obj, dict):
                return {str(k): sanitize_for_jsonb(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [sanitize_for_jsonb(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, (int, float, str, bool)):
                return obj
            else:
                return str(obj)

        import json

        with get_session() as session:
            # Convert JSONB fields to JSON strings for raw SQL
            stage3_json = json.dumps(sanitize_for_jsonb(candidate.subtheme_results)) if candidate.subtheme_results else None
            stage4_json = json.dumps(sanitize_for_jsonb(candidate.validation_results)) if candidate.validation_results else None
            final_conf_json = json.dumps(sanitize_for_jsonb(final_conf_scores))

            # Use CAST() instead of :: to avoid SQLAlchemy parameter parsing issues
            session.execute(
                text("""
                    UPDATE theme_classifications
                    SET theme_ids = :theme_ids,
                        subtheme_ids = :subtheme_ids,
                        high_confidence_themes = :high_confidence_themes,
                        stage3_results = CAST(:stage3_results AS jsonb),
                        stage4_validations = CAST(:stage4_validations AS jsonb),
                        final_confidence_scores = CAST(:final_confidence_scores AS jsonb),
                        embedding = :embedding,
                        updated_at = :updated_at
                    WHERE id = :tc_id
                """),
                {
                    'theme_ids': [str(t) for t in candidate.theme_ids] if candidate.theme_ids else [],
                    'subtheme_ids': subtheme_ids,
                    'high_confidence_themes': high_conf_themes,
                    'stage3_results': stage3_json,
                    'stage4_validations': stage4_json,
                    'final_confidence_scores': final_conf_json,
                    'embedding': embedding,
                    'updated_at': datetime.now(),
                    'tc_id': tc_id
                }
            )
            session.commit()

    # ============ Stage 5: FINAL CHECK Methods ============

    def get_pending_stage5_count(self) -> int:
        """
        Get count of HIGH CONFIDENCE candidates that haven't had Stage 5 FINAL CHECK.

        Only processes segments with high_confidence_themes (confidence >= 0.75).

        Returns:
            Number of pending Stage 5 candidates
        """
        if not self.schema_id:
            raise ValueError("Schema not loaded. Call load_or_create_schema() first.")

        with get_session() as session:
            result = session.execute(
                text("""
                    SELECT COUNT(*)
                    FROM theme_classifications
                    WHERE schema_id = :schema_id
                      AND stage3_results IS NOT NULL
                      AND stage3_results != 'null'::jsonb
                      AND high_confidence_themes IS NOT NULL
                      AND high_confidence_themes != '{}'
                      AND (stage5_final_check IS NULL OR stage5_final_check = 'null'::jsonb)
                """),
                {'schema_id': self.schema_id}
            )
            return result.scalar() or 0

    def get_pending_stage5_candidates(self, batch_size: int = 100) -> List[Dict]:
        """
        Get HIGH CONFIDENCE candidates that need Stage 5 FINAL CHECK.

        Only fetches segments with high_confidence_themes (confidence >= 0.75).

        Args:
            batch_size: Number of candidates to fetch

        Returns:
            List of dicts with candidate data and text for final check
        """
        if not self.schema_id:
            raise ValueError("Schema not loaded. Call load_or_create_schema() first.")

        with get_session() as session:
            query = text("""
                SELECT
                    tc.id as tc_id,
                    tc.segment_id,
                    tc.theme_ids,
                    tc.subtheme_ids,
                    tc.high_confidence_themes,
                    es.text as segment_text,
                    es.speaker_positions,
                    c.main_language
                FROM theme_classifications tc
                JOIN embedding_segments es ON tc.segment_id = es.id
                JOIN content c ON es.content_id = c.id
                WHERE tc.schema_id = :schema_id
                  AND tc.stage3_results IS NOT NULL
                  AND tc.stage3_results != 'null'::jsonb
                  AND tc.high_confidence_themes IS NOT NULL
                  AND tc.high_confidence_themes != '{}'
                  AND (tc.stage5_final_check IS NULL OR tc.stage5_final_check = 'null'::jsonb)
                ORDER BY tc.id
                LIMIT :limit
            """)
            result = session.execute(query, {
                'schema_id': self.schema_id,
                'limit': batch_size
            })
            return [dict(row._mapping) for row in result.fetchall()]

    def update_stage5_result(self, tc_id: int, is_relevant: bool, reasoning: str, model_used: str):
        """
        Update a theme_classification row with Stage 5 FINAL CHECK result.

        Args:
            tc_id: ID of the theme_classifications row
            is_relevant: Whether segment is relevant to gender-based violence
            reasoning: LLM reasoning for the decision
            model_used: Model used for final check (e.g., 'tier_1')
        """
        import json

        stage5_result = {
            'is_relevant': is_relevant,
            'reasoning': reasoning,
            'model': model_used,
            'checked_at': datetime.now().isoformat()
        }

        with get_session() as session:
            session.execute(
                text("""
                    UPDATE theme_classifications
                    SET stage5_final_check = CAST(:stage5_result AS jsonb),
                        updated_at = :updated_at
                    WHERE id = :tc_id
                """),
                {
                    'stage5_result': json.dumps(stage5_result),
                    'updated_at': datetime.now(),
                    'tc_id': tc_id
                }
            )
            session.commit()

    def bulk_update_stage5_results(self, results: List[Dict]):
        """
        Bulk update Stage 5 results for multiple candidates.

        Args:
            results: List of dicts with 'tc_id', 'is_relevant', 'reasoning', 'model'
        """
        import json

        if not results:
            return

        with get_session() as session:
            for r in results:
                stage5_result = {
                    'is_relevant': r['is_relevant'],
                    'reasoning': r.get('reasoning', ''),
                    'relevance': r.get('relevance'),
                    'model': r.get('model', 'tier_1'),
                    'checked_at': datetime.now().isoformat()
                }

                session.execute(
                    text("""
                        UPDATE theme_classifications
                        SET stage5_final_check = CAST(:stage5_result AS jsonb),
                            updated_at = :updated_at
                        WHERE id = :tc_id
                    """),
                    {
                        'stage5_result': json.dumps(stage5_result),
                        'updated_at': datetime.now(),
                        'tc_id': r['tc_id']
                    }
                )

            session.commit()
            logger.info(f"Updated {len(results)} Stage 5 results")

    # ============ Stage 6: FALSE POSITIVE DETECTION Methods ============

    def get_pending_stage6_count(self) -> int:
        """
        Get count of candidates that passed Stage 5 but haven't had Stage 6 false positive check.

        Only processes segments where stage5_final_check.is_relevant = true.

        Returns:
            Number of pending Stage 6 candidates
        """
        if not self.schema_id:
            raise ValueError("Schema not loaded. Call load_or_create_schema() first.")

        with get_session() as session:
            result = session.execute(
                text("""
                    SELECT COUNT(*)
                    FROM theme_classifications
                    WHERE schema_id = :schema_id
                      AND stage5_final_check IS NOT NULL
                      AND stage5_final_check->>'is_relevant' = 'true'
                      AND stage6_false_positive_check IS NULL
                """),
                {'schema_id': self.schema_id}
            )
            return result.scalar() or 0

    def get_pending_stage6_candidates(self, batch_size: int = 100) -> List[Dict]:
        """
        Get candidates that passed Stage 5 and need Stage 6 false positive detection.

        Args:
            batch_size: Number of candidates to fetch

        Returns:
            List of dicts with candidate data and text for false positive check
        """
        if not self.schema_id:
            raise ValueError("Schema not loaded. Call load_or_create_schema() first.")

        with get_session() as session:
            query = text("""
                SELECT
                    tc.id as tc_id,
                    tc.segment_id,
                    tc.theme_ids,
                    tc.subtheme_ids,
                    tc.high_confidence_themes,
                    tc.stage5_final_check,
                    es.text as segment_text,
                    es.speaker_positions,
                    c.main_language
                FROM theme_classifications tc
                JOIN embedding_segments es ON tc.segment_id = es.id
                JOIN content c ON es.content_id = c.id
                WHERE tc.schema_id = :schema_id
                  AND tc.stage5_final_check IS NOT NULL
                  AND tc.stage5_final_check->>'is_relevant' = 'true'
                  AND tc.stage6_false_positive_check IS NULL
                ORDER BY tc.id
                LIMIT :limit
            """)
            result = session.execute(query, {
                'schema_id': self.schema_id,
                'limit': batch_size
            })
            return [dict(row._mapping) for row in result.fetchall()]

    def bulk_update_stage6_results(self, results: List[Dict]):
        """
        Bulk update Stage 6 false positive detection results.

        Args:
            results: List of dicts with:
                - tc_id: int
                - is_false_positive: bool
                - problematic_content: str|None (description of detected content)
                - reasoning: str
                - speaker_stance: str ('strongly_holds'|'holds'|'leans_holds'|'neutral'|'leans_rejects'|'rejects'|'strongly_rejects')
                - model: str
        """
        import json

        if not results:
            return

        with get_session() as session:
            for r in results:
                stage6_result = {
                    'is_false_positive': r['is_false_positive'],
                    'problematic_content': r.get('problematic_content'),
                    'reasoning': r.get('reasoning', ''),
                    'speaker_stance': r.get('speaker_stance'),
                    'model': r.get('model', 'tier_1'),
                    'checked_at': datetime.now().isoformat()
                }

                session.execute(
                    text("""
                        UPDATE theme_classifications
                        SET stage6_false_positive_check = CAST(:stage6_result AS jsonb),
                            updated_at = :updated_at
                        WHERE id = :tc_id
                    """),
                    {
                        'stage6_result': json.dumps(stage6_result),
                        'updated_at': datetime.now(),
                        'tc_id': r['tc_id']
                    }
                )

            session.commit()
            logger.info(f"Updated {len(results)} Stage 6 results")

    # ============ Stage 7: EXPANDED CONTEXT RE-CHECK Methods ============

    def get_pending_stage7_count(self) -> int:
        """
        Get count of Stage 6 false positives that haven't had Stage 7 expanded context check.

        Only processes segments where stage6 speaker_stance is in reject categories.

        Returns:
            Number of pending Stage 7 candidates
        """
        if not self.schema_id:
            raise ValueError("Schema not loaded. Call load_or_create_schema() first.")

        with get_session() as session:
            result = session.execute(
                text("""
                    SELECT COUNT(*)
                    FROM theme_classifications
                    WHERE schema_id = :schema_id
                      AND stage6_false_positive_check IS NOT NULL
                      AND stage6_false_positive_check->>'speaker_stance' IN ('neutral', 'leans_rejects', 'rejects', 'strongly_rejects')
                      AND stage7_expanded_context IS NULL
                """),
                {'schema_id': self.schema_id}
            )
            return result.scalar() or 0

    def get_pending_stage7_candidates(self, batch_size: int = 50, context_window_seconds: int = 20) -> List[Dict]:
        """
        Get Stage 6 false positives that need Stage 7 expanded context check.

        Fetches the segment plus surrounding context within ±context_window_seconds.

        Args:
            batch_size: Number of candidates to fetch
            context_window_seconds: Seconds of context to include before/after segment

        Returns:
            List of dicts with candidate data and expanded text for re-check
        """
        if not self.schema_id:
            raise ValueError("Schema not loaded. Call load_or_create_schema() first.")

        with get_session() as session:
            # First get the base candidates
            query = text("""
                SELECT
                    tc.id as tc_id,
                    tc.segment_id,
                    tc.theme_ids,
                    tc.subtheme_ids,
                    tc.high_confidence_themes,
                    tc.stage6_false_positive_check,
                    es.text as segment_text,
                    es.speaker_positions,
                    es.content_id,
                    es.start_time,
                    es.end_time,
                    c.main_language
                FROM theme_classifications tc
                JOIN embedding_segments es ON tc.segment_id = es.id
                JOIN content c ON es.content_id = c.id
                WHERE tc.schema_id = :schema_id
                  AND tc.stage6_false_positive_check IS NOT NULL
                  AND tc.stage6_false_positive_check->>'speaker_stance' IN ('neutral', 'leans_rejects', 'rejects', 'strongly_rejects')
                  AND tc.stage7_expanded_context IS NULL
                ORDER BY tc.id
                LIMIT :limit
            """)
            result = session.execute(query, {
                'schema_id': self.schema_id,
                'limit': batch_size
            })
            candidates = [dict(row._mapping) for row in result.fetchall()]

            # For each candidate, fetch expanded context
            for candidate in candidates:
                content_id = candidate['content_id']
                start_time = candidate['start_time']
                end_time = candidate['end_time']

                # Get surrounding segments within the time window
                context_query = text("""
                    SELECT
                        es.text,
                        es.start_time,
                        es.end_time,
                        es.speaker_positions
                    FROM embedding_segments es
                    WHERE es.content_id = :content_id
                      AND (
                          (es.end_time >= :window_start AND es.start_time <= :window_end)
                      )
                    ORDER BY es.start_time
                """)
                context_result = session.execute(context_query, {
                    'content_id': content_id,
                    'window_start': start_time - context_window_seconds,
                    'window_end': end_time + context_window_seconds
                })
                context_segments = [dict(row._mapping) for row in context_result.fetchall()]

                # Combine text from all context segments
                expanded_texts = []
                all_speaker_positions = {}
                text_offset = 0

                for seg in context_segments:
                    seg_text = seg['text']
                    expanded_texts.append(seg_text)

                    # Merge speaker positions with offset
                    if seg.get('speaker_positions'):
                        for speaker_id, positions in seg['speaker_positions'].items():
                            if speaker_id not in all_speaker_positions:
                                all_speaker_positions[speaker_id] = []
                            for start, end in positions:
                                all_speaker_positions[speaker_id].append([
                                    start + text_offset,
                                    end + text_offset
                                ])

                    text_offset += len(seg_text) + 1  # +1 for space

                candidate['expanded_text'] = ' '.join(expanded_texts)
                candidate['expanded_speaker_positions'] = all_speaker_positions
                candidate['context_window_seconds'] = context_window_seconds
                candidate['original_stance'] = candidate['stage6_false_positive_check'].get('speaker_stance')

            return candidates

    def bulk_update_stage7_results(self, results: List[Dict]):
        """
        Bulk update Stage 7 expanded context re-check results.

        Args:
            results: List of dicts with:
                - tc_id: int
                - is_false_positive: bool
                - speaker_stance: str
                - reasoning: str
                - original_stance: str (from stage6)
                - context_window_seconds: int
                - model: str
        """
        import json

        if not results:
            return

        with get_session() as session:
            for r in results:
                stage7_result = {
                    'is_false_positive': r['is_false_positive'],
                    'speaker_stance': r.get('speaker_stance'),
                    'reasoning': r.get('reasoning', ''),
                    'original_stance': r.get('original_stance'),
                    'context_window_seconds': r.get('context_window_seconds', 20),
                    'model': r.get('model', 'tier_1'),
                    'checked_at': datetime.now().isoformat()
                }

                session.execute(
                    text("""
                        UPDATE theme_classifications
                        SET stage7_expanded_context = CAST(:stage7_result AS jsonb),
                            updated_at = :updated_at
                        WHERE id = :tc_id
                    """),
                    {
                        'stage7_result': json.dumps(stage7_result),
                        'updated_at': datetime.now(),
                        'tc_id': r['tc_id']
                    }
                )

            session.commit()
            logger.info(f"Updated {len(results)} Stage 7 results")