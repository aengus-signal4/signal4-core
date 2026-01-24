"""
Content database queries for the system monitoring dashboard.
"""

from datetime import datetime, timezone

import streamlit as st
from sqlalchemy import text, func, and_, or_

from src.database.session import get_session
from src.database.models import Content
from src.utils.logger import setup_worker_logger
from ..config import load_config

logger = setup_worker_logger('system_monitoring')


@st.cache_data(ttl=60)
def get_pipeline_progress_from_db() -> dict:
    """Query pipeline progress directly from database"""
    try:
        with get_session() as session:
            emb_result = session.execute(text("""
                SELECT
                    COUNT(*) as total_segments,
                    COUNT(embedding) FILTER (WHERE embedding IS NOT NULL) as with_primary,
                    COUNT(embedding_alt) FILTER (WHERE embedding_alt IS NOT NULL) as with_alternative
                FROM embedding_segments
            """)).fetchone()

            spk_result = session.execute(text("""
                SELECT
                    COUNT(*) as total_speakers,
                    COUNT(speaker_identity_id) FILTER (WHERE speaker_identity_id IS NOT NULL) as with_identity,
                    COUNT(*) FILTER (WHERE text_evidence_status IS NOT NULL AND text_evidence_status NOT IN ('not_processed', 'unprocessed')) as phase2_processed,
                    COUNT(*) FILTER (WHERE text_evidence_status = 'certain') as phase2_certain,
                    COUNT(*) FILTER (WHERE text_evidence_status = 'none') as phase2_no_evidence,
                    COUNT(*) FILTER (WHERE duration > 60) as significant_duration
                FROM speakers
            """)).fetchone()

            identity_result = session.execute(text("""
                SELECT
                    COUNT(*) as total_identities,
                    COUNT(CASE WHEN primary_name IS NOT NULL AND primary_name != '' THEN 1 END) as named_identities
                FROM speaker_identities
            """)).fetchone()

            content_result = session.execute(text("""
                SELECT
                    COUNT(*) as total_content,
                    COUNT(*) FILTER (WHERE is_stitched = true) as stitched,
                    COUNT(*) FILTER (WHERE is_embedded = true) as embedded,
                    COUNT(*) FILTER (WHERE is_stitched = true AND is_embedded = false) as needs_embedding
                FROM content
                WHERE blocked_download = false AND is_duplicate = false AND is_short = false
            """)).fetchone()

        total_segments = emb_result.total_segments or 1
        total_speakers = spk_result.total_speakers or 1
        significant_speakers = spk_result.significant_duration or 1
        total_content = content_result.total_content or 1

        return {
            "embedding": {
                "total_segments": emb_result.total_segments,
                "primary": {
                    "completed": emb_result.with_primary,
                    "percent": round(100 * emb_result.with_primary / total_segments, 2)
                },
                "alternative": {
                    "completed": emb_result.with_alternative,
                    "percent": round(100 * emb_result.with_alternative / total_segments, 2)
                }
            },
            "speaker_identification": {
                "total_speakers": spk_result.total_speakers,
                "significant_duration_speakers": spk_result.significant_duration,
                "with_identity": {
                    "count": spk_result.with_identity,
                    "percent": round(100 * spk_result.with_identity / total_speakers, 2)
                },
                "phase2_text_evidence": {
                    "processed": spk_result.phase2_processed,
                    "certain": spk_result.phase2_certain,
                    "no_evidence": spk_result.phase2_no_evidence,
                    "percent_of_significant": round(100 * spk_result.phase2_processed / significant_speakers, 2) if spk_result.phase2_processed else 0
                },
                "identities": {
                    "total": identity_result.total_identities,
                    "named": identity_result.named_identities
                }
            },
            "content": {
                "total": content_result.total_content,
                "stitched": content_result.stitched,
                "embedded": content_result.embedded,
                "needs_embedding": content_result.needs_embedding,
                "percent_embedded": round(100 * content_result.embedded / total_content, 2)
            }
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=60)
def get_global_content_status() -> dict:
    """Get content processing status breakdown by project for top-level summary."""
    with get_session() as session:
        try:
            config = load_config()
            active_projects = [project for project, settings in config.get('active_projects', {}).items()
                             if settings.get('enabled', False)]

            if not active_projects:
                return {}

            project_data = {}
            total_content = 0

            for project in active_projects:
                project_settings = config.get('active_projects', {}).get(project, {})
                project_start_str = project_settings.get('start_date')
                project_end_str = project_settings.get('end_date')

                project_filters = [Content.projects.any(project)]

                if project_start_str:
                    project_start = datetime.strptime(project_start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    project_filters.append(Content.publish_date >= project_start)

                if project_end_str:
                    project_end = datetime.strptime(project_end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    project_filters.append(Content.publish_date <= project_end)

                if project_start_str or project_end_str:
                    project_filters.append(Content.publish_date.isnot(None))

                project_total = session.query(func.count(Content.content_id)).filter(*project_filters).scalar() or 0

                if project_total == 0:
                    continue

                total_content += project_total

                segment_embeddings = session.query(func.count()).filter(
                    *project_filters,
                    Content.is_embedded == True
                ).scalar() or 0

                stitched = session.query(func.count()).filter(
                    *project_filters,
                    Content.is_stitched == True,
                    or_(Content.is_embedded != True, Content.is_embedded.is_(None))
                ).scalar() or 0

                diarized_transcribed = session.query(func.count()).filter(
                    *project_filters,
                    Content.is_diarized == True,
                    Content.is_transcribed == True,
                    Content.is_stitched == False,
                    or_(Content.is_embedded != True, Content.is_embedded.is_(None))
                ).scalar() or 0

                audio_extracted = session.query(func.count()).filter(
                    *project_filters,
                    Content.is_converted == True,
                    Content.is_transcribed == False,
                    Content.is_diarized == False,
                    Content.is_stitched == False,
                    or_(Content.is_embedded != True, Content.is_embedded.is_(None))
                ).scalar() or 0

                downloaded_only = session.query(func.count()).filter(
                    *project_filters,
                    Content.is_downloaded == True,
                    Content.is_converted == False,
                    Content.is_transcribed == False,
                    Content.is_diarized == False,
                    Content.is_stitched == False,
                    or_(Content.is_embedded != True, Content.is_embedded.is_(None))
                ).scalar() or 0

                pending_download = session.query(func.count()).filter(
                    *project_filters,
                    Content.is_downloaded == False,
                    Content.blocked_download == False,
                    Content.is_converted == False,
                    Content.is_transcribed == False,
                    Content.is_diarized == False,
                    Content.is_stitched == False,
                    Content.is_embedded == False,
                    Content.is_compressed == False,
                    or_(
                        Content.duration >= 180,
                        Content.duration.is_(None),
                        and_(Content.platform != 'podcast', Content.duration < 180)
                    )
                ).scalar() or 0

                status_counts = {
                    'Segment Embeddings': segment_embeddings,
                    'Stitched': stitched,
                    'Diarized & Transcribed': diarized_transcribed,
                    'Audio Extracted': audio_extracted,
                    'Downloaded Only': downloaded_only,
                    'Pending Download': pending_download,
                }

                project_data[project] = {
                    'status_counts': status_counts,
                    'total_content': project_total,
                    'start_date': project_start_str,
                    'end_date': project_end_str
                }

            return {
                'project_data': project_data,
                'total_content': total_content,
                'active_projects': list(project_data.keys())
            }

        except Exception as e:
            logger.error(f"Database error in get_global_content_status: {str(e)}")
            session.rollback()
            return {}
