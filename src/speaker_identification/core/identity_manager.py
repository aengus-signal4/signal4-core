#!/usr/bin/env python3
"""
Identity Manager for Speaker Identification
============================================

CRUD operations for SpeakerIdentity records.
Creates identities, assigns speakers, and manages verification status.
"""

from typing import Dict, List, Optional
import json
import numpy as np
from sqlalchemy import text
from datetime import datetime

from src.database.session import get_session
from src.database.models import SpeakerIdentity, IdentificationStatus
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('speaker_identification.identity_manager')


class IdentityManager:
    """Manage SpeakerIdentity records and speaker assignments."""

    def create_or_match_identity(
        self,
        name: str,
        role: str,
        confidence: float,
        method: str = "metadata_identification",
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Create new identity or match existing by name.

        Args:
            name: Full name of speaker
            role: 'host' or 'guest'
            confidence: Confidence score (0.0-1.0)
            method: Identification method
            metadata: Additional metadata

        Returns:
            identity_id of created or matched identity
        """
        with get_session() as session:
            # Try to find existing identity with exact name match
            query = text("""
                SELECT id
                FROM speaker_identities
                WHERE LOWER(primary_name) = LOWER(:name)
                  AND is_active = TRUE
                LIMIT 1
            """)

            existing = session.execute(query, {'name': name}).fetchone()

            if existing:
                logger.info(f"Matched existing identity: '{name}' (ID: {existing.id})")
                return existing.id

            # Create new identity
            verification_status = self._get_verification_status(confidence)

            # Convert numpy types to native Python types
            def convert_numpy(obj):
                """Recursively convert numpy types to native Python types."""
                import numpy as np
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj

            metadata_dict = convert_numpy(metadata or {})
            metadata_dict.update({
                'role': role,
                'identified_at': datetime.now().isoformat(),
                'identification_method': method,
                'initial_confidence': float(confidence)
            })

            identity = SpeakerIdentity(
                primary_name=name,
                confidence_score=float(confidence),
                verification_status=verification_status,
                verification_metadata=metadata_dict,
                role=role,
                is_active=True
            )

            session.add(identity)
            session.commit()
            session.refresh(identity)

            logger.info(
                f"Created identity: '{name}' (ID: {identity.id}, "
                f"status: {verification_status}, confidence: {confidence:.2f})"
            )

            return identity.id

    def assign_speaker_to_identity(
        self,
        speaker_id: int,
        identity_id: int,
        confidence: float,
        method: str,
        metadata: Optional[Dict] = None
    ):
        """
        Assign a speaker to an identity.

        Args:
            speaker_id: Speaker ID
            identity_id: SpeakerIdentity ID
            confidence: Assignment confidence (0.0-1.0)
            method: Assignment method
            metadata: Additional metadata
        """
        with get_session() as session:
            # Update speaker record
            import json

            # Store assignment reasoning in meta_data
            metadata_json = json.dumps(metadata or {})

            query = text("""
                UPDATE speakers
                SET
                    speaker_identity_id = :identity_id,
                    assignment_confidence = :confidence,
                    assignment_method = :method,
                    identification_status = :status,
                    meta_data = COALESCE(meta_data, '{}'::jsonb) || CAST(:metadata AS jsonb)
                WHERE id = :speaker_id
            """)

            session.execute(query, {
                'speaker_id': int(speaker_id),
                'identity_id': int(identity_id),
                'confidence': float(confidence),
                'method': method,
                'status': IdentificationStatus.ASSIGNED,
                'metadata': metadata_json
            })
            session.commit()

            logger.debug(f"Assigned speaker {speaker_id} to identity {identity_id}")

    def bulk_assign_speakers(
        self,
        assignments: list[Dict],
        method: str
    ):
        """
        Bulk assign multiple speakers to identities.

        Args:
            assignments: List of dicts with keys:
                        {speaker_id, identity_id, confidence, metadata}
            method: Assignment method for all
        """
        if not assignments:
            return

        with get_session() as session:
            # Use UNNEST for efficient bulk update
            import json
            speaker_ids = [int(a['speaker_id']) for a in assignments]
            identity_ids = [int(a['identity_id']) for a in assignments]
            confidences = [float(a['confidence']) for a in assignments]
            metadata_list = [json.dumps(a.get('metadata', {})) for a in assignments]

            query = text("""
                UPDATE speakers s
                SET
                    speaker_identity_id = u.identity_id,
                    assignment_confidence = u.confidence,
                    assignment_method = :method,
                    identification_status = :status,
                    meta_data = COALESCE(s.meta_data, '{}'::jsonb) || CAST(u.metadata AS jsonb)
                FROM (
                    SELECT
                        UNNEST(CAST(:speaker_ids AS int[])) as speaker_id,
                        UNNEST(CAST(:identity_ids AS int[])) as identity_id,
                        UNNEST(CAST(:confidences AS float[])) as confidence,
                        UNNEST(CAST(:metadata_list AS text[])) as metadata
                ) u
                WHERE s.id = u.speaker_id
            """)

            session.execute(query, {
                'speaker_ids': speaker_ids,
                'identity_ids': identity_ids,
                'confidences': confidences,
                'metadata_list': metadata_list,
                'method': method,
                'status': IdentificationStatus.ASSIGNED
            })
            session.commit()

            logger.info(f"Bulk assigned {len(assignments)} speakers using {method}")

    def get_identity_by_name(self, name: str) -> Optional[Dict]:
        """
        Get identity by name.

        Args:
            name: Speaker name (case-insensitive)

        Returns:
            Identity dict or None
        """
        with get_session() as session:
            query = text("""
                SELECT
                    id,
                    primary_name,
                    confidence_score,
                    verification_status,
                    verification_metadata,
                    bio,
                    occupation
                FROM speaker_identities
                WHERE LOWER(primary_name) = LOWER(:name)
                  AND is_active = TRUE
                LIMIT 1
            """)

            result = session.execute(query, {'name': name}).fetchone()

            if not result:
                return None

            return {
                'id': result.id,
                'primary_name': result.primary_name,
                'confidence_score': result.confidence_score,
                'verification_status': result.verification_status,
                'verification_metadata': result.verification_metadata or {},
                'bio': result.bio,
                'occupation': result.occupation
            }

    def get_identity_speaker_count(self, identity_id: int) -> int:
        """
        Get count of speakers assigned to an identity.

        Args:
            identity_id: Identity ID

        Returns:
            Number of assigned speakers
        """
        with get_session() as session:
            query = text("""
                SELECT COUNT(*)
                FROM speakers
                WHERE speaker_identity_id = :identity_id
            """)

            result = session.execute(query, {'identity_id': identity_id}).scalar()
            return result or 0

    def _get_verification_status(self, confidence: float) -> str:
        """
        Determine verification status based on confidence score.

        Args:
            confidence: Confidence score (0.0-1.0)

        Returns:
            Verification status string
        """
        if confidence >= 0.85:
            return 'verified'
        elif confidence >= 0.70:
            return 'llm_identified'
        elif confidence >= 0.60:
            return 'low_confidence'
        else:
            return 'pending_review'

    def update_identity_confidence(
        self,
        identity_id: int,
        new_confidence: float,
        reason: str
    ):
        """
        Update identity confidence score.

        Args:
            identity_id: Identity ID
            new_confidence: New confidence score
            reason: Reason for update
        """
        with get_session() as session:
            verification_status = self._get_verification_status(new_confidence)

            query = text("""
                UPDATE speaker_identities
                SET
                    confidence_score = :confidence,
                    verification_status = :status,
                    verification_metadata = verification_metadata ||
                        jsonb_build_object(
                            'confidence_updated_at', NOW(),
                            'confidence_update_reason', :reason
                        )
                WHERE id = :identity_id
            """)

            session.execute(query, {
                'identity_id': identity_id,
                'confidence': new_confidence,
                'status': verification_status,
                'reason': reason
            })
            session.commit()

            logger.info(
                f"Updated identity {identity_id} confidence to {new_confidence:.2f} "
                f"(status: {verification_status})"
            )

    def _should_update_centroid(
        self,
        existing_meta: Dict,
        new_quality: float,
        new_sample_count: int
    ) -> bool:
        """
        Determine if new centroid is better than existing.

        Args:
            existing_meta: Existing verification_metadata
            new_quality: Quality of new centroid
            new_sample_count: Sample count of new centroid

        Returns:
            True if we should update to the new centroid
        """
        if not existing_meta.get('centroid'):
            return True  # No existing centroid

        existing_quality = existing_meta.get('centroid_quality', 0)
        existing_samples = existing_meta.get('centroid_sample_count', 0)

        # Better quality (by margin of 0.05)
        if new_quality > existing_quality + 0.05:
            return True

        # Significantly more samples (1.5x) and quality not much worse
        if new_sample_count > existing_samples * 1.5 and new_quality >= existing_quality - 0.02:
            return True

        return False

    def store_centroid(
        self,
        identity_id: int,
        centroid: np.ndarray,
        quality: float,
        sample_count: int,
        source_channel_id: int
    ) -> bool:
        """
        Store centroid if it's better than existing.
        Also ensures channel_roles is populated for consistency with get_centroids_for_channel.

        Args:
            identity_id: SpeakerIdentity ID
            centroid: 512-dim embedding centroid
            quality: Average quality of contributing embeddings
            sample_count: Number of speakers contributing to centroid
            source_channel_id: Channel ID where this centroid came from

        Returns:
            True if centroid was updated, False if existing was kept
        """
        with get_session() as session:
            # Get existing identity data including role
            existing = session.execute(text("""
                SELECT verification_metadata, role
                FROM speaker_identities
                WHERE id = :identity_id
            """), {'identity_id': identity_id}).fetchone()

            if not existing:
                logger.warning(f"Identity {identity_id} not found")
                return False

            existing_meta = existing.verification_metadata or {}
            identity_role = existing.role or 'host'

            # Check if we should update
            if not self._should_update_centroid(existing_meta, quality, sample_count):
                logger.info(
                    f"Keeping existing centroid for identity {identity_id} "
                    f"(quality={existing_meta.get('centroid_quality'):.3f}, "
                    f"samples={existing_meta.get('centroid_sample_count')})"
                )
                return False

            # Convert centroid to list for JSON storage
            centroid_list = centroid.tolist() if isinstance(centroid, np.ndarray) else list(centroid)

            centroid_data = {
                'centroid': centroid_list,
                'centroid_quality': float(quality),
                'centroid_sample_count': int(sample_count),
                'centroid_source_channel': int(source_channel_id),
                'centroid_updated_at': datetime.now().isoformat()
            }

            query = text("""
                UPDATE speaker_identities
                SET verification_metadata = COALESCE(verification_metadata, '{}'::jsonb) || CAST(:centroid_data AS jsonb)
                WHERE id = :identity_id
            """)

            session.execute(query, {
                'identity_id': identity_id,
                'centroid_data': json.dumps(centroid_data)
            })
            session.commit()

            logger.info(
                f"Updated centroid for identity {identity_id}: "
                f"quality={quality:.3f}, samples={sample_count}, source_channel={source_channel_id}"
            )

        # Also ensure channel_roles is populated for this channel
        # This ensures get_centroids_for_channel can find this identity
        self.store_channel_role(
            identity_id=identity_id,
            channel_id=source_channel_id,
            role=identity_role,
            episode_count=sample_count
        )

        return True

    def store_host_centroid(
        self,
        identity_id: int,
        centroid: np.ndarray,
        quality: float,
        sample_count: int,
        episode_ids: List[str],
        channel_id: int
    ):
        """
        Store host centroid in verification_metadata.
        DEPRECATED: Use store_centroid() instead. This method is kept for backward compatibility.

        Args:
            identity_id: SpeakerIdentity ID
            centroid: 512-dim embedding centroid
            quality: Average quality of contributing embeddings
            sample_count: Number of speakers contributing to centroid
            episode_ids: Episode IDs that contributed to centroid (ignored in new version)
            channel_id: Channel ID for this centroid
        """
        # Call the new method
        self.store_centroid(
            identity_id=identity_id,
            centroid=centroid,
            quality=quality,
            sample_count=sample_count,
            source_channel_id=channel_id
        )

    def get_channel_host_centroids(self, channel_id: int) -> List[Dict]:
        """
        Load all host identities with centroids for a channel.

        Args:
            channel_id: Channel ID

        Returns:
            List of dicts with identity info and centroid:
            [
                {
                    'identity_id': 123,
                    'name': 'Andrew Huberman',
                    'centroid': np.array([...]),
                    'centroid_quality': 0.82,
                    'centroid_sample_count': 45
                }
            ]
        """
        with get_session() as session:
            query = text("""
                SELECT
                    id,
                    primary_name,
                    verification_metadata
                FROM speaker_identities
                WHERE is_active = TRUE
                  AND role = 'host'
                  AND verification_metadata ? 'centroid'
                  AND (verification_metadata->>'centroid_channel_id')::int = :channel_id
            """)

            results = session.execute(query, {'channel_id': channel_id}).fetchall()

            centroids = []
            for row in results:
                metadata = row.verification_metadata or {}
                centroid_list = metadata.get('centroid')

                if centroid_list:
                    centroids.append({
                        'identity_id': row.id,
                        'name': row.primary_name,
                        'centroid': np.array(centroid_list, dtype=np.float32),
                        'centroid_quality': metadata.get('centroid_quality', 0.0),
                        'centroid_sample_count': metadata.get('centroid_sample_count', 0)
                    })

            logger.info(f"Loaded {len(centroids)} host centroids for channel {channel_id}")
            return centroids

    def get_or_create_host_identity(
        self,
        name: str,
        channel_id: int,
        confidence: float = 0.80,
        role: str = 'host'
    ) -> int:
        """
        Get existing host identity or create new one.

        Unlike create_or_match_identity, this specifically looks for
        host identities associated with a channel.

        Args:
            name: Host name
            channel_id: Channel ID
            confidence: Initial confidence if creating
            role: Role for the identity (default 'host', can be 'co_host')

        Returns:
            Identity ID
        """
        # First try exact match
        existing = self.get_identity_by_name(name)
        if existing:
            return existing['id']

        # Create new host identity
        return self.create_or_match_identity(
            name=name,
            role=role,
            confidence=confidence,
            method='host_frequency_analysis',
            metadata={'channel_id': channel_id}
        )

    def store_channel_role(
        self,
        identity_id: int,
        channel_id: int,
        role: str,
        episode_count: int = 0,
        total_duration: float = 0.0
    ):
        """
        Store or update role for a specific channel.

        Args:
            identity_id: SpeakerIdentity ID
            channel_id: Channel ID
            role: Role on this channel ('host', 'co_host', 'recurring_guest', 'guest')
            episode_count: Number of episodes appeared in
            total_duration: Total speaking duration in seconds
        """
        with get_session() as session:
            role_data = {
                'role': role,
                'episode_count': int(episode_count),
                'total_duration': float(total_duration),
                'updated_at': datetime.now().isoformat()
            }

            # Use nested jsonb_set to create channel_roles if it doesn't exist
            query = text("""
                UPDATE speaker_identities
                SET verification_metadata = jsonb_set(
                    jsonb_set(
                        COALESCE(verification_metadata, '{}'::jsonb),
                        '{channel_roles}',
                        COALESCE(verification_metadata->'channel_roles', '{}'::jsonb)
                    ),
                    ARRAY['channel_roles', :channel_id_str],
                    CAST(:role_data AS jsonb)
                )
                WHERE id = :identity_id
            """)

            session.execute(query, {
                'identity_id': identity_id,
                'channel_id_str': str(channel_id),
                'role_data': json.dumps(role_data)
            })
            session.commit()

            logger.info(
                f"Stored channel role for identity {identity_id}: "
                f"channel={channel_id}, role={role}, episodes={episode_count}"
            )

    def clear_channel_role(self, channel_id: int) -> int:
        """
        Remove role data for a channel from ALL identities.
        Used by --reset to clear channel-specific data without losing centroids.

        Args:
            channel_id: Channel ID to clear

        Returns:
            Number of identities updated
        """
        with get_session() as session:
            query = text("""
                UPDATE speaker_identities
                SET verification_metadata = verification_metadata #- ARRAY['channel_roles', :channel_id_str]
                WHERE verification_metadata->'channel_roles' ? :channel_id_str
            """)

            result = session.execute(query, {'channel_id_str': str(channel_id)})
            count = result.rowcount
            session.commit()

            logger.info(f"Cleared channel role for channel {channel_id} from {count} identities")
            return count

    def get_identity_channels(self, identity_id: int) -> List[Dict]:
        """
        Get all channels where this identity has a role.

        Args:
            identity_id: SpeakerIdentity ID

        Returns:
            List of dicts with channel info:
            [{'channel_id': '6569', 'role': 'host', 'episode_count': 120}, ...]
        """
        with get_session() as session:
            query = text("""
                SELECT
                    key as channel_id,
                    value->>'role' as role,
                    (value->>'episode_count')::int as episode_count,
                    (value->>'total_duration')::float as total_duration
                FROM speaker_identities,
                     jsonb_each(verification_metadata->'channel_roles')
                WHERE id = :identity_id
                ORDER BY (value->>'episode_count')::int DESC
            """)

            results = session.execute(query, {'identity_id': identity_id}).fetchall()

            return [
                {
                    'channel_id': row.channel_id,
                    'role': row.role,
                    'episode_count': row.episode_count or 0,
                    'total_duration': row.total_duration or 0.0
                }
                for row in results
            ]

    def get_centroids_for_channel(
        self,
        channel_id: int,
        roles: List[str] = None
    ) -> List[Dict]:
        """
        Get identities that have a role on this channel and have a centroid.
        Returns the identity's best centroid (not channel-specific).

        Args:
            channel_id: Channel ID
            roles: Filter by roles (default: ['host', 'co_host'])

        Returns:
            List of dicts with identity info and centroid
        """
        roles = roles or ['host', 'co_host']

        with get_session() as session:
            # Find identities with centroids for this channel
            # Check both channel_roles and centroid_source_channel for compatibility
            query = text("""
                SELECT
                    id,
                    primary_name,
                    role,
                    verification_metadata
                FROM speaker_identities
                WHERE is_active = TRUE
                  AND verification_metadata ? 'centroid'
                  AND (
                      -- Check channel_roles
                      (verification_metadata->'channel_roles' ? :channel_id_str
                       AND verification_metadata->'channel_roles'->:channel_id_str->>'role' = ANY(:roles))
                      OR
                      -- Fallback: check centroid_source_channel (for older data)
                      (verification_metadata->>'centroid_source_channel' = :channel_id_str
                       AND role = ANY(:roles))
                  )
            """)

            results = session.execute(query, {
                'channel_id_str': str(channel_id),
                'roles': roles
            }).fetchall()

            centroids = []
            for row in results:
                meta = row.verification_metadata or {}
                centroid_list = meta.get('centroid')

                if centroid_list:
                    channel_roles = meta.get('channel_roles', {})
                    channel_role_info = channel_roles.get(str(channel_id), {})

                    # Get role from channel_roles or fall back to identity.role
                    role = channel_role_info.get('role') or row.role

                    centroids.append({
                        'identity_id': row.id,
                        'name': row.primary_name,
                        'role': role,
                        'centroid': np.array(centroid_list, dtype=np.float32),
                        'centroid_quality': meta.get('centroid_quality', 0.0),
                        'centroid_sample_count': meta.get('centroid_sample_count', 0)
                    })

            logger.info(f"Loaded {len(centroids)} centroids for channel {channel_id} with roles {roles}")
            return centroids

    def update_identification_status(
        self,
        speaker_id: int,
        status: str
    ):
        """
        Update identification status for a single speaker.

        Args:
            speaker_id: Speaker ID
            status: Status from IdentificationStatus
        """
        with get_session() as session:
            query = text("""
                UPDATE speakers
                SET identification_status = :status, updated_at = NOW()
                WHERE id = :speaker_id
            """)
            session.execute(query, {
                'speaker_id': speaker_id,
                'status': status
            })
            session.commit()

    def bulk_update_identification_status(
        self,
        speaker_ids: List[int],
        status: str
    ):
        """
        Efficiently update identification status for multiple speakers.

        Args:
            speaker_ids: List of speaker IDs
            status: Status from IdentificationStatus
        """
        if not speaker_ids:
            return

        with get_session() as session:
            query = text("""
                UPDATE speakers
                SET identification_status = :status, updated_at = NOW()
                WHERE id = ANY(:ids)
            """)
            session.execute(query, {
                'status': status,
                'ids': speaker_ids
            })
            session.commit()

            logger.debug(f"Updated {len(speaker_ids)} speakers to status '{status}'")
