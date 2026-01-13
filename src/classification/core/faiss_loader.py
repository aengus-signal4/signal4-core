"""
FAISS index loader for semantic search.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

try:
    import faiss
except ImportError:
    raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")

from src.database.session import get_session
from sqlalchemy import text

logger = logging.getLogger(__name__)


class FAISSIndexLoader:
    """Load embeddings and build FAISS index for fast semantic search"""

    def __init__(self, project: str, embedding_dim: int = 2000, limit: Optional[int] = None):
        self.project = project
        self.embedding_dim = embedding_dim
        self.limit = limit
        self.index = None
        self.segment_id_mapping = {}  # array_index -> segment_id
        self.reverse_mapping = {}  # segment_id -> array_index
        self.embeddings = None
        self.metadata_df: Optional[pd.DataFrame] = None  # Segment metadata

    def load_embeddings_from_db(self) -> Tuple[np.ndarray, Dict[int, int]]:
        """Load embeddings from PostgreSQL for the specified project using streaming/chunked approach"""
        if self.limit:
            logger.info(f"Loading up to {self.limit} embeddings for project {self.project} from database (test mode)...")
        else:
            logger.info(f"Loading all embeddings for project {self.project} from database...")

        with get_session() as session:
            # Add LIMIT clause if specified
            limit_clause = f"LIMIT {self.limit}" if self.limit else ""

            # PASS 1: Get count to pre-allocate array
            count_query = text(f"""
                SELECT COUNT(*)
                FROM embedding_segments es
                JOIN content c ON es.content_id = c.id
                WHERE :project = ANY(c.projects)
                AND es.embedding_alt IS NOT NULL
                {limit_clause}
            """)

            count_result = session.execute(count_query, {'project': self.project})
            total_count = count_result.scalar()

            logger.info(f"Found {total_count:,} segments with embeddings")

            if total_count == 0:
                logger.warning("No segments found!")
                return np.array([]).reshape(0, self.embedding_dim).astype(np.float32), {}

            # Pre-allocate numpy array (saves memory compared to list + vstack)
            embeddings = np.zeros((total_count, self.embedding_dim), dtype=np.float32)
            logger.info(f"Pre-allocated array: {embeddings.shape}, {embeddings.nbytes / 1024 / 1024:.2f} MB")

            # PASS 2: Stream embeddings in chunks using server-side cursor
            data_query = text(f"""
                SELECT es.id, es.embedding_alt
                FROM embedding_segments es
                JOIN content c ON es.content_id = c.id
                WHERE :project = ANY(c.projects)
                AND es.embedding_alt IS NOT NULL
                ORDER BY es.id
                {limit_clause}
            """)

            # Use yield_per for server-side cursor (streaming)
            chunk_size = 50000
            result = session.execute(data_query, {'project': self.project})
            result = result.yield_per(chunk_size)

            idx = 0
            with tqdm(total=total_count, desc="Loading embeddings", unit="emb") as pbar:
                for segment_id, embedding_vector in result:
                    # Convert vector to numpy array
                    if isinstance(embedding_vector, str):
                        embedding_str = embedding_vector.strip('[]')
                        embedding_array = np.array([float(x) for x in embedding_str.split(',')], dtype=np.float32)
                    else:
                        embedding_array = np.array(embedding_vector, dtype=np.float32)

                    # Direct write to pre-allocated array (no intermediate list)
                    if idx >= total_count:
                        logger.warning(f"Received more rows ({idx+1}) than COUNT query returned ({total_count}), stopping")
                        break

                    embeddings[idx] = embedding_array
                    self.segment_id_mapping[idx] = segment_id
                    self.reverse_mapping[segment_id] = idx

                    idx += 1
                    pbar.update(1)

                    # Progress logging every 100k
                    if idx % 100000 == 0:
                        logger.info(f"  Loaded {idx:,}/{total_count:,} embeddings ({idx/total_count*100:.1f}%)")

            logger.info(f"Loaded embeddings array: {embeddings.shape}, {embeddings.nbytes / 1024 / 1024:.2f} MB")
            return embeddings, self.segment_id_mapping

    def build_index(self, embeddings: np.ndarray):
        """Build FAISS CPU index for cosine similarity search"""
        logger.info(f"Building FAISS CPU index for {embeddings.shape[0]} embeddings...")
        logger.info(f"  Embeddings dtype: {embeddings.dtype}, shape: {embeddings.shape}")
        logger.info(f"  Memory size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
        dimension = embeddings.shape[1]

        # Ensure embeddings are contiguous in memory and C-ordered
        if not embeddings.flags['C_CONTIGUOUS']:
            logger.info("Converting embeddings to C-contiguous array...")
            embeddings = np.ascontiguousarray(embeddings)

        # Normalize embeddings for cosine similarity in chunks to avoid memory issues
        logger.info("Normalizing embeddings in chunks...")
        norm_chunk_size = 100000
        for i in range(0, embeddings.shape[0], norm_chunk_size):
            end_idx = min(i + norm_chunk_size, embeddings.shape[0])
            chunk = embeddings[i:end_idx]
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings[i:end_idx] = chunk / norms
            if (i // norm_chunk_size) % 10 == 0:
                logger.info(f"  Normalized {end_idx}/{embeddings.shape[0]} vectors...")
        logger.info(f"Normalization complete, all vectors now have unit length")

        # Use IndexFlatIP (Inner Product) for cosine similarity with normalized vectors
        logger.info("Creating FAISS IndexFlatIP for Inner Product search...")
        index = faiss.IndexFlatIP(dimension)

        # Add embeddings in chunks
        chunk_size = 100000  # Process 100k at a time
        logger.info("Adding embeddings to index in chunks...")
        for i in range(0, embeddings.shape[0], chunk_size):
            end_idx = min(i + chunk_size, embeddings.shape[0])
            chunk = embeddings[i:end_idx]
            # Ensure chunk is contiguous
            if not chunk.flags['C_CONTIGUOUS']:
                chunk = np.ascontiguousarray(chunk)
            index.add(chunk)
            logger.info(f"  Added {end_idx}/{embeddings.shape[0]} embeddings to index ({(end_idx/embeddings.shape[0]*100):.1f}%)")

        logger.info(f"FAISS index built successfully: {index.ntotal} vectors, dimension {dimension}")
        return index, embeddings

    def load_or_build_index(self, index_path: Optional[str] = None):
        """Load existing FAISS index or build new one

        Note: In test mode with a limit, we won't save the index to avoid overwriting
        a full production index with a limited test index.
        """
        # Try to load existing index if path provided and not in test mode
        if index_path and Path(index_path).exists() and not self.limit:
            logger.info(f"Loading existing FAISS index from {index_path}...")
            try:
                self.index = faiss.read_index(index_path)
                # Also need to load mappings
                mapping_path = str(Path(index_path).with_suffix('.mapping.json'))
                if Path(mapping_path).exists():
                    with open(mapping_path, 'r') as f:
                        mapping_data = json.load(f)
                        self.segment_id_mapping = {int(k): v for k, v in mapping_data['segment_id_mapping'].items()}
                        self.reverse_mapping = {v: int(k) for k, v in mapping_data['segment_id_mapping'].items()}
                    logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                    return self.index
                else:
                    logger.warning(f"Mapping file not found at {mapping_path}, rebuilding index")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}, rebuilding...")

        # Build new index
        logger.info("Building new FAISS index from database embeddings...")
        self.embeddings, self.segment_id_mapping = self.load_embeddings_from_db()
        self.reverse_mapping = {v: k for k, v in self.segment_id_mapping.items()}
        self.index, self.embeddings = self.build_index(self.embeddings)

        # Save index if path provided and not in test mode
        if index_path and not self.limit:
            self.save_index(index_path)

        logger.info("FAISS index ready for CPU search")
        return self.index

    def save_index(self, index_path: str):
        """Save FAISS index and mappings to disk"""
        if self.index is None:
            logger.warning("No index to save")
            return

        try:
            # Save FAISS index
            logger.info(f"Saving FAISS index to {index_path}...")
            faiss.write_index(self.index, index_path)

            # Save mappings
            mapping_path = str(Path(index_path).with_suffix('.mapping.json'))
            with open(mapping_path, 'w') as f:
                json.dump({
                    'segment_id_mapping': {str(k): v for k, v in self.segment_id_mapping.items()}
                }, f)

            logger.info(f"Saved FAISS index and mappings")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def search(self, query_embeddings: np.ndarray, k: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Search FAISS index

        Args:
            query_embeddings: Query vectors, shape (n_queries, dim)
            k: Number of nearest neighbors to return

        Returns:
            scores: Similarity scores, shape (n_queries, k)
            indices: Array indices, shape (n_queries, k)
        """
        if self.index is None:
            raise ValueError("Index not loaded")

        # Normalize query embeddings for cosine similarity
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)

        # Search - FAISS returns (distances, indices) as numpy arrays
        scores, indices = self.index.search(query_embeddings, k)

        return scores, indices

    def get_segment_ids(self, array_indices: np.ndarray) -> List[int]:
        """Convert array indices to segment IDs"""
        return [self.segment_id_mapping.get(idx, -1) for idx in array_indices]
