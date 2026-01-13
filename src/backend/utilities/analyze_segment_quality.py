#!/usr/bin/env python3
"""
Analyze segment quality and similarity scores
"""
import sys
import os
import asyncio
import numpy as np

# Load environment
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), '../../../.env')
load_dotenv(env_path)

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.dashboard_config import load_dashboard_config
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService


async def main():
    config = load_dashboard_config('cprmv-practitioner')
    llm = LLMService(dashboard_id='cprmv-practitioner', config=config)
    embedding_service = EmbeddingService(config=config, dashboard_id='cprmv-practitioner')
    search_service = SearchService('cprmv-practitioner', config)

    # Test query - the balanced immigration one
    query = 'What balanced immigration reforms can secure borders while providing pathways to citizenship for essential workers?'

    print(f'Query: {query}')
    print()

    # Get embedding
    embeddings = await embedding_service.encode_queries([query])
    if not embeddings:
        print('Failed to generate embedding')
        return

    embedding = embeddings[0]
    print(f'Embedding shape: {embedding.shape}')
    print(f'Embedding norm: {np.linalg.norm(embedding):.4f}')
    print()

    # Search with different thresholds
    print("=" * 80)
    print("SEGMENTS BY THRESHOLD (7-day window)")
    print("=" * 80)
    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        segments = search_service.search(
            embedding,
            k=100,
            time_window_days=7,
            threshold=threshold
        )

        if segments:
            top3_sims = [f"{seg.get('similarity', 0):.3f}" for seg in segments[:3]]
            print(f'Threshold {threshold:.2f}: {len(segments):3d} segments | Top 3: [{", ".join(top3_sims)}]')
        else:
            print(f'Threshold {threshold:.2f}:   0 segments')

    print()
    print('='*80)
    print('TOP 15 SEGMENTS (threshold=0.35)')
    print('='*80)

    segments = search_service.search(embedding, k=100, time_window_days=7, threshold=0.35)

    for i, seg in enumerate(segments[:15], 1):
        sim = seg.get('similarity', 0)
        text = seg.get('text', '')[:250]
        channel = seg.get('channel_name', 'Unknown')
        title = seg.get('title', 'Unknown')[:60]

        print(f'\n[{i}] Similarity: {sim:.4f} | Channel: {channel}')
        print(f'Title: {title}')
        print(f'Text: {text}...')

        # Quality assessment
        relevance = "???"
        if any(word in text.lower() for word in ['immigration', 'immigr', 'border', 'citizenship', 'refugee', 'migrant']):
            relevance = "HIGH - Direct match"
        elif any(word in text.lower() for word in ['policy', 'reform', 'law', 'work', 'security']):
            relevance = "MEDIUM - Related policy"
        else:
            relevance = "LOW - Off topic"

        print(f'Assessment: {relevance}')


if __name__ == "__main__":
    asyncio.run(main())
