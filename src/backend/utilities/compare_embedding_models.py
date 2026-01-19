#!/usr/bin/env python3
"""
Compare 1024-dim vs 2000-dim Embedding Models
=============================================

Tests the same queries with both embedding models to compare retrieval
performance: result counts, similarity scores, and quality of matches.

Usage:
    cd ~/signal4/core/src/backend
    python utilities/compare_embedding_models.py [OPTIONS]

Examples:
    python utilities/compare_embedding_models.py              # Default 7 days
    python utilities/compare_embedding_models.py --days 30    # Last 30 days

Options:
    --days INT    Number of days to look back (default: 7)

Models compared:
    - 1024-dim: Qwen3-Embedding-0.6B (faster, smaller)
    - 2000-dim: Qwen3-Embedding-4B truncated (more nuanced, larger)

What it does:
    1. Loads both embedding models on MPS/CPU
    2. Runs test queries through LLM query2doc_stances
    3. Generates embeddings with both models
    4. Searches at threshold 0.35
    5. Compares results, similarity scores, and overlap

Test queries (hardcoded):
    - immigration policy
    - climate change
    - economic policy
    - healthcare reform

Output:
    Console comparison showing per-query and overall statistics
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta, timezone
from collections import Counter
from typing import List, Dict

# Load environment
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), '../../../.env')
load_dotenv(env_path)

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.dashboard_config import load_dashboard_config, DashboardConfig
from app.services.llm_service import LLMService
from app.services.search_service import SearchService
from app.database.connection import SessionLocal
from sentence_transformers import SentenceTransformer
import torch
import numpy as np


async def test_embedding_comparison(time_window_days: int = 7):
    """
    Compare 1024-dim vs 2000-dim embeddings with same queries.
    """
    print("=" * 80)
    print("EMBEDDING MODEL COMPARISON: 1024-dim vs 2000-dim")
    print("=" * 80)

    # Test queries
    test_queries = [
        "immigration policy",
        "climate change",
        "economic policy",
        "healthcare reform"
    ]

    # Calculate date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=time_window_days)

    print(f"\nTime window: Past {time_window_days} days")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Threshold: 0.35")
    print(f"Test queries: {len(test_queries)}")
    print()

    # Initialize services for both models
    dashboard_id = "cprmv-practitioner"
    config = load_dashboard_config(dashboard_id)

    # Model 1: 1024-dim (Qwen3-0.6B)
    print("[Setup] Loading 1024-dim embedding model (Qwen3-0.6B)...")
    config.search['use_alt_embeddings'] = False
    config.search['embedding_dim'] = 1024
    search_service_1024 = SearchService(dashboard_id, config)

    embedding_model_1024 = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    embedding_model_1024 = embedding_model_1024.to(device)
    print(f"  Loaded on {device}")

    # Model 2: 2000-dim (Qwen3-4B truncated)
    print("[Setup] Loading 2000-dim embedding model (Qwen3-4B, truncated)...")
    config_alt = load_dashboard_config(dashboard_id)
    config_alt.search['use_alt_embeddings'] = True
    config_alt.search['embedding_dim'] = 2000
    search_service_2000 = SearchService(dashboard_id, config_alt)

    embedding_model_2000 = SentenceTransformer('Qwen/Qwen3-Embedding-4B', trust_remote_code=True, truncate_dim=2000)
    embedding_model_2000 = embedding_model_2000.to(device)
    print(f"  Loaded on {device}")
    print()

    # LLM service for query generation
    llm_service = LLMService(dashboard_id=dashboard_id, config=config)

    db_session = SessionLocal()

    try:
        results_comparison = []

        for query in test_queries:
            print(f"\n{'=' * 80}")
            print(f"Query: '{query}'")
            print(f"{'=' * 80}")

            # Get embedding using query2doc_stances
            stances = llm_service.query2doc_stances(query, n_stances=3)

            if not stances or len(stances) < 3:
                print(f"⚠️  Failed to generate stances for query: {query}")
                continue

            # Test with first stance (balanced view)
            test_stance = stances[2] if len(stances) > 2 else stances[0]
            print(f"Test stance: {test_stance[:80]}...")
            print()

            # Generate embeddings with BOTH models
            # Add query prefix for Qwen models
            query_text = f"Instruct: Retrieve relevant passages.\nQuery: {test_stance}"

            # 1024-dim embedding
            print("[1/2] Generating 1024-dim embedding and searching...")
            embedding_1024 = embedding_model_1024.encode([query_text], convert_to_numpy=True)[0]
            embedding_1024 = np.array(embedding_1024, dtype=np.float32)

            results_1024 = search_service_1024.search(
                embedding_1024,
                k=100,
                time_window_days=time_window_days,
                threshold=0.35
            )

            top_sim_1024 = max([r.get('similarity', 0) for r in results_1024]) if results_1024 else 0
            avg_sim_1024 = sum([r.get('similarity', 0) for r in results_1024]) / len(results_1024) if results_1024 else 0

            print(f"  Results: {len(results_1024)}")
            print(f"  Top similarity: {top_sim_1024:.4f}")
            print(f"  Avg similarity: {avg_sim_1024:.4f}")

            # 2000-dim embedding
            print("[2/2] Generating 2000-dim embedding and searching...")
            embedding_2000 = embedding_model_2000.encode([query_text], convert_to_numpy=True)[0]
            embedding_2000 = np.array(embedding_2000, dtype=np.float32)

            results_2000 = search_service_2000.search(
                embedding_2000,
                k=100,
                time_window_days=time_window_days,
                threshold=0.35
            )

            top_sim_2000 = max([r.get('similarity', 0) for r in results_2000]) if results_2000 else 0
            avg_sim_2000 = sum([r.get('similarity', 0) for r in results_2000]) / len(results_2000) if results_2000 else 0

            print(f"  Results: {len(results_2000)}")
            print(f"  Top similarity: {top_sim_2000:.4f}")
            print(f"  Avg similarity: {avg_sim_2000:.4f}")

            # Comparison
            print()
            print("Comparison:")
            print(f"  Δ Results: {len(results_2000) - len(results_1024):+d} ({len(results_2000) / len(results_1024) * 100 if results_1024 else 0:.1f}% of 1024)")
            print(f"  Δ Top sim: {top_sim_2000 - top_sim_1024:+.4f}")
            print(f"  Δ Avg sim: {avg_sim_2000 - avg_sim_1024:+.4f}")

            # Store results
            results_comparison.append({
                'query': query,
                'stance': test_stance,
                'results_1024': len(results_1024),
                'results_2000': len(results_2000),
                'top_sim_1024': top_sim_1024,
                'top_sim_2000': top_sim_2000,
                'avg_sim_1024': avg_sim_1024,
                'avg_sim_2000': avg_sim_2000
            })

        # Overall summary
        print("\n" + "=" * 80)
        print("OVERALL COMPARISON")
        print("=" * 80)

        if results_comparison:
            total_1024 = sum(r['results_1024'] for r in results_comparison)
            total_2000 = sum(r['results_2000'] for r in results_comparison)
            avg_top_1024 = sum(r['top_sim_1024'] for r in results_comparison) / len(results_comparison)
            avg_top_2000 = sum(r['top_sim_2000'] for r in results_comparison) / len(results_comparison)

            print(f"\nTotal results across {len(results_comparison)} queries:")
            print(f"  1024-dim (Qwen3-0.6B): {total_1024} segments")
            print(f"  2000-dim (Qwen3-4B):   {total_2000} segments")
            print(f"  Difference: {total_2000 - total_1024:+d} ({(total_2000/total_1024*100 if total_1024 else 0):.1f}%)")

            print(f"\nAverage top similarity:")
            print(f"  1024-dim: {avg_top_1024:.4f}")
            print(f"  2000-dim: {avg_top_2000:.4f}")
            print(f"  Difference: {avg_top_2000 - avg_top_1024:+.4f}")

            print("\n" + "-" * 80)
            print("CONCLUSION:")
            print("-" * 80)

            if total_2000 > total_1024 * 1.2:
                print("✓ 2000-dim embeddings retrieve significantly more results (+20%)")
            elif total_2000 > total_1024:
                print("~ 2000-dim embeddings retrieve slightly more results")
            else:
                print("~ Similar number of results for both models")

            if avg_top_2000 > avg_top_1024 + 0.05:
                print("✓ 2000-dim embeddings show higher similarity scores")
            elif avg_top_2000 > avg_top_1024:
                print("~ 2000-dim embeddings show slightly higher similarity")
            else:
                print("~ Similar similarity scores for both models")

            print("\nNote: Higher-dimensional embeddings (2000-dim) capture more nuance")
            print("      but require more storage and may be more sensitive to noise.")

    finally:
        db_session.close()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare 1024-dim vs 2048-dim embeddings")
    parser.add_argument("--days", type=int, default=7, help="Number of days to look back (default: 7)")
    args = parser.parse_args()

    asyncio.run(test_embedding_comparison(time_window_days=args.days))
