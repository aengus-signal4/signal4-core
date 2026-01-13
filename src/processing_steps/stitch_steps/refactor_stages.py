#!/usr/bin/env python3
"""
Script to help refactor stage ordering in stitch.py
"""

# Stage mapping: old -> new
STAGE_MAPPING = {
    6: "speaker_embeddings",     # was stage 9
    7: "speaker_centroids",      # was stage 10
    8: "good_grammar_multi",     # was stage 6
    9: "bad_grammar_multi",      # was stage 7
    10: "llm_resolution",        # was stage 8
}

# Function names mapping
FUNCTION_MAPPING = {
    6: ("speaker_embeddings_stage", "speaker_embeddings"),
    7: ("speaker_centroids_stage", "speaker_centroids"),
    8: ("good_grammar_multi_speaker_stage", "good_grammar_multi_analysis"),
    9: ("bad_grammar_multi_speaker_stage", "bad_grammar_multi_analysis"),
    10: ("stage10_llm_resolution", "llm_resolution"),
}

# Cache key mapping
CACHE_KEY_MAPPING = {
    6: "stage6_speaker_embeddings",
    7: "stage7_speaker_centroids", 
    8: "stage8_good_grammar_multi",
    9: "stage9_bad_grammar_multi",
    10: "stage10_llm_resolution",
}

# Print the refactoring plan
def print_refactoring_plan():
    print("Stage Refactoring Plan:")
    print("-" * 50)
    print("Stage 6: Good Grammar Multi -> Speaker Embeddings")
    print("Stage 7: Bad Grammar Multi -> Speaker Centroids")
    print("Stage 8: LLM Resolution -> Good Grammar Multi")
    print("Stage 9: Speaker Embeddings -> Bad Grammar Multi")
    print("Stage 10: Speaker Centroids -> LLM Resolution")
    print("-" * 50)
    
if __name__ == "__main__":
    print_refactoring_plan()