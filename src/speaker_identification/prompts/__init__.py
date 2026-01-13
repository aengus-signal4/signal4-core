"""
Speaker Identification Prompts
==============================

Centralized prompt management for the speaker identification pipeline.

This module provides:
- Reusable instruction blocks (shared across multiple prompts)
- Complete prompt templates for each pipeline phase
- Easy customization and A/B testing of prompts

Usage:
    from src.speaker_identification.prompts import PromptRegistry

    # Get a complete prompt
    prompt = PromptRegistry.get_phase1a_prompt(channel_name, description, platform)

    # Access shared blocks for custom prompts
    from src.speaker_identification.prompts.blocks import ADDRESSING_RULE
"""

from .registry import PromptRegistry
from . import blocks

__all__ = ['PromptRegistry', 'blocks']
