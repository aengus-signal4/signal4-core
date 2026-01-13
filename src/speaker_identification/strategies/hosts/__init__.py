"""
Host Identification Strategies
==============================

Strategies for identifying hosts in podcast/video channels.

- SingleHostStrategy: For channels with one primary host
- MultiHostStrategy: For channels with multiple co-hosts
"""

from .single_host import SingleHostStrategy
from .multi_host import MultiHostStrategy

__all__ = ['SingleHostStrategy', 'MultiHostStrategy']
