"""
Metrics module for Kosmic Lab.

This module provides coherence and K-index calculation utilities.
"""

from __future__ import annotations

from fre.metrics.k_index import (
    k_index,
    bootstrap_k_ci,
    verify_k_bounds,
    k_index_robust,
    k_index_with_ci,
)
from fre.metrics.k_lag import k_lag, verify_causal_direction

__all__ = [
    "k_index",
    "bootstrap_k_ci",
    "verify_k_bounds",
    "k_index_robust",
    "k_index_with_ci",
    "k_lag",
    "verify_causal_direction",
]
