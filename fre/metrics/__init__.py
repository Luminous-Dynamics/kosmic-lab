"""
Metrics module for Kosmic Lab.

This module provides coherence and K-index calculation utilities.
"""

from __future__ import annotations

from fre.metrics.k_index import (
    compute_k_index,
    bootstrap_k_ci,
    validate_k_bounds,
)
from fre.metrics.k_lag import k_lag, compute_lagged_correlation

__all__ = [
    "compute_k_index",
    "bootstrap_k_ci",
    "validate_k_bounds",
    "k_lag",
    "compute_lagged_correlation",
]
