"""
Kosmic Lab Visualization Library

Publication-quality visualization tools for K-Index, K-Lag, and experiment results.

This module provides:
- One-liner functions for common plots
- Publication-ready styling (Nature, Science, etc.)
- Interactive visualizations (Plotly)
- Reusable components

Usage:
    from core.visualization import plot_k_index_ci, publication_style

    # Quick K-Index plot with CI
    fig = plot_k_index_ci(observed, actual, title="My Experiment")
    fig.savefig("figure1.png", dpi=300)

    # Apply publication styling globally
    with publication_style("nature"):
        # All plots now use Nature style
        plt.plot(x, y)
        plt.show()
"""

from .k_index_plots import (
    plot_k_index_ci,
    plot_k_index_comparison,
    plot_k_lag,
    plot_k_distribution
)

from .publication import (
    publication_style,
    STYLE_PRESETS,
    save_publication_figure
)

from .utils import (
    setup_axes,
    add_confidence_band,
    format_pvalue,
    create_color_palette
)

__all__ = [
    # K-Index plots
    'plot_k_index_ci',
    'plot_k_index_comparison',
    'plot_k_lag',
    'plot_k_distribution',

    # Publication styling
    'publication_style',
    'STYLE_PRESETS',
    'save_publication_figure',

    # Utilities
    'setup_axes',
    'add_confidence_band',
    'format_pvalue',
    'create_color_palette',
]

__version__ = "1.0.0"
