"""
Publication-Ready Styling for Matplotlib Figures

Provides style presets for major scientific journals and utilities
for saving publication-quality figures.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union, Dict


# Publication style presets
STYLE_PRESETS = {
    "nature": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 10,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    },

    "science": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.titlesize": 9,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    },

    "plos": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.5,
        "patch.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    },

    "presentation": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.titlesize": 20,
        "axes.linewidth": 2.0,
        "grid.linewidth": 1.0,
        "lines.linewidth": 3.0,
        "patch.linewidth": 2.0,
        "xtick.major.width": 2.0,
        "ytick.major.width": 2.0,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    },

    "poster": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 24,
        "axes.linewidth": 2.5,
        "grid.linewidth": 1.5,
        "lines.linewidth": 3.5,
        "patch.linewidth": 2.5,
        "xtick.major.width": 2.5,
        "ytick.major.width": 2.5,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    },
}


@contextmanager
def publication_style(style: str = "nature"):
    """
    Context manager for applying publication-ready styling to plots.

    Args:
        style: Style preset name (nature, science, plos, presentation, poster)

    Example:
        >>> with publication_style("nature"):
        ...     plt.plot(x, y)
        ...     plt.xlabel("X")
        ...     plt.ylabel("Y")
        ...     plt.savefig("figure.png")
    """
    if style not in STYLE_PRESETS:
        raise ValueError(f"Unknown style '{style}'. Available: {list(STYLE_PRESETS.keys())}")

    # Save current style
    original_rcParams = mpl.rcParams.copy()

    try:
        # Apply publication style
        mpl.rcParams.update(STYLE_PRESETS[style])
        yield
    finally:
        # Restore original style
        mpl.rcParams.update(original_rcParams)


def save_publication_figure(
    fig,
    filename: Union[str, Path],
    style: str = "nature",
    dpi: Optional[int] = None,
    format: str = "png",
    bbox_inches: str = "tight",
    **kwargs
):
    """
    Save figure with publication-ready settings.

    Args:
        fig: Matplotlib figure object
        filename: Output filename
        style: Style preset (nature, science, plos, presentation, poster)
        dpi: DPI override (uses style default if None)
        format: Output format (png, pdf, svg, eps)
        bbox_inches: Bbox setting (default: "tight")
        **kwargs: Additional arguments for fig.savefig()

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x, y)
        >>> save_publication_figure(fig, "figure1.png", style="nature")
    """
    if style not in STYLE_PRESETS:
        raise ValueError(f"Unknown style '{style}'. Available: {list(STYLE_PRESETS.keys())}")

    # Use style's DPI if not overridden
    if dpi is None:
        dpi = STYLE_PRESETS[style]["savefig.dpi"]

    # Ensure parent directory exists
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    fig.savefig(
        filename,
        dpi=dpi,
        format=format,
        bbox_inches=bbox_inches,
        **kwargs
    )

    print(f"✅ Figure saved: {filename} ({style} style, {dpi} DPI, {format} format)")


def apply_publication_style(style: str = "nature"):
    """
    Permanently apply publication style to current session.

    Args:
        style: Style preset name

    Example:
        >>> apply_publication_style("nature")
        >>> # All subsequent plots use Nature style
        >>> plt.plot(x, y)
    """
    if style not in STYLE_PRESETS:
        raise ValueError(f"Unknown style '{style}'. Available: {list(STYLE_PRESETS.keys())}")

    mpl.rcParams.update(STYLE_PRESETS[style])
    print(f"✅ Applied '{style}' publication style globally")


def get_figure_size(
    width_mm: float,
    height_mm: Optional[float] = None,
    aspect_ratio: float = 1.6  # Golden ratio
) -> tuple:
    """
    Convert figure size from mm to inches for matplotlib.

    Many journals specify figure dimensions in mm.

    Args:
        width_mm: Figure width in mm
        height_mm: Figure height in mm (calculated from aspect ratio if None)
        aspect_ratio: Width/height ratio if height_mm is None

    Returns:
        Tuple of (width_inches, height_inches)

    Example:
        >>> # Nature single column: 89mm
        >>> figsize = get_figure_size(89)
        >>> fig, ax = plt.subplots(figsize=figsize)

        >>> # Nature double column: 183mm
        >>> figsize = get_figure_size(183)

        >>> # Custom size
        >>> figsize = get_figure_size(width_mm=120, height_mm=80)
    """
    mm_to_inch = 1 / 25.4

    width_inch = width_mm * mm_to_inch

    if height_mm is None:
        height_inch = width_inch / aspect_ratio
    else:
        height_inch = height_mm * mm_to_inch

    return (width_inch, height_inch)


# Common journal figure sizes (in mm)
JOURNAL_SIZES = {
    "nature_single": 89,
    "nature_double": 183,
    "nature_full": 247,
    "science_single": 89,
    "science_double": 183,
    "plos_single": 83,
    "plos_double": 173,
}


def get_journal_figsize(journal: str, columns: int = 1) -> tuple:
    """
    Get standard figure size for a journal.

    Args:
        journal: Journal name (nature, science, plos)
        columns: Number of columns (1 or 2)

    Returns:
        Tuple of (width_inches, height_inches)

    Example:
        >>> figsize = get_journal_figsize("nature", columns=1)
        >>> fig, ax = plt.subplots(figsize=figsize)
    """
    key = f"{journal}_{'single' if columns == 1 else 'double'}"

    if key not in JOURNAL_SIZES:
        raise ValueError(f"Unknown journal/column combo: {journal}, {columns}")

    width_mm = JOURNAL_SIZES[key]
    return get_figure_size(width_mm)
