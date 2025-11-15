"""
Visualization Utility Functions

Helper functions for common plotting tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Optional, Tuple, List


def setup_axes(
    ax: Axes,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    grid: bool = True,
    grid_alpha: float = 0.3
) -> Axes:
    """
    Setup axes with common styling.

    Args:
        ax: Matplotlib axes object
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        grid: Whether to show grid
        grid_alpha: Grid transparency

    Returns:
        Modified axes object

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax = setup_axes(ax, "Time (s)", "Amplitude", "Signal")
        >>> ax.plot(x, y)
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='bold')
    if title:
        ax.set_title(title, fontweight='bold')
    if grid:
        ax.grid(alpha=grid_alpha)

    return ax


def add_confidence_band(
    ax: Axes,
    x: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    color: str = 'gray',
    alpha: float = 0.3,
    label: Optional[str] = None
):
    """
    Add confidence band to a plot.

    Args:
        ax: Matplotlib axes object
        x: X coordinates
        y_lower: Lower confidence bound
        y_upper: Upper confidence bound
        color: Band color
        alpha: Band transparency
        label: Legend label

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x, y_mean, 'b-', label='Mean')
        >>> add_confidence_band(ax, x, y_lower, y_upper, label='95% CI')
    """
    ax.fill_between(x, y_lower, y_upper,
                    color=color, alpha=alpha, label=label)


def format_pvalue(p: float) -> str:
    """
    Format p-value for display.

    Args:
        p: P-value

    Returns:
        Formatted string

    Example:
        >>> format_pvalue(0.00123)
        'p = 0.001'
        >>> format_pvalue(0.456)
        'p = 0.456'
        >>> format_pvalue(0.000001)
        'p < 0.001'
    """
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f}"


def add_significance_bar(
    ax: Axes,
    x1: float,
    x2: float,
    y: float,
    p_value: float,
    height: float = 0.05,
    linewidth: float = 1.5
):
    """
    Add significance bar between two points.

    Args:
        ax: Matplotlib axes object
        x1, x2: X coordinates of points to compare
        y: Y coordinate for the bar
        p_value: P-value for comparison
        height: Height of vertical bars
        linewidth: Line width

    Example:
        >>> ax.bar([0, 1], [val1, val2])
        >>> add_significance_bar(ax, 0, 1, max(val1, val2) + 0.1, p_value=0.001)
    """
    # Horizontal line
    ax.plot([x1, x2], [y, y], 'k-', linewidth=linewidth)

    # Vertical lines
    ax.plot([x1, x1], [y - height, y], 'k-', linewidth=linewidth)
    ax.plot([x2, x2], [y - height, y], 'k-', linewidth=linewidth)

    # Significance stars
    if p_value < 0.001:
        text = "***"
    elif p_value < 0.01:
        text = "**"
    elif p_value < 0.05:
        text = "*"
    else:
        text = "ns"

    ax.text((x1 + x2) / 2, y, text, ha='center', va='bottom',
            fontsize=12, fontweight='bold')


def create_color_palette(n_colors: int, cmap: str = 'viridis') -> List[str]:
    """
    Create a color palette with n distinct colors.

    Args:
        n_colors: Number of colors
        cmap: Matplotlib colormap name

    Returns:
        List of color hex codes

    Example:
        >>> colors = create_color_palette(5, 'viridis')
        >>> for i, color in enumerate(colors):
        ...     plt.plot(x, y[i], color=color)
    """
    cmap_obj = plt.cm.get_cmap(cmap)
    colors = [cmap_obj(i / (n_colors - 1)) for i in range(n_colors)]
    return [plt.matplotlib.colors.rgb2hex(c) for c in colors]


def add_subplot_labels(
    fig,
    labels: Optional[List[str]] = None,
    xy: Tuple[float, float] = (-0.1, 1.05),
    fontsize: int = 14,
    fontweight: str = 'bold'
):
    """
    Add (A), (B), (C) labels to subplots.

    Args:
        fig: Matplotlib figure object
        labels: Custom labels (default: A, B, C, ...)
        xy: Position relative to axes (default: top-left)
        fontsize: Label font size
        fontweight: Label font weight

    Example:
        >>> fig, axes = plt.subplots(2, 2)
        >>> # ... create plots ...
        >>> add_subplot_labels(fig)  # Adds (A), (B), (C), (D)
    """
    axes = fig.get_axes()

    if labels is None:
        labels = [chr(65 + i) for i in range(len(axes))]  # A, B, C, ...

    for ax, label in zip(axes, labels):
        ax.text(xy[0], xy[1], f"({label})",
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight=fontweight,
                va='top', ha='right')


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """
    Truncate a colormap to a subset of its range.

    Args:
        cmap: Colormap name or object
        minval: Minimum value (0-1)
        maxval: Maximum value (0-1)
        n: Number of colors in new colormap

    Returns:
        New colormap

    Example:
        >>> # Use only the upper half of viridis
        >>> new_cmap = truncate_colormap('viridis', 0.5, 1.0)
        >>> plt.imshow(data, cmap=new_cmap)
    """
    import matplotlib.colors as mcolors

    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
        cmap(np.linspace(minval, maxval, n))
    )

    return new_cmap


def add_colorbar_to_axis(
    im,
    ax: Axes,
    label: str = "",
    location: str = "right",
    size: str = "5%",
    pad: float = 0.05
):
    """
    Add a colorbar to a specific axis (not the whole figure).

    Args:
        im: Mappable object (e.g., from plt.imshow or plt.scatter)
        ax: Axes object
        label: Colorbar label
        location: Colorbar location (right, left, top, bottom)
        size: Colorbar size
        pad: Padding between axes and colorbar

    Example:
        >>> im = ax.imshow(data, cmap='viridis')
        >>> add_colorbar_to_axis(im, ax, label="Temperature (°C)")
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad)
    cbar = plt.colorbar(im, cax=cax)

    if label:
        cbar.set_label(label, fontweight='bold')

    return cbar
