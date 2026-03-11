"""
plotting.py

# Copyright (c) 2026, Manudeo Singh          #
# Author: Manudeo Singh, March 2026          #
-----------
Convenience functions for visualising WetlandMapper outputs.

All functions:
- Respect geographic coordinates when present (``x``/``y`` or ``lon``/``lat``),
  so axes ticks show real coordinate values rather than pixel indices.
- Detect y-axis direction (ascending vs descending) and set ``origin``
  accordingly so images are never vertically flipped.
- Place the class legend outside the plot area to avoid obscuring data.
- Return (fig, ax) so callers can further customise or save.

Coordinate conventions
----------------------
:func:`wetlandmapper.gee.fetch`
    Returns arrays with dims ``(time, y, x)`` where ``y`` is **descending**
    (north → south, standard raster convention).
:func:`wetlandmapper.gee.fetch_xee`
    After the built-in ``sortby`` fix, also returns ``y`` descending.
    If you have an older array with ``lat``/``lon`` dims, rename them first:
    ``da = da.rename({"lat": "y", "lon": "x"})``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "plot_dynamics",
    "plot_wct",
    "plot_index",
    "plot_wet_frequency",
]


# ---------------------------------------------------------------------------
# Lazy matplotlib import
# ---------------------------------------------------------------------------

def _get_mpl():
    try:
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        return plt, mcolors, mpatches
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with:  pip install matplotlib"
        )


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _spatial_coords(da):
    """Return (x_values, y_values) arrays from a DataArray, or (None, None)."""
    # Support both (y, x) and (lat, lon) naming conventions
    x_coord = None
    y_coord = None
    for xname in ("x", "lon", "longitude"):
        if xname in da.coords:
            x_coord = da.coords[xname].values
            break
    for yname in ("y", "lat", "latitude"):
        if yname in da.coords:
            y_coord = da.coords[yname].values
            break
    return x_coord, y_coord


def _imshow_extent(da):
    """Compute the (left, right, bottom, top) extent for imshow from coords.

    Returns None if no spatial coordinates are present, in which case imshow
    falls back to pixel-index axes.
    """
    x, y = _spatial_coords(da)
    if x is None or y is None:
        return None

    # Half-pixel expansion so the extent covers the full pixel area
    dx = float(np.abs(x[1] - x[0])) / 2 if len(x) > 1 else 0
    dy = float(np.abs(y[1] - y[0])) / 2 if len(y) > 1 else 0

    left   = float(x.min()) - dx
    right  = float(x.max()) + dx
    bottom = float(y.min()) - dy
    top    = float(y.max()) + dy

    return (left, right, bottom, top)


def _imshow_origin(da):
    """Return 'upper' if y is descending (north→south), else 'lower'."""
    _, y = _spatial_coords(da)
    if y is not None and len(y) > 1:
        return "upper" if y[0] > y[-1] else "lower"
    return "upper"   # safe default for plain numpy arrays


def _get_2d(da):
    """Squeeze or select to get a 2-D array for imshow."""
    if "time" in da.dims:
        da = da.isel(time=0)
    # Drop any remaining length-1 dims
    for dim in list(da.dims):
        if dim not in ("y", "x", "lat", "lon", "latitude", "longitude"):
            if da.sizes[dim] == 1:
                da = da.isel({dim: 0})
    return da


# ---------------------------------------------------------------------------
# Colormap / norm helpers
# ---------------------------------------------------------------------------

def _build_cmap_and_norm(class_codes, class_colors):
    """Build a discrete Matplotlib colormap from class code → color dicts."""
    _, mcolors, _ = _get_mpl()

    codes  = sorted(class_codes.keys())
    colors = [class_colors[c] for c in codes]
    cmap   = mcolors.ListedColormap(colors)
    bounds = [codes[0] - 0.5] + [c + 0.5 for c in codes]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm, codes


# ---------------------------------------------------------------------------
# Legend helpers
# ---------------------------------------------------------------------------

def _add_outside_legend(fig, ax, patches, title, legend_loc):
    """Add a patch legend either inside (loc string)
    or outside ('outside right' / 'outside bottom')."""
    _, _, mpatches = _get_mpl()

    if legend_loc == "outside right":
        ax.legend(
            handles=patches,
            title=title,
            fontsize=8,
            title_fontsize=8.5,
            framealpha=0.9,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
        )
        fig.tight_layout(rect=[0, 0, 0.80, 1])
    elif legend_loc == "outside bottom":
        n = len(patches)
        ncol = min(n, 4)
        ax.legend(
            handles=patches,
            title=title,
            fontsize=8,
            title_fontsize=8.5,
            framealpha=0.9,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=ncol,
            borderaxespad=0,
        )
        fig.tight_layout(rect=[0, 0.12, 1, 1])
    else:
        # Standard matplotlib loc string
        ax.legend(
            handles=patches,
            title=title,
            fontsize=8,
            title_fontsize=8.5,
            framealpha=0.9,
            loc=legend_loc,
        )
        fig.tight_layout()


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_dynamics(
    dynamics,
    ax=None,
    title: str = "Wetland Dynamics",
    figsize: tuple = (8, 7),
    add_colorbar: bool = True,
    legend_loc: str = "outside right",
    savepath: str | None = None,
    dpi: int = 150,
):
    """Plot a wetland dynamics classification raster.

    Parameters
    ----------
    dynamics : xr.DataArray
        Output of :func:`wetlandmapper.classify_dynamics`. Spatial coordinates
        (``x``/``y`` or ``lon``/``lat``) are used for axis ticks when present.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into.  Created if not provided.
    title : str
        Plot title.
    figsize : tuple
        Figure size in inches.
    add_colorbar : bool
        Add a class legend.
    legend_loc : str
        Legend placement.  Use ``"outside right"`` (default), ``"outside bottom"``,
        or any standard Matplotlib ``loc`` string (e.g. ``"lower right"``).
        ``"outside right"`` / ``"outside bottom"`` never overlap the data.
    savepath : str, optional
        If given, save the figure to this path (PNG, PDF, TIFF, etc.).
    dpi : int
        Resolution for saved figure.  Default 150.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    from .dynamics import DYNAMICS_CLASSES, DYNAMICS_COLORS
    plt, mcolors, mpatches = _get_mpl()

    cmap, norm, codes = _build_cmap_and_norm(DYNAMICS_CLASSES, DYNAMICS_COLORS)
    fig, ax = _ensure_axes(ax, figsize)

    da2d   = _get_2d(dynamics)
    extent = _imshow_extent(da2d)
    origin = _imshow_origin(da2d)

    ax.imshow(
        da2d.values,
        cmap=cmap, norm=norm,
        origin=origin,
        extent=extent,
        interpolation="nearest",
        aspect="equal" if extent is None else "auto",
    )
    _add_xy_labels(ax, da2d)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)

    if add_colorbar:
        patches = [
            mpatches.Patch(color=DYNAMICS_COLORS[c], label=DYNAMICS_CLASSES[c])
            for c in sorted(DYNAMICS_CLASSES.keys(), reverse=True)
        ]
        _add_outside_legend(fig, ax, patches, "Dynamics Class", legend_loc)
    else:
        fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_wct(
    wct,
    ax=None,
    title: str = "Wetland Cover Types",
    figsize: tuple = (8, 7),
    add_colorbar: bool = True,
    legend_loc: str = "outside right",
    savepath: str | None = None,
    dpi: int = 150,
):
    """Plot a Wetland Cover Type classification raster.

    Parameters
    ----------
    wct : xr.DataArray
        Output of :func:`wetlandmapper.classify_wct` or
        :func:`wetlandmapper.classify_wct_ema`.
    ax, title, figsize, add_colorbar, legend_loc, savepath, dpi
        Same as :func:`plot_dynamics`.

    Returns
    -------
    fig, ax
    """
    from .wct import WCT_CLASSES, WCT_COLORS
    plt, mcolors, mpatches = _get_mpl()

    cmap, norm, codes = _build_cmap_and_norm(WCT_CLASSES, WCT_COLORS)
    fig, ax = _ensure_axes(ax, figsize)

    da2d   = _get_2d(wct)
    extent = _imshow_extent(da2d)
    origin = _imshow_origin(da2d)

    ax.imshow(
        da2d.values,
        cmap=cmap, norm=norm,
        origin=origin,
        extent=extent,
        interpolation="nearest",
        aspect="equal" if extent is None else "auto",
    )
    _add_xy_labels(ax, da2d)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)

    if add_colorbar:
        ordered = [c for c in sorted(WCT_CLASSES.keys()) if c != 0] + [0]
        patches = [
            mpatches.Patch(color=WCT_COLORS[c], label=WCT_CLASSES[c])
            for c in ordered
        ]
        _add_outside_legend(fig, ax, patches, "Cover Type", legend_loc)
    else:
        fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_index(
    da,
    index_name: str = "Index",
    ax=None,
    figsize: tuple = (8, 7),
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdYlGn",
    time_step: int | None = None,
    savepath: str | None = None,
    dpi: int = 150,
):
    """Plot a single spectral index (MNDWI, NDVI, or NDTI).

    Parameters
    ----------
    da : xr.DataArray
        2-D or 3-D (time, y, x) index DataArray.
    index_name : str
        Used in the title and colorbar label.
    time_step : int, optional
        Select this time index (0-based) when ``da`` has a time dim.
        Defaults to the temporal mean if not provided.
    savepath, dpi
        Same as :func:`plot_dynamics`.

    Returns
    -------
    fig, ax
    """
    plt, _, _ = _get_mpl()
    fig, ax   = _ensure_axes(ax, figsize)

    if "time" in da.dims:
        if time_step is not None:
            da2d     = da.isel(time=time_step)
            subtitle = f"t={time_step}"
        else:
            da2d     = da.mean(dim="time")
            subtitle = "temporal mean"
        title_full = f"{index_name} ({subtitle})"
    else:
        da2d       = da
        title_full = index_name

    extent = _imshow_extent(da2d)
    origin = _imshow_origin(da2d)

    im = ax.imshow(
        da2d.values,
        cmap=cmap, vmin=vmin, vmax=vmax,
        origin=origin,
        extent=extent,
        interpolation="bilinear",
        aspect="equal" if extent is None else "auto",
    )
    plt.colorbar(im, ax=ax, label=index_name, shrink=0.75, pad=0.02)
    _add_xy_labels(ax, da2d)
    ax.set_title(title_full, fontsize=12, fontweight="bold", pad=6)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_wet_frequency(
    mndwi,
    ax=None,
    figsize: tuple = (8, 7),
    mndwi_threshold: float = 0.0,
    savepath: str | None = None,
    dpi: int = 150,
):
    """Plot wet frequency (%) derived from an MNDWI time series.

    Parameters
    ----------
    mndwi : xr.DataArray
        Multi-temporal MNDWI with a ``time`` dimension.
    mndwi_threshold : float
        Pixels with MNDWI above this value are counted as wet.
    savepath, dpi
        Same as :func:`plot_dynamics`.

    Returns
    -------
    fig, ax
    """
    from .dynamics import compute_wet_frequency
    plt, _, _ = _get_mpl()

    freq   = compute_wet_frequency(mndwi, mndwi_threshold=mndwi_threshold)
    fig, ax = _ensure_axes(ax, figsize)

    da2d   = _get_2d(freq)
    extent = _imshow_extent(da2d)
    origin = _imshow_origin(da2d)

    im = ax.imshow(
        da2d.values,
        cmap="Blues", vmin=0, vmax=100,
        origin=origin,
        extent=extent,
        interpolation="bilinear",
        aspect="equal" if extent is None else "auto",
    )
    plt.colorbar(im, ax=ax, label="Wet Frequency (%)", shrink=0.75, pad=0.02)
    _add_xy_labels(ax, da2d)
    ax.set_title("Wet Frequency (%)", fontsize=12, fontweight="bold", pad=6)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig, ax


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_axes(ax, figsize):
    plt, _, _ = _get_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _add_xy_labels(ax, da):
    x, y = _spatial_coords(da)
    if x is not None:
        ax.set_xlabel("Longitude", fontsize=9)
    if y is not None:
        ax.set_ylabel("Latitude", fontsize=9)
    # Rotate x tick labels if they look like decimal degrees
    if x is not None:
        ax.tick_params(axis="x", labelsize=8, rotation=30)
    if y is not None:
        ax.tick_params(axis="y", labelsize=8)
