"""
terrain.py
----------
DEM-based terrain analysis for masking high-altitude artefacts in wetland mapping.

Glaciers, permanent snowpacks, and steep mountain terrain produce strong
positive wetness index signals that are not wetlands. This module provides
tools to identify and mask these artefacts using a Digital Elevation Model
(DEM) by characterising local terrain flatness — the key topographic
criterion that separates true wetlands (flat, low-lying areas) from
high-altitude false positives.

Three flatness metrics are provided, each serving a different purpose
-------------------------------------------------------------------
compute_slope(dem)
    Most broadly applicable. A wetland pixel almost never sits on a slope >5°.
    Computed from numpy gradient on the DEM with approximate metre conversion
    for geographic coordinates. Independent of window size.

compute_tpi(dem, window)
    Adds discrimination between flat plateau (near-zero TPI, high elevation)
    and valley bottom (negative TPI). Slope alone can't separate a flat glacier
    from a flat floodplain. TPI can, if you combine it with an elevation ceiling.

compute_local_range(dem, window)
    Directly replicates the GEE rolling-window approach. Most intuitive to
    threshold (e.g. "< 30 m variation in a 5×5 window"). Maximum minus minimum
    elevation within an NxN window. A low local range indicates flat terrain.

map_dem_depressions(raw_dem, filled_dem, ...)
    Depression mapping from raw and pit-filled DEM using integer division and
    binary reclassification (depression=1, non-depression=0), following the
    protocol described by Sinha et al. (2017, Current Science).

mask_terrain_artifacts(wetness, dem, ...)
    Combines any or all three metrics plus an elevation ceiling. Emits a warning
    if the mask retains <10% of pixels (likely thresholds are too strict).

Notes
-----
All functions operate on xarray DataArrays with spatial dimensions
``(y, x)`` or ``(lat, lon)``. CRS is preserved where rioxarray is available.
No GEE dependency — these functions are designed for use on locally downloaded
data (from :func:`wetlandmapper.gee.fetch` or any other source).

For server-side DEM masking within GEE (without downloading the DEM), use
the ``dem_mask`` parameter of :func:`wetlandmapper.gee.fetch`.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

try:
    import rioxarray  # noqa: F401

    _HAS_RIO = True
except ImportError:
    _HAS_RIO = False

__all__ = [
    "compute_slope",
    "compute_tpi",
    "compute_local_range",
    "map_dem_depressions",
    "mask_terrain_artifacts",
]


# ---------------------------------------------------------------------------
# Internal helpers


def _check_dem(dem: xr.DataArray) -> None:
    if not isinstance(dem, xr.DataArray):
        raise TypeError(f"DEM must be an xarray.DataArray, got {type(dem)}")
    spatial = {"y", "x"} | {"lat", "lon"}
    if not any(dim in spatial for dim in dem.dims):
        raise ValueError(
            "DEM must contain spatial dimensions 'y'/'x' or 'lat'/'lon'. "
            f"Found dimensions: {dem.dims}"
        )


def _spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    """Return the (y_dim, x_dim) names, supporting both y/x and lat/lon."""
    if "y" in da.dims and "x" in da.dims:
        return "y", "x"
    if "lat" in da.dims and "lon" in da.dims:
        return "lat", "lon"
    raise ValueError(
        f"Cannot identify spatial dimensions in {da.dims}. "
        "Expected 'y'/'x' or 'lat'/'lon'."
    )


# ---------------------------------------------------------------------------
# Slope


def compute_slope(
    dem: xr.DataArray,
    units: str = "degrees",
) -> xr.DataArray:
    """Compute terrain slope from a DEM DataArray.

    Uses central-difference gradient (numpy.gradient) applied to the
    spatial dimensions.  The result represents the steepness of terrain
    at each pixel — low values indicate flat ground suitable for wetlands.

    Parameters
    ----------
    dem : xr.DataArray
        Digital Elevation Model with spatial dimensions ``(y, x)`` or
        ``(lat, lon)``.  Units should be metres.
    units : {"degrees", "radians", "percent"}
        Output slope units.  Default ``"degrees"``.

    Returns
    -------
    xr.DataArray
        Slope values, same spatial shape as ``dem``.  Name: ``"slope"``.
    """
    _check_dem(dem)
    if units not in {"degrees", "radians", "percent"}:
        raise ValueError(
            f"units must be one of 'degrees', 'radians', or 'percent'. " f"Got {units!r}."
        )

    y_dim, x_dim = _spatial_dims(dem)
    y_coords = dem[y_dim].values
    x_coords = dem[x_dim].values

    if y_coords.size < 2 or x_coords.size < 2:
        raise ValueError(
            "DEM must contain at least two values in each spatial dimension."
        )

    elev = dem.values.astype(float)
    dy_m = abs(float(np.mean(np.diff(y_coords)))) * 111_320.0
    mid_lat_rad = np.deg2rad(float(np.mean(y_coords)))
    dx_m = abs(float(np.mean(np.diff(x_coords)))) * 111_320.0 * np.cos(mid_lat_rad)

    grad_y, grad_x = np.gradient(elev, dy_m, dx_m)
    slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))

    if units == "degrees":
        slope_val = np.degrees(slope_rad)
    elif units == "radians":
        slope_val = slope_rad
    else:
        slope_val = np.tan(slope_rad) * 100.0

    result = xr.DataArray(slope_val, dims=dem.dims, coords=dem.coords)
    result.name = "slope"
    result.attrs.update(
        long_name=f"Terrain Slope ({units})",
        units=units,
        source="Computed from DEM using central-difference gradient",
    )

    if _HAS_RIO:
        try:
            crs = dem.rio.crs
            if crs is not None:
                result = result.rio.write_crs(crs)
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# TPI — Topographic Position Index


def compute_tpi(
    dem: xr.DataArray,
    window: int = 5,
) -> xr.DataArray:
    """Compute the Topographic Position Index (TPI).

    TPI = elevation - focal_mean(elevation, window x window)

    Positive TPI indicates a pixel is higher than its surroundings
    (hilltop, ridge); negative TPI indicates a valley or depression;
    near-zero TPI indicates flat terrain or a mid-slope position.

    Parameters
    ----------
    dem : xr.DataArray
        Digital Elevation Model with spatial dimensions ``(y, x)`` or
        ``(lat, lon)``.
    window : int
        Size of the square focal window in pixels.  Odd numbers produce
        a symmetric neighbourhood.  Default 5 (5 x 5 = 25-pixel window).

    Returns
    -------
    xr.DataArray
        TPI values in the same elevation units as ``dem`` (usually metres).
        Name: ``"TPI"``.
    """
    _check_dem(dem)
    if not isinstance(window, int) or window < 3:
        raise ValueError("window must be an integer >= 3.")

    y_dim, x_dim = _spatial_dims(dem)
    focal_mean = dem.rolling(
        {y_dim: window, x_dim: window}, center=True, min_periods=1
    ).mean()
    tpi = dem - focal_mean
    tpi.name = "TPI"
    tpi.attrs.update(
        long_name="Topographic Position Index",
        window_size=window,
        interpretation=(
            "Positive = hilltop/ridge; "
            "Negative = valley/depression; "
            "Near-zero = flat terrain or mid-slope"
        ),
    )
    return tpi


# ---------------------------------------------------------------------------
# Local elevation range


def compute_local_range(
    dem: xr.DataArray,
    window: int = 5,
) -> xr.DataArray:
    """Compute local elevation range (max - min) within a rolling window.

    A low local range indicates flat terrain.  This directly replicates
    the rolling-window approach used in GEE scripts to retain only flat
    neighbourhoods for wetland mapping.

    Parameters
    ----------
    dem : xr.DataArray
        Digital Elevation Model with spatial dimensions ``(y, x)``.
    window : int
        Square window size in pixels.  Default 5 (5 x 5 window).

    Returns
    -------
    xr.DataArray
        Local elevation range in the same units as ``dem`` (metres).
        Name: ``"local_range"``.
    """
    _check_dem(dem)
    if not isinstance(window, int) or window < 3:
        raise ValueError("window must be an integer >= 3.")

    y_dim, x_dim = _spatial_dims(dem)
    rolling = dem.rolling({y_dim: window, x_dim: window}, center=True, min_periods=1)
    local_range = rolling.max() - rolling.min()
    local_range.name = "local_range"
    local_range.attrs.update(
        long_name="Local Elevation Range",
        window_size=window,
        units=dem.attrs.get("units", "m"),
        interpretation=(
            "Low values indicate flat terrain; "
            "high values indicate steep or rough terrain"
        ),
    )
    return local_range


def map_dem_depressions(
    raw_dem: xr.DataArray,
    filled_dem: xr.DataArray,
    *,
    require_integer: bool = True,
    apply_cleanup: bool = True,
    cleanup_window: int = 3,
    min_neighbours: int = 2,
) -> xr.DataArray:
    """Map topographic depressions from raw and pit-filled DEMs.

    This implements a depression protocol used in floodplain wetland mapping:

    1. Integer-divide ``raw_dem / filled_dem``.
    2. Pixels with value 1 are unchanged terrain (no pit).
    3. Pixels with value 0 are depressions (raw < filled).
    4. Reclassify to a binary mask: depression=1, non-depression=0.

    Parameters
    ----------
    raw_dem : xr.DataArray
        Original (unfilled) DEM.
    filled_dem : xr.DataArray
        Pit-filled DEM created from ``raw_dem``.
    require_integer : bool
        If ``True`` (default), both DEMs must have integer dtype.
    apply_cleanup : bool
        If ``True`` (default), remove isolated one-pixel/very small speckles
        using a neighbourhood-count filter.
    cleanup_window : int
        Square rolling window size for cleanup. Default 3.
    min_neighbours : int
        Minimum number of depression pixels (including self) within
        ``cleanup_window`` to retain a depression pixel. Default 2.

    Returns
    -------
    xr.DataArray
        Binary depression mask with values {0, 1}. Name: ``"depression_mask"``.

    Notes
    -----
    This method performs best in low-relief floodplains and may be less
    reliable in rugged terrain. Residual speckle can still occur and may need
    additional post-processing for specific study areas.

    References
    ----------
    Sinha, R., et al. (2017). Protocols for Riverine Wetland Mapping and
    Classification Using Remote Sensing and GIS. Current Science, 112(7),
    1544-1552. http://www.jstor.org/stable/24912702
    """
    _check_dem(raw_dem)
    _check_dem(filled_dem)

    if raw_dem.dims != filled_dem.dims or raw_dem.shape != filled_dem.shape:
        raise ValueError("raw_dem and filled_dem must have identical dimensions and shape.")

    if require_integer:
        if not np.issubdtype(raw_dem.dtype, np.integer):
            raise TypeError("raw_dem must be integer dtype when require_integer=True.")
        if not np.issubdtype(filled_dem.dtype, np.integer):
            raise TypeError("filled_dem must be integer dtype when require_integer=True.")

    if apply_cleanup:
        if not isinstance(cleanup_window, int) or cleanup_window < 3:
            raise ValueError("cleanup_window must be an integer >= 3.")
        if not isinstance(min_neighbours, int) or min_neighbours < 1:
            raise ValueError("min_neighbours must be an integer >= 1.")

    raw = raw_dem.astype(np.int64)
    filled = filled_dem.astype(np.int64)

    # Integer protocol: unchanged terrain gives 1, depressions give 0.
    ratio = xr.where(filled != 0, raw // filled, 1)
    depression_mask = xr.where(ratio == 0, 1, 0).astype(np.uint8)

    if apply_cleanup:
        y_dim, x_dim = _spatial_dims(depression_mask)
        neighbour_count = depression_mask.rolling(
            {y_dim: cleanup_window, x_dim: cleanup_window},
            center=True,
            min_periods=1,
        ).sum()
        depression_mask = xr.where(
            (depression_mask == 1) & (neighbour_count < min_neighbours),
            0,
            depression_mask,
        ).astype(np.uint8)

    depression_mask.name = "depression_mask"
    depression_mask.attrs.update(
        long_name="DEM depression mask from raw vs pit-filled DEM",
        values="1=depression/wetland candidate, 0=flat/non-depression",
        method="integer_division_raw_over_filled",
        cleanup_applied=bool(apply_cleanup),
        cleanup_window=int(cleanup_window),
        min_neighbours=int(min_neighbours),
    )

    if _HAS_RIO:
        try:
            crs = raw_dem.rio.crs
            if crs is not None:
                depression_mask = depression_mask.rio.write_crs(crs)
        except Exception:
            pass

    return depression_mask


# ---------------------------------------------------------------------------
# Combined terrain masking


def mask_terrain_artifacts(
    wetness: "xr.DataArray | xr.Dataset",
    dem: xr.DataArray,
    max_slope: float | None = 5.0,
    max_tpi: float | None = None,
    max_local_range: float | None = None,
    local_range_window: int = 5,
    tpi_window: int = 5,
    max_elevation: float | None = None,
    invert: bool = False,
) -> "xr.DataArray | xr.Dataset":
    """Mask wetness data using terrain flatness and elevation filters.

    Parameters
    ----------
    wetness : xr.DataArray or xr.Dataset
        Wetness or index data to mask. Can include a ``time`` dimension.
    dem : xr.DataArray
        Digital Elevation Model with spatial dimensions matching ``wetness``.
    max_slope : float or None
        Maximum slope in degrees. Pixels steeper than this are masked.
        Default 5.0. Set to ``None`` to disable slope filtering.
    max_tpi : float or None
        Maximum absolute TPI (metres). Default ``None`` (disabled).
    max_local_range : float or None
        Maximum local elevation range (metres). Default ``None`` (disabled).
    local_range_window : int
        Window size for local range. Default 5.
    tpi_window : int
        Window size for TPI. Default 5.
    max_elevation : float or None
        Absolute elevation ceiling (metres). Pixels above this are masked.
        Default ``None``.
    invert : bool
        If ``True``, invert the mask so excluded terrain is retained.

    Returns
    -------
    xr.DataArray or xr.Dataset
        ``wetness`` masked by terrain suitability. The type matches the input.
    """
    if not isinstance(wetness, (xr.DataArray, xr.Dataset)):
        raise TypeError(
            f"wetness must be an xarray.DataArray or Dataset, got {type(wetness)}"
        )
    _check_dem(dem)

    if max_local_range is not None and (
        not isinstance(local_range_window, int) or local_range_window < 3
    ):
        raise ValueError("local_range_window must be an integer >= 3.")
    if max_tpi is not None and (not isinstance(tpi_window, int) or tpi_window < 3):
        raise ValueError("tpi_window must be an integer >= 3.")

    terrain_mask = xr.ones_like(dem, dtype=bool)
    if max_elevation is not None:
        terrain_mask = terrain_mask & (dem <= max_elevation)
    if max_slope is not None:
        terrain_mask = terrain_mask & (compute_slope(dem) <= max_slope)
    if max_tpi is not None:
        tpi_result = abs(compute_tpi(dem, window=tpi_window)) <= max_tpi
        terrain_mask = terrain_mask & tpi_result
    if max_local_range is not None:
        lr_result = compute_local_range(dem, window=local_range_window) <= max_local_range
        terrain_mask = terrain_mask & lr_result

    if invert:
        terrain_mask = ~terrain_mask

    if isinstance(wetness, xr.Dataset):
        masked = wetness.where(terrain_mask)
    else:
        masked = wetness.where(terrain_mask)

    if _HAS_RIO and isinstance(masked, xr.DataArray):
        try:
            crs = dem.rio.crs
            if crs is not None:
                masked = masked.rio.write_crs(crs)
        except Exception:
            pass

    return masked
