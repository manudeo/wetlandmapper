"""
analysis.py

Post-processing and analysis utilities for wetland indices xarray data.

Functions
---------
linear_trend
    Fit per-pixel linear trend against time (fractional years) and return
    slope, intercept, and p-value.
last_occurrence
    Find the year-fraction and index value of the last time each pixel was "on"
    (above a threshold) for one or more indices.
class_summary
    Summarise class-code rasters as counts, percentages, and optional area.
summarize_dynamics
    Convenience wrapper around :func:`class_summary` for dynamics outputs.
summarize_wct
    Convenience wrapper around :func:`class_summary` for WCT outputs.
trend_products
    Convenience wrapper around :func:`linear_trend` with significance masking,
    trend classes, and optional raster export.
class_area_timeseries
    Compute per-time class counts, percentages, and optional area metrics.
class_transition_matrix
    Compute class-to-class transition matrix between two timestamps.
detect_wet_events
    Detect first/last wet occurrence and persistence streak metrics.
summarize_by_polygons
    Compute polygon-level summary statistics for raster stacks.
quality_uncertainty_summary
    Compute pixelwise QA metrics such as valid support and wet fraction.
build_run_manifest
    Create and optionally persist a reproducibility manifest in JSON.
"""

from __future__ import annotations

import json
import platform
import warnings
from datetime import datetime, timezone
from hashlib import sha256
from math import erfc
from pathlib import Path
from typing import cast

import numpy as np
import xarray as xr


def _extract_class_dataarray(
    classes: xr.DataArray | xr.Dataset,
    variable: str | None,
) -> xr.DataArray:
    if isinstance(classes, xr.DataArray):
        return classes

    if variable is not None:
        if variable not in classes.data_vars:
            raise ValueError(
                f"Variable {variable!r} not found in Dataset. "
                f"Available: {list(classes.data_vars)}"
            )
        return classes[variable]

    preferred = [
        "wetland_cover_type_level2",
        "wetland_cover_type",
        "dynamics",
        "wetland_dynamics",
    ]
    for name in preferred:
        if name in classes.data_vars:
            return classes[name]

    if len(classes.data_vars) == 1:
        return next(iter(classes.data_vars.values()))

    raise ValueError(
        "Could not infer class variable from Dataset. Pass variable=... "
        f"Available: {list(classes.data_vars)}"
    )


def _area_factor(area_unit: str) -> float:
    factors = {
        "m2": 1.0,
        "km2": 1e-6,
        "ha": 1e-4,
    }
    if area_unit not in factors:
        raise ValueError(
            f"Unsupported area_unit {area_unit!r}. "
            "Use one of: 'm2', 'km2', 'ha'."
        )
    return factors[area_unit]


def _time_to_year_fraction(time_coord: xr.DataArray) -> xr.DataArray:
    """Convert a datetime-like time coordinate to decimal years."""
    import pandas as pd

    try:
        t = pd.to_datetime(time_coord.values)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(
            "Time coordinate must be datetime-like for linear_trend()."
        ) from exc

    year = np.asarray(t.year, dtype=float)
    day_of_year = np.asarray(t.dayofyear, dtype=float)
    is_leap = np.asarray(t.is_leap_year, dtype=bool)
    days_in_year = np.where(is_leap, 366.0, 365.0)

    sec = np.asarray(t.hour, dtype=float) * 3600.0
    sec += np.asarray(t.minute, dtype=float) * 60.0
    sec += np.asarray(t.second, dtype=float)
    sec += np.asarray(t.microsecond, dtype=float) * 1e-6

    frac = (day_of_year - 1.0 + sec / 86400.0) / days_in_year
    year_frac = year + frac

    return xr.DataArray(
        year_frac,
        dims=time_coord.dims,
        coords=time_coord.coords,
        name="year_fraction",
    )


def linear_trend(
    data: xr.DataArray | xr.Dataset,
    variable: str | None = None,
    time_dim: str = "time",
) -> xr.Dataset:
    """Fit per-pixel linear trend against fractional year time.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Time-series input with a datetime-like ``time_dim``.
    variable : str, optional
        Variable name when ``data`` is a Dataset. If omitted and Dataset has
        one variable, that variable is used.
    time_dim : str
        Name of time dimension. Default ``"time"``.

    Returns
    -------
    xr.Dataset
        Dataset with variables:
        ``slope`` (units per year), ``intercept`` and ``p_value``.

    Notes
    -----
    Computation is vectorized with xarray broadcasting across all non-time
    dimensions. p-values are computed from a two-sided t-test on slope.
    If SciPy is unavailable, a normal approximation is used.
    """
    if isinstance(data, xr.Dataset):
        if variable is not None:
            if variable not in data.data_vars:
                raise ValueError(
                    f"Variable {variable!r} not found in Dataset. "
                    f"Available: {list(data.data_vars)}"
                )
            da = data[variable]
        elif len(data.data_vars) == 1:
            da = next(iter(data.data_vars.values()))
        else:
            raise ValueError(
                "Dataset input requires variable=... when multiple variables exist. "
                f"Available: {list(data.data_vars)}"
            )
    else:
        da = data

    if not isinstance(da, xr.DataArray):
        raise TypeError(f"data must be xr.DataArray or xr.Dataset, got {type(data)}")

    if time_dim not in da.dims:
        raise ValueError(
            f"Input data must contain time dimension {time_dim!r}. Found {da.dims}."
        )

    t = _time_to_year_fraction(da[time_dim])
    y = da.astype(float)

    # Broadcast time coordinate across spatial dims and mask invalid values.
    t_b, y_b = xr.broadcast(t, y)
    mask = xr.ufuncs.isfinite(y_b) & xr.ufuncs.isfinite(t_b)
    n = mask.sum(dim=time_dim)

    t_mask = xr.where(mask, t_b, np.nan)
    y_mask = xr.where(mask, y_b, np.nan)

    t_mean = t_mask.mean(dim=time_dim, skipna=True)
    y_mean = y_mask.mean(dim=time_dim, skipna=True)

    dt = t_mask - t_mean
    dy = y_mask - y_mean
    sxx = (dt * dt).sum(dim=time_dim, skipna=True)
    sxy = (dt * dy).sum(dim=time_dim, skipna=True)

    slope = sxy / sxx
    intercept = y_mean - slope * t_mean

    y_hat = intercept + slope * t_b
    resid = xr.where(mask, y_b - y_hat, np.nan)
    sse = (resid * resid).sum(dim=time_dim, skipna=True)

    df = n - 2
    stderr = xr.apply_ufunc(np.sqrt, (sse / df) / sxx)
    t_stat = slope / stderr

    base_valid = (n >= 3) & xr.ufuncs.isfinite(sxx) & (sxx > 0)

    p_value = xr.full_like(slope, np.nan, dtype=float)

    try:
        from scipy.stats import t as student_t  # type: ignore

        p_calc = xr.apply_ufunc(
            lambda ts, d: 2.0 * student_t.sf(np.abs(ts), d),
            t_stat,
            df,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
    except Exception:
        warnings.warn(
            "SciPy not available: p-values computed using normal approximation.",
            UserWarning,
            stacklevel=2,
        )
        p_calc = xr.apply_ufunc(
            lambda z: erfc(abs(float(z)) / np.sqrt(2.0)),
            t_stat,
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

    zero_stderr = base_valid & xr.ufuncs.isfinite(stderr) & (stderr == 0)
    finite_stderr = base_valid & xr.ufuncs.isfinite(stderr) & (stderr > 0)

    p_zero = xr.where(xr.ufuncs.fabs(slope) > 0, 0.0, 1.0)
    p_value = xr.where(zero_stderr, p_zero, np.nan)
    p_value = xr.where(finite_stderr, p_calc, p_value)

    slope = xr.where(base_valid, slope, np.nan)
    intercept = xr.where(base_valid, intercept, np.nan)

    out = xr.Dataset(
        {
            "slope": slope,
            "intercept": intercept,
            "p_value": p_value,
        }
    )

    base_units = da.attrs.get("units")
    if base_units:
        out["slope"].attrs["units"] = f"{base_units}/year"
        out["intercept"].attrs["units"] = base_units
    out["slope"].attrs["long_name"] = "Linear trend slope"
    out["intercept"].attrs["long_name"] = "Linear trend intercept"
    out["p_value"].attrs["long_name"] = "Two-sided p-value for slope"
    out.attrs["time_basis"] = "fractional_year"
    out.attrs["source_variable"] = da.name if da.name is not None else "unnamed"

    return out


def class_summary(
    classes: xr.DataArray | xr.Dataset,
    variable: str | None = None,
    class_labels: dict[int, str] | None = None,
    pixel_area: float | None = None,
    area_unit: str = "km2",
    include_all_labels: bool = True,
) -> xr.Dataset:
    """Summarise class-code rasters as counts, percentages, and optional area.

    Parameters
    ----------
    classes : xr.DataArray or xr.Dataset
        Class-coded raster.
    variable : str, optional
        Variable name when ``classes`` is a Dataset.
    class_labels : dict[int, str], optional
        Mapping of class code to class name. When provided and
        ``include_all_labels=True``, rows for missing classes are included
        with zero counts.
    pixel_area : float, optional
        Area represented by one pixel in square metres. If provided,
        area statistics are included.
    area_unit : {"m2", "km2", "ha"}
        Unit for the output area column.
    include_all_labels : bool
        Include all keys in ``class_labels`` even if absent in the raster.

    Returns
    -------
    xr.Dataset
        Dataset indexed by ``class_code`` with variables:
        ``pixel_count``, ``percent_of_valid``, ``class_name`` and,
        when ``pixel_area`` is provided, ``area_<unit>``.
    """
    da = _extract_class_dataarray(classes, variable=variable)

    vals = np.asarray(da.values).ravel()
    valid = vals[np.isfinite(vals)]
    if valid.size == 0:
        raise ValueError("No finite class values found in input.")

    valid_int = valid.astype(np.int64)
    observed_codes, observed_counts = np.unique(valid_int, return_counts=True)

    observed_map = {
        int(code): int(count)
        for code, count in zip(observed_codes.tolist(), observed_counts.tolist())
    }

    if class_labels is not None and include_all_labels:
        class_codes = sorted(class_labels.keys())
    else:
        class_codes = sorted(observed_map.keys())

    counts = np.array([observed_map.get(c, 0) for c in class_codes], dtype=np.int64)
    total = int(valid_int.size)
    pct = (counts.astype(float) / float(total)) * 100.0

    if class_labels is None:
        names = np.array([str(c) for c in class_codes], dtype=object)
    else:
        names = np.array(
            [class_labels.get(c, f"Unknown ({c})") for c in class_codes],
            dtype=object,
        )

    out = xr.Dataset(
        data_vars={
            "pixel_count": ("class_code", counts),
            "percent_of_valid": ("class_code", pct),
            "class_name": ("class_code", names),
        },
        coords={"class_code": np.array(class_codes, dtype=np.int64)},
    )

    if pixel_area is not None:
        if pixel_area <= 0:
            raise ValueError("pixel_area must be > 0 when provided.")
        factor = _area_factor(area_unit)
        out[f"area_{area_unit}"] = (
            "class_code",
            counts.astype(float) * pixel_area * factor,
        )

    out.attrs.update(
        total_valid_pixels=total,
        source_variable=da.name if da.name is not None else "unnamed",
    )
    return out


def summarize_dynamics(
    dynamics: xr.DataArray | xr.Dataset,
    variable: str | None = None,
    pixel_area: float | None = None,
    area_unit: str = "km2",
) -> xr.Dataset:
    """Summarise dynamics classes with canonical class labels."""
    from .dynamics import DYNAMICS_CLASSES

    return class_summary(
        dynamics,
        variable=variable,
        class_labels=DYNAMICS_CLASSES,
        pixel_area=pixel_area,
        area_unit=area_unit,
        include_all_labels=True,
    )


def summarize_wct(
    wct: xr.DataArray | xr.Dataset,
    variable: str | None = None,
    level2: bool = False,
    pixel_area: float | None = None,
    area_unit: str = "km2",
) -> xr.Dataset:
    """Summarise WCT classes with Level-1 or Level-2 canonical labels."""
    from .wct import WCT_CLASSES, WCT_LEVEL2_CLASSES

    labels = WCT_LEVEL2_CLASSES if level2 else WCT_CLASSES

    return class_summary(
        wct,
        variable=variable,
        class_labels=labels,
        pixel_area=pixel_area,
        area_unit=area_unit,
        include_all_labels=True,
    )


def last_occurrence(
    data: xr.DataArray | xr.Dataset,
    indices: str | list[str],
    threshold: float = 0.0,
) -> tuple[xr.DataArray | xr.Dataset, xr.DataArray | xr.Dataset]:
    """Find the last time each pixel exceeded a threshold and its value then.

    For each spatial pixel and each requested index, scans the time series
    in reverse chronological order to find the most recent timestep where
    the index value is >= threshold. Returns two arrays:

    1. **year_fraction**: The time of last occurrence as a decimal year
       (e.g., 2025.3 = April 15, 2025).
    2. **value_at_last_on**: The index value at that time.

    Pixels that never exceed the threshold are set to NaN in both outputs.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Xarray object with a 'time' dimension (e.g. from ``fetch()`` or
        ``fetch_xee().compute()``). Can be lazy (dask-backed) or in-memory.

        For DataArray: single index band.
        For Dataset: multiple named variables (bands).
    indices : str or list of str
        Index name(s) to analyze. Must be present in ``data``.
        If ``data`` is a DataArray, this should match its name or be a single
        index name.
    threshold : float
        Threshold value; pixels with index >= threshold are considered "on".
        Default 0.0 (standard for normalized-difference indices).

    Returns
    -------
    year_fraction : xr.DataArray or xr.Dataset
        Spatial arrays (dims: ``y``, ``x``) with the decimal year of last
        occurrence for each pixel and index. Shape matches input spatial dims.
        NaN for pixels never exceeding threshold.

        If ``indices`` is a single string:
            Returns a single DataArray.
        If ``indices`` is a list:
            Returns a Dataset with one variable per index, suffixed ``_year``.

    value_at_last_on : xr.DataArray or xr.Dataset
        Spatial arrays with the index value at last occurrence, same structure
        as ``year_fraction``. Suffixed ``_value`` for Dataset outputs.

    Raises
    ------
    ValueError
        If 'time' dimension is missing or if requested index is not found.
    TypeError
        If ``data`` is neither DataArray nor Dataset.

    Notes
    -----
    **Year fraction calculation**: Based on the time coordinate value and
    the calendar. For example:

    - 2025-01-01 12:00 → 2025.0 (Jan 1)
    - 2025-04-15 00:00 → 2025.28 (≈ Apr 15)
    - 2025-12-31 23:59 → 2025.997 (Dec 31)

    **Lazy (dask) support**: The function works on both in-memory and lazy
    xarray objects. For lazy data, use ``.compute()`` on the result to
    materialize the output if needed.

    **Multiple indices**: When analyzing multiple indices from a Dataset,
    each is processed independently. The output Dataset has separate
    ``{index}_year`` and ``{index}_value`` variables.

    Examples
    --------
    Single index from a DataArray (from ``fetch()``):

    >>> mndwi = fetch(aoi, "1984-01-01", "2023-12-31", index="MNDWI")
    >>> year_last_wet, value_last_wet = last_occurrence(mndwi, threshold=0.0)
    >>> print(year_last_wet.values)  # shape (ny, nx), values like 2023.45

    Multiple indices from a Dataset (from ``fetch()`` with list of indices):

    >>> indices_ds = fetch(aoi, "1984-01-01", "2023-12-31",
    ...                     index=["MNDWI", "NDVI"])
    >>> year_last, value_last = last_occurrence(
    ...     indices_ds,
    ...     indices=["MNDWI", "NDVI"],
    ...     threshold=0.0
    ... )
    >>> print(year_last)  # Dataset with MNDWI_year, NDVI_year
    >>> print(value_last)  # Dataset with MNDWI_value, NDVI_value

    Lazy xarray from ``fetch_xee()``:

    >>> mndwi_lazy = fetch_xee(aoi, "1984-01-01", "2023-12-31")
    >>> year, value = last_occurrence(mndwi_lazy, "MNDWI", threshold=0.0)
    >>> year_computed = year.compute()  # materialize if needed
    """
    # ──────────────────────────────────────────────────────────────────────
    # Validate input and convert time coordinate to year fraction
    # ──────────────────────────────────────────────────────────────────────
    if not isinstance(data, (xr.DataArray, xr.Dataset)):
        raise TypeError(
            f"data must be xr.DataArray or xr.Dataset, got {type(data).__name__}"
        )

    if "time" not in data.dims:
        raise ValueError(
            f"data must have a 'time' dimension. Available dims: {data.dims}"
        )

    # Normalize indices to a list
    if isinstance(indices, str):
        indices_list = [indices]
        is_single = True
    elif isinstance(indices, (list, tuple)):
        indices_list = list(indices)
        is_single = False
    else:
        raise TypeError(
            f"indices must be str or list of str, got {type(indices).__name__}"
        )

    # Verify all indices exist in data
    if isinstance(data, xr.DataArray):
        if len(indices_list) > 1:
            raise ValueError(
                "Cannot request multiple indices from a single DataArray. "
                f"DataArray name: {data.name!r}, requested: {indices_list}"
            )
        if indices_list[0] != data.name and len(indices_list) == 1:
            data.name = indices_list[0]
    else:  # Dataset
        missing = set(indices_list) - set(data.data_vars)
        if missing:
            raise ValueError(
                f"Index/indices not found in Dataset: {missing}. "
                f"Available: {set(data.data_vars)}"
            )

    # Convert time coordinate to decimal year
    time_vals = data["time"].values

    # Handle time conversion using pandas
    import pandas as pd

    try:
        time_pd = pd.to_datetime(time_vals)
        # Extract as numpy arrays for reliable indexing
        years = np.asarray(time_pd.year, dtype=float)
        day_of_year = np.asarray(time_pd.dayofyear, dtype=float)
        is_leap = np.asarray(time_pd.is_leap_year, dtype=bool)
        days_in_year = np.where(is_leap, 366, 365).astype(float)
        year_fraction_array = years + (day_of_year - 1) / days_in_year
    except Exception as e:
        raise ValueError(
            f"Could not parse time coordinate: {e}. "
            "Time must be convertible to datetime via pandas.to_datetime()."
        )

    # ──────────────────────────────────────────────────────────────────────
    # Process each index
    # ──────────────────────────────────────────────────────────────────────
    result_years = {}
    result_values = {}

    for idx_name in indices_list:
        if isinstance(data, xr.DataArray):
            index_da = data
        else:
            index_da = data[idx_name]

        # index_da has dims (time, y, x) or similar spatial dims
        # We want to find, for each (y, x), the last time where index_da >= threshold

        # Create a binary mask: True where index >= threshold
        above_threshold = index_da >= threshold

        # Find the last True along time dimension.
        # Reverse time and find first True (latest in original time).
        # then map back to original time indices
        above_threshold_reversed = above_threshold.isel(time=slice(None, None, -1))
        first_along_time_reversed = cast(
            xr.DataArray,
            above_threshold_reversed.argmax(dim="time", skipna=False),
        )

        # Index in reversed array (0=last in original, 1=second-to-last, ...).
        # Convert back to original time index
        n_time = len(above_threshold.time)
        original_time_index = n_time - 1 - first_along_time_reversed

        # If no True exists, argmax returns 0; mask those pixels after selection.
        # We need to mark those as NaN. Check if any True exists along time.
        has_any_above = above_threshold.any(dim="time")

        # Extract the year_fraction at the last occurrence
        year_last = xr.DataArray(
            year_fraction_array[original_time_index.values],
            dims=original_time_index.dims,
            coords=original_time_index.coords,
        )

        # Extract the index value at the last occurrence
        value_last = index_da.isel(time=original_time_index)

        # Mask out pixels that never exceeded threshold
        year_last = year_last.where(has_any_above, np.nan)
        value_last = value_last.where(has_any_above, np.nan)

        # Store results
        if is_single:
            result_years[idx_name] = year_last
            result_values[idx_name] = value_last
        else:
            result_years[f"{idx_name}_year"] = year_last
            result_values[f"{idx_name}_value"] = value_last

    # ──────────────────────────────────────────────────────────────────────
    # Assemble output
    # ──────────────────────────────────────────────────────────────────────
    if is_single:
        year_result = result_years[indices_list[0]]
        value_result = result_values[indices_list[0]]
    else:
        # When combining multiple indices into a Dataset, use xr.Dataset directly
        # with explicit variable assignment to avoid coordinate merging issues
        year_vars = {k: (["y", "x"], v.values) for k, v in result_years.items()}
        value_vars = {k: (["y", "x"], v.values) for k, v in result_values.items()}

        # Use coordinates from the first result
        first_year = list(result_years.values())[0]
        coords = {
            "y": first_year.coords["y"],
            "x": first_year.coords["x"],
        }

        year_result = xr.Dataset(year_vars, coords=coords)
        value_result = xr.Dataset(value_vars, coords=coords)

    return year_result, value_result


def trend_products(
    data: xr.DataArray | xr.Dataset,
    variable: str | None = None,
    time_dim: str = "time",
    alpha: float = 0.05,
    stable_epsilon: float = 0.0,
    output_netcdf: str | Path | None = None,
    output_geotiff_dir: str | Path | None = None,
    geotiff_prefix: str = "trend",
) -> xr.Dataset:
    """Create a trend product bundle from a time-series variable.

    This function wraps :func:`linear_trend` and adds practical map products
    used in reporting workflows:

    - ``is_significant``: significance mask from p-value threshold
    - ``slope_significant``: slope retained only where significant
    - ``trend_class``: directional class map (-1, 0, 1)

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Input time-series data.
    variable : str, optional
        Variable to analyze when ``data`` is a Dataset.
    time_dim : str
        Name of the temporal dimension.
    alpha : float
        Two-sided significance threshold for slope p-values.
    stable_epsilon : float
        Deadband around zero slope for assigning the stable class.
        Slopes in ``[-stable_epsilon, stable_epsilon]`` are mapped to class 0.
    output_netcdf : str or pathlib.Path, optional
        If provided, writes all output layers to a NetCDF file.
    output_geotiff_dir : str or pathlib.Path, optional
        If provided, writes each output variable as a separate GeoTIFF.
        Requires ``rioxarray`` and spatial metadata on the input.
    geotiff_prefix : str
        Prefix used for GeoTIFF filenames when ``output_geotiff_dir`` is set.

    Returns
    -------
    xr.Dataset
        Dataset containing ``slope``, ``intercept``, ``p_value``,
        ``is_significant``, ``slope_significant``, and ``trend_class``.

    Notes
    -----
    ``trend_class`` encoding:

    - ``-1`` decreasing
    - ``0`` stable
    - ``1`` increasing
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if stable_epsilon < 0:
        raise ValueError("stable_epsilon must be >= 0.")

    trend = linear_trend(data=data, variable=variable, time_dim=time_dim)

    sig = trend["p_value"] <= alpha
    slope_sig = xr.where(sig, trend["slope"], np.nan)
    trend_class = xr.where(
        slope_sig > stable_epsilon,
        1,
        xr.where(slope_sig < -stable_epsilon, -1, 0),
    ).astype(np.int8)

    trend["is_significant"] = sig.astype(np.int8)
    trend["slope_significant"] = slope_sig
    trend["trend_class"] = trend_class

    trend["trend_class"].attrs["class_encoding"] = (
        "-1:decreasing,0:stable,1:increasing"
    )
    trend.attrs["alpha"] = alpha
    trend.attrs["stable_epsilon"] = stable_epsilon

    if output_netcdf is not None:
        nc_path = Path(output_netcdf)
        nc_path.parent.mkdir(parents=True, exist_ok=True)
        trend.to_netcdf(nc_path)
        trend.attrs["output_netcdf"] = str(nc_path)

    if output_geotiff_dir is not None:
        out_dir = Path(output_geotiff_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            import rioxarray  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "GeoTIFF export requires rioxarray and spatial metadata on input."
            ) from exc

        for name in trend.data_vars:
            trend[name].rio.to_raster(out_dir / f"{geotiff_prefix}_{name}.tif")

        trend.attrs["output_geotiff_dir"] = str(out_dir)

    return trend


def _resolve_class_codes(
    observed_codes: np.ndarray,
    class_labels: dict[int, str] | None,
    include_all_labels: bool,
) -> list[int]:
    observed = sorted(int(v) for v in observed_codes.tolist())
    if class_labels is not None and include_all_labels:
        return sorted(int(k) for k in class_labels.keys())
    return observed


def class_area_timeseries(
    classes: xr.DataArray | xr.Dataset,
    variable: str | None = None,
    time_dim: str = "time",
    class_labels: dict[int, str] | None = None,
    pixel_area: float | None = None,
    area_unit: str = "km2",
    include_all_labels: bool = True,
) -> xr.Dataset:
    """Summarize class composition for each time step.

    Parameters
    ----------
    classes : xr.DataArray or xr.Dataset
        Class-coded raster stack with ``time_dim`` and spatial dimensions.
    variable : str, optional
        Variable to use when ``classes`` is a Dataset.
    time_dim : str
        Name of the temporal dimension.
    class_labels : dict[int, str], optional
        Mapping from class code to class label for named outputs.
    pixel_area : float, optional
        Pixel area in square meters. If provided, area columns are added.
    area_unit : {"m2", "km2", "ha"}
        Unit for output area values.
    include_all_labels : bool
        If True and ``class_labels`` is provided, include all listed classes
        even when absent at a given timestep.

    Returns
    -------
    xr.Dataset
        Dataset indexed by ``time_dim`` and ``class_code`` with
        ``pixel_count``, ``percent_of_valid``, ``total_valid_pixels`` and
        optional ``area_<unit>`` plus ``class_name``.
    """
    da = _extract_class_dataarray(classes, variable=variable)
    if time_dim not in da.dims:
        raise ValueError(
            f"Input classes must contain time dimension {time_dim!r}. Found {da.dims}."
        )

    nt = da.sizes[time_dim]
    reshaped = np.asarray(da.values).reshape(nt, -1)

    observed_codes = np.unique(reshaped[np.isfinite(reshaped)].astype(np.int64))
    if observed_codes.size == 0:
        raise ValueError("No finite class values found in input.")
    class_codes = _resolve_class_codes(observed_codes, class_labels, include_all_labels)

    counts = np.zeros((nt, len(class_codes)), dtype=np.int64)
    totals = np.zeros(nt, dtype=np.int64)
    code_to_idx = {code: i for i, code in enumerate(class_codes)}

    for i in range(nt):
        vals = reshaped[i]
        valid = vals[np.isfinite(vals)]
        totals[i] = valid.size
        if valid.size == 0:
            continue
        codes, cts = np.unique(valid.astype(np.int64), return_counts=True)
        for code, cnt in zip(codes.tolist(), cts.tolist()):
            idx = code_to_idx.get(int(code))
            if idx is not None:
                counts[i, idx] = int(cnt)

    with np.errstate(invalid="ignore", divide="ignore"):
        percent = (counts / totals[:, None]) * 100.0

    out = xr.Dataset(
        {
            "pixel_count": ((time_dim, "class_code"), counts),
            "percent_of_valid": ((time_dim, "class_code"), percent),
            "total_valid_pixels": (time_dim, totals),
        },
        coords={
            time_dim: da[time_dim],
            "class_code": np.asarray(class_codes, dtype=np.int64),
        },
    )

    if class_labels is not None:
        names = [class_labels.get(c, f"Unknown ({c})") for c in class_codes]
        out["class_name"] = ("class_code", np.asarray(names, dtype=object))

    if pixel_area is not None:
        if pixel_area <= 0:
            raise ValueError("pixel_area must be > 0 when provided.")
        factor = _area_factor(area_unit)
        out[f"area_{area_unit}"] = (
            (time_dim, "class_code"),
            counts.astype(float) * pixel_area * factor,
        )

    out.attrs["source_variable"] = da.name if da.name is not None else "unnamed"
    return out


def class_transition_matrix(
    classes: xr.DataArray | xr.Dataset,
    start_time: object,
    end_time: object,
    variable: str | None = None,
    time_dim: str = "time",
    class_labels: dict[int, str] | None = None,
    pixel_area: float | None = None,
    area_unit: str = "km2",
    include_all_labels: bool = True,
) -> xr.Dataset:
    """Compute class transitions between two timestamps.

    Parameters
    ----------
    classes : xr.DataArray or xr.Dataset
        Class-coded raster stack with ``time_dim``.
    start_time : object
        Start timestamp selector value used with ``xarray.DataArray.sel``.
    end_time : object
        End timestamp selector value used with ``xarray.DataArray.sel``.
    variable : str, optional
        Variable to use when ``classes`` is a Dataset.
    time_dim : str
        Name of the temporal dimension.
    class_labels : dict[int, str], optional
        Mapping from class code to class label.
    pixel_area : float, optional
        Pixel area in square meters to derive area transition matrix.
    area_unit : {"m2", "km2", "ha"}
        Unit for output area values.
    include_all_labels : bool
        If True and ``class_labels`` is provided, includes all listed classes
        in both from/to axes.

    Returns
    -------
    xr.Dataset
        Transition matrix with dimensions ``from_class`` and ``to_class`` and
        variables ``pixel_count`` and ``percent_of_valid`` plus optional area.
    """
    da = _extract_class_dataarray(classes, variable=variable)
    if time_dim not in da.dims:
        raise ValueError(
            f"Input classes must contain time dimension {time_dim!r}. Found {da.dims}."
        )

    a = da.sel({time_dim: start_time})
    b = da.sel({time_dim: end_time})

    av = np.asarray(a.values).ravel()
    bv = np.asarray(b.values).ravel()
    mask = np.isfinite(av) & np.isfinite(bv)
    av = av[mask].astype(np.int64)
    bv = bv[mask].astype(np.int64)

    if av.size == 0:
        raise ValueError("No overlapping finite class values for selected times.")

    observed = np.unique(np.concatenate([av, bv]))
    class_codes = _resolve_class_codes(observed, class_labels, include_all_labels)
    code_to_idx = {code: i for i, code in enumerate(class_codes)}

    mat = np.zeros((len(class_codes), len(class_codes)), dtype=np.int64)
    for c_from, c_to in zip(av.tolist(), bv.tolist()):
        i = code_to_idx.get(int(c_from))
        j = code_to_idx.get(int(c_to))
        if i is not None and j is not None:
            mat[i, j] += 1

    total = int(mat.sum())
    percent = (mat.astype(float) / float(total)) * 100.0

    out = xr.Dataset(
        {
            "pixel_count": (("from_class", "to_class"), mat),
            "percent_of_valid": (("from_class", "to_class"), percent),
        },
        coords={
            "from_class": np.asarray(class_codes, dtype=np.int64),
            "to_class": np.asarray(class_codes, dtype=np.int64),
        },
    )

    if class_labels is not None:
        out["from_name"] = (
            "from_class",
            np.asarray([class_labels.get(c, f"Unknown ({c})") for c in class_codes]),
        )
        out["to_name"] = (
            "to_class",
            np.asarray([class_labels.get(c, f"Unknown ({c})") for c in class_codes]),
        )

    if pixel_area is not None:
        if pixel_area <= 0:
            raise ValueError("pixel_area must be > 0 when provided.")
        factor = _area_factor(area_unit)
        out[f"area_{area_unit}"] = (
            ("from_class", "to_class"),
            mat.astype(float) * pixel_area * factor,
        )

    out.attrs.update(
        total_valid_pixels=total,
        source_variable=da.name if da.name is not None else "unnamed",
        start_time=str(np.asarray(a[time_dim].values)),
        end_time=str(np.asarray(b[time_dim].values)),
    )
    return out


def detect_wet_events(
    data: xr.DataArray | xr.Dataset,
    variable: str | None = None,
    threshold: float = 0.0,
    time_dim: str = "time",
) -> xr.Dataset:
    """Derive wet-event timing and persistence indicators per pixel.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Time-series variable to evaluate against ``threshold``.
    variable : str, optional
        Variable to use when ``data`` is a Dataset.
    threshold : float
        Wetness threshold. Values greater than or equal to this threshold are
        treated as wet observations.
    time_dim : str
        Name of the temporal dimension.

    Returns
    -------
    xr.Dataset
        Dataset with:
        ``first_on_year``, ``last_on_year``, ``on_count``, ``on_fraction``,
        and ``longest_on_streak``.

    Notes
    -----
    ``first_on_year`` and ``last_on_year`` are expressed in fractional years.
    """
    if isinstance(data, xr.Dataset):
        if variable is None:
            if len(data.data_vars) == 1:
                da = next(iter(data.data_vars.values()))
            else:
                raise ValueError(
                    "Dataset input requires variable=... when multiple variables exist."
                )
        else:
            if variable not in data.data_vars:
                raise ValueError(
                    f"Variable {variable!r} not found in Dataset. "
                    f"Available: {list(data.data_vars)}"
                )
            da = data[variable]
    else:
        da = data

    if time_dim not in da.dims:
        raise ValueError(
            f"Input data must contain time dimension {time_dim!r}. Found {da.dims}."
        )

    valid = xr.ufuncs.isfinite(da)
    on = (da >= threshold) & valid

    n_total = valid.sum(dim=time_dim)
    n_on = on.sum(dim=time_dim)
    on_fraction = xr.where(n_total > 0, n_on / n_total, np.nan)

    t_year = _time_to_year_fraction(da[time_dim])
    t_b, _ = xr.broadcast(t_year, da)
    first_on = t_b.where(on).min(dim=time_dim, skipna=True)
    last_on = t_b.where(on).max(dim=time_dim, skipna=True)

    def _max_run_1d(arr: np.ndarray) -> int:
        best = 0
        cur = 0
        for value in arr:
            if bool(value):
                cur += 1
                if cur > best:
                    best = cur
            else:
                cur = 0
        return best

    longest = xr.apply_ufunc(
        _max_run_1d,
        on.fillna(False).astype(np.int8),
        input_core_dims=[[time_dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int64],
    )

    out = xr.Dataset(
        {
            "first_on_year": first_on,
            "last_on_year": last_on,
            "on_count": n_on.astype(np.int64),
            "on_fraction": on_fraction,
            "longest_on_streak": longest.astype(np.int64),
        }
    )
    out.attrs["threshold"] = threshold
    out.attrs["time_basis"] = "fractional_year"
    return out


def summarize_by_polygons(
    data: xr.DataArray | xr.Dataset,
    polygons: object,
    variable: str | None = None,
    time_dim: str = "time",
    polygon_id_col: str | None = None,
    stats: tuple[str, ...] = ("mean", "median", "std", "count"),
):
    """Compute zonal summary statistics by polygon feature.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Spatial raster or raster stack with ``x`` and ``y`` coordinates.
    polygons : path-like or GeoDataFrame
        Polygon source used to define zones.
    variable : str, optional
        Variable to use when ``data`` is a Dataset.
    time_dim : str
        Temporal dimension name. If present, results are returned per polygon
        and per time step.
    polygon_id_col : str, optional
        Column in the GeoDataFrame used as stable polygon identifier.
        If omitted, the row index is used.
    stats : tuple[str, ...]
        Statistics to compute. Supported values are ``mean``, ``median``,
        ``std``, and ``count``.

    Returns
    -------
    pandas.DataFrame
        Rows represent polygon/time combinations (or polygon only when there
        is no time dimension).

    Raises
    ------
    ImportError
        If ``geopandas`` or ``shapely>=2`` is not available.
    """
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("summarize_by_polygons requires geopandas.") from exc

    try:
        from shapely import contains_xy
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("summarize_by_polygons requires shapely>=2.") from exc

    if isinstance(data, xr.Dataset):
        if variable is None:
            if len(data.data_vars) == 1:
                da = next(iter(data.data_vars.values()))
            else:
                raise ValueError(
                    "Dataset input requires variable=... when multiple variables exist."
                )
        else:
            if variable not in data.data_vars:
                raise ValueError(
                    f"Variable {variable!r} not found in Dataset. "
                    f"Available: {list(data.data_vars)}"
                )
            da = data[variable]
    else:
        da = data

    if "y" not in da.dims or "x" not in da.dims:
        raise ValueError("Data must include spatial dimensions 'y' and 'x'.")

    if isinstance(polygons, (str, Path)):
        gdf = gpd.read_file(polygons)
    elif hasattr(polygons, "geometry"):
        gdf = polygons.copy()
    else:
        raise TypeError("polygons must be a path-like or GeoDataFrame.")

    if gdf.empty:
        raise ValueError("No polygons provided.")

    xv = np.asarray(da["x"].values, dtype=float)
    yv = np.asarray(da["y"].values, dtype=float)
    xx, yy = np.meshgrid(xv, yv)

    has_time = time_dim in da.dims
    rows: list[dict[str, object]] = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        mask = contains_xy(geom, xx, yy)
        if not np.any(mask):
            continue

        pid = row[polygon_id_col] if polygon_id_col and polygon_id_col in row else idx
        masked = da.where(mask)

        if has_time:
            for t in masked[time_dim].values.tolist():
                vals = np.asarray(masked.sel({time_dim: t}).values).ravel()
                vals = vals[np.isfinite(vals)]
                rec: dict[str, object] = {"polygon_id": pid, time_dim: t}
                for stat in stats:
                    if vals.size == 0:
                        rec[stat] = np.nan
                    elif stat == "mean":
                        rec[stat] = float(np.mean(vals))
                    elif stat == "median":
                        rec[stat] = float(np.median(vals))
                    elif stat == "std":
                        rec[stat] = float(np.std(vals))
                    elif stat == "count":
                        rec[stat] = int(vals.size)
                    else:
                        raise ValueError(f"Unsupported stat {stat!r}.")
                rows.append(rec)
        else:
            vals = np.asarray(masked.values).ravel()
            vals = vals[np.isfinite(vals)]
            rec = {"polygon_id": pid}
            for stat in stats:
                if vals.size == 0:
                    rec[stat] = np.nan
                elif stat == "mean":
                    rec[stat] = float(np.mean(vals))
                elif stat == "median":
                    rec[stat] = float(np.median(vals))
                elif stat == "std":
                    rec[stat] = float(np.std(vals))
                elif stat == "count":
                    rec[stat] = int(vals.size)
                else:
                    raise ValueError(f"Unsupported stat {stat!r}.")
            rows.append(rec)

    import pandas as pd

    return pd.DataFrame(rows)


def quality_uncertainty_summary(
    data: xr.DataArray | xr.Dataset,
    variable: str | None = None,
    time_dim: str = "time",
    wet_threshold: float = 0.0,
    low_support_threshold: float = 0.5,
) -> xr.Dataset:
    """Compute support and uncertainty diagnostics for time-series pixels.

    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Time-series data array to evaluate.
    variable : str, optional
        Variable to use when ``data`` is a Dataset.
    time_dim : str
        Name of the temporal dimension.
    wet_threshold : float
        Threshold used to compute ``wet_fraction``.
    low_support_threshold : float
        Valid-support threshold in ``[0, 1]`` used for the ``low_support`` flag.

    Returns
    -------
    xr.Dataset
        Dataset containing ``n_total``, ``n_valid``, ``valid_fraction``,
        ``missing_fraction``, ``wet_fraction``, and ``low_support``.
        If a time-aligned ``sensor`` coordinate exists, also returns
        ``sensor_dominant_fraction`` and ``sensor_count``.
    """
    if isinstance(data, xr.Dataset):
        if variable is None:
            if len(data.data_vars) == 1:
                da = next(iter(data.data_vars.values()))
            else:
                raise ValueError(
                    "Dataset input requires variable=... when multiple variables exist."
                )
        else:
            if variable not in data.data_vars:
                raise ValueError(
                    f"Variable {variable!r} not found in Dataset. "
                    f"Available: {list(data.data_vars)}"
                )
            da = data[variable]
    else:
        da = data

    if time_dim not in da.dims:
        raise ValueError(
            f"Input data must contain time dimension {time_dim!r}. Found {da.dims}."
        )
    if not (0.0 <= low_support_threshold <= 1.0):
        raise ValueError("low_support_threshold must be in [0, 1].")

    valid = xr.ufuncs.isfinite(da)
    n_total = xr.full_like(valid.isel({time_dim: 0}), da.sizes[time_dim], dtype=np.int64)
    n_valid = valid.sum(dim=time_dim).astype(np.int64)
    valid_fraction = n_valid / n_total
    missing_fraction = 1.0 - valid_fraction

    wet = (da >= wet_threshold) & valid
    wet_fraction = xr.where(n_valid > 0, wet.sum(dim=time_dim) / n_valid, np.nan)

    out = xr.Dataset(
        {
            "n_total": n_total,
            "n_valid": n_valid,
            "valid_fraction": valid_fraction,
            "missing_fraction": missing_fraction,
            "wet_fraction": wet_fraction,
            "low_support": (valid_fraction < low_support_threshold).astype(np.int8),
        }
    )

    if "sensor" in da.coords and da["sensor"].dims == (time_dim,):
        sensor_coord = xr.DataArray(da["sensor"].values, dims=[time_dim])
        observed = xr.where(valid, sensor_coord, "")

        def _dominant_sensor_ratio(values: np.ndarray) -> float:
            items = [v for v in values.tolist() if v != ""]
            if not items:
                return np.nan
            _, counts = np.unique(items, return_counts=True)
            return float(np.max(counts) / np.sum(counts))

        def _sensor_type_count(values: np.ndarray) -> int:
            items = [v for v in values.tolist() if v != ""]
            return int(len(np.unique(items)))

        out["sensor_dominant_fraction"] = xr.apply_ufunc(
            _dominant_sensor_ratio,
            observed,
            input_core_dims=[[time_dim]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        out["sensor_count"] = xr.apply_ufunc(
            _sensor_type_count,
            observed,
            input_core_dims=[[time_dim]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int64],
        )

    out.attrs["wet_threshold"] = wet_threshold
    out.attrs["low_support_threshold"] = low_support_threshold
    return out


def build_run_manifest(
    parameters: dict[str, object] | None = None,
    input_paths: list[str | Path] | None = None,
    output_path: str | Path | None = None,
    extras: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build and optionally write a reproducibility manifest.

    Parameters
    ----------
    parameters : dict, optional
        Runtime parameters and thresholds used in the workflow.
    input_paths : list[str or pathlib.Path], optional
        Input files to hash with SHA-256 for provenance tracking.
    output_path : str or pathlib.Path, optional
        If provided, writes the manifest as JSON to this location.
    extras : dict, optional
        Additional structured metadata to store in the manifest.

    Returns
    -------
    dict[str, object]
        Manifest dictionary containing timestamp, software versions, platform,
        parameter payload, and input file hashes.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        wm_version = version("wetlandmapper")
    except PackageNotFoundError:
        wm_version = "unknown"

    hashes: dict[str, str] = {}
    for pathlike in input_paths or []:
        path = Path(pathlike)
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")
        if path.is_file():
            hashes[str(path)] = sha256(path.read_bytes()).hexdigest()

    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wetlandmapper_version": wm_version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "parameters": parameters or {},
        "input_hashes_sha256": hashes,
        "extras": extras or {},
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest
