"""
analysis.py

Post-processing and analysis utilities for wetland indices xarray data.

Functions
---------
last_occurrence
    Find the year-fraction and index value of the last time each pixel was "on"
    (above a threshold) for one or more indices.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


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
        first_along_time_reversed = above_threshold_reversed.argmax(
            dim="time", skipna=False
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
