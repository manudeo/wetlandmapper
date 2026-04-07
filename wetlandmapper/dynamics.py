"""Wetland dynamics classification and temporal aggregation utilities."""

# Copyright (c) 2026, Manudeo Singh
# Author: Manudeo Singh, March 2026

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

try:
    import xarray as xr
except ImportError as e:
    raise ImportError(
        "xarray is required. Install: pip install wetlandmapper"
    ) from e

try:
    from rioxarray import exceptions as _rio_exc  # noqa: F401
    _HAS_RIO = True
except ImportError:
    _HAS_RIO = False

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

DYNAMICS_CLASSES: dict[int, str] = {
    0:  "Non-wetland",
    2:  "New",
    3:  "Lost",
    4:  "Diminishing",
    5:  "Intensifying",
    6:  "Intermittent",
    10: "Persistent",
}

DYNAMICS_COLORS: dict[int, str] = {
    0:  "#d5d8dc",
    2:  "#8e44ad",
    3:  "#c0392b",
    4:  "#e67e22",
    5:  "#2ecc71",
    6:  "#76d7c4",
    10: "#1a5276",
}


# ---------------------------------------------------------------------------
# Main classification function
# ---------------------------------------------------------------------------

def classify_dynamics(
    water_index: xr.DataArray,
    nYear: int = 3,
    thresholdWet: float = 25.0,
    thresholdPersis: float = 75.0,
    water_threshold: float = 0.0,
    nan_policy: str = "total",
    min_valid_obs: int | None = None,
    # ── backward-compatibility aliases ────────────────────────────────────
    mndwi: xr.DataArray | None = None,
    mndwi_threshold: float | None = None,
) -> xr.DataArray:
    """Classify wetland pixels into six temporal dynamics classes.

    Parameters
    ----------
    water_index : xr.DataArray
        Multi-temporal water index time series with a ``time`` dimension.
        Can be MNDWI, AWEIsh, AWEInsh, or any index where positive values
        indicate water.

        .. deprecated::
            Passing the array as keyword argument ``mndwi=`` is still
            accepted for backward compatibility but will be removed in a
            future release.  Use the first positional argument instead.

    nYear : int
        Number of years at each end of the record used to define the
        "historic" and "recent" windows for trend detection.  Default 3.
    thresholdWet : float
        Minimum wet frequency (%) for a pixel to be classified as any
        wetland class.  Default 25.
    thresholdPersis : float
        Wet frequency (%) above which a pixel is classified as Persistent.
        Must be greater than ``thresholdWet``.  Default 75.
    water_threshold : float
        Index value above which a pixel is counted as wet each year.
        Default 0.0 (positive MNDWI = water-dominated; Xu 2006).

        .. deprecated::
            Keyword argument ``mndwi_threshold=`` is still accepted for
            backward compatibility but will be removed in a future release.
            Use ``water_threshold=`` instead.

    nan_policy : {"total", "valid"}
        How to handle missing (NaN) observations when computing wet
        frequency and trend windows.

        ``"total"`` (default)
            Use the total number of time steps as the denominator.
            NaN pixels count as dry.  This is the original behaviour from
            Singh & Sinha (2022) and is appropriate when NaN pixels are
            rare and randomly distributed.

        ``"valid"``
            Use the per-pixel count of non-NaN observations as the
            denominator.  Wet frequency is therefore the fraction of
            *cloud-free* observations that were wet.  Trend windows
            (historic, recent) are also normalised by their own valid
            counts, and ``delta`` is expressed as a fraction in [−1, +1]
            rather than a raw count.  Use this when cloud masking produces
            substantial or spatially clustered NaN values.

    min_valid_obs : int, optional
        Minimum number of valid (non-NaN) observations required for a
        pixel to receive a classification under ``nan_policy="valid"``.
        Pixels below this threshold are set to NaN in the output.
        If ``None`` (default), no minimum is enforced.  Ignored when
        ``nan_policy="total"``.

    mndwi : xr.DataArray, optional
        **Deprecated.** Pass the water index as the first positional
        argument instead.

    mndwi_threshold : float, optional
        **Deprecated.** Use ``water_threshold`` instead.

    Returns
    -------
    xr.DataArray
        Integer DataArray (dtype int8) of shape ``(y, x)`` with class
        codes defined in :data:`DYNAMICS_CLASSES`.

    References
    ----------
    Singh, M. & Sinha, R. (2022). Remote Sensing Letters, 13(1), 1–13.
    https://doi.org/10.1080/2150704X.2021.1980919
    """
    # ------------------------------------------------------------------
    # Backward-compatibility shims
    # ------------------------------------------------------------------
    if mndwi is not None:
        warnings.warn(
            "The 'mndwi' keyword argument is deprecated and will be removed "
            "in a future release. Pass the water index as the first positional "
            "argument: classify_dynamics(my_mndwi, ...).",
            DeprecationWarning,
            stacklevel=2,
        )
        water_index = mndwi

    if mndwi_threshold is not None:
        warnings.warn(
            "The 'mndwi_threshold' keyword argument is deprecated and will be "
            "removed in a future release. Use 'water_threshold' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        water_threshold = mndwi_threshold

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if "time" not in water_index.dims:
        raise ValueError(
            "Input 'water_index' must have a 'time' dimension. "
            f"Found dimensions: {water_index.dims}"
        )
    n_time = len(water_index.time)
    if nYear * 2 > n_time:
        raise ValueError(
            f"nYear={nYear} requires at least {nYear * 2} time steps, "
            f"but the input has only {n_time}."
        )
    if not (0 <= thresholdWet <= 100) or not (0 <= thresholdPersis <= 100):
        raise ValueError("Thresholds must be in the range [0, 100].")
    if thresholdPersis <= thresholdWet:
        raise ValueError(
            f"thresholdPersis ({thresholdPersis}) must be greater than "
            f"thresholdWet ({thresholdWet})."
        )
    if nan_policy not in ("total", "valid"):
        raise ValueError(
            f"nan_policy must be 'total' or 'valid', got {nan_policy!r}."
        )

    # ------------------------------------------------------------------
    # Stage 1: Binary water mask per time step
    # ------------------------------------------------------------------
    if nan_policy == "total":
        # Original behaviour: NaN treated as dry, denominator = n_time
        water_binary = xr.where(water_index > water_threshold, 1, 0)

        wall      = water_binary.sum(dim="time")
        whistoric = water_binary.isel(time=slice(0, nYear)).sum(dim="time")
        wrecent   = water_binary.isel(time=slice(-nYear, None)).sum(dim="time")

        w_percent = (wall / n_time) * 100
        delta_w   = wrecent - whistoric          # raw count in [-nYear, +nYear]

        # Classification conditions — same as original
        def _is_new(dw):        return dw == nYear
        def _is_lost(dw):       return dw == -nYear
        def _is_intensifying(wp, dw): return (wp >= thresholdWet) & (dw > 0) & (dw < nYear)
        def _is_diminishing(wp, dw):  return (wp >= thresholdWet) & (dw < 0) & (dw > -nYear)

    else:
        # nan_policy == "valid": per-pixel denominator
        # Cast to float so NaN propagates correctly (int > threshold gives bool,
        # which loses NaN information).
        wet_mask = water_index > water_threshold        # bool, NaN stays NaN
        wet_float = wet_mask.where(water_index.notnull())   # NaN where input is NaN

        n_valid   = water_index.count(dim="time")
        n_his_valid = water_index.isel(time=slice(0, nYear)).count(dim="time")
        n_rec_valid = water_index.isel(time=slice(-nYear, None)).count(dim="time")

        wall      = wet_float.sum(dim="time", skipna=True)
        whistoric = wet_float.isel(time=slice(0, nYear)).sum(dim="time", skipna=True)
        wrecent   = wet_float.isel(time=slice(-nYear, None)).sum(dim="time", skipna=True)

        # Safe division: avoid /0 for pixels with no valid obs
        w_percent = (wall / n_valid.where(n_valid > 0)) * 100

        # Normalise each window by its own valid count → fraction [0, 1]
        f_his = whistoric / n_his_valid.where(n_his_valid > 0)  # NaN if no data
        f_rec = wrecent   / n_rec_valid.where(n_rec_valid > 0)

        delta_f = f_rec - f_his                  # fraction in [-1, +1]

        # Classification conditions use fraction-based logic
        def _is_new(df):         return (f_his == 0) & (f_rec > 0)
        def _is_lost(df):        return (f_his > 0)  & (f_rec == 0)
        def _is_intensifying(wp, df): return (wp >= thresholdWet) & (df > 0) & ~_is_new(df)
        def _is_diminishing(wp, df):  return (wp >= thresholdWet) & (df < 0) & ~_is_lost(df)

        delta_w = delta_f   # unified name for the classification stage below

    # ------------------------------------------------------------------
    # Stage 3: Dynamics classification (additive integer encoding)
    # ------------------------------------------------------------------
    classification = xr.zeros_like(w_percent, dtype=np.int8)

    # Priority 1 — Persistent
    classification = xr.where(
        w_percent >= thresholdPersis,
        classification + 10,
        classification,
    )

    # Priority 2 — New
    classification = xr.where(
        _is_new(delta_w),
        classification + 2,
        classification,
    )

    # Priority 3 — Intensifying
    classification = xr.where(
        _is_intensifying(w_percent, delta_w),
        classification + 5,
        classification,
    )

    # Priority 4 — Diminishing
    classification = xr.where(
        _is_diminishing(w_percent, delta_w),
        classification + 4,
        classification,
    )

    # Priority 5 — Lost
    classification = xr.where(
        _is_lost(delta_w),
        classification + 3,
        classification,
    )

    # Priority 6 — Intermittent
    classification = xr.where(
        (w_percent >= thresholdWet) & (classification == 0),
        classification + 6,
        classification,
    )

    # ------------------------------------------------------------------
    # Stage 4: Mask insufficient-data pixels (valid mode only)
    # ------------------------------------------------------------------
    if nan_policy == "valid" and min_valid_obs is not None:
        enough_data = n_valid >= min_valid_obs
        classification = classification.where(enough_data)

    # ------------------------------------------------------------------
    # Preserve spatial reference
    # ------------------------------------------------------------------
    if _HAS_RIO:
        try:
            crs = water_index.rio.crs
            if crs is not None:
                classification = classification.rio.write_crs(crs)
        except Exception as e:
            warnings.warn(
                f"Could not write CRS to output: {e}.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    name = f"dynamics_nYear{nYear}_wet{thresholdWet}_persis{thresholdPersis}"
    classification.name = name
    classification.attrs.update(
        long_name="Wetland Temporal Dynamics Class",
        nYear=nYear,
        thresholdWet=thresholdWet,
        thresholdPersis=thresholdPersis,
        water_threshold=water_threshold,
        nan_policy=nan_policy,
        n_timesteps=int(n_time),
        class_codes=str(DYNAMICS_CLASSES),
        references=(
            "Singh & Sinha (2022). Remote Sensing Letters, 13(1), 1-13. "
            "https://doi.org/10.1080/2150704X.2021.1980919"
        ),
    )

    return classification


# ---------------------------------------------------------------------------
# Wet frequency
# ---------------------------------------------------------------------------

def compute_wet_frequency(
    water_index: xr.DataArray,
    water_threshold: float = 0.0,
    nan_policy: str = "total",
    # backward-compat alias
    mndwi_threshold: float | None = None,
) -> xr.DataArray:
    """Return the pixel-wise wet frequency (%) across the full time series.

    Parameters
    ----------
    water_index : xr.DataArray
        Multi-temporal water index with a ``time`` dimension.
    water_threshold : float
        Index value above which a pixel is counted as wet.  Default 0.0.

        .. deprecated::
            ``mndwi_threshold`` is still accepted for backward compatibility
            but will be removed in a future release.

    nan_policy : {"total", "valid"}
        ``"total"`` (default): NaN pixels count as dry; denominator is the
        total number of time steps.  Matches the original method.

        ``"valid"``: denominator is the per-pixel count of non-NaN
        observations.  Wet frequency is the fraction of cloud-free years
        that were wet.

    Returns
    -------
    xr.DataArray
        Wet frequency in percent (0–100), or NaN where all observations
        are missing under ``nan_policy="valid"``.
    """
    if mndwi_threshold is not None:
        warnings.warn(
            "The 'mndwi_threshold' keyword argument is deprecated. "
            "Use 'water_threshold' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        water_threshold = mndwi_threshold

    if nan_policy not in ("total", "valid"):
        raise ValueError(
            f"nan_policy must be 'total' or 'valid', got {nan_policy!r}."
        )

    if nan_policy == "total":
        water_binary = xr.where(water_index > water_threshold, 1, 0)
        freq = (water_binary.sum(dim="time") / len(water_index.time)) * 100
    else:
        wet_float = (water_index > water_threshold).where(water_index.notnull())
        n_valid   = water_index.count(dim="time")
        freq      = (
            wet_float.sum(dim="time", skipna=True)
            / n_valid.where(n_valid > 0)
        ) * 100

    freq.name = "wet_frequency_pct"
    freq.attrs["long_name"] = "Wet Frequency (%)"
    freq.attrs["nan_policy"] = nan_policy
    return freq


# ---------------------------------------------------------------------------
# Temporal aggregation utility
# ---------------------------------------------------------------------------

def aggregate_time(
    da: "xr.DataArray | xr.Dataset",
    freq: str = "YE",
    method: str = "median",
) -> "xr.DataArray | xr.Dataset":
    """Resample a time-series DataArray or Dataset to a lower frequency.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Input with a ``time`` dimension.
    freq : str
        Pandas offset alias, e.g. ``"YE"`` (annual), ``"ME"`` (monthly),
        ``"QE"`` (quarterly).  Default ``"YE"``.
    method : {"median", "mean", "max", "min"}
        Reduction method.  Default ``"median"``.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Resampled object.  NaNs (``skipna=True`` is the xarray default)
        are excluded from the reduction within each period.
    """
    _valid_methods = ("median", "mean", "max", "min")
    if method not in _valid_methods:
        raise ValueError(
            f"method must be one of {_valid_methods}, got {method!r}."
        )

    resampled = da.resample(time=freq)
    return getattr(resampled, method)()
