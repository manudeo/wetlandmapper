"""Wetland dynamics classification and temporal aggregation utilities.

Copyright (c) 2026, Manudeo Singh
Author: Manudeo Singh, March 2026
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

try:
    import xarray as xr
except ImportError as e:
    raise ImportError("xarray is required. Install: pip install wetlandmapper") from e

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

_VALID_NAN_POLICIES = ("total", "valid")


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
    # backward-compatibility alias only — no mndwi= alias (function is index-agnostic)
    mndwi_threshold: float | None = None,
) -> xr.DataArray:
    """Classify wetland pixels into six mutually exclusive temporal dynamics classes.

    Each pixel is assigned to exactly one class based on three temporal
    summary statistics derived from a multi-year water index stack:

    * Overall wet frequency  *W%*  (% of years above ``water_threshold``)
    * Historic wet count  *W_historic*  (first ``nYear`` years)
    * Recent wet count    *W_recent*    (last  ``nYear`` years)

    **Class priority (strictly exclusive — a pixel receives exactly one class):**

    =========  ====  ===================================================
    Class      Code  Primary condition
    =========  ====  ===================================================
    Persistent  10   W% ≥ thresholdPersis
    New          2   W_historic = 0  AND  W_recent > 0  (newly appeared)
    Lost         3   W_historic > 0  AND  W_recent = 0  (fully gone)
    Intensifying 5   W% ≥ thresholdWet  AND  delta > 0  (not New)
    Diminishing  4   W% ≥ thresholdWet  AND  delta < 0  (not Lost)
    Intermittent 6   W% ≥ thresholdWet  AND  no directional signal
    Non-wetland  0   W% < thresholdWet
    =========  ====  ===================================================

    Each priority is applied only to **unclassified** pixels (code = 0),
    preventing any pixel from receiving more than one class code even when
    multiple conditions are simultaneously true (e.g. a pixel that is both
    Persistent *and* Intensifying will be classified as Persistent only).

    Parameters
    ----------
    water_index : xr.DataArray
        Multi-temporal water index time series with a ``time`` dimension.
        Accepts MNDWI, AWEIsh, AWEInsh, NDWI, or any index where positive
        values indicate surface water.
    nYear : int
        Length of the historic and recent windows in years.  Default 3.
    thresholdWet : float
        Minimum wet frequency (%) for a pixel to be any wetland class.
        Default 25.
    thresholdPersis : float
        Wet frequency (%) above which a pixel is Persistent.  Must be
        greater than ``thresholdWet``.  Default 75.
    water_threshold : float
        Index value above which a pixel is counted as wet in a given year.
        Default 0.0 (positive MNDWI = water-dominated; Xu 2006).
    nan_policy : {"total", "valid"}
        Denominator used when computing wet frequency.

        ``"total"`` *(default)*
            Denominator = total number of time steps.  NaN pixels count as
            dry.  Reproduces the original Singh & Sinha (2022) method.
            Appropriate when NaN values are rare or randomly distributed.

        ``"valid"``
            Denominator = per-pixel count of non-NaN observations.  Wet
            frequency is the fraction of *cloud-free* years that were wet.
            The historic/recent windows are also normalised by their own
            valid counts so that ``delta`` is expressed as a fraction
            in [−1, +1].  Use when cloud masking produces substantial
            or spatially clustered NaN values.

    min_valid_obs : int, optional
        *(Only active when nan_policy="valid".)* Minimum number of non-NaN
        observations required for classification.  Pixels with fewer valid
        observations are set to NaN in the output.  Default ``None``
        (no minimum enforced).
    mndwi_threshold : float, optional
        **Deprecated.** Use ``water_threshold`` instead.

    Returns
    -------
    xr.DataArray
        Integer DataArray (dtype int8) of shape ``(y, x)`` with class
        codes from :data:`DYNAMICS_CLASSES`.  Guaranteed to contain only
        valid class codes — no additive artefacts.

    Raises
    ------
    ValueError
        If ``water_index`` lacks a ``time`` dimension, if ``nYear * 2 >
        n_time``, if thresholds are out of range, or if ``nan_policy`` is
        not one of the accepted values.

    References
    ----------
    Singh, M. & Sinha, R. (2022). Remote Sensing Letters, 13(1), 1–13.
    https://doi.org/10.1080/2150704X.2021.1980919
    """
    # ------------------------------------------------------------------
    # Backward-compatibility shim
    # ------------------------------------------------------------------
    if mndwi_threshold is not None:
        warnings.warn(
            "'mndwi_threshold' is deprecated; use 'water_threshold' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        water_threshold = mndwi_threshold

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if "time" not in water_index.dims:
        raise ValueError(
            "'water_index' must have a 'time' dimension. "
            f"Found: {water_index.dims}"
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
    if nan_policy not in _VALID_NAN_POLICIES:
        raise ValueError(
            f"nan_policy must be one of {_VALID_NAN_POLICIES}, "
            f"got {nan_policy!r}."
        )

    # ------------------------------------------------------------------
    # Stage 1: Summary statistics
    # ------------------------------------------------------------------
    his_slice = water_index.isel(time=slice(0, nYear))
    rec_slice = water_index.isel(time=slice(-nYear, None))

    if nan_policy == "total":
        # Original method: NaN → dry, denominator = n_time
        wb       = xr.where(water_index > water_threshold, 1, 0)
        wb_his   = xr.where(his_slice   > water_threshold, 1, 0)
        wb_rec   = xr.where(rec_slice   > water_threshold, 1, 0)

        wall      = wb.sum(dim="time")
        whistoric = wb_his.sum(dim="time")
        wrecent   = wb_rec.sum(dim="time")

        w_percent = (wall / n_time) * 100

        # Trend window: raw counts in [0, nYear]
        # ── Classification helpers (total mode) ───────────────────────
        def _new(wh, wr):    return (wh == 0)  & (wr > 0)
        def _lost(wh, wr):   return (wh > 0)   & (wr == 0)
        def _intens(wp, wh, wr): return (wp >= thresholdWet) & (wr > wh) & ~_new(wh, wr)
        def _dimin(wp, wh, wr):  return (wp >= thresholdWet) & (wr < wh) & ~_lost(wh, wr)

        wh_arg, wr_arg = whistoric, wrecent

    else:
        # Valid mode: per-pixel denominator
        def _safe_mean(da, dim):
            """Fraction of valid observations that were wet."""
            wet   = (da > water_threshold).where(da.notnull())
            n_v   = da.count(dim=dim)
            return wet.sum(dim=dim, skipna=True) / n_v.where(n_v > 0)

        f_total = _safe_mean(water_index, "time")
        f_his   = _safe_mean(his_slice,   "time")
        f_rec   = _safe_mean(rec_slice,   "time")

        n_valid = water_index.count(dim="time")
        w_percent = f_total * 100

        # ── Classification helpers (valid mode) ───────────────────────
        def _new(fh, fr):    return (fh == 0)  & (fr > 0)
        def _lost(fh, fr):   return (fh > 0)   & (fr == 0)
        def _intens(wp, fh, fr): return (wp >= thresholdWet) & (fr > fh) & ~_new(fh, fr)
        def _dimin(wp, fh, fr):  return (wp >= thresholdWet) & (fr < fh) & ~_lost(fh, fr)

        wh_arg, wr_arg = f_his, f_rec

    # ------------------------------------------------------------------
    # Stage 2: Exclusive priority classification
    #
    # Each rule uses the guard `classification == 0` so that a pixel
    # already assigned a class is never overwritten.  This prevents
    # additive artefacts (e.g. Persistent=10 + Intensifying=5 → 15)
    # that arise when multiple conditions are simultaneously true.
    #
    # Priority order (high → low):
    #   Persistent (10) → New (2) → Lost (3) → Intensifying (5)
    #   → Diminishing (4) → Intermittent (6) → Non-wetland (0)
    # ------------------------------------------------------------------
    unset = lambda c: c == 0   # noqa: E731  helper for readability

    classification = xr.zeros_like(w_percent, dtype=np.int8)

    # 1 — Persistent
    classification = xr.where(
        unset(classification) & (w_percent >= thresholdPersis),
        np.int8(10),
        classification,
    )

    # 2 — New
    classification = xr.where(
        unset(classification) & _new(wh_arg, wr_arg),
        np.int8(2),
        classification,
    )

    # 3 — Lost
    classification = xr.where(
        unset(classification) & _lost(wh_arg, wr_arg),
        np.int8(3),
        classification,
    )

    # 4 — Intensifying
    classification = xr.where(
        unset(classification) & _intens(w_percent, wh_arg, wr_arg),
        np.int8(5),
        classification,
    )

    # 5 — Diminishing
    classification = xr.where(
        unset(classification) & _dimin(w_percent, wh_arg, wr_arg),
        np.int8(4),
        classification,
    )

    # 6 — Intermittent
    classification = xr.where(
        unset(classification) & (w_percent >= thresholdWet),
        np.int8(6),
        classification,
    )

    # 0 — Non-wetland: pixels still 0 (below thresholdWet) remain

    # ------------------------------------------------------------------
    # Stage 3: Mask insufficient-data pixels (valid mode only)
    # ------------------------------------------------------------------
    if nan_policy == "valid" and min_valid_obs is not None:
        classification = classification.where(n_valid >= min_valid_obs)

    # ------------------------------------------------------------------
    # Sanity check: no pixel should have a code outside valid set
    # ------------------------------------------------------------------
    valid_codes = np.array(list(DYNAMICS_CLASSES.keys()), dtype=np.int8)
    # (this is a no-cost check on the non-NaN values)
    assert (
        np.isin(
            classification.values[~np.isnan(classification.values.astype(float))],
            valid_codes
        ).all()
    ), (
        "classify_dynamics produced invalid class codes — this is a bug. "
        f"Unexpected values: "
        f"{set(np.unique(classification.values.astype(int)).tolist()) - 
           set(DYNAMICS_CLASSES.keys())}"
    )

    # ------------------------------------------------------------------
    # Preserve CRS if available
    # ------------------------------------------------------------------
    if _HAS_RIO:
        try:
            crs = water_index.rio.crs
            if crs is not None:
                classification = classification.rio.write_crs(crs)
        except Exception as e:
            warnings.warn(f"Could not write CRS to output: {e}.", stacklevel=2)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    classification.name = (
        f"dynamics_nYear{nYear}_wet{thresholdWet}_persis{thresholdPersis}"
    )
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
    mndwi_threshold: float | None = None,
) -> xr.DataArray:
    """Return the pixel-wise wet frequency (%) across the full time series.

    Parameters
    ----------
    water_index : xr.DataArray
        Multi-temporal water index with a ``time`` dimension.
    water_threshold : float
        Index value above which a pixel is counted as wet.  Default 0.0.
    nan_policy : {"total", "valid"}
        ``"total"`` (default): denominator = total time steps; NaN = dry.
        ``"valid"``: denominator = per-pixel non-NaN count.
    mndwi_threshold : float, optional
        **Deprecated.** Use ``water_threshold`` instead.

    Returns
    -------
    xr.DataArray
        Wet frequency in percent (0–100), shape ``(y, x)``.
    """
    if mndwi_threshold is not None:
        warnings.warn(
            "'mndwi_threshold' is deprecated; use 'water_threshold' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        water_threshold = mndwi_threshold

    if nan_policy not in _VALID_NAN_POLICIES:
        raise ValueError(
            f"nan_policy must be one of {_VALID_NAN_POLICIES}, got {nan_policy!r}."
        )

    if nan_policy == "total":
        wb   = xr.where(water_index > water_threshold, 1, 0)
        freq = (wb.sum(dim="time") / len(water_index.time)) * 100
    else:
        wet   = (water_index > water_threshold).where(water_index.notnull())
        n_v   = water_index.count(dim="time")
        freq  = wet.sum(dim="time", skipna=True) / n_v.where(n_v > 0) * 100

    freq.name = "wet_frequency_pct"
    freq.attrs.update(
        long_name="Wet Frequency (%)",
        nan_policy=nan_policy,
        water_threshold=water_threshold,
    )
    return freq


# ---------------------------------------------------------------------------
# Temporal aggregation utility
# ---------------------------------------------------------------------------

def aggregate_time(
    da: "xr.DataArray | xr.Dataset",
    freq: str = "annual",
    method: str = "median",
) -> "xr.DataArray | xr.Dataset":
    """Temporally aggregate a multi-temporal xarray object before classification.

    Reduces a time series to one composite per chosen period by computing a
    pixel-wise statistic within each period.  Useful for:

    - **Dynamics**: produce annual composites from all available scenes rather
      than using every raw overpass.
    - **WCT**: produce monthly or seasonal composites, then classify each with
      :func:`~wetlandmapper.classify_wct` / :func:`~wetlandmapper.classify_wct_ema`.

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        Input data with a ``time`` dimension.  Accepts both GEE-fetched and
        locally constructed objects.
    freq : {"annual", "monthly", "seasonal", "all"}
        Aggregation period:

        ``"annual"``
            One composite per calendar year (resampled to year-end).
        ``"monthly"``
            One composite per calendar month.
        ``"seasonal"``
            One composite per meteorological season per year:
            DJF (Dec–Jan–Feb), MAM (Mar–Apr–May),
            JJA (Jun–Jul–Aug), SON (Sep–Oct–Nov).
            Uses ``pandas`` quarterly resampling anchored to December.
        ``"all"``
            No aggregation — returns ``da`` unchanged.

    method : {"median", "mean", "max", "min"}
        Pixel-wise statistic computed within each period. Default ``"median"``.

    Returns
    -------
    xr.DataArray or xr.Dataset
        Same type as ``da`` with a reduced ``time`` dimension (one step per
        period).  For ``"seasonal"``, the time coordinate is labelled with the
        first day of each quarter (e.g., ``2003-12-01`` for DJF 2004).

    Raises
    ------
    ValueError
        If ``freq`` or ``method`` is not one of the valid options.

    Examples
    --------
    Produce annual MNDWI composites from a dense time series:

    >>> from wetlandmapper.dynamics import aggregate_time
    >>> mndwi_annual = aggregate_time(mndwi_ts, freq="annual")
    >>> dynamics = classify_dynamics(mndwi_annual, nYear=3)

    Produce seasonal composites for WCT classification:

    >>> from wetlandmapper import compute_indices, classify_wct_ema
    >>> from wetlandmapper.dynamics import aggregate_time
    >>> indices_ts = fetch(aoi, "2010-01-01", "2023-12-31",
    ...                    index=["MNDWI","NDVI","NDTI"])
    >>> seasonal = aggregate_time(indices_ts, freq="seasonal")
    >>> # Classify each season independently
    >>> for t in seasonal.time:
    ...     wct = classify_wct_ema(seasonal.sel(time=t))

    Notes
    -----
    Pixels that are NaN (masked by cloud or no-data) in all scenes within a
    period remain NaN in the composite.  The statistic is computed ignoring
    NaNs (``skipna=True`` is the xarray default).
    """
    _VALID_FREQ = {"annual", "monthly", "seasonal", "all"}
    _VALID_METHOD = {"median", "mean", "max", "min"}

    if freq not in _VALID_FREQ:
        raise ValueError(f"freq must be one of {_VALID_FREQ}. Got {freq!r}.")
    if method not in _VALID_METHOD:
        raise ValueError(f"method must be one of {_VALID_METHOD}. Got {method!r}.")

    if freq == "all":
        return da

    _RESAMPLE_RULE = {
        "annual": "YE",
        "monthly": "ME",
        # QS-DEC anchors quarters to December: DJF / MAM / JJA / SON
        "seasonal": "QS-DEC",
    }

    resampled = da.resample(time=_RESAMPLE_RULE[freq])

    _AGG = {
        "median": resampled.median,
        "mean": resampled.mean,
        "max": resampled.max,
        "min": resampled.min,
    }
    result = _AGG[method]()

    # Carry forward the name (DataArray only)
    if isinstance(da, xr.DataArray) and da.name:
        result.name = da.name

    # Descriptive attribute
    result.attrs["temporal_aggregation"] = freq
    result.attrs["aggregation_method"] = method

    return result
