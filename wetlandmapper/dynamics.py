"""
dynamics.py

# Copyright (c) 2026, Manudeo Singh          #
# Author: Manudeo Singh, March 2026          #
-----------
Wetland detection and temporal dynamics classification from a water index time series.

Implements the basin-scale inventory and hydrodynamics framework of:
    Singh & Sinha (2022). A basin-scale inventory and hydrodynamics of floodplain
    wetlands based on time-series of remote sensing data.
    Remote Sensing Letters, 13(1), 1–13.
    https://doi.org/10.1080/2150704X.2021.1980919

Supports any water index (MNDWI, AWEIsh, AWEInsh, etc.) as input. The classification
is based on temporal patterns of wetness frequency and directional change over
the time series.

Six dynamics classes
--------------------
Class code  Name          Description
----------  ------------- -------------------------------------------------------
10          Persistent    Wet for >= thresholdPersis % of all time steps.
                          Perennially or near-perennially inundated.
6           Intermittent  Wet for >= thresholdWet % of time steps but does not
                          meet any directional change criterion. Seasonally active
                          without a clear trend.
5           Intensifying  Wet frequency increasing: more wet in recent period than
                          historic, but not fully new.
4           Diminishing   Wet frequency decreasing: less wet recently than
                          historically, but not fully lost.
3           Lost          Completely dry in the recent period (delta_w == -nYear).
                          Previously wet, now absent.
2           New           Completely dry in the historic period (delta_w == +nYear).
                          Newly emerged wetland.
0           Non-wetland   Pixel does not meet the minimum wet-frequency threshold.

Notes
-----
- Class codes are additive integers so that multiple conditions can be inspected
  if needed; the dominant class is selected by priority order in the code.
- 'Persistent' (10) takes highest priority because a consistently wet pixel
  subsumes directional change categories.
- Water index thresholds: Wet pixels are identified where index > 0 (default).
  This works for MNDWI, AWEIsh, AWEInsh, and most water indices.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

try:
    import rioxarray  # noqa: F401 — registers .rio accessor

    _HAS_RIO = True
except ImportError:
    _HAS_RIO = False

__all__ = [
    "classify_dynamics",
    "DYNAMICS_CLASSES",
    "DYNAMICS_COLORS",
    "compute_wet_frequency",
    "aggregate_time",
]


# ---------------------------------------------------------------------------
# Class metadata (useful for plotting and legend generation)
# ---------------------------------------------------------------------------

DYNAMICS_CLASSES: dict[int, str] = {
    10: "Persistent",
    6: "Intermittent",
    5: "Intensifying",
    4: "Diminishing",
    3: "Lost",
    2: "New",
    0: "Non-wetland",
}

DYNAMICS_COLORS: dict[int, str] = {
    10: "#1a5276",  # deep blue    — Persistent
    6: "#76d7c4",  # teal         — Intermittent
    5: "#2ecc71",  # green        — Intensifying
    4: "#e67e22",  # orange       — Diminishing
    3: "#c0392b",  # red          — Lost
    2: "#8e44ad",  # purple       — New
    0: "#f2f3f4",  # light grey   — Non-wetland
}


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def classify_dynamics(
    water_index: xr.DataArray,
    nYear: int = 3,
    thresholdWet: float = 25.0,
    thresholdPersis: float = 75.0,
    water_threshold: float = 0.0,
    mndwi_threshold: float | None = None,   # deprecated alias
) -> xr.DataArray:
    """Classify wetland pixels into six temporal dynamics classes.

    The function aggregates a multi-temporal water index raster stack into three
    summary statistics (overall wet frequency, historic wet count, recent wet
    count) and then applies threshold-based rules to assign each pixel a
    dynamics class.

    Parameters
    ----------
    water_index : xr.DataArray
        Multi-temporal water index time series with a ``time`` dimension.
        Can be MNDWI, AWEIsh, AWEInsh, or any water index where positive values
        indicate water. Values should typically be in the range [-1, 1].
        Must have a CRS set (via rioxarray) to preserve spatial reference in output.
    nYear : int, optional
        Number of time steps (e.g., years or seasons) used to define each
        temporal window. The first ``nYear`` time steps form the *historic*
        period; the last ``nYear`` form the *recent* period. Default: 3.
    thresholdWet : float, optional
        Minimum wet-frequency percentage (0–100) for a pixel to be considered
        a wetland at all. Pixels below this threshold are labelled Non-wetland (0).
        Default: 25.
    thresholdPersis : float, optional
        Wet-frequency percentage (0–100) above which a pixel is classified as
        Persistent. Should be > thresholdWet. Default: 75.
    water_threshold : float, optional
        Water index value above which a pixel is counted as 'wet' at any time step.
        Default: 0.0 (positive index = water-dominated pixel).

    Returns
    -------
    xr.DataArray
        Integer raster of dynamics class codes (dtype int8).
        Class codes are defined in ``DYNAMICS_CLASSES``.
        The CRS from ``water_index`` is preserved if rioxarray is available.
        The DataArray name encodes the parameter values used, e.g.
        ``"dynamics_nYear3_wet25_persis75"``.

    Raises
    ------
    ValueError
        If ``nYear * 2 > len(water_index.time)``, i.e. the historic and recent windows
        would overlap or exceed the available time series.
    ValueError
        If ``thresholdPersis <= thresholdWet``.

    Examples
    --------
    Classify a 20-year Landsat MNDWI time series:

    >>> dynamics = classify_dynamics(
    ...     mndwi,
    ...     nYear=3,
    ...     thresholdWet=25,
    ...     thresholdPersis=75,
    ... )
    >>> dynamics.rio.to_raster("wetland_dynamics.tif")

    Use AWEIsh instead of MNDWI for better shadow suppression:

    >>> water_indices = compute_water_indices(ds)
    >>> dynamics = classify_dynamics(
    ...     water_indices["AWEIsh"],
    ...     nYear=3,
    ...     thresholdWet=25,
    ...     thresholdPersis=75,
    ... )

    Notes
    -----
    All classification steps use ``xr.where`` for vectorised, Dask-compatible
    execution — no Python-level pixel iteration is performed.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if "time" not in water_index.dims:
        raise ValueError(
            "Input 'water_index' must have a 'time' dimension. "
            f"Found dimensions: {water_index.dims}"
        )
    if mndwi_threshold is not None:
        import warnings
        warnings.warn(
            "mndwi_threshold is deprecated; use water_threshold instead.",
            DeprecationWarning, stacklevel=2
        )
        water_threshold = mndwi_threshold
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

    # ------------------------------------------------------------------
    # Stage 1: Binary water mask per time step
    # ------------------------------------------------------------------
    water_binary = xr.where(water_index > water_threshold, 1, 0)

    # ------------------------------------------------------------------
    # Stage 2: Temporal aggregation
    # ------------------------------------------------------------------
    wall = water_binary.sum(dim="time")
    whistoric = water_binary.isel(time=slice(0, nYear)).sum(dim="time")
    wrecent = water_binary.isel(time=slice(-nYear, None)).sum(dim="time")

    w_percent = (wall / n_time) * 100
    delta_w = wrecent - whistoric

    # ------------------------------------------------------------------
    # Stage 3: Dynamics classification (additive integer encoding)
    # ------------------------------------------------------------------
    classification = xr.zeros_like(w_percent, dtype=np.int8)

    # Priority 1 — Persistent (highest, overrides everything)
    classification = xr.where(
        w_percent >= thresholdPersis,
        classification + 10,
        classification,
    )

    # Priority 2 — New (fully absent in historic, present in recent)
    classification = xr.where(
        delta_w == nYear,
        classification + 2,
        classification,
    )

    # Priority 3 — Intensifying
    classification = xr.where(
        (w_percent >= thresholdWet) & (delta_w > 0) & (delta_w < nYear),
        classification + 5,
        classification,
    )

    # Priority 4 — Diminishing
    classification = xr.where(
        (w_percent >= thresholdWet) & (delta_w > -nYear) & (delta_w < 0),
        classification + 4,
        classification,
    )

    # Priority 5 — Lost (fully present in historic, absent in recent)
    classification = xr.where(
        delta_w == -nYear,
        classification + 3,
        classification,
    )

    # Priority 6 — Intermittent (wet enough but no directional class assigned yet)
    # Must check classification == 0, not classification < 4, because New (2)
    # and Lost (3) are both < 4 and would incorrectly absorb the +6 otherwise.
    classification = xr.where(
        (w_percent >= thresholdWet) & (classification == 0),
        classification + 6,
        classification,
    )

    # Non-wetland pixels remain at 0 (below thresholdWet)

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
                f"Could not write CRS to output: {e}. "
                "Install rioxarray for full CRS support.",
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
        n_timesteps=int(n_time),
        class_codes=str(DYNAMICS_CLASSES),
        references=(
            "Singh & Sinha (2022). Remote Sensing Letters, 13(1), 1-13. "
            "https://doi.org/10.1080/2150704X.2021.1980919"
        ),
    )

    return classification


# ---------------------------------------------------------------------------
# Convenience: intermediate rasters (useful for diagnostics)
# ---------------------------------------------------------------------------


def compute_wet_frequency(
    water_index: xr.DataArray,
    water_threshold: float = 0.0,
    mndwi_threshold: float | None = None,   # deprecated alias
) -> xr.DataArray:
    """Return the pixel-wise wet frequency (%) across the full time series.

    Parameters
    ----------
    water_index : xr.DataArray
        Multi-temporal water index with a ``time`` dimension.
    water_threshold : float
        Water index value above which a pixel is counted as wet. Default: 0.0.

    Returns
    -------
    xr.DataArray
        Wet frequency in percent (0–100).
    """
    if mndwi_threshold is not None:
        import warnings
        warnings.warn(
            "mndwi_threshold is deprecated; use water_threshold instead.",
            DeprecationWarning, stacklevel=2
        )
        water_threshold = mndwi_threshold
    water_binary = xr.where(water_index > water_threshold, 1, 0)
    freq = (water_binary.sum(dim="time") / len(water_index.time)) * 100
    freq.name = "wet_frequency_pct"
    freq.attrs["long_name"] = "Wet Frequency (%)"
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
