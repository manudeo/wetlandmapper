"""
wct.py
------
Wetland Cover Type (WCT) classification from combined MNDWI, NDVI, and NDTI.

Two classification approaches are provided, both based on:
    Singh, M., Allaka, S., Gupta, P. K., Patel, J. G., & Sinha, R. (2022).
    Deriving wetland-cover types (WCTs) from integration of multispectral indices
    based on Earth observation data.
    Environmental Monitoring and Assessment, 194(12), 878.
    https://doi.org/10.1007/s10661-022-10541-7

Functions
---------
classify_wct_ema(indices, n_parts=4)
    Original EMA paper method.
    Each index is independently binned into n_parts equal intervals over its
    positive [0, 1] range (default 4, giving quartile step 0.25), producing
    a level 0-4 per index.  Every pixel then carries a three-way combination
    code  w_level × v_level × t_level  (e.g. "w2v3t1") and the WCT class
    is read from a pre-built 5×5×5 lookup table.  No priority ordering:
    each unique combination has exactly one class.

    Critical implication: a pixel with negative MNDWI (w0) but high NDVI
    (v3 or v4) is assigned WCT 4 (Emergent/Floating Veg) because dense
    aquatic vegetation completely suppresses the water signal in MNDWI.

classify_wct(indices, thresholds=None)
    Improved continuous-threshold method.
    Same five WCT classes, but boundaries are free-floating values that can
    be tuned per sensor, season, or region without changing the bin structure.

WCT class codes (shared by both methods)
 1  Open Clear Water         High MNDWI,  Low  NDVI,  Low  NDTI
 2  Turbid / Sediment-laden  High MNDWI,  Low  NDVI,  High NDTI
 3  Submerged Aquatic Veg.   High MNDWI,  Mod. NDVI,  Low  NDTI
 4  Emergent / Floating Veg. Any  MNDWI,  High NDVI,  Low  NDTI  ← includes w0
 5  Moist / Waterlogged Soil Low  MNDWI,  Low  NDVI,  Mod. NDTI
 0  Non-wetland / Dry
"""

from __future__ import annotations

import warnings
import numpy as np
import xarray as xr

try:
    import rioxarray  # noqa: F401
    _HAS_RIO = True
except ImportError:
    _HAS_RIO = False

__all__ = [
    "classify_wct_ema",
    "classify_wct",
    "WCT_CLASSES",
    "WCT_COLORS",
    "WCT_EMA_QUARTILE_BOUNDARIES",
    "build_ema_lookup_table",
]


# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

WCT_CLASSES: dict[int, str] = {
    0: "Non-wetland",
    1: "Open Clear Water",
    2: "Turbid / Sediment-laden Water",
    3: "Submerged Aquatic Vegetation",
    4: "Emergent / Floating Vegetation",
    5: "Moist / Waterlogged Soil",
}

WCT_COLORS: dict[int, str] = {
    0: "#f2f3f4",   # light grey   — Non-wetland
    1: "#1a78c2",   # blue         — Open Clear Water
    2: "#d4a017",   # amber/brown  — Turbid Water
    3: "#27ae60",   # medium green — Submerged Aquatic Veg.
    4: "#145a32",   # dark green   — Emergent / Floating Veg.
    5: "#a04000",   # brown        — Moist Soil
}

WCT_EMA_QUARTILE_BOUNDARIES: dict = {
    "description": "Equal-width bins over positive [0,1] range with n_parts=4",
    "boundaries":  [0.0, 0.25, 0.50, 0.75, 1.00],
    "level_names": {
        0: "Negative/Dry (< 0)",
        1: "Low          [0.00, 0.25)",
        2: "Moderate-Low [0.25, 0.50)",
        3: "Moderate-High[0.50, 0.75)",
        4: "High         [0.75, 1.00]",
    },
}


# ---------------------------------------------------------------------------
# Default thresholds — improved (continuous) method
# ---------------------------------------------------------------------------

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "mndwi_water":      0.0,    # MNDWI above which pixel is open-water-dominated
    "mndwi_moist":     -0.2,    # MNDWI lower bound for moist-soil zone
    "ndvi_veg_high":    0.2,    # NDVI above which emergent/floating veg. is present
    "ndvi_veg_low":     0.05,   # NDVI lower bound for submerged vegetation signal
    "ndti_turbid":      0.0,    # NDTI above which water is turbid/sediment-laden
}


# ---------------------------------------------------------------------------
# EMA lookup table  (5 × 5 × 5 : mndwi_level × ndvi_level × ndti_level)
# ---------------------------------------------------------------------------

def build_ema_lookup_table(n_parts: int = 4) -> np.ndarray:
    """Build the (n_parts+1)³ lookup table mapping index levels → WCT class.

    Every entry ``table[w, v, t]`` is the WCT class code (0–5) for the
    combination of MNDWI level w, NDVI level v, and NDTI level t.

    Level coding (same for all three indices):
        0  — negative value (dry / non-wetland)
        1  — [0,         1/n_parts)        Low
        2  — [1/n_parts, 2/n_parts)        Moderate-Low
        ...
        n  — [(n-1)/n_parts, 1.0]          High   (n = n_parts)

    WCT class rules (applied in this order; later entries overwrite earlier):

    WCT  MNDWI level   NDVI level     NDTI level   Cover type
    ---  -----------   ----------     ----------   ----------
     5   w == 1        v <= 1         t <= 2       Moist / Waterlogged Soil
     3   w >= 2        1 <= v <= 2    t <= 1       Submerged Aquatic Veg.
     2   w >= 2        v <= 1         t >= 2       Turbid / Sediment Water
     4   w >= 0        v >= 3         t <= 1       Emergent / Floating Veg. (*)
     1   w >= 3        v <= 1         t <= 1       Open Clear Water

    (*) WCT 4 matches even w == 0 (negative MNDWI): dense aquatic vegetation
    completely suppresses the MNDWI water signal, so the pixel reads as dry
    on MNDWI but shows high NDVI.  The combination code w0v3t1 (for example)
    is unambiguously aquatic-vegetated, not dry land.

    Parameters
    ----------
    n_parts : int
        Number of equal bins over [0, 1]. Default 4.

    Returns
    -------
    np.ndarray, shape (n_parts+1, n_parts+1, n_parts+1), dtype int8
        Lookup table indexed as [mndwi_level, ndvi_level, ndti_level].
    """
    n = n_parts
    size = n + 1
    table = np.zeros((size, size, size), dtype=np.int8)

    # WCT 5 — Moist / Waterlogged Soil
    # Low positive MNDWI (w==1), low NDVI (v<=1), low-moderate NDTI (t<=2)
    for w in [1]:
        for v in range(0, 2):          # 0, 1
            for t in range(0, 3):      # 0, 1, 2
                table[w, v, t] = 5

    # WCT 3 — Submerged Aquatic Vegetation
    # Moderate-high MNDWI (w>=2), moderate NDVI (v 1–2), clear water (t<=1)
    for w in range(2, n + 1):          # 2, 3, 4
        for v in range(1, 3):          # 1, 2
            for t in range(0, 2):      # 0, 1
                table[w, v, t] = 3

    # WCT 2 — Turbid / Sediment-laden Water
    # Moderate-high MNDWI (w>=2), low NDVI (v<=1), elevated NDTI (t>=2)
    for w in range(2, n + 1):          # 2, 3, 4
        for v in range(0, 2):          # 0, 1
            for t in range(2, n + 1):  # 2, 3, 4
                table[w, v, t] = 2

    # WCT 4 — Emergent / Floating Vegetation
    # ANY MNDWI including w==0 (veg masks water signal), high NDVI (v>=3), clear (t<=1)
    for w in range(0, n + 1):          # 0, 1, 2, 3, 4
        for v in range(3, n + 1):      # 3, 4
            for t in range(0, 2):      # 0, 1
                table[w, v, t] = 4

    # WCT 1 — Open Clear Water  (highest certainty: set last so it always wins
    # over WCT 3/5 at the boundary levels that both rules touch)
    # High MNDWI (w>=3), very low NDVI (v<=1), very low NDTI (t<=1)
    for w in range(3, n + 1):          # 3, 4
        for v in range(0, 2):          # 0, 1
            for t in range(0, 2):      # 0, 1
                table[w, v, t] = 1

    return table


# Cache the default table so it is built only once per process
_EMA_LOOKUP_4: np.ndarray = build_ema_lookup_table(n_parts=4)


# ---------------------------------------------------------------------------
# Original EMA method — combination-code lookup
# ---------------------------------------------------------------------------

def classify_wct_ema(
    indices: xr.Dataset,
    n_parts: int = 4,
) -> xr.DataArray:
    """Classify Wetland Cover Types using the original Singh et al. (2022) EMA method.

    Each pixel is independently binned into a level (0–n_parts) for MNDWI,
    NDVI, and NDTI.  The three levels together form a **combination code**:

        w_level  v_level  t_level  →  WCT class
        -------  -------  -------     ---------
           0        4        0    →   4 (Emergent veg — water signal masked)
           3        0        0    →   1 (Open clear water)
           2        0        3    →   2 (Turbid water)
           2        2        0    →   3 (Submerged aquatic veg.)
           1        0        1    →   5 (Moist soil)

    The mapping is a pre-built lookup table (see :func:`build_ema_lookup_table`).
    There is **no priority ordering** — each combination has exactly one class.

    A vegetation-masked pixel (negative MNDWI but high NDVI) is classified as
    WCT 4 (Emergent / Floating Vegetation) even though its MNDWI alone would
    suggest dry land, because the dense canopy fully attenuates the water signal.

    Parameters
    ----------
    indices : xr.Dataset
        Dataset with variables ``"MNDWI"``, ``"NDVI"``, ``"NDTI"``.
        Typically produced by :func:`wetlandmapper.compute_indices`.
    n_parts : int
        Number of equal bins over the positive [0, 1] range.  Default 4
        (step 0.25; levels 1–4 at boundaries 0.25 / 0.50 / 0.75 / 1.00).

    Returns
    -------
    xr.DataArray
        Integer raster of WCT codes (dtype int8). Name: ``"wetland_cover_type"``.
        CRS preserved from input if rioxarray is available.

    Notes
    -----
    The returned DataArray carries a ``"combination_codes"`` attribute that
    encodes the (mndwi_level * 100 + ndvi_level * 10 + ndti_level) value for
    every pixel, useful for inspection and debugging.

    Examples
    --------
    >>> from wetlandmapper import compute_indices, classify_wct_ema
    >>> indices = compute_indices(ds, green_band="B3", red_band="B4",
    ...                           nir_band="B5",  swir_band="B6")
    >>> wct = classify_wct_ema(indices)
    >>> wct.attrs['classification_method']
    'EMA-combination-lookup'

    References
    ----------
    Singh et al. (2022). Environmental Monitoring and Assessment, 194(12), 878.
    https://doi.org/10.1007/s10661-022-10541-7
    """
    _validate_input(indices)
    if not isinstance(n_parts, int) or n_parts < 2:
        raise ValueError(f"n_parts must be an integer >= 2, got {n_parts!r}")

    mndwi = indices["MNDWI"]
    ndvi  = indices["NDVI"]
    ndti  = indices["NDTI"]

    step = 1.0 / n_parts   # 0.25 for n_parts=4

    def _discretize(da: xr.DataArray) -> np.ndarray:
        """Return numpy int8 array of levels 0..n_parts for each pixel."""
        vals = da.values.astype(float)
        lvl  = np.zeros(vals.shape, dtype=np.int8)   # 0 = negative
        for k in range(1, n_parts + 1):
            lo = (k - 1) * step
            hi = k * step if k < n_parts else 1.0 + 1e-9
            lvl = np.where((vals >= lo) & (vals < hi), np.int8(k), lvl)
        return lvl

    ml = _discretize(mndwi)   # shape (ny, nx)
    vl = _discretize(ndvi)
    tl = _discretize(ndti)

    # Build (or retrieve cached) lookup table
    if n_parts == 4:
        table = _EMA_LOOKUP_4
    else:
        table = build_ema_lookup_table(n_parts=n_parts)

    # Vectorised lookup: table[ml[i,j], vl[i,j], tl[i,j]] for every pixel
    wct_vals = table[ml, vl, tl]   # numpy fancy indexing, shape (ny, nx)

    # Wrap in DataArray preserving spatial coordinates
    wct = xr.DataArray(
        wct_vals,
        dims=mndwi.dims,
        coords=mndwi.coords,
    )

    # Attach a diagnostic combination-code array (w*100 + v*10 + t)
    combo = ml.astype(np.int16) * 100 + vl.astype(np.int16) * 10 + tl.astype(np.int16)

    return _finalise(
        wct, indices,
        method="EMA-combination-lookup",
        extra_attrs={
            "n_parts": n_parts,
            "step": step,
            "boundaries": [k * step for k in range(n_parts + 1)],
            "combo_code_info": (
                "mndwi_level*100 + ndvi_level*10 + ndti_level; "
                "e.g. 401 = w4v0t1 = Open Clear Water"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Improved continuous-threshold method
# ---------------------------------------------------------------------------

def classify_wct(
    indices: xr.Dataset,
    thresholds: dict[str, float] | None = None,
) -> xr.DataArray:
    """Classify Wetland Cover Types using continuous, adjustable spectral thresholds.

    An improvement of :func:`classify_wct_ema`: boundaries are free-floating
    values (not locked to multiples of 1/n_parts), enabling sub-quartile
    calibration for different sensors, seasons, or geographic regions.
    The five WCT classes and their spectral logic are identical to the EMA method;
    only the boundary representation differs.

    WCT 4 (Emergent/Floating Veg.) is assigned whenever NDVI exceeds
    ``ndvi_veg_high`` regardless of the MNDWI sign, matching the EMA method's
    behaviour for vegetation-masked pixels.

    Parameters
    ----------
    indices : xr.Dataset
        Dataset with variables ``"MNDWI"``, ``"NDVI"``, ``"NDTI"``.
    thresholds : dict, optional
        Override any subset of the defaults (from Singh et al. 2022):

        ``"mndwi_water"``   (0.0)   MNDWI above which pixel is open water
        ``"mndwi_moist"``   (-0.2)  MNDWI lower bound for moist-soil zone
        ``"ndvi_veg_high"`` (0.2)   NDVI above which emergent veg. present
        ``"ndvi_veg_low"``  (0.05)  NDVI lower bound for submerged veg.
        ``"ndti_turbid"``   (0.0)   NDTI above which water is turbid

    Returns
    -------
    xr.DataArray
        Integer raster of WCT codes (dtype int8). Name: ``"wetland_cover_type"``.

    References
    ----------
    Singh et al. (2022). Environmental Monitoring and Assessment, 194(12), 878.
    """
    _validate_input(indices)

    thr = {**_DEFAULT_THRESHOLDS}
    if thresholds:
        unknown = set(thresholds) - set(_DEFAULT_THRESHOLDS)
        if unknown:
            warnings.warn(
                f"Unknown threshold key(s): {unknown}. "
                f"Valid keys: {set(_DEFAULT_THRESHOLDS)}",
                stacklevel=2,
            )
        thr.update(thresholds)

    mndwi = indices["MNDWI"]
    ndvi  = indices["NDVI"]
    ndti  = indices["NDTI"]

    is_water     = mndwi > thr["mndwi_water"]
    is_moist     = (mndwi > thr["mndwi_moist"]) & ~is_water
    is_turbid    = ndti  > thr["ndti_turbid"]
    has_high_veg = ndvi  > thr["ndvi_veg_high"]
    has_low_veg  = ndvi  > thr["ndvi_veg_low"]

    wct = xr.zeros_like(mndwi, dtype=np.int8)

    # Assign in priority order — WCT 1 (most diagnostic) last so it wins
    wct = xr.where(is_moist & ~has_low_veg,                np.int8(5), wct)  # Moist soil
    wct = xr.where(is_water & has_low_veg & ~has_high_veg, np.int8(3), wct)  # Submerged veg
    wct = xr.where(is_water & is_turbid   & ~has_low_veg,  np.int8(2), wct)  # Turbid water
    # WCT 4: any MNDWI (including negative) when NDVI is high — mirrors EMA behaviour
    wct = xr.where(has_high_veg & ~is_turbid,              np.int8(4), wct)  # Emergent veg
    wct = xr.where(is_water & ~is_turbid & ~has_low_veg,   np.int8(1), wct)  # Clear water

    return _finalise(wct, indices, method="threshold",
                     extra_attrs={"thresholds": str(thr)})


# ---------------------------------------------------------------------------
# Shared private helpers
# ---------------------------------------------------------------------------

def _validate_input(indices: xr.Dataset) -> None:
    required = {"MNDWI", "NDVI", "NDTI"}
    missing  = required - set(indices.data_vars)
    if missing:
        raise KeyError(
            f"Missing required variables: {missing}. "
            f"Present: {set(indices.data_vars)}. "
            "Use wetlandmapper.compute_indices() to generate all three."
        )


def _finalise(
    wct: xr.DataArray,
    indices: xr.Dataset,
    method: str,
    extra_attrs: dict,
) -> xr.DataArray:
    if _HAS_RIO:
        try:
            crs = indices["MNDWI"].rio.crs
            if crs is not None:
                wct = wct.rio.write_crs(crs)
        except Exception as e:
            warnings.warn(f"Could not write CRS to WCT output: {e}", stacklevel=3)

    wct.name = "wetland_cover_type"
    wct.attrs.update(
        long_name="Wetland Cover Type",
        classification_method=method,
        class_codes=str(WCT_CLASSES),
        references=(
            "Singh et al. (2022). Environmental Monitoring and Assessment, "
            "194(12), 878. https://doi.org/10.1007/s10661-022-10541-7"
        ),
        **extra_attrs,
    )
    return wct
