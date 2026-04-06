"""
gee.py

# Copyright (c) 2026, Manudeo Singh          #
# Author: Manudeo Singh, March 2026          #
------
Optional Google Earth Engine (GEE) data acquisition module.

Supports **all Landsat missions** (4, 5, 7, 8, 9), Sentinel-2, and **MODIS** (Terra/Aqua),
including a ``"LandsatAll"`` option that automatically merges available missions for any
requested date range.

Two retrieval functions
-----------------------
fetch(aoi, ...)
    Downloads data immediately via GEE's ``getDownloadURL`` API.
    Best for small-to-medium AOIs (up to ~100 km²).

fetch_xee(aoi, ...)
    Opens the GEE collection as a **lazy Dask-backed xarray** via ``xee``.
    No data is transferred until ``.compute()`` is called.
    Suitable for large AOIs and long time series.

Temporal aggregation
--------------------
Both functions accept a ``temporal_aggregation`` parameter:

    ``"all"``      — every individual scene (default)
    ``"annual"``   — one median composite per calendar year
    ``"monthly"``  — one median composite per calendar month
    ``"seasonal"`` — one composite per meteorological season (DJF/MAM/JJA/SON)

Sensors and date coverage
------------------------------------
+--------------+---------------------+---------------------------+--------------------+
| sensor=      | GEE collection      | Operational dates         | Band family        |
+--------------+---------------------+---------------------------+--------------------+
| "Landsat4"   | LT04/C02/T1_L2      | 1982-08-22 – 1993-12-14   | TM (SR_B1–B5,B7)   |
| "Landsat5"   | LT05/C02/T1_L2      | 1984-03-16 – 2013-06-05   | TM (SR_B1–B5,B7)   |
| "Landsat7"   | LE07/C02/T1_L2      | 1999-04-15 – 2022-04-06   | ETM+ (SR_B1–B5,B7) |
|              |                     |   SLC failure: 2003-06-01 | use_slc_off=False  |
| "Landsat8"   | LC08/C02/T1_L2      | 2013-04-11 – present      | OLI (SR_B2–B7)     |
| "Landsat9"   | LC09/C02/T1_L2      | 2021-10-31 – present      | OLI-2 (SR_B2–B7)   |
| "LandsatAll" | merged above        | 1982 – present            | auto-harmonised    |
| "Sentinel2"  | S2_SR_HARMONIZED    | 2015-06-27 – present      | MSI (B2–B12)       |
| "MODIS_Terra"| MOD09A1             | 2000-02-24 – present      | MODIS (500m)       |
| "MODIS_Aqua" | MYD09A1             | 2002-07-04 – present      | MODIS (500m)       |
| "MODISAll"   | merged MODIS        | 2002-07-04 – present      | MODIS (500m)       |
+-------------+---------------------+---------------------------+--------------------+

``"Landsat"`` is an alias for ``"Landsat8"`` for backward compatibility.

Landsat 7 SLC-off note
-----------------------
The Scan Line Corrector (SLC) on Landsat 7 ETM+ failed on 2003-05-31.
Images acquired after this date have wedge-shaped data gaps covering roughly
22 % of each scene.  Use ``use_slc_off=False`` (default) to exclude these
images and use only the good-quality 1999–2003 record.  Set ``use_slc_off=True``
to include post-failure images (useful when other sensors have no coverage, e.g.
1999–2012 before Landsat 8).

MODIS note
----------
MODIS provides 500m resolution surface reflectance composites (8-day intervals).
Use ``"MODISAll"`` to automatically merge Terra and Aqua for continuous coverage.
MODIS has coarser resolution but longer temporal record (2000–present) compared
to Landsat. Suitable for regional-scale studies where 500m resolution is adequate.
Band mapping differs from Landsat — MODIS stores Red as Band 1, NIR as Band 2,
Blue as Band 3, Green as Band 4. Cloud masking uses StateQA bits 0-1 (cloud state)
and bit 2 (shadow). Scale factor is 0.0001 with no offset (unlike Landsat C02 L2's
-0.2 offset). AWEIsh and AWEInsh are computed server-side for MODIS since all
required bands are available.

DEM masking
-----------
Server-side DEM masking using Copernicus GLO-30 can be activated with the
``dem_mask`` parameter in :func:`fetch`. This applies terrain filters directly
in GEE using ``ee.Terrain.slope()`` and ``reduceNeighborhood``, avoiding the
need to download the DEM. The snow question is addressed by ``min_temp_c`` in
climate-adaptive mode (ERA5 precipitation includes snowfall; filtering to months
≥5°C removes cold months where precipitation is frozen) and by ``max_elevation_m``
(above the local glaciation line, always mask regardless of other criteria).

Requirements
------------
- earthengine-api  : always required
- rasterio         : required by fetch()
- xee + dask       : required by fetch_xee()

Install:  ``pip install wetlandmapper[gee]``
Authenticate once:  ``earthengine authenticate``
"""

from __future__ import annotations

import datetime
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Optional dependency guard
# ---------------------------------------------------------------------------
try:
    import ee

    _HAS_EE = True
except ImportError:
    _HAS_EE = False
    ee = None  # type: ignore

__all__ = ["fetch", "fetch_xee", "authenticate", "init"]


# ---------------------------------------------------------------------------
# Sensor configuration
# ---------------------------------------------------------------------------

# Internal band-family keys (not all exposed to the user)
_BAND_MAP: dict[str, dict[str, str]] = {
    # Landsat 4/5 TM and Landsat 7 ETM+ — SR_B1..B5, B7; B6 is thermal
    "LandsatTM_ETM": {
        "blue": "SR_B1",
        "green": "SR_B2",
        "red": "SR_B3",
        "nir": "SR_B4",
        "swir": "SR_B5",  # SWIR1 (band 5 on TM/ETM+)
        "swir2": "SR_B7",  # SWIR2 (band 7 on TM/ETM+)
        "qa": "QA_PIXEL",
    },
    # Landsat 8/9 OLI — SR_B2..B7 (extra coastal/aerosol B1 pushed others up by one)
    "LandsatOLI": {
        "blue": "SR_B2",
        "green": "SR_B3",
        "red": "SR_B4",
        "nir": "SR_B5",
        "swir": "SR_B6",  # SWIR1
        "swir2": "SR_B7",
        "qa": "QA_PIXEL",
    },
    # Sentinel-2 MSI
    "Sentinel2": {
        "blue": "B2",
        "green": "B3",
        "red": "B4",
        "nir": "B8",
        "swir": "B11",  # SWIR1 (20 m; resampled by GEE to 10 m)
        "swir2": "B12",
        "qa": "QA60",
    },
    # MODIS Terra/Aqua MOD09A1 / MYD09A1 (8-day 500m surface reflectance)
    # Scale: multiply by 0.0001 (stored as int16, range -100 to 16000)
    "MODIS_500m": {
        "blue": "sur_refl_b03",   # 459-479 nm
        "green": "sur_refl_b04",  # 545-565 nm
        "red": "sur_refl_b01",    # 620-670 nm
        "nir": "sur_refl_b02",    # 841-876 nm
        "swir": "sur_refl_b06",   # SWIR1 1628-1652 nm
        "swir2": "sur_refl_b07",  # SWIR2 2105-2155 nm
        "qa": "StateQA",
    },
    # Common renamed bands used internally after harmonising LandsatAll
    "_harmonised": {
        "blue": "blue",
        "green": "green",
        "red": "red",
        "nir": "nir",
        "swir": "swir1",
        "swir2": "swir2",
        "qa": "qa",
    },
}

_COLLECTION_ID: dict[str, str] = {
    "Landsat4": "LANDSAT/LT04/C02/T1_L2",
    "Landsat5": "LANDSAT/LT05/C02/T1_L2",
    "Landsat7": "LANDSAT/LE07/C02/T1_L2",
    "Landsat8": "LANDSAT/LC08/C02/T1_L2",
    "Landsat9": "LANDSAT/LC09/C02/T1_L2",
    "Sentinel2": "COPERNICUS/S2_SR_HARMONIZED",
    "MODIS_Terra": "MODIS/061/MOD09A1",
    "MODIS_Aqua": "MODIS/061/MYD09A1",
}

# Map user-facing sensor name → internal band-family key
_SENSOR_BAND_FAMILY: dict[str, str] = {
    "Landsat4": "LandsatTM_ETM",
    "Landsat5": "LandsatTM_ETM",
    "Landsat7": "LandsatTM_ETM",
    "Landsat8": "LandsatOLI",
    "Landsat9": "LandsatOLI",
    "Sentinel2": "Sentinel2",
    "MODIS_Terra": "MODIS_500m",
    "MODIS_Aqua": "MODIS_500m",
    # LandsatAll and MODISAll handled separately
}

# Backward-compat alias
_SENSOR_ALIASES: dict[str, str] = {
    "Landsat": "Landsat8",
}

# Scale factors: all Landsat C02 L2 use the same formula; S2/MODIS divide by 10000
_SCALE_FACTOR: dict[str, dict[str, float]] = {
    "LandsatTM_ETM": {"scale": 0.0000275, "offset": -0.2},
    "LandsatOLI": {"scale": 0.0000275, "offset": -0.2},
    "Sentinel2": {"scale": 0.0001, "offset": 0.0},
    "MODIS_500m": {"scale": 0.0001, "offset": 0.0},
}

# Cloud-cover image property per sensor family
_CLOUD_COVER_PROP: dict[str, str] = {
    "LandsatTM_ETM": "CLOUD_COVER",
    "LandsatOLI": "CLOUD_COVER",
    "Sentinel2": "CLOUDY_PIXEL_PERCENTAGE",
    "MODIS_500m": "CLOUD_COVER",  # not used — MODIS uses pixel-level QA
}

# Approximate operational date ranges for LandsatAll auto-selection
_LANDSAT_DATE_RANGES: dict[str, tuple[str, str]] = {
    "Landsat4": ("1982-08-22", "1993-12-15"),
    "Landsat5": ("1984-03-16", "2013-06-06"),
    "Landsat7": ("1999-04-15", "2022-04-07"),
    "Landsat8": ("2013-04-11", "2099-01-01"),
    "Landsat9": ("2021-10-31", "2099-01-01"),
}

# Date after which Landsat 7 SLC-off images should be excluded
_L7_SLC_FAILURE_DATE = "2003-06-01"

# Meteorological seasons: name → (months, label_month, label_day)
_SEASONS: dict[str, tuple[list[int], int, int]] = {
    "DJF": ([12, 1, 2], 1, 15),
    "MAM": ([3, 4, 5], 4, 15),
    "JJA": ([6, 7, 8], 7, 15),
    "SON": ([9, 10, 11], 10, 15),
}

_VALID_AGGREGATIONS = {"all", "annual", "monthly", "seasonal"}
_VALID_SINGLE_SENSORS = set(_COLLECTION_ID.keys()) | set(_SENSOR_ALIASES.keys())
_ALL_VALID_SENSORS = _VALID_SINGLE_SENSORS | {"LandsatAll", "MODISAll"}


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------


def authenticate() -> None:
    """Run interactive GEE authentication (opens browser; only needed once)."""
    _require_ee()
    ee.Authenticate()


def init(project: str | None = None) -> None:
    """Initialise the GEE Python client.

    Parameters
    ----------
    project : str, optional
        GEE cloud project ID.  Required for accounts created after 2023.
    """
    _require_ee()
    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()


# ---------------------------------------------------------------------------
# Cloud masking
# ---------------------------------------------------------------------------


def _mask_landsat_clouds(image: "ee.Image") -> "ee.Image":
    """Mask clouds and cloud shadows using QA_PIXEL bits 3 and 4 (Landsat C02 L2)."""
    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(1 << 3)
        .eq(0)  # bit 3 = cloud
        .And(qa.bitwiseAnd(1 << 4).eq(0))  # bit 4 = cloud shadow
    )
    return image.updateMask(mask)


def _mask_sentinel2_clouds(image: "ee.Image") -> "ee.Image":
    """Mask opaque clouds (bit 10) and cirrus (bit 11) using QA60 (S2 SR)."""
    qa = image.select("QA60")
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask).divide(10000)


def _mask_modis_clouds(image: "ee.Image") -> "ee.Image":
    """Mask clouds and cloud shadows using MODIS StateQA bits.

    StateQA bit layout (MOD09A1 / MYD09A1):
        Bits 0-1: cloud state  (00 = clear, 01 = cloudy, 10 = mixed)
        Bit 2:    cloud shadow (1 = shadow present)

    Only pixels where bits 0-1 == 0 (clear) AND bit 2 == 0 (no shadow)
    are retained.
    """
    qa = image.select("StateQA")
    cloud_state = qa.bitwiseAnd(3).eq(0)  # bits 0-1: clear only
    cloud_shadow = qa.bitwiseAnd(1 << 2).eq(0)  # bit 2: no shadow
    return image.updateMask(cloud_state.And(cloud_shadow))


# ---------------------------------------------------------------------------
# Index computation (server-side, on already-scaled reflectance)
# ---------------------------------------------------------------------------


def _add_indices(image: "ee.Image", bands: dict[str, str]) -> "ee.Image":
    """Add MNDWI, NDVI, and NDTI bands to a GEE image using normalizedDifference.

    Indices are evaluated server-side as (A - B) / (A + B).

    MNDWI = (Green - SWIR1) / (Green + SWIR1)   — sensitive to open water
    NDVI  = (NIR   - Red  ) / (NIR   + Red  )   — sensitive to green vegetation
    NDTI  = (Red   - Green) / (Red   + Green)   — sensitive to turbid / sediment water

    Band assignments per sensor family:

    +-------+-----------------------+-----------------------+-------------------+
    | Index | TM / ETM+             | OLI (L8/L9)           | Sentinel-2        |
    +-------+-----------------------+-----------------------+-------------------+
    | MNDWI | (SR_B2 - SR_B5) / sum | (SR_B3 - SR_B6) / sum | (B3 - B11) / sum  |
    | NDVI  | (SR_B4 - SR_B3) / sum | (SR_B5 - SR_B4) / sum | (B8 - B4 ) / sum  |
    | NDTI  | (SR_B3 - SR_B2) / sum | (SR_B4 - SR_B3) / sum | (B4 - B3 ) / sum  |
    +-------+-----------------------+-----------------------+-------------------+
    """
    mndwi = image.normalizedDifference([bands["green"], bands["swir"]]).rename("MNDWI")
    ndvi = image.normalizedDifference([bands["nir"], bands["red"]]).rename("NDVI")
    ndti = image.normalizedDifference([bands["red"], bands["green"]]).rename("NDTI")
    # AWEIsh = Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2
    # Note: Landsat C02 L2 reflectance is already scaled (0.0000275*DN - 0.2);
    # the 0.0001 constant from the original Feyisa et al. (2014) formula is omitted.
    aweish = (
        image.select(bands["blue"])
        .add(image.select(bands["green"]).multiply(2.5))
        .subtract(
            image.select(bands["nir"])
                 .add(image.select(bands["swir"]))
                 .multiply(1.5)
        )
        .subtract(image.select(bands["swir2"]).multiply(0.25))
        .rename("AWEIsh")
    )
    # AWEInsh = 4*(Green - SWIR1) - (0.25*NIR + 2.75*SWIR1)
    aweinsh = (
        image.select(bands["green"])
        .subtract(image.select(bands["swir"]))
        .multiply(4.0)
        .subtract(
            image.select(bands["nir"]).multiply(0.25)
            .add(image.select(bands["swir"]).multiply(2.75))
        )
        .rename("AWEInsh")
    )
    return image.addBands([mndwi, ndvi, ndti, aweish, aweinsh])



# ---------------------------------------------------------------------------
# LandsatAll: build merged, harmonised collection
# ---------------------------------------------------------------------------


def _build_landsat_all(
    ee_geom: "ee.Geometry",
    start: str,
    end: str,
    max_cloud_cover: float,
    use_slc_off: bool,
) -> "ee.ImageCollection":
    """Build a merged Landsat 4-9 collection with harmonised band names.

    Each mission's sub-collection is filtered to the requested date range
    (intersected with the mission's operational window), cloud-masked, scaled
    to surface reflectance, and bands renamed to a common scheme:

        blue, green, red, nir, swir1, swir2, qa

    The merged collection can then be processed identically regardless of
    which missions contributed images to a given period.

    Parameters
    ----------
    ee_geom : ee.Geometry
        Study area geometry (used for ``filterBounds``).
    start, end : str
        ISO 8601 date strings.
    max_cloud_cover : float
        Maximum cloud cover (%) per image.
    use_slc_off : bool
        If False, Landsat 7 images after 2003-05-31 are excluded.

    Returns
    -------
    ee.ImageCollection
        Merged, harmonised collection with bands:
        ``MNDWI``, ``NDVI``, ``NDTI`` (only index bands are kept).
    """
    sub_collections = []

    for mission in ["Landsat4", "Landsat5", "Landsat7", "Landsat8", "Landsat9"]:
        op_start, op_end = _LANDSAT_DATE_RANGES[mission]

        # Intersect requested range with operational window
        eff_start = max(start[:10], op_start)
        eff_end = min(end[:10], op_end)

        if eff_start >= eff_end:
            continue  # mission not active in requested period

        # For Landsat 7, optionally exclude SLC-off data
        if mission == "Landsat7" and not use_slc_off:
            eff_end = min(eff_end, _L7_SLC_FAILURE_DATE)
            if eff_start >= eff_end:
                continue  # SLC-on period entirely outside requested range

        col = (
            ee.ImageCollection(_COLLECTION_ID[mission])
            .filterBounds(ee_geom)
            .filterDate(eff_start, eff_end)
            .filter(ee.Filter.lt("CLOUD_COVER", max_cloud_cover))
        )

        band_family = _SENSOR_BAND_FAMILY[mission]
        sf = _SCALE_FACTOR[band_family]
        bm = _BAND_MAP[band_family]

        # Cloud mask + scale to reflectance
        col = col.map(_mask_landsat_clouds)
        col = col.map(
            lambda img: (
                img.multiply(sf["scale"])
                .add(sf["offset"])
                .copyProperties(img, ["system:time_start"])
            )
        )

        # Rename to common harmonised band names
        col = col.map(
            lambda img: img.select(
                [
                    bm["blue"],
                    bm["green"],
                    bm["red"],
                    bm["nir"],
                    bm["swir"],
                    bm["swir2"],
                    bm["qa"],
                ],
                ["blue", "green", "red", "nir", "swir1", "swir2", "qa"],
            )
        )

        sub_collections.append(col)

    if not sub_collections:
        raise RuntimeError(
            "No Landsat missions have data in the requested date range "
            f"[{start}, {end}] with use_slc_off={use_slc_off}."
        )

    # Merge all sub-collections
    merged = sub_collections[0]
    for col in sub_collections[1:]:
        merged = merged.merge(col)

    # Add spectral indices using harmonised band names
    harm_bands = _BAND_MAP["_harmonised"]
    merged = merged.map(lambda img: _add_indices(img, harm_bands))

    return merged


def _build_modis_all(
    ee_geom: "ee.Geometry",
    start: str,
    end: str,
    max_cloud_cover: float,
) -> "ee.ImageCollection":
    """Merge MODIS Terra (MOD09A1) and Aqua (MYD09A1) into one collection.

    Both collections use the same MODIS_500m band family.  Terra has an
    equatorial crossing time of ~10:30 AM and Aqua ~1:30 PM, giving
    approximately twice the sampling frequency when combined.  Both are
    8-day composites so the merged collection has ~16-day sampling.

    Parameters
    ----------
    ee_geom : ee.Geometry
    start, end : str
        ISO 8601 date strings.
    max_cloud_cover : float
        Not used for per-image filtering (MODIS uses pixel-level QA only);
        kept for API consistency.

    Returns
    -------
    ee.ImageCollection
        Merged Terra + Aqua collection with MNDWI / NDVI / NDTI bands.
    """
    bm = _BAND_MAP["MODIS_500m"]
    sf = _SCALE_FACTOR["MODIS_500m"]

    def _prep(collection_id):
        col = (
            ee.ImageCollection(collection_id).filterBounds(ee_geom).filterDate(start, end)
        )
        col = col.map(_mask_modis_clouds)
        col = col.map(
            lambda img: (
                img.multiply(sf["scale"]).copyProperties(img, ["system:time_start"])
            )
        )
        col = col.map(lambda img: _add_indices(img, bm))
        return col

    terra = _prep("MODIS/061/MOD09A1")
    aqua = _prep("MODIS/061/MYD09A1")
    return terra.merge(aqua)


# ---------------------------------------------------------------------------
# Server-side temporal compositing with empty-period safeguard
# ---------------------------------------------------------------------------


def _make_nan_image(bands: list[str], timestamp: "ee.Number") -> "ee.Image":
    """Create a constant all-masked image with the specified band names.

    Used as a fallback when a compositing period contains no valid images.
    The image has all pixels masked so NaN propagates naturally into the
    xarray output — no actual NaN constant is needed.
    """
    # Build a multi-band constant image, then mask all pixels
    img = ee.Image.cat([ee.Image.constant(0).rename(b) for b in bands]).updateMask(
        ee.Image.constant(0)
    )  # mask = 0 everywhere → all pixels masked
    return img.set("system:time_start", timestamp)


def _build_composites(
    collection: "ee.ImageCollection",
    temporal_aggregation: str,
    start: str,
    end: str,
    index_bands: list[str],
) -> "ee.ImageCollection":
    """Reduce an ImageCollection to one median composite per chosen period.

    Empty periods (no cloud-free scenes) produce a fully-masked image rather
    than raising a "band not found" error.  This is implemented using GEE's
    ``ee.Algorithms.If`` so the check runs server-side without extra
    ``.getInfo()`` calls.

    Parameters
    ----------
    collection : ee.ImageCollection
        Pre-processed collection with index bands added (MNDWI, NDVI, NDTI).
    temporal_aggregation : str
        One of ``"all"``, ``"annual"``, ``"monthly"``, ``"seasonal"``.
    start, end : str
        ISO 8601 date strings for the full requested range.
    index_bands : list of str
        Band names present in the collection (e.g. ``["MNDWI", "NDVI"]``).
        Used to construct the fallback masked image for empty periods.

    Returns
    -------
    ee.ImageCollection
        One image per period, ``system:time_start`` set to the period midpoint.
        For ``"all"``, the input collection is returned unchanged.
    """
    if temporal_aggregation == "all":
        # Ensure system:time_start is a proper property xee can read
        collection = collection.map(
            lambda img: img.set("system:time_start", img.get("system:time_start"))
        )
        return collection

    start_dt = datetime.date.fromisoformat(start[:10])
    end_dt = datetime.date.fromisoformat(end[:10])
    start_yr = start_dt.year
    end_yr = end_dt.year
    images = []

    def _safe_composite(
        period_col: "ee.ImageCollection", timestamp: "ee.Number"
    ) -> "ee.Image":
        """Return a median composite or a masked fallback image, server-side."""
        real = period_col.median().set("system:time_start", timestamp)
        fallback = _make_nan_image(index_bands, timestamp)
        return ee.Image(ee.Algorithms.If(period_col.size().gt(0), real, fallback))

    if temporal_aggregation == "annual":
        for yr in range(start_yr, end_yr + 1):
            col = collection.filterDate(f"{yr}-01-01", f"{yr + 1}-01-01")
            ts = ee.Date.fromYMD(yr, 7, 1).millis()
            images.append(_safe_composite(col, ts))

    elif temporal_aggregation == "monthly":
        cur = start_dt.replace(day=1)
        while cur.year < end_yr or (cur.year == end_yr and cur.month <= end_dt.month):
            yr, mo = cur.year, cur.month
            nxt = (
                datetime.date(yr + 1, 1, 1) if mo == 12 else datetime.date(yr, mo + 1, 1)
            )
            col = collection.filterDate(cur.isoformat(), nxt.isoformat())
            ts = ee.Date.fromYMD(yr, mo, 15).millis()
            images.append(_safe_composite(col, ts))
            cur = nxt

    elif temporal_aggregation == "seasonal":
        for yr in range(start_yr, end_yr + 1):
            for sname, (months, tag_mo, tag_day) in _SEASONS.items():
                season_parts = []
                for mo in months:
                    mo_yr = yr - 1 if (sname == "DJF" and mo == 12) else yr
                    mo_start = datetime.date(mo_yr, mo, 1)
                    mo_end = (
                        datetime.date(mo_yr + 1, 1, 1)
                        if mo == 12
                        else datetime.date(mo_yr, mo + 1, 1)
                    )
                    season_parts.append(
                        collection.filterDate(mo_start.isoformat(), mo_end.isoformat())
                    )
                merged = season_parts[0]
                for sc in season_parts[1:]:
                    merged = merged.merge(sc)
                ts = ee.Date.fromYMD(yr, tag_mo, tag_day).millis()
                images.append(_safe_composite(merged, ts))

    if not images:
        raise RuntimeError(
            f"No composites generated for temporal_aggregation={temporal_aggregation!r}, "
            f"date range [{start}, {end}]."
        )

    return ee.ImageCollection.fromImages(images)


# ---------------------------------------------------------------------------
# Core download helpers
# ---------------------------------------------------------------------------


def _parse_aoi(aoi: "dict | str | Path") -> "ee.Geometry":
    """Convert an AOI to an ``ee.Geometry``.

    Accepts three forms:

    1. **GeoJSON dict** — plain geometry, Feature, or FeatureCollection.
    2. **Shapefile path** — ``str`` or ``pathlib.Path`` pointing to a ``.shp``
       file (or any format readable by ``geopandas``/``fiona``).  All features
       are dissolved into a single geometry (union) so the AOI always has one
       contiguous boundary.
    3. **GeoJSON file path** — a ``.geojson`` / ``.json`` file is read with
       ``geopandas`` and treated the same as a shapefile.

    Parameters
    ----------
    aoi : dict, str, or Path
        The area of interest.  If a file path, the file is read with
        ``geopandas`` (must be installed: ``pip install geopandas``).
        The CRS is reprojected to WGS 84 (EPSG:4326) if necessary.

    Returns
    -------
    ee.Geometry
        GEE geometry suitable for ``filterBounds`` and ``getDownloadURL``.

    Raises
    ------
    ImportError
        If a file path is provided but ``geopandas`` is not installed.
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the dissolved geometry is empty (e.g. empty shapefile).
    """
    from pathlib import Path as _Path

    # ── File path branch ────────────────────────────────────────────────────
    if isinstance(aoi, (str, _Path)):
        path = _Path(aoi)
        if not path.exists():
            raise FileNotFoundError(f"AOI file not found: {path}")

        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError(
                "Reading a shapefile or GeoJSON file requires geopandas.\n"
                "Install it with:  pip install geopandas"
            )

        gdf = gpd.read_file(path)

        # Reproject to WGS 84 if needed
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        # Dissolve all features into a single geometry (union)
        dissolved = gdf.dissolve()
        if dissolved.empty or dissolved.geometry.iloc[0] is None:
            raise ValueError(f"AOI file is empty or has no geometry: {path}")

        geom_json = dissolved.geometry.iloc[0].__geo_interface__
        return ee.Geometry(geom_json)

    # ── GeoJSON dict branch ─────────────────────────────────────────────────
    geom_type = aoi.get("type", "")
    if geom_type == "FeatureCollection":
        return ee.FeatureCollection(aoi).geometry()
    elif geom_type == "Feature":
        return ee.Geometry(aoi["geometry"])
    else:
        return ee.Geometry(aoi)


def _resolve_sensor(sensor: str) -> str:
    """Resolve sensor aliases and validate the sensor name."""
    sensor = _SENSOR_ALIASES.get(sensor, sensor)
    if sensor not in _ALL_VALID_SENSORS:
        raise ValueError(
            f"Unknown sensor {sensor!r}. " f"Valid options: {sorted(_ALL_VALID_SENSORS)}"
        )
    return sensor


def _build_single_sensor_collection(
    sensor: str,
    ee_geom: "ee.Geometry",
    start: str,
    end: str,
    max_cloud_cover: float,
    use_slc_off: bool,
) -> tuple["ee.ImageCollection", dict[str, str]]:
    """Build a cloud-masked, scaled collection for a single sensor.

    Returns
    -------
    collection : ee.ImageCollection
        Pre-processed collection (cloud-masked, reflectance-scaled) with
        MNDWI / NDVI / NDTI index bands added.
    bands : dict
        Band-name mapping used (needed only for documentation; _add_indices
        has already been applied).
    """
    band_family = _SENSOR_BAND_FAMILY[sensor]
    bm = _BAND_MAP[band_family]
    sf = _SCALE_FACTOR[band_family]
    cloud_prop = _CLOUD_COVER_PROP[band_family]

    # For Landsat 7 SLC-off filter
    effective_end = end
    if sensor == "Landsat7" and not use_slc_off:
        effective_end = min(end[:10], _L7_SLC_FAILURE_DATE)
        if start[:10] >= effective_end:
            raise RuntimeError(
                "Landsat 7 SLC-on data (before 2003-06-01) is not available "
                f"in the requested date range [{start}, {end}]. "
                "Set use_slc_off=True to include post-failure data."
            )
        if end[:10] > _L7_SLC_FAILURE_DATE:
            warnings.warn(
                f"Landsat 7 SLC-off images after {_L7_SLC_FAILURE_DATE} "
                "are excluded (use_slc_off=False). "
                "Only the 1999-2003 good-quality record will be used.",
                UserWarning,
                stacklevel=4,
            )

    collection = ee.ImageCollection(_COLLECTION_ID[sensor])
    collection = collection.filterBounds(ee_geom).filterDate(start, effective_end)
    if sensor not in ("MODIS_Terra", "MODIS_Aqua"):
        collection = collection.filter(ee.Filter.lt(cloud_prop, max_cloud_cover))

    if sensor == "Sentinel2":
        collection = collection.map(_mask_sentinel2_clouds)
        collection = collection.map(
            lambda img: (
                img.multiply(sf["scale"]).copyProperties(img, ["system:time_start"])
            )
        )
    elif sensor in ("MODIS_Terra", "MODIS_Aqua"):
        collection = collection.map(_mask_modis_clouds)
        collection = collection.map(
            lambda img: (
                img.multiply(sf["scale"]).copyProperties(img, ["system:time_start"])
            )
        )
    else:
        collection = collection.map(_mask_landsat_clouds)
        collection = collection.map(
            lambda img: (
                img.multiply(sf["scale"])
                .add(sf["offset"])
                .copyProperties(img, ["system:time_start"])
            )
        )

    collection = collection.map(lambda img: _add_indices(img, bm))
    return collection, bm


def _ee_image_to_dataarray(
    image: "ee.Image",
    ee_geom: "ee.Geometry",
    scale: int,
) -> "xr.DataArray":
    """Download a single-band GEE image to a numpy-backed xr.DataArray.

    Uses GEE's ``getDownloadURL`` + rasterio.  Returns a DataArray with
    dims ``(y, x)`` and CRS written if rioxarray is available.

    Raises
    ------
    RuntimeError
        Re-raises any download or rasterio error with an informative message
        so callers can catch and fill with NaN.
    """
    import os
    import tempfile
    import urllib.request

    import rasterio
    import xarray as xr

    try:
        import rioxarray  # noqa: F401

        _rio = True
    except ImportError:
        _rio = False

    try:
        url = image.getDownloadURL(
            {
                "scale": scale,
                "region": ee_geom,
                "format": "GEO_TIFF",
                "crs": "EPSG:4326",
            }
        )
    except Exception as exc:
        raise RuntimeError(f"GEE getDownloadURL failed: {exc}") from exc

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urllib.request.urlretrieve(url, tmp_path)
        with rasterio.open(tmp_path) as src:
            data = src.read(1).astype(float)
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            transform = src.transform
            crs = src.crs

        ny, nx = data.shape
        xs = [transform.c + transform.a * (j + 0.5) for j in range(nx)]
        ys = [transform.f + transform.e * (i + 0.5) for i in range(ny)]

        da = xr.DataArray(data, dims=["y", "x"], coords={"y": ys, "x": xs})
        if _rio and crs is not None:
            da = da.rio.write_crs(crs.to_epsg() or str(crs))
        return da
    except Exception as exc:
        raise RuntimeError(f"Download/read failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _build_dem_mask(
    ee_geom: "ee.Geometry",
    max_slope_deg: float | None = 5.0,
    max_tpi_m: float | None = None,
    tpi_window_px: int = 5,
    max_local_range_m: float | None = None,
    local_range_window_px: int = 5,
    max_elevation_m: float | None = None,
) -> "ee.Image":
    """Build a server-side terrain flatness mask using Copernicus GLO-30 DEM.

    Returns a binary mask image (1 = valid flat terrain, 0 = steep/high).
    Used by :func:`fetch` when ``dem_mask=True`` to suppress glacier and
    snowpack artefacts server-side before download.

    Parameters
    ----------
    ee_geom : ee.Geometry
        Area of interest for DEM loading.
    max_slope_deg : float or None
        Maximum slope in degrees.  Pixels steeper than this are masked.
        Default 5.0.
    max_tpi_m : float or None
        Maximum absolute TPI (metres).  Uses a focal mean kernel.
        Default None (disabled).
    tpi_window_px : int
        Kernel radius in pixels for TPI focal mean. Default 5.
    max_local_range_m : float or None
        Maximum local elevation range (metres). Default None (disabled).
    local_range_window_px : int
        Kernel radius for local range. Default 5.
    max_elevation_m : float or None
        Absolute elevation ceiling (metres).  Pixels above this are
        always masked. Default None (disabled).

    Returns
    -------
    ee.Image
        Single-band mask: 1 = valid terrain, 0 = artefact.
    """
    dem = (
        ee.ImageCollection("COPERNICUS/DEM/GLO30")
        .filterBounds(ee_geom)
        .select("DEM")
        .mean()
    )

    mask = ee.Image.constant(1)

    # ── Absolute elevation ceiling ──────────────────────────────────────────
    if max_elevation_m is not None:
        mask = mask.And(dem.lte(max_elevation_m))

    # ── Slope ───────────────────────────────────────────────────────────────
    if max_slope_deg is not None:
        slope = ee.Terrain.slope(dem)
        mask = mask.And(slope.lte(max_slope_deg))

    # ── TPI ─────────────────────────────────────────────────────────────────
    if max_tpi_m is not None:
        kernel    = ee.Kernel.square(tpi_window_px, "pixels")
        focal_mean = dem.reduceNeighborhood(
            reducer=ee.Reducer.mean(), kernel=kernel
        )
        tpi = dem.subtract(focal_mean).abs()
        mask = mask.And(tpi.lte(max_tpi_m))

    # ── Local elevation range ────────────────────────────────────────────────
    if max_local_range_m is not None:
        kernel    = ee.Kernel.square(local_range_window_px, "pixels")
        local_max = dem.reduceNeighborhood(ee.Reducer.max(), kernel)
        local_min = dem.reduceNeighborhood(ee.Reducer.min(), kernel)
        local_rng = local_max.subtract(local_min)
        mask = mask.And(local_rng.lte(max_local_range_m))

    return mask.rename("terrain_mask")


# ---------------------------------------------------------------------------
# Public API: fetch()
# ---------------------------------------------------------------------------


def fetch(
    aoi: "dict | str",
    start: str,
    end: str,
    sensor: str = "Landsat8",
    index: "str | Sequence[str]" = "MNDWI",
    scale: int = 30,
    max_cloud_cover: float = 20.0,
    temporal_aggregation: str = "all",
    use_slc_off: bool = False,
    project: str | None = None,
    climate_adaptive: bool = False,
    min_precip_mm: float = 20.0,
    min_temp_c: float = 5.0,
    hydroperiod_months: int = 1,
    wetness_index: str = "MNDWI",
    wetness_threshold: float = 0.0,
    dem_mask: bool = False,
    max_slope_deg: float | None = 5.0,
    max_tpi_m: float | None = None,
    tpi_window_px: int = 5,
    max_local_range_m: float | None = None,
    local_range_window_px: int = 5,
    max_elevation_m: float | None = None,
) -> "xr.DataArray | xr.Dataset":
    """Retrieve spectral indices from GEE as an xarray object (immediate download).

    Fetches an image collection, applies cloud masking and surface-reflectance
    scaling, computes MNDWI / NDVI / NDTI server-side, optionally composites
    by time period, and downloads the result.

    Parameters
    ----------
    aoi : dict, str, or Path
        Area of interest.  Accepts:

        - **GeoJSON dict** — plain geometry, Feature, or FeatureCollection.
        - **Shapefile path** — ``str`` / ``Path`` to a ``.shp`` (or any
          format readable by ``geopandas``).  Multiple features are dissolved
          into one boundary.  Requires ``pip install geopandas``.
        - **GeoJSON file path** — a ``.geojson`` / ``.json`` file.
    start : str
        Start date ISO 8601, e.g. ``"2000-01-01"``.
    end : str
        End date (inclusive) ISO 8601, e.g. ``"2023-12-31"``.
    sensor : str
        Satellite sensor.  One of:
        ``"Landsat4"``, ``"Landsat5"``, ``"Landsat7"``, ``"Landsat8"``
        (default), ``"Landsat9"``, ``"LandsatAll"``, ``"Sentinel2"``,
        ``"MODIS_Terra"``, ``"MODIS_Aqua"``, ``"MODISAll"``.
        ``"Landsat"`` is an alias for ``"Landsat8"``.
    index : str or list of str
        One or more of ``"MNDWI"``, ``"NDVI"``, ``"NDTI"``.
        Single str → DataArray; list → Dataset (both with time dim).
    scale : int
        Spatial resolution in metres.  Default 30 (Landsat native).
    max_cloud_cover : float
        Maximum cloud cover (%) per image.  Default 20.
    temporal_aggregation : {"all", "annual", "monthly", "seasonal"}
        Server-side temporal compositing.  Default ``"all"`` (every scene).
        Using ``"annual"`` or ``"monthly"`` greatly reduces download volume.
    use_slc_off : bool
        Include Landsat 7 SLC-off images (acquired after 2003-05-31)?
        Default ``False``.  Only relevant when ``sensor`` is ``"Landsat7"``
        or ``"LandsatAll"``.
    project : str, optional
        GEE cloud project ID.
    climate_adaptive : bool
        If ``True``, replace the standard temporal composite with a
        climate-adaptive annual composite guided by ERA5-Land precipitation
        and temperature. When enabled, ``temporal_aggregation`` is ignored
        (output is always one image per year). Default ``False``.

        The algorithm:

        1. Builds monthly Landsat composites internally.
        2. Joins with ERA5-Land monthly precipitation and 2m temperature.
        3. Filters months where precip >= ``min_precip_mm`` AND
           temp >= ``min_temp_c`` (excludes dry season and snow months).
        4. For each year, selects per-pixel values from the month with
           peak precipitation using ``qualityMosaic`` (captures maximum
           wetness rather than an arbitrary median).
        5. Masks pixels wet for fewer than ``hydroperiod_months`` months
           per year on average (removes transient waterlogging).

    min_precip_mm : float
        Minimum monthly precipitation (mm) to include a month in the
        composite window. Used only when ``climate_adaptive=True``.
        Default 20 mm.
    min_temp_c : float
        Minimum monthly mean 2m temperature (degrees C) to include a
        month. Excludes frozen-ground and snow months. Used only when
        ``climate_adaptive=True``. Default 5 degrees C.
    hydroperiod_months : int
        Minimum number of months per year a pixel must be wet (index
        above ``wetness_threshold``) on average across the full record
        to be retained. Pixels below this are masked as transient
        waterlogging. Used only when ``climate_adaptive=True``.
        Default 1. Increase to 2-3 for stricter wetland delineation.
    wetness_index : str
        Which index band to use as the wetness indicator for hydroperiod
        counting and qualityMosaic selection. Must be one of the bands
        in ``index``. Default ``"MNDWI"``.
    wetness_threshold : float
        Index value above which a pixel is counted as wet for the
        hydroperiod calculation. Default 0.0.
    dem_mask : bool
        If ``True``, apply a server-side terrain flatness mask using the
        Copernicus GLO-30 DEM before compositing. Masks out glaciers,
        snowpacks, and steep mountain terrain that produce false wetness
        signals. Default ``False``.
    max_slope_deg : float or None
        Maximum terrain slope (degrees) to retain when ``dem_mask=True``.
        Default 5.0. Set to ``None`` to disable slope filtering.
    max_tpi_m : float or None
        Maximum absolute TPI (metres) when ``dem_mask=True``.
        Default ``None`` (disabled).
    tpi_window_px : int
        Focal window radius in pixels for TPI. Default 5.
    max_local_range_m : float or None
        Maximum local elevation range (metres) in the rolling window
        when ``dem_mask=True``. Default ``None`` (disabled).
    local_range_window_px : int
        Window radius for local elevation range. Default 5.
    max_elevation_m : float or None
        Absolute elevation ceiling (metres). Pixels above this elevation
        are always masked when ``dem_mask=True``. Default ``None``.


    Returns
    -------
    xr.DataArray
        DataArray with dims ``(time, y, x)`` when a single index is requested.
    xr.Dataset
        Dataset with one variable per index, dims ``(time, y, x)``.

    Notes
    -----
    Time steps where no valid cloud-free pixels exist are skipped with a
    ``UserWarning`` rather than raising an error.  The returned object will have
    fewer time steps than requested periods in such cases.

    Examples
    --------
    Long-record annual MNDWI for dynamics using all available Landsat missions:

    >>> mndwi = fetch(aoi, "1984-01-01", "2023-12-31",
    ...               sensor="LandsatAll",
    ...               temporal_aggregation="annual")
    >>> dynamics = classify_dynamics(mndwi, nYear=3)

    Post-monsoon WCT composite from Landsat 5 era:

    >>> indices = fetch(aoi, "2005-10-01", "2005-12-31",
    ...                 sensor="Landsat5",
    ...                 index=["MNDWI", "NDVI", "NDTI"])

    Annual MNDWI from MODIS for regional-scale analysis:

    >>> mndwi_modis = fetch(aoi, "2000-01-01", "2023-12-31",
    ...                     sensor="MODISAll",
    ...                     temporal_aggregation="annual",
    ...                     scale=500)
    """
    _require_ee()

    if temporal_aggregation not in _VALID_AGGREGATIONS:
        raise ValueError(
            f"temporal_aggregation must be one of {_VALID_AGGREGATIONS}, "
            f"got {temporal_aggregation!r}."
        )

    try:
        ee.Number(1).getInfo()
    except Exception:
        init(project=project)

    sensor = _resolve_sensor(sensor)

    indices_list = [index] if isinstance(index, str) else list(index)
    _VALID_INDICES = {"MNDWI", "NDVI", "NDTI", "AWEIsh", "AWEInsh"}
    bad = set(indices_list) - _VALID_INDICES
    if bad:
        raise ValueError(
            f"Unknown index/indices: {bad}. Valid: {_VALID_INDICES}"
        )   

    ee_geom = _parse_aoi(aoi)

    # Build collection (merged or single sensor)
    if sensor == "LandsatAll":
        collection = _build_landsat_all(ee_geom, start, end, max_cloud_cover, use_slc_off)
    elif sensor == "MODISAll":
        collection = _build_modis_all(ee_geom, start, end, max_cloud_cover)
    elif sensor in ("MODIS_Terra", "MODIS_Aqua"):
        bm = _BAND_MAP["MODIS_500m"]
        sf = _SCALE_FACTOR["MODIS_500m"]
        raw = (
            ee.ImageCollection(_COLLECTION_ID[sensor])
            .filterBounds(ee_geom)
            .filterDate(start, end)
        )
        raw = raw.map(_mask_modis_clouds)
        raw = raw.map(
            lambda img: (
                img.multiply(sf["scale"])
                   .copyProperties(img, ["system:time_start"])
            )
        )
        collection = raw.map(lambda img: _add_indices(img, bm))
    
    else:
        collection, _ = _build_single_sensor_collection(
            sensor, ee_geom, start, end, max_cloud_cover, use_slc_off
        )

    # ── Server-side DEM terrain mask ────────────────────────────────────────
    if dem_mask:
        terrain_mask = _build_dem_mask(
            ee_geom,
            max_slope_deg=max_slope_deg,
            max_tpi_m=max_tpi_m,
            tpi_window_px=tpi_window_px,
            max_local_range_m=max_local_range_m,
            local_range_window_px=local_range_window_px,
            max_elevation_m=max_elevation_m,
        )
        collection = collection.map(
            lambda img: img.updateMask(terrain_mask)
        )
            
    # Keep only requested index bands
    collection = collection.select(indices_list)

    # Server-side temporal compositing
    if climate_adaptive:
        # Monthly composites first (needed as input to climate-adaptive fn)
        monthly = _build_composites(
            collection, "monthly", start, end, indices_list
        )
        collection = _build_climate_adaptive_composites(
            monthly,
            start=start,
            end=end,
            index_bands=indices_list,
            wetness_index=wetness_index,
            wetness_threshold=wetness_threshold,
            min_precip_mm=min_precip_mm,
            min_temp_c=min_temp_c,
            hydroperiod_months=hydroperiod_months,
        )
    else:        
    collection = _build_composites(
        collection, temporal_aggregation, start, end, indices_list
    )

    n_images = collection.size().getInfo()
    if n_images == 0:
        raise RuntimeError(
            f"No images found for sensor={sensor!r}, [{start}, {end}], "
            f"max_cloud_cover={max_cloud_cover}%, "
            f"temporal_aggregation={temporal_aggregation!r}."
        )

    import xarray as xr

    image_list = collection.toList(n_images)
    result_ds_list: list["xr.Dataset"] = []
    skipped = 0

    for i in range(n_images):
        img = ee.Image(image_list.get(i))
        ts = img.get("system:time_start").getInfo()
        dt = np.datetime64(datetime.datetime.utcfromtimestamp(ts / 1000))

        band_arrays: dict[str, "xr.DataArray"] = {}
        failed = False

        for idx in indices_list:
            try:
                da_band = _ee_image_to_dataarray(img.select(idx), ee_geom, scale)
                band_arrays[idx] = da_band
            except RuntimeError as exc:
                warnings.warn(
                    f"Skipping time step {dt} (index '{idx}'): {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                failed = True
                break

        if failed:
            skipped += 1
            continue

        ds_t = xr.Dataset({k: v.expand_dims(time=[dt]) for k, v in band_arrays.items()})
        result_ds_list.append(ds_t)

    if not result_ds_list:
        raise RuntimeError(
            f"All {n_images} time steps failed to download. "
            "Check your AOI size, date range, and cloud cover threshold."
        )

    if skipped:
        warnings.warn(
            f"{skipped} of {n_images} time step(s) were skipped due to "
            "download errors or empty composites.",
            UserWarning,
            stacklevel=2,
        )

    combined = xr.concat(result_ds_list, dim="time")

    if isinstance(index, str):
        result = combined[index]
        result.name = index
        return result
    return combined


# ---------------------------------------------------------------------------
# Public API: fetch_xee()
# ---------------------------------------------------------------------------

def fetch_xee(
    aoi: "dict | str",
    start: str,
    end: str,
    sensor: str = "Landsat8",
    index: "str | list[str]" = "MNDWI",
    scale: int = 30,
    max_cloud_cover: float = 20.0,
    temporal_aggregation: str = "all",
    use_slc_off: bool = False,
    project: str | None = None,
    chunks: dict | None = None,
) -> "xr.DataArray | xr.Dataset":
    """Retrieve spectral indices from GEE as a **lazy Dask-backed xarray** via xee.

    Unlike :func:`fetch`, no pixels are transferred until ``.compute()`` is
    called.  Practical for large AOIs and multi-decade time series.

    Parameters
    ----------
    aoi : dict, str, or Path
        Area of interest — same formats as :func:`fetch`:
        GeoJSON dict, shapefile path, or GeoJSON file path.
    start : str
        Start date ISO 8601.
    end : str
        End date ISO 8601.
    sensor : str
        See :func:`fetch` for the full list.  Default ``"Landsat8"``.
    index : str or list of str
        One or more of ``"MNDWI"``, ``"NDVI"``, ``"NDTI"``.
        Single str → DataArray; list → Dataset.
    scale : int
        Spatial resolution in metres.  Default 30.
    max_cloud_cover : float
        Maximum cloud cover (%) per image.  Default 20.
    temporal_aggregation : {"all", "annual", "monthly", "seasonal"}
        Server-side compositing.  Default ``"all"``.
        For large collections, ``"annual"`` or ``"monthly"`` is strongly
        recommended to limit the number of lazy time steps.
    use_slc_off : bool
        Include Landsat 7 SLC-off images?  Default ``False``.
    project : str, optional
        GEE cloud project ID.
    chunks : dict, optional
        Dask chunk sizes, e.g. ``{"time": 1, "lon": 512, "lat": 512}``.
        Defaults to ``{"time": 1, "lon": 512, "lat": 512}``.

    Returns
    -------
    xr.DataArray
        Lazy DataArray with dims ``(time, lat, lon)`` for a single index.
    xr.Dataset
        Lazy Dataset, dims ``(time, lat, lon)`` for multiple indices.

    Notes
    -----
    **Bounding box requirement:**
    xee grids pixels over a rectangle.  Arbitrary polygon → centroid only
    (one pixel).  This function extracts the AOI bounding box via
    ``ee_geom.bounds()`` and passes that to xee.  Server-side filtering
    (``filterBounds``) still uses the original polygon.

    **Dimension names:**
    xee uses ``"lat"`` / ``"lon"`` not ``"y"`` / ``"x"``.  Rename after
    computing if needed: ``da.rename({"lat": "y", "lon": "x"})``.

    **Empty periods:**
    Periods with no valid imagery produce fully-masked time slices (NaN)
    rather than a "band not found" error.

    Examples
    --------
    >>> mndwi_lazy = fetch_xee(
    ...     aoi, "1984-01-01", "2023-12-31",
    ...     sensor="LandsatAll",
    ...     temporal_aggregation="annual",
    ...     chunks={"time": 1, "lon": 512, "lat": 512},
    ... )
    >>> mndwi_lazy.isel(time=0).compute()   # first year only
    >>> dynamics = classify_dynamics(
    ...     mndwi_lazy.rename({"lat": "y", "lon": "x"}).compute(),
    ...     nYear=3,
    ... )
    """
    _require_ee()
    try:
        import xee  # noqa: F401
    except ImportError:
        raise ImportError(
            "xee is required for fetch_xee(). "
            "Install:  pip install 'wetlandmapper[gee]'"
        )
    try:
        import dask  # noqa: F401
    except ImportError:
        raise ImportError(
            "dask is required for fetch_xee(). " 
            "Install:  pip install dask"
        )

    if temporal_aggregation not in _VALID_AGGREGATIONS:
        raise ValueError(
            f"temporal_aggregation must be one of {_VALID_AGGREGATIONS}, "
            f"got {temporal_aggregation!r}."
        )

    try:
        ee.Number(1).getInfo()
    except Exception:
        init(project=project)

    sensor = _resolve_sensor(sensor)

    indices_list = [index] if isinstance(index, str) else list(index)
    bad = set(indices_list) - {"MNDWI", "NDVI", "NDTI"}
    if bad:
        raise ValueError(f"Unknown index/indices: {bad}. Valid: MNDWI, NDVI, NDTI")

    ee_geom = _parse_aoi(aoi)

    # ---------------------------------------------------------------
    # xee requires a bounding box — polygon centroid → one pixel bug
    # ---------------------------------------------------------------
    bounds_info = ee_geom.bounds().getInfo()["coordinates"][0]
    lons = [c[0] for c in bounds_info]
    lats = [c[1] for c in bounds_info]
    ee_bbox = ee.Geometry.BBox(min(lons), min(lats), max(lons), max(lats))

    # Build collection
    if sensor == "LandsatAll":
        collection = _build_landsat_all(ee_geom, start, end, max_cloud_cover, use_slc_off)
    elif sensor == "MODISAll":
        collection = _build_modis_all(ee_geom, start, end, max_cloud_cover)
    else:
        collection, _ = _build_single_sensor_collection(
            sensor, ee_geom, start, end, max_cloud_cover, use_slc_off
        )

    collection = collection.select(indices_list)
    collection = _build_composites(
        collection, temporal_aggregation, start, end, indices_list
    )

    import xarray as xr

    # xee needs projection with embedded scale, not a bare scale kwarg
    projection = ee.Projection("EPSG:4326").atScale(scale)
    default_chunks = chunks or {"time": 1, "lon": 512, "lat": 512}

    ds_lazy = xr.open_dataset(
        collection,
        engine="ee",
        projection=projection,
        geometry=ee_bbox,
        chunks=default_chunks,
    )

    # ---------------------------------------------------------------
    # xee returns lat ascending (south → north).  Sort descending so
    # the spatial orientation matches fetch() (north → south, standard
    # raster convention).  sortby is lazy — no compute triggered.
    # ---------------------------------------------------------------
    if "lat" in ds_lazy.dims:
        ds_lazy = ds_lazy.sortby("lat", ascending=False)

    if isinstance(index, str):
        da = ds_lazy[index]
        da.name = index
        return da
    return ds_lazy[indices_list]


# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------

def _require_ee() -> None:
    if not _HAS_EE:
        raise ImportError(
            "earthengine-api is required for the GEE module.\n"
            "Install:  pip install 'wetlandmapper[gee]'\n"
            "Auth:     earthengine authenticate"
        )


# ---------------------------------------------------------------------------
# Climate-adaptive annual compositing (ERA5-Land guided)
# ---------------------------------------------------------------------------

def _build_climate_adaptive_composites(
    collection: "ee.ImageCollection",
    start: str,
    end: str,
    index_bands: list[str],
    wetness_index: str = "MNDWI",
    wetness_threshold: float = 0.0,
    min_precip_mm: float = 20.0,
    min_temp_c: float = 5.0,
    hydroperiod_months: int = 1,
) -> "ee.ImageCollection":
    """Build climate-adaptive annual composites guided by ERA5-Land.

    This function addresses two limitations of a naive annual median:

    1. **Season selection**: Instead of compositing all months equally,
       it identifies for each year the month with peak precipitation that
       also meets minimum temperature and rainfall thresholds. Using
       ``qualityMosaic`` on precipitation selects, per pixel, the index
       value from the wettest climatically-valid month. This avoids
       selecting snow-covered or drought-period images as representative
       of annual wetness.

    2. **Hydroperiod filtering**: Transient waterlogging (e.g. flooded
       fields after a storm) produces a water signal for only one or two
       months per year. A true wetland is inundated for a sustained period.
       Pixels that are wet during fewer than ``hydroperiod_months`` months
       per year on average across the full record are masked out.

    Parameters
    ----------
    collection : ee.ImageCollection
        Pre-processed monthly composite collection with index bands
        (e.g. MNDWI, NDVI, NDTI) already computed server-side.
        Should be a monthly composite (``temporal_aggregation="monthly"``
        applied upstream).
    start, end : str
        ISO 8601 date strings for the full requested range.
    index_bands : list of str
        Band names to composite (e.g. ``["MNDWI"]``).
    wetness_index : str
        Which band to use as the wetness indicator for both the
        ``qualityMosaic`` quality band and the hydroperiod count.
        Default ``"MNDWI"``. Can also be ``"AWEIsh"`` or ``"AWEInsh"``
        if those bands were computed server-side.
    wetness_threshold : float
        Index value above which a pixel is considered wet for the
        hydroperiod count. Default 0.0 (standard MNDWI water threshold).
    min_precip_mm : float
        Minimum monthly total precipitation (mm) for a month to be
        included in the composite window. Months drier than this are
        skipped (dry season filter). Default 20 mm.
    min_temp_c : float
        Minimum monthly mean 2m air temperature (degrees C) for a month
        to be included. Months colder than this are skipped (snow/ice
        filter). Default 5 degrees C. ERA5-Land temperature is in Kelvin
        internally; this parameter is in Celsius for user convenience.
    hydroperiod_months : int
        Minimum number of months per year (on average across the full
        record) that a pixel must be wet to be retained as a wetland
        pixel. Pixels below this threshold are masked — they represent
        transient waterlogging rather than persistent wetland. Default 1.
        Increase to 2 or 3 for stricter wetland delineation.

    Returns
    -------
    ee.ImageCollection
        One image per year, ``system:time_start`` set to July 1 of each
        year. Pixels failing the hydroperiod test are masked in all images.

    Notes
    -----
    ERA5-Land data
        - Collection: ``ECMWF/ERA5_LAND/MONTHLY_AGGR``
        - ``total_precipitation_sum``: monthly total precipitation in
          metres (multiplied by 1000 to convert to mm).
        - ``temperature_2m``: monthly mean 2m air temperature in Kelvin
          (273.15 subtracted to convert to degrees C).
        - Available from 1950-01-01 to near-present at 0.1 degree
          (~11 km) resolution; GEE resamples to the Landsat grid.

    Snow exclusion
        ERA5-Land ``total_precipitation_sum`` includes both rainfall and
        snowfall. The ``min_temp_c`` filter effectively excludes months
        where precipitation falls primarily as snow, without requiring a
        separate snowfall band.

    Hydroperiod vs wet_percent_threshold
        The ``hydroperiod_months`` parameter operates at monthly
        resolution and is applied during data acquisition. The
        ``thresholdWet`` parameter in :func:`classify_dynamics` operates
        at annual-composite resolution and is applied during
        classification. Both can be used together for a two-stage filter.

    Examples
    --------
    Dryland application with strict hydroperiod:

    >>> mndwi = fetch(
    ...     aoi, "1984-01-01", "2023-12-31",
    ...     sensor="LandsatAll",
    ...     climate_adaptive=True,
    ...     min_precip_mm=25.0,   # wet season only
    ...     min_temp_c=10.0,      # exclude cold months
    ...     hydroperiod_months=2, # at least 2 wet months/year
    ... )

    Temperate wetland (less strict):

    >>> mndwi = fetch(
    ...     aoi, "2000-01-01", "2023-12-31",
    ...     sensor="LandsatAll",
    ...     climate_adaptive=True,
    ...     min_precip_mm=10.0,
    ...     min_temp_c=2.0,
    ...     hydroperiod_months=1,
    ... )
    """
    start_yr = datetime.date.fromisoformat(start[:10]).year
    end_yr   = datetime.date.fromisoformat(end[:10]).year

    # ── ERA5-Land monthly climate data ──────────────────────────────────────
    # Convert units server-side:
    #   precipitation: m -> mm  (*1000)
    #   temperature:   K -> C   (-273.15)
    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(start, end)
        .select(
            ["total_precipitation_sum", "temperature_2m"],
            ["precip_m",               "temp_k"],
        )
    )

    # Add year, month, and precip_mm / temp_c as image properties + bands
    def _prepare_era5(img):
        date    = img.date()
        yr      = date.get("year")
        mo      = date.get("month")
        precip  = img.select("precip_m").multiply(1000)   # m -> mm
        temp    = img.select("temp_k").subtract(273.15)   # K -> C
        return (
            img
            .addBands(precip.rename("precip_mm"))
            .addBands(temp.rename("temp_c"))
            .set("year", yr)
            .set("month", mo)
            .set("ym", ee.String(yr).cat("_").cat(
                ee.Number(mo).format("%02d")))
        )

    era5 = era5.map(_prepare_era5)

    # ── Add year/month labels to the Landsat monthly composites ─────────────
    def _label_landsat(img):
        date = img.date()
        yr   = date.get("year")
        mo   = date.get("month")
        return img.set(
            "year",  yr,
            "month", mo,
            "ym",    ee.String(yr).cat("_").cat(
                ee.Number(mo).format("%02d")),
        )

    collection = collection.map(_label_landsat)

    # ── Inner join on year-month string ─────────────────────────────────────
    join_filter  = ee.Filter.equals(leftField="ym", rightField="ym")
    inner_join   = ee.Join.inner("landsat", "era5")
    joined       = inner_join.apply(collection, era5, join_filter)

    # Merge each pair into one image carrying both Landsat indices and
    # climate bands, then apply climate filters
    def _merge_pair(feature):
        ls  = ee.Image(feature.get("landsat"))
        clm = ee.Image(feature.get("era5"))
        return (
            ee.Image.cat(ls, clm.select(["precip_mm", "temp_c"]))
            .copyProperties(ls, ls.propertyNames())
        )

    joined_col = ee.ImageCollection(joined.map(_merge_pair))

    # Filter to climate-valid months: warm enough AND wet enough
    climate_valid = joined_col.filter(
        ee.Filter.And(
            ee.Filter.gte("precip_mm", min_precip_mm),  # server-side property?
            # Property-level filter won't work for raster values;
            # use a pixel-level mask instead (applied per image below)
        )
    )

    # Apply pixel-level climate mask (ERA5 at ~11 km resamples to Landsat grid)
    def _apply_climate_mask(img):
        precip_ok = img.select("precip_mm").gte(min_precip_mm)
        temp_ok   = img.select("temp_c").gte(min_temp_c)
        valid     = precip_ok.And(temp_ok)
        return img.updateMask(valid).copyProperties(img, img.propertyNames())

    climate_valid = joined_col.map(_apply_climate_mask)

    # ── Hydroperiod mask ─────────────────────────────────────────────────────
    # For each year, count how many climate-valid months each pixel is wet.
    # Average across years. Mask pixels below hydroperiod_months.
    years = ee.List.sequence(start_yr, end_yr)

    def _wet_months_in_year(yr):
        yr_col  = climate_valid.filter(ee.Filter.eq("year", yr))
        wet_col = yr_col.map(
            lambda img: img.select(wetness_index)
                           .gt(wetness_threshold)
                           .rename("wet")
                           .unmask(0)
        )
        return wet_col.sum().rename("wet_months")

    wet_per_year  = ee.ImageCollection(years.map(_wet_months_in_year))
    mean_wet_mths = wet_per_year.mean()
    hydro_mask    = mean_wet_mths.gte(hydroperiod_months)

    # ── Per-year best-month composite via qualityMosaic on precipitation ────
    # qualityMosaic picks, per pixel, the values from the image with the
    # highest value of the quality band (here: precip_mm). This selects the
    # index values from the wettest climate-valid month of each year.
    def _annual_composite(yr):
        yr_col = climate_valid.filter(ee.Filter.eq("year", yr))
        # qualityMosaic on precip_mm: highest precipitation month wins
        composite = yr_col.qualityMosaic("precip_mm")
        # Keep only the requested index bands
        composite = (
            composite.select(index_bands)
                     .updateMask(hydro_mask)        # apply hydroperiod mask
                     .set("system:time_start",
                          ee.Date.fromYMD(yr, 7, 1).millis())
                     .set("year", yr)
        )
        fallback = _make_nan_image(index_bands, ee.Date.fromYMD(yr, 7, 1).millis())
        return ee.Image(
            ee.Algorithms.If(yr_col.size().gt(0), composite, fallback)
        )

    images = years.map(_annual_composite)
    return ee.ImageCollection(images)
