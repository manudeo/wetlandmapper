"""
sensor_harmonize.py

# Copyright (c) 2026, Manudeo Singh          #
# Author: Manudeo Singh, November 2026       #
------
Local inter-sensor calibration of Landsat spectral indices.

Landsat 5 TM, Landsat 7 ETM+ and Landsat 8/9 OLI do not return the same index
value for the same ground target (different band passbands, and LEDAPS for TM/ETM+
against LaSRC for OLI in Collection 2 surface reflectance).  The difference is
small, but it is one-sided in time: the historic part of a long record comes from
TM/ETM+ and the recent part from OLI, so an unremoved offset is read as a trend
(Diminishing or Lost wetlands when TM/ETM+ read wetter than OLI).

This module estimates the offset **in the user's own working window**, because
the offset is not constant in space (Ganga Plain test blocks, MNDWI, September to
November: Landsat 7 to Landsat 8 between -0.01 and -0.08, Landsat 5 to Landsat 7
between -0.02 and -0.07).  Landsat 8 is the reference sensor; Landsat 9 is brought
onto Landsat 8, Landsat 7 and Landsat 5 (and Landsat 4, assumed identical to
Landsat 5) onto Landsat 8 through the chain 5 -> 7 -> 8.

Method
------
For each pair of sensors, in years in which both operate (``CALIBRATION_PAIRS``),
the yearly composite of each sensor (same months, same cloud filter and same
reducer settings as the user's request) is computed server-side.  Pixels valid in
both sensors in the same year are kept; each sensor's per-pixel median over the
overlap years is formed (this averages out differences in acquisition dates between
the sensors); the offset is the median over the window of (second minus first).
A year-level bootstrap of the yearly offsets gives a 95 % interval.  Only a few
numbers leave Earth Engine.

The offset is additive on the index value.  In the Ganga Plain test blocks the
orthogonal regression slope between sensors was within 0.05 to 0.12 of 1, so a
constant offset was preferred to a slope that cannot be estimated reliably from a
few overlap years.

Landsat 7 is calibrated only in the years before its orbit drifted
(``CALIBRATION_PAIRS``: 2013 to 2016).  Landsat 7 after the start of Landsat 8 can
be left out of a series with ``landsat7_end`` in :func:`wetlandmapper.gee.fetch_xee`.

If the working window has too few valid pixels or years for a pair, the estimate
falls back to ``GLOBAL_PAIR_OFFSETS`` (median of seven Ganga Plain blocks) with a
warning, or raises, depending on ``fallback``.
"""

from __future__ import annotations

import datetime
import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CALIBRATION_PAIRS",
    "GLOBAL_PAIR_OFFSETS",
    "REFERENCE_SENSOR",
    "SensorOffsets",
    "chain_corrections",
    "estimate_sensor_offsets",
    "needed_pairs",
]

#: Sensor onto whose scale all others are brought.  Landsat 9 is treated as a
#: second OLI and is brought onto Landsat 8.
REFERENCE_SENSOR = "Landsat8"

#: (first sensor, second sensor, first year, last year) of the overlap used for
#: each pair.  Landsat 7 is used only up to 2016, before its orbit drift.
CALIBRATION_PAIRS: tuple[tuple[str, str, int, int], ...] = (
    ("Landsat5", "Landsat7", 1999, 2011),
    ("Landsat7", "Landsat8", 2013, 2016),
    ("Landsat8", "Landsat9", 2022, 2099),
)

#: Fallback offsets (second minus first sensor) used only when the working window
#: cannot support a local estimate.  MNDWI, September to November, median of seven
#: 0.3-degree blocks in the Ganga Plain (Yamuna, Ganga-Gomti, Ghaghara-Rapti,
#: Kosi-Gandak, Chambal-Ken, Kiul-Sone, Bhagwanpur).  Other indices have no
#: fallback: they are left uncorrected when a local estimate is not possible.
GLOBAL_PAIR_OFFSETS: dict[str, dict[tuple[str, str], float]] = {
    "MNDWI": {
        ("Landsat5", "Landsat7"): -0.030,
        ("Landsat7", "Landsat8"): -0.040,
        ("Landsat8", "Landsat9"): -0.006,
    }
}

_OPERATIONAL: dict[str, tuple[str, str]] = {
    "Landsat4": ("1982-08-22", "1993-12-15"),
    "Landsat5": ("1984-03-16", "2013-06-06"),
    "Landsat7": ("1999-04-15", "2022-04-07"),
    "Landsat8": ("2013-03-18", "2099-01-01"),
    "Landsat9": ("2021-10-31", "2099-01-01"),
}

_CACHE: dict[str, "SensorOffsets"] = {}


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass
class SensorOffsets:
    """Offsets of one index between Landsat sensors in one working window.

    Attributes
    ----------
    index : str
        Index the offsets belong to (for example ``"MNDWI"``).
    table : pandas.DataFrame
        One row per sensor pair: ``first``, ``second``, ``years`` (number of
        overlap years used), ``pixels`` (pixels in the comparison), ``offset``
        (median of second minus first), ``ci_low``, ``ci_high`` (95 % interval
        from a year-level bootstrap) and ``source`` (``"local"`` or ``"global"``).
    correction : dict[str, float]
        Value added to the index of each sensor to bring it onto the Landsat 8
        scale.  Landsat 8 is 0 by definition.
    """

    index: str
    table: pd.DataFrame
    correction: dict[str, float]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "reference": REFERENCE_SENSOR,
            "correction": dict(self.correction),
            "pairs": self.table.to_dict(orient="records"),
            "meta": dict(self.meta),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=float)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        c = ", ".join(f"{k} {v:+.3f}" for k, v in self.correction.items())
        return f"SensorOffsets({self.index}, reference {REFERENCE_SENSOR}: {c})"


# ---------------------------------------------------------------------------
# Pure helpers (no Earth Engine)
# ---------------------------------------------------------------------------


def chain_corrections(pair_offsets: Mapping[tuple[str, str], float]) -> dict[str, float]:
    """Convert pairwise offsets (second minus first) to additive corrections onto Landsat 8.

    Parameters
    ----------
    pair_offsets : mapping
        ``{("Landsat5", "Landsat7"): o57, ("Landsat7", "Landsat8"): o78,
        ("Landsat8", "Landsat9"): o89}``.  Pairs that are not needed may be
        missing; a sensor whose chain is incomplete is left out of the result.

    Returns
    -------
    dict
        Correction to add to each sensor's index.  Landsat 8 is 0, Landsat 9 is
        ``-o89``, Landsat 7 is ``o78``, Landsat 5 is ``o57 + o78`` and Landsat 4
        equals Landsat 5.
    """
    out: dict[str, float] = {REFERENCE_SENSOR: 0.0}
    o57 = pair_offsets.get(("Landsat5", "Landsat7"))
    o78 = pair_offsets.get(("Landsat7", "Landsat8"))
    o89 = pair_offsets.get(("Landsat8", "Landsat9"))
    if o89 is not None:
        out["Landsat9"] = -float(o89)
    if o78 is not None:
        out["Landsat7"] = float(o78)
        if o57 is not None:
            out["Landsat5"] = float(o57) + float(o78)
            out["Landsat4"] = out["Landsat5"]
    return out


def needed_pairs(missions: Iterable[str]) -> list[tuple[str, str]]:
    """Sensor pairs whose offsets are required to bring ``missions`` onto Landsat 8."""
    ms = set(missions)
    pairs: list[tuple[str, str]] = []
    if ms & {"Landsat4", "Landsat5"}:
        pairs += [("Landsat5", "Landsat7"), ("Landsat7", "Landsat8")]
    elif "Landsat7" in ms:
        pairs += [("Landsat7", "Landsat8")]
    if "Landsat9" in ms:
        pairs += [("Landsat8", "Landsat9")]
    return pairs


def missions_in_window(start: str, end: str, landsat7_end: str | None = None) -> set[str]:
    """Landsat missions that contribute scenes to ``[start, end]`` (Landsat 7 cut at ``landsat7_end``)."""
    out = set()
    for m, (a, b) in _OPERATIONAL.items():
        if m == "Landsat7" and landsat7_end:
            b = min(b, landsat7_end[:10])
        if max(start[:10], a) < min(end[:10], b):
            out.add(m)
    return out


def _bootstrap_median_ci(values: Sequence[float], n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan)
    if len(v) == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    boots = [np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def _pair_row(first: str, second: str, info: dict | None, min_years: int, min_pixels: int,
              global_value: float | None, fallback: str) -> dict[str, Any]:
    """Turn the server-side summary of one pair into a table row, applying the quality gates."""
    years_used = 0 if info is None else len(info.get("years", []))
    pixels = 0 if info is None else int(info.get("count") or 0)
    offset = None if info is None else info.get("median")
    ok = offset is not None and np.isfinite(offset) and years_used >= min_years and pixels >= min_pixels
    if ok:
        lo, hi = _bootstrap_median_ci(info["yearly"], seed=len(first) + len(second))
        return dict(first=first, second=second, years=years_used, pixels=pixels, offset=float(offset),
                    ci_low=lo, ci_high=hi, source="local")
    if fallback == "global" and global_value is not None:
        warnings.warn(
            f"Sensor offset {first}->{second}: only {years_used} overlap years and {pixels} pixels in the "
            f"working window (need {min_years} years and {min_pixels} pixels); using the global fallback "
            f"{global_value:+.3f} from seven Ganga Plain blocks. Enlarge the window or set "
            f"estimation_buffer_m to estimate locally.",
            UserWarning, stacklevel=3)
        return dict(first=first, second=second, years=years_used, pixels=pixels, offset=float(global_value),
                    ci_low=np.nan, ci_high=np.nan, source="global")
    raise RuntimeError(
        f"Cannot estimate the sensor offset {first}->{second} in this window "
        f"({years_used} overlap years, {pixels} pixels) and no global fallback is available "
        f"for this index. Use harmonize=False or pass offsets explicitly.")


# ---------------------------------------------------------------------------
# Server-side estimation
# ---------------------------------------------------------------------------


def _sensor_year_composite(ee, gee, sensor: str, year: int, geom, index: str,
                           months: Sequence[int] | None, max_cloud_cover: float):
    col, _ = gee._build_single_sensor_collection(
        sensor, geom, f"{year}-01-01", f"{year + 1}-01-01", max_cloud_cover, True)
    col = col.select([index])
    if months:
        parts = [col.filter(ee.Filter.calendarRange(m, m, "month")) for m in months]
        col = parts[0]
        for p in parts[1:]:
            col = col.merge(p)
    empty = ee.Image.constant(0).rename(index).updateMask(ee.Image.constant(0)).float()
    return ee.Image(ee.Algorithms.If(col.size().gt(0), col.median().float(), empty))


def _get_info_retry(obj, tries: int = 5, wait: float = 20.0):
    """``getInfo`` with a back-off when Earth Engine reports too many concurrent requests (HTTP 429)."""
    import time

    for k in range(tries):
        try:
            return obj.getInfo()
        except Exception as exc:  # EEException, HttpError
            msg = str(exc)
            if k < tries - 1 and ("429" in msg or "Too many" in msg or "concurrent" in msg
                                  or "Connection" in msg or "Temporary failure" in msg):
                time.sleep(wait * (k + 1))
                continue
            raise


def _estimate_pair_server(ee, gee, geom, index: str, first: str, second: str, years: Sequence[int],
                          months: Sequence[int] | None, max_cloud_cover: float, scale: float) -> dict | None:
    """Offset of one pair in the geometry, computed in Earth Engine; returns a small dict."""
    if len(years) < 1:
        return None
    a_list, b_list, d_list = [], [], []
    for y in years:
        A = _sensor_year_composite(ee, gee, first, y, geom, index, months, max_cloud_cover)
        B = _sensor_year_composite(ee, gee, second, y, geom, index, months, max_cloud_cover)
        a_y = A.updateMask(B.mask())
        b_y = B.updateMask(A.mask())
        a_list.append(a_y)
        b_list.append(b_y)
        d_list.append(b_y.subtract(a_y).rename("d"))
    ca, cb = ee.ImageCollection.fromImages(a_list), ee.ImageCollection.fromImages(b_list)
    n_years = ca.count()
    diff = cb.median().subtract(ca.median()).updateMask(n_years.gte(min(2, len(years)))).rename("pooled")
    # One multiband image (pooled offset and one band per year) -> a single reduction in Earth Engine,
    # which keeps the number of concurrent aggregations low (restricted accounts allow few).
    stack = ee.Image.cat([diff] + [d.rename(f"y{y}") for d, y in zip(d_list, years)])
    reducer = ee.Reducer.median().combine(ee.Reducer.count(), "", True)
    res = _get_info_retry(stack.reduceRegion(
        reducer=reducer, geometry=geom, scale=scale, bestEffort=True, maxPixels=int(1e9), tileScale=4))
    med = res.get("pooled_median", res.get("median"))
    cnt = res.get("pooled_count", res.get("count", 0))
    ys, yo = [], []
    for y in years:
        v = res.get(f"y{y}_median")
        if v is not None:
            ys.append(y)
            yo.append(float(v))
    return {"median": med, "count": cnt, "years": ys, "yearly": yo}


def _cache_key(aoi: Any, index: str, months: Sequence[int] | None, max_cloud_cover: float, scale: float,
               min_years: int, min_pixels: int, buffer_m: float, missions: Iterable[str]) -> str:
    try:
        a = json.dumps(aoi, sort_keys=True, default=str)
    except TypeError:
        a = str(aoi)
    return json.dumps([a, index, sorted(months) if months else None, max_cloud_cover, scale, min_years,
                       min_pixels, buffer_m, sorted(missions)], default=str)


def estimate_sensor_offsets(
    aoi: "dict | str",
    index: str = "MNDWI",
    months: Sequence[int] | None = None,
    missions: Iterable[str] | None = None,
    max_cloud_cover: float = 20.0,
    scale: float = 90.0,
    min_years: int = 3,
    min_pixels: int = 300,
    estimation_buffer_m: float = 0.0,
    fallback: str = "global",
    project: str | None = None,
    use_cache: bool = True,
) -> SensorOffsets:
    """Estimate the Landsat 5/7/8/9 offsets of an index inside a working window.

    Parameters
    ----------
    aoi : dict or str
        GeoJSON dict, shapefile path or GeoJSON file path (as for ``fetch_xee``).
        The offsets are estimated over this geometry, buffered by
        ``estimation_buffer_m``.
    index : str
        Built-in index to calibrate (``"MNDWI"``, ``"NDWI"``, ``"NDVI"``,
        ``"NDTI"``, ``"AWEIsh"``, ``"AWEInsh"``).
    months : list of int, optional
        Calendar months entering the yearly composites.  Use the same months as
        the later fetch: the offset depends on the season and on the surface
        state.
    missions : iterable of str, optional
        Sensors that the later series will contain; decides which pairs are
        needed.  Default: all of Landsat 5, 7, 8 and 9.
    max_cloud_cover : float
        Scene cloud-cover limit, as in ``fetch_xee``.
    scale : float
        Pixel size in metres at which the offset is summarised (default 90 m).
    min_years, min_pixels : int
        Quality gates for a local estimate of a pair.
    estimation_buffer_m : float
        Buffer around the geometry in metres.  A buffer lets small windows reach
        ``min_pixels``.
    fallback : {"global", "error"}
        What to do when a pair fails the quality gates: use ``GLOBAL_PAIR_OFFSETS``
        with a warning (MNDWI only), or raise.
    project : str, optional
        Earth Engine project id.
    use_cache : bool
        Reuse the result for the same window, index, months and settings
        within the Python session.

    Returns
    -------
    SensorOffsets
        Table of pair offsets (second minus first) with 95 % intervals and the
        additive ``correction`` of each sensor onto Landsat 8.
    """
    from . import gee

    if fallback not in ("global", "error"):
        raise ValueError("fallback must be 'global' or 'error'.")
    if index not in gee._VALID_INDICES:
        raise ValueError(f"Offsets can only be estimated for built-in indices {sorted(gee._VALID_INDICES)}, "
                         f"got {index!r}.")
    gee._require_ee()
    ee = gee.ee
    try:
        ee.Number(1).getInfo()
    except Exception:
        gee.init(project=project)

    ms = set(missions) if missions is not None else {"Landsat5", "Landsat7", "Landsat8", "Landsat9"}
    pairs = needed_pairs(ms)
    key = _cache_key(aoi, index, months, max_cloud_cover, scale, min_years, min_pixels,
                     estimation_buffer_m, ms)
    if use_cache and key in _CACHE:
        return _CACHE[key]

    geom = gee._parse_aoi(aoi)
    if estimation_buffer_m:
        geom = geom.buffer(estimation_buffer_m)
    this_year = datetime.date.today().year
    spans = {(a, b): (y0, min(y1, this_year)) for a, b, y0, y1 in CALIBRATION_PAIRS}

    rows, offsets = [], {}
    for first, second in pairs:
        y0, y1 = spans[(first, second)]
        info = _estimate_pair_server(ee, gee, geom, index, first, second, list(range(y0, y1 + 1)),
                                     months, max_cloud_cover, scale)
        g = GLOBAL_PAIR_OFFSETS.get(index, {}).get((first, second))
        row = _pair_row(first, second, info, min_years, min_pixels, g, fallback)
        rows.append(row)
        offsets[(first, second)] = row["offset"]

    table = pd.DataFrame(rows, columns=["first", "second", "years", "pixels", "offset", "ci_low",
                                        "ci_high", "source"])
    result = SensorOffsets(index=index, table=table, correction=chain_corrections(offsets),
                           meta=dict(months=list(months) if months else None, scale=scale,
                                     max_cloud_cover=max_cloud_cover, buffer_m=estimation_buffer_m,
                                     reference=REFERENCE_SENSOR))
    if use_cache:
        _CACHE[key] = result
    return result
