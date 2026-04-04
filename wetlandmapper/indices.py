"""
indices.py
----------
Spectral index computation from multispectral xarray DataArrays.

Supported indices
-----------------
- MNDWI  Modified Normalised Difference Water Index  (Xu 2006)
         = (Green - SWIR) / (Green + SWIR)
- NDWI   Normalised Difference Water Index  (McFeeters 1996)
         = (Green - NIR) / (Green + NIR)
- NDVI   Normalised Difference Vegetation Index
         = (NIR - Red)   / (NIR + Red)
- NDTI   Normalised Difference Turbidity Index
         = (Red - Green) / (Red + Green)
- AWEIsh Automated Water Extraction Index Shadow   (Feyisa et al. 2014)
         = Blue + 2.5*Green - 1.5*(NIR + SWIR) - 0.25*SWIR2
- AWEInsh Automated Water Extraction Index No Shadow (Feyisa et al. 2014)
         = Blue + 2.5*Green - 1.5*(NIR + SWIR)
Functions
---------
- compute_mndwi()      : Single MNDWI computation
- compute_ndwi()       : Single NDWI computation
- compute_ndvi()       : Single NDVI computation
- compute_ndti()       : Single NDTI computation
- compute_aweish()     : Single AWEIsh computation
- compute_aweinsh()    : Single AWEInsh computation
- compute_indices()    : MNDWI, NDVI, NDTI (+ optionally AWEIsh, AWEInsh for WCT)
- compute_water_indices() : All water indices (MNDWI, NDWI, AWEIsh, AWEInsh)"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "compute_mndwi",
    "compute_ndwi",
    "compute_ndvi",
    "compute_ndti",
    "compute_aweish",
    "compute_aweinsh",
    "compute_indices",
    "compute_water_indices",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_band(ds: xr.Dataset | xr.DataArray, band: str) -> xr.DataArray:
    """Return a single band from a Dataset or DataArray.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        If a Dataset, ``band`` should be a variable name.
        If a DataArray with a ``band`` coordinate, ``band`` should be one of
        the coordinate labels.
    band : str
        Band name or coordinate label.
    """
    if isinstance(ds, xr.Dataset):
        if band not in ds:
            raise KeyError(
                f"Band '{band}' not found in Dataset. "
                f"Available variables: {list(ds.data_vars)}"
            )
        return ds[band].astype(float)

    if isinstance(ds, xr.DataArray):
        if "band" in ds.coords and band in ds.band.values:
            return ds.sel(band=band).astype(float)
        raise KeyError(
            f"Band '{band}' not found. DataArray bands: {ds.band.values.tolist()}"
        )

    raise TypeError(f"Expected xr.Dataset or xr.DataArray, got {type(ds)}")


def _normalised_difference(a: xr.DataArray, b: xr.DataArray) -> xr.DataArray:
    """(a - b) / (a + b) with safe zero-division → NaN."""
    denom = a + b
    return xr.where(denom != 0, (a - b) / denom, np.nan)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compute_mndwi(
    ds: xr.Dataset | xr.DataArray,
    green_band: str = "green",
    swir_band: str = "swir",
) -> xr.DataArray:
    """Compute the Modified Normalised Difference Water Index (MNDWI).

    MNDWI = (Green - SWIR) / (Green + SWIR)

    Values range from -1 to +1. Positive values indicate surface water.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Multi-spectral dataset containing at least a green and a SWIR band.
    green_band : str
        Name of the green band (default: ``"green"``).
        Common alternatives: ``"B3"`` (Landsat 8/9), ``"B03"`` (Sentinel-2).
    swir_band : str
        Name of the SWIR band (default: ``"swir"``).
        Common alternatives: ``"B6"`` (Landsat 8/9 SWIR1), ``"B11"`` (Sentinel-2).

    Returns
    -------
    xr.DataArray
        MNDWI values in the range [-1, 1], preserving input dimensions.

    References
    ----------
    Xu, H. (2006). Modification of normalised difference water index (NDWI) to
    enhance open water features in remotely sensed imagery.
    International Journal of Remote Sensing, 27(14), 3025–3033.
    https://doi.org/10.1080/01431160600589179
    """
    green = _get_band(ds, green_band)
    swir = _get_band(ds, swir_band)
    mndwi = _normalised_difference(green, swir)
    mndwi.name = "MNDWI"
    mndwi.attrs.update(
        long_name="Modified Normalised Difference Water Index",
        valid_min=-1.0,
        valid_max=1.0,
        references="Xu (2006) Int. J. Remote Sens. 27(14):3025-3033",
    )
    return mndwi


def compute_ndwi(
    ds: xr.Dataset | xr.DataArray,
    green_band: str = "green",
    nir_band: str = "nir",
) -> xr.DataArray:
    """Compute the Normalised Difference Water Index (NDWI).

    NDWI = (Green - NIR) / (Green + NIR)

    Values range from -1 to +1. Positive values indicate surface water.
    Less sensitive to built-up land and vegetation than MNDWI.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Multi-spectral dataset containing at least a green and a NIR band.
    green_band : str
        Name of the green band (default: ``"green"``).
        Common alternatives: ``"B3"`` (Landsat 8/9), ``"B03"`` (Sentinel-2).
    nir_band : str
        Name of the NIR band (default: ``"nir"``).
        Common alternatives: ``"B5"`` (Landsat 8/9), ``"B08"`` (Sentinel-2).

    Returns
    -------
    xr.DataArray
        NDWI values in the range [-1, 1], preserving input dimensions.

    References
    ----------
    McFeeters, S. K. (1996). The use of the Normalized Difference Water Index
    (NDWI) in the delineation of open water features.
    International Journal of Remote Sensing, 17(7), 1425–1432.
    https://doi.org/10.1080/01431169608948714
    """
    green = _get_band(ds, green_band)
    nir = _get_band(ds, nir_band)
    ndwi = _normalised_difference(green, nir)
    ndwi.name = "NDWI"
    ndwi.attrs.update(
        long_name="Normalised Difference Water Index",
        valid_min=-1.0,
        valid_max=1.0,
        references="McFeeters (1996) Int. J. Remote Sens. 17(7):1425-1432",
    )
    return ndwi


def compute_ndvi(
    ds: xr.Dataset | xr.DataArray,
    nir_band: str = "nir",
    red_band: str = "red",
) -> xr.DataArray:
    """Compute the Normalised Difference Vegetation Index (NDVI).

    NDVI = (NIR - Red) / (NIR + Red)

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
    nir_band : str
        Name of the NIR band (default: ``"nir"``).
        Common alternatives: ``"B5"`` (Landsat 8/9), ``"B08"`` (Sentinel-2).
    red_band : str
        Name of the red band (default: ``"red"``).
        Common alternatives: ``"B4"`` (Landsat 8/9), ``"B04"`` (Sentinel-2).

    Returns
    -------
    xr.DataArray
        NDVI values in the range [-1, 1].
    """
    nir = _get_band(ds, nir_band)
    red = _get_band(ds, red_band)
    ndvi = _normalised_difference(nir, red)
    ndvi.name = "NDVI"
    ndvi.attrs.update(
        long_name="Normalised Difference Vegetation Index",
        valid_min=-1.0,
        valid_max=1.0,
    )
    return ndvi


def compute_ndti(
    ds: xr.Dataset | xr.DataArray,
    red_band: str = "red",
    green_band: str = "green",
) -> xr.DataArray:
    """Compute the Normalised Difference Turbidity Index (NDTI).

    NDTI = (Red - Green) / (Red + Green)

    Higher values indicate more turbid (suspended-sediment-rich) water.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
    red_band : str
        Name of the red band (default: ``"red"``).
    green_band : str
        Name of the green band (default: ``"green"``).

    Returns
    -------
    xr.DataArray
        NDTI values in the range [-1, 1].
    """
    red = _get_band(ds, red_band)
    green = _get_band(ds, green_band)
    ndti = _normalised_difference(red, green)
    ndti.name = "NDTI"
    ndti.attrs.update(
        long_name="Normalised Difference Turbidity Index",
        valid_min=-1.0,
        valid_max=1.0,
    )
    return ndti


def compute_aweish(
    ds: xr.Dataset | xr.DataArray,
    blue_band: str = "blue",
    green_band: str = "green",
    nir_band: str = "nir",
    swir_band: str = "swir",
    swir2_band: str = "swir2",
) -> xr.DataArray:
    """Compute the Automated Water Extraction Index with shadow suppression.

    AWEIsh = Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2

    Designed to suppress topographic shadow and built-up surface confusion
    that affects simpler water indices such as MNDWI. Positive values
    indicate surface water. Preferred over MNDWI in mountainous terrain
    or urban areas.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Multi-spectral dataset. Input bands should be surface reflectance
        in the range [0, 1] (not raw DN values).
    blue_band : str
        Default ``"blue"``. Landsat 8/9: ``"SR_B2"``;
        Sentinel-2: ``"B2"``.
    green_band : str
        Default ``"green"``.
    nir_band : str
        Default ``"nir"``.
    swir_band : str
        SWIR1. Default ``"swir"``.
    swir2_band : str
        SWIR2. Default ``"swir2"``. Landsat 8/9: ``"SR_B7"``;
        Sentinel-2: ``"B12"``.

    Returns
    -------
    xr.DataArray
        AWEIsh values. A threshold of 0.0 separates water (positive)
        from non-water (negative).

    Notes
    -----
    The original Feyisa et al. (2014) formula was defined for raw Landsat
    DN values and includes a 0.0001 scaling factor. This implementation
    operates on surface reflectance already scaled to [0, 1], so the
    constant is omitted.
    """
    blue  = _get_band(ds, blue_band)
    green = _get_band(ds, green_band)
    nir   = _get_band(ds, nir_band)
    swir  = _get_band(ds, swir_band)
    swir2 = _get_band(ds, swir2_band)
    aweish = blue + 2.5 * green - 1.5 * (nir + swir) - 0.25 * swir2
    aweish.name = "AWEIsh"
    aweish.attrs.update(
        long_name="Automated Water Extraction Index (with shadow suppression)",
        water_threshold=0.0,
    )
    return aweish


def compute_aweinsh(
    ds: xr.Dataset | xr.DataArray,
    green_band: str = "green",
    nir_band: str = "nir",
    swir_band: str = "swir",
) -> xr.DataArray:
    """Compute the Automated Water Extraction Index without shadow suppression.

    AWEInsh = 4*(Green - SWIR1) - (0.25*NIR + 2.75*SWIR1)

    A simpler form that does not require the blue band. Less effective at
    suppressing shadow than AWEIsh, but suitable when the blue band is
    unavailable, noisy, or affected by atmospheric haze.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Multi-spectral dataset. Input bands should be surface reflectance
        in the range [0, 1].
    green_band : str
        Default ``"green"``.
    nir_band : str
        Default ``"nir"``.
    swir_band : str
        SWIR1. Default ``"swir"``.

    Returns
    -------
    xr.DataArray
        AWEInsh values. A threshold of 0.0 separates water from non-water.
    """
    green = _get_band(ds, green_band)
    nir   = _get_band(ds, nir_band)
    swir  = _get_band(ds, swir_band)
    aweinsh = 4.0 * (green - swir) - (0.25 * nir + 2.75 * swir)
    aweinsh.name = "AWEInsh"
    aweinsh.attrs.update(
        long_name="Automated Water Extraction Index (no shadow suppression)",
        water_threshold=0.0,
    )
    return aweinsh


def compute_indices(
    ds: xr.Dataset | xr.DataArray,
    green_band: str = "green",
    red_band: str = "red",
    nir_band: str = "nir",
    swir_band: str = "swir",
    swir2_band: str = "swir2",
    blue_band: str = "blue",
    include_awei: bool = False,
) -> xr.Dataset:
    """Compute spectral indices and return as a Dataset.

    Always computes MNDWI, NDVI, and NDTI (required for WCT classification).
    Optionally adds AWEIsh and AWEInsh.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Multi-spectral image (single-date or seasonal composite).
    green_band, red_band, nir_band, swir_band : str
        Band names within ``ds``. Defaults match the wetlandmapper
        harmonised naming convention.
    swir2_band : str
        SWIR2 band name. Required only when ``include_awei=True``.
        Default ``"swir2"``.
    blue_band : str
        Blue band name. Required only for AWEIsh when
        ``include_awei=True``. Default ``"blue"``.
    include_awei : bool
        If ``True``, also compute AWEIsh and AWEInsh and include them
        in the returned Dataset. Default ``False`` for backward
        compatibility.

    Returns
    -------
    xr.Dataset
        Dataset with variables ``MNDWI``, ``NDVI``, ``NDTI`` (always),
        and optionally ``AWEIsh``, ``AWEInsh``.
    """
    mndwi = compute_mndwi(ds, green_band=green_band, swir_band=swir_band)
    ndvi  = compute_ndvi(ds, nir_band=nir_band, red_band=red_band)
    ndti  = compute_ndti(ds, red_band=red_band, green_band=green_band)
    result = {"MNDWI": mndwi, "NDVI": ndvi, "NDTI": ndti}

    if include_awei:
        result["AWEIsh"] = compute_aweish(
            ds,
            blue_band=blue_band,
            green_band=green_band,
            nir_band=nir_band,
            swir_band=swir_band,
            swir2_band=swir2_band,
        )
        result["AWEInsh"] = compute_aweinsh(
            ds,
            green_band=green_band,
            nir_band=nir_band,
            swir_band=swir_band,
        )

    return xr.Dataset(result)


def compute_water_indices(
    ds: xr.Dataset | xr.DataArray,
    green_band: str = "green",
    red_band: str = "red",
    nir_band: str = "nir",
    swir_band: str = "swir",
    swir2_band: str = "swir2",
    blue_band: str = "blue",
) -> xr.Dataset:
    """Compute all available water indices and return as a Dataset.

    Computes MNDWI, NDWI, AWEIsh, and AWEInsh for comprehensive water detection.
    Useful for comparing different water indices or selecting the best
    performing one for a specific study area.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Multi-spectral image (single-date or time series).
    green_band, red_band, nir_band, swir_band : str
        Band names within ``ds``.
    swir2_band : str
        SWIR2 band name. Default ``"swir2"``.
    blue_band : str
        Blue band name. Default ``"blue"``.

    Returns
    -------
    xr.Dataset
        Dataset with variables ``MNDWI``, ``NDWI``, ``AWEIsh``, ``AWEInsh``.

    Examples
    --------
    >>> water_indices = compute_water_indices(
    ...     ds,
    ...     blue_band="B2", green_band="B3", red_band="B4",
    ...     nir_band="B5", swir_band="B6", swir2_band="B7",  # Landsat 8/9
    ... )
    >>> water_indices
    <xarray.Dataset>
    Dimensions:  (time: ..., y: ..., x: ...)
    Data variables:
        MNDWI    ...
        NDWI     ...
        AWEIsh   ...
        AWEInsh  ...
    """
    mndwi = compute_mndwi(ds, green_band=green_band, swir_band=swir_band)
    ndwi = compute_ndwi(ds, green_band=green_band, nir_band=nir_band)
    aweish = compute_aweish(
        ds,
        blue_band=blue_band,
        green_band=green_band,
        nir_band=nir_band,
        swir_band=swir_band,
        swir2_band=swir2_band,
    )
    aweinsh = compute_aweinsh(
        ds,
        green_band=green_band,
        nir_band=nir_band,
        swir_band=swir_band,
    )
    return xr.Dataset({"MNDWI": mndwi, "NDWI": ndwi, "AWEIsh": aweish, "AWEInsh": aweinsh})
