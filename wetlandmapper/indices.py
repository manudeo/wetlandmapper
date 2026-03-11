"""
indices.py
----------
Spectral index computation from multispectral xarray DataArrays.

Supported indices
-----------------
- MNDWI  Modified Normalised Difference Water Index  (Xu 2006)
         = (Green - SWIR) / (Green + SWIR)
- NDVI   Normalised Difference Vegetation Index
         = (NIR - Red)   / (NIR + Red)
- NDTI   Normalised Difference Turbidity Index
         = (Red - Green) / (Red + Green)
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["compute_mndwi", "compute_ndvi", "compute_ndti", "compute_indices"]


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


def compute_indices(
    ds: xr.Dataset | xr.DataArray,
    green_band: str = "green",
    red_band: str = "red",
    nir_band: str = "nir",
    swir_band: str = "swir",
) -> xr.Dataset:
    """Compute MNDWI, NDVI, and NDTI together and return as a Dataset.

    This is the recommended entry point for the WCT classification workflow,
    which requires all three indices.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Multi-spectral image (single-date or seasonal composite).
    green_band, red_band, nir_band, swir_band : str
        Band names within ``ds``.

    Returns
    -------
    xr.Dataset
        Dataset with variables ``MNDWI``, ``NDVI``, and ``NDTI``.

    Examples
    --------
    >>> indices = compute_indices(
    ...     ds,
    ...     green_band="B3", red_band="B4",
    ...     nir_band="B5",   swir_band="B6",   # Landsat 8/9
    ... )
    >>> indices
    <xarray.Dataset>
    Dimensions:  (y: ..., x: ...)
    Data variables:
        MNDWI    ...
        NDVI     ...
        NDTI     ...
    """
    mndwi = compute_mndwi(ds, green_band=green_band, swir_band=swir_band)
    ndvi  = compute_ndvi(ds, nir_band=nir_band, red_band=red_band)
    ndti  = compute_ndti(ds, red_band=red_band, green_band=green_band)
    return xr.Dataset({"MNDWI": mndwi, "NDVI": ndvi, "NDTI": ndti})
