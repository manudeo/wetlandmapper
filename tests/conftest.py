"""
Shared test fixtures for WetlandMapper.

All fixtures create small synthetic rasters so that tests run fast and
without any external data dependencies.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
NY, NX = 20, 20          # spatial grid size
N_TIMES = 12             # number of time steps (e.g., annual scenes)
DATES = pd.date_range("2010-01-01", periods=N_TIMES, freq="YE")


# ---------------------------------------------------------------------------
# MNDWI fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mndwi_all_wet():
    """MNDWI time series where every pixel is wet at every time step."""
    data = np.full((N_TIMES, NY, NX), 0.5, dtype=float)
    return xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": DATES,
                "y": np.linspace(25.0, 25.19, NY),
                "x": np.linspace(80.0, 80.19, NX)},
        name="MNDWI",
    )


@pytest.fixture
def mndwi_all_dry():
    """MNDWI time series where every pixel is dry at every time step."""
    data = np.full((N_TIMES, NY, NX), -0.5, dtype=float)
    return xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": DATES,
                "y": np.linspace(25.0, 25.19, NY),
                "x": np.linspace(80.0, 80.19, NX)},
        name="MNDWI",
    )


@pytest.fixture
def mndwi_mixed():
    """
    MNDWI time series with four spatial quadrants of different behaviour:
      TL (top-left)     : always wet  → should be Persistent
      TR (top-right)    : wet only in recent half → should be New
      BL (bottom-left)  : wet only in historic half → should be Lost
      BR (bottom-right) : always dry → should be Non-wetland (0)
    """
    rng = np.random.default_rng(42)
    data = np.full((N_TIMES, NY, NX), -0.5)

    half = N_TIMES // 2
    hy, hx = NY // 2, NX // 2

    # TL — always wet
    data[:, :hy, :hx] = 0.6

    # TR — wet only in recent half
    data[half:, :hy, hx:] = 0.6

    # BL — wet only in historic half
    data[:half, hy:, :hx] = 0.6

    # BR — always dry (default -0.5)

    return xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={"time": DATES,
                "y": np.linspace(25.0, 25.19, NY),
                "x": np.linspace(80.0, 80.19, NX)},
        name="MNDWI",
    )


# ---------------------------------------------------------------------------
# Multi-band dataset fixture (for index computation and WCT tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def multispectral_ds():
    """
    Synthetic single-date multispectral Dataset with four spatial zones
    representing different wetland cover types:

      Zone 1 (rows 0–4):   Open clear water    — high MNDWI, low NDVI, low NDTI
      Zone 2 (rows 5–9):   Turbid water        — high MNDWI, low NDVI, high NDTI
      Zone 3 (rows 10–14): Emergent vegetation  — moderate MNDWI, high NDVI
      Zone 4 (rows 15–19): Non-wetland (dry)   — low MNDWI
    """
    shape = (NY, NX)
    green = np.full(shape, 0.05)
    red   = np.full(shape, 0.05)
    nir   = np.full(shape, 0.10)
    swir  = np.full(shape, 0.05)

    # Zone 1: open clear water — high green, low swir → positive MNDWI
    #         low red relative to green → negative NDTI (clear)
    green[0:5, :]  = 0.15
    swir[0:5, :]   = 0.04
    red[0:5, :]    = 0.04
    nir[0:5, :]    = 0.06

    # Zone 2: turbid water — high red relative to green → positive NDTI
    green[5:10, :] = 0.12
    swir[5:10, :]  = 0.04
    red[5:10, :]   = 0.15
    nir[5:10, :]   = 0.08

    # Zone 3: emergent vegetation — high NIR, moderate green
    green[10:15, :] = 0.10
    swir[10:15, :]  = 0.08
    red[10:15, :]   = 0.05
    nir[10:15, :]   = 0.45

    # Zone 4: non-wetland — low green, high swir → strongly negative MNDWI
    green[15:, :]  = 0.05
    swir[15:, :]   = 0.25
    red[15:, :]    = 0.10
    nir[15:, :]    = 0.15

    y = np.linspace(25.0, 25.19, NY)
    x = np.linspace(80.0, 80.19, NX)

    return xr.Dataset(
        {
            "green": xr.DataArray(green, dims=["y", "x"],
                                  coords={"y": y, "x": x}),
            "red":   xr.DataArray(red,   dims=["y", "x"],
                                  coords={"y": y, "x": x}),
            "nir":   xr.DataArray(nir,   dims=["y", "x"],
                                  coords={"y": y, "x": x}),
            "swir":  xr.DataArray(swir,  dims=["y", "x"],
                                  coords={"y": y, "x": x}),
        }
    )
