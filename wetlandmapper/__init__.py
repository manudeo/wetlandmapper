"""
WetlandMapper
=============
# Copyright (c) 2026, Manudeo Singh          #
# Author: Manudeo Singh, March 2026          #

Automatic wetland detection, dynamics classification, and Wetland Cover Type (WCT)
characterisation from multispectral satellite time-series data.

Two core workflows:
  1. Wetland dynamics  — MNDWI time-series → 6 temporal dynamics classes
                         (Singh & Sinha 2022, Remote Sensing Letters)
  2. Wetland Cover Types — MNDWI + NDVI + NDTI → biophysical surface classes
                         (Singh et al. 2022, Environmental Monitoring and Assessment)

Quick start
-----------
>>> from wetlandmapper import compute_mndwi, classify_dynamics
>>> from wetlandmapper import compute_indices, classify_wct

See the package documentation and demo notebook for full usage examples.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("wetlandmapper")
except PackageNotFoundError:
    __version__ = "unknown"

from .dynamics import (
    DYNAMICS_CLASSES,
    DYNAMICS_COLORS,
    aggregate_time,
    classify_dynamics,
    compute_wet_frequency,
)
from .indices import (
    compute_indices,
    compute_mndwi,
    compute_ndwi,
    compute_ndti,
    compute_ndvi,
    compute_aweish,
    compute_aweinsh,
    compute_water_indices,
)
from .terrain import (
    compute_slope,
    compute_tpi,
    compute_local_range,
    mask_terrain_artifacts,
)
from .wct import (
    WCT_CLASSES,
    WCT_COLORS,
    WCT_EMA_QUARTILE_BOUNDARIES,
    build_ema_lookup_table,
    classify_wct,
    classify_wct_ema,
)

__all__ = [
    # Index computation
    "compute_mndwi",
    "compute_ndwi",
    "compute_ndvi",
    "compute_ndti",
    "compute_aweish",
    "compute_aweinsh",
    "compute_indices",
    "compute_water_indices",
    # Dynamics classification
    "classify_dynamics",
    "DYNAMICS_CLASSES",
    "DYNAMICS_COLORS",
    "compute_wet_frequency",
    "aggregate_time",
    # Terrain masking
    "compute_slope",
    "compute_tpi",
    "compute_local_range",
    "mask_terrain_artifacts",
    # WCT classification
    "classify_wct_ema",
    "classify_wct",
    "WCT_CLASSES",
    "WCT_COLORS",
    "WCT_EMA_QUARTILE_BOUNDARIES",
    "build_ema_lookup_table",
]
