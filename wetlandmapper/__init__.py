"""
WetlandMapper
=============
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

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("wetlandmapper")
except PackageNotFoundError:
    __version__ = "unknown"

from .indices import compute_mndwi, compute_ndvi, compute_ndti, compute_indices
from .dynamics import classify_dynamics, DYNAMICS_CLASSES, DYNAMICS_COLORS, compute_wet_frequency, aggregate_time
from .wct import classify_wct, classify_wct_ema, WCT_CLASSES, WCT_COLORS, WCT_EMA_QUARTILE_BOUNDARIES, build_ema_lookup_table

__all__ = [
    # Index computation
    "compute_mndwi",
    "compute_ndvi",
    "compute_ndti",
    "compute_indices",
    # Dynamics classification
    "classify_dynamics",
    "DYNAMICS_CLASSES",
    "DYNAMICS_COLORS",
    "compute_wet_frequency",
    "aggregate_time",
    # WCT classification
    "classify_wct_ema",
    "classify_wct",
    "WCT_CLASSES",
    "WCT_COLORS",
    "WCT_EMA_QUARTILE_BOUNDARIES",
    "build_ema_lookup_table",
]
