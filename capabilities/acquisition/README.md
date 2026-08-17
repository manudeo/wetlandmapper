# Data Acquisition Capabilities

WetlandMapper supports user-provided xarray stacks and direct acquisition from
Google Earth Engine (GEE).

## Local data path

- Use xarray-compatible rasters/cubes from local or cloud files.
- Compute indices and classify directly.

## GEE path

- `fetch`: eager retrieval for small to medium AOIs.
- `fetch_xee`: lazy xarray loading for larger AOIs and long records.
- Supports Landsat 4-9, LandsatAll, Sentinel-2, MODIS Terra/Aqua, MODISAll.

## Typical output flow

1. Acquire composites or time-series.
2. Compute indices.
3. Run dynamics or WCT classification.
4. Apply analysis utilities for trend and reporting.
