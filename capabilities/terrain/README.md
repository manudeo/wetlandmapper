# Terrain and Conditioning Capabilities

WetlandMapper includes DEM-driven helpers to improve wetland mapping in
complex terrain.

## Functions

- `compute_slope`: slope in degrees from DEM.
- `compute_tpi`: topographic position index.
- `compute_local_range`: local elevation range in moving windows.
- `mask_terrain_artifacts`: mask wetness artefacts by terrain thresholds.
- `map_dem_depressions`: derive closed depression masks from raw and filled DEM.

## Why this matters

Terrain filters reduce false positives in steep or noisy landscapes, while
closed-depression products support hydro-geomorphic interpretation.
