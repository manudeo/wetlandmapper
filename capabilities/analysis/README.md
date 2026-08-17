# Analysis Utilities

WetlandMapper includes analysis helpers for trend, change, zonal summary,
quality diagnostics, and reproducibility metadata.

## Key functions

- `linear_trend`: per-pixel slope, intercept, and p-value against fractional year.
- `trend_products`: significance masking and trend class layers.
- `class_summary`: class counts and percentages for one classified map.
- `class_area_timeseries`: class counts and areas through time.
- `class_transition_matrix`: class-to-class transition matrix between two dates.
- `detect_wet_events`: first/last wet year, wet count/fraction, longest streak.
- `summarize_by_polygons`: polygon-level zonal stats over rasters.
- `quality_uncertainty_summary`: support and missingness diagnostics.
- `build_run_manifest`: reproducibility JSON with parameters and input hashes.

## Typical sequence

1. Run classification or index generation.
2. Generate trend and event diagnostics.
3. Summarize area and transitions.
4. Aggregate over polygons for reporting.
5. Save a run manifest for provenance.
