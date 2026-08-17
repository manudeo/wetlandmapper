# End-to-End Workflows

This guide maps common objectives to concrete WetlandMapper workflows.

## Workflow A: Dynamics mapping

1. Acquire time-series (`fetch`, `fetch_xee`, or local xarray stack).
2. Compute MNDWI.
3. Run `classify_dynamics`.
4. Summarize with `summarize_dynamics` or `class_area_timeseries`.

## Workflow B: WCT mapping

1. Acquire multispectral composites.
2. Compute MNDWI, NDVI, NDTI (`compute_indices`).
3. Run `classify_wct_ema` or `classify_wct_ema_level2`.
4. Summarize with `summarize_wct` and optional transition analysis.

## Workflow C: Trend and change diagnostics

1. Fit trends using `trend_products`.
2. Detect event timing with `detect_wet_events`.
3. Compare periods with `class_transition_matrix`.
4. Aggregate by polygons using `summarize_by_polygons`.
5. Save provenance via `build_run_manifest`.
