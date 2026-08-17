# Classification Capabilities

WetlandMapper supports two primary classification families.

## Wetland dynamics

- Input: time-series water index (commonly MNDWI).
- Output: 7-class dynamics map (non-wetland + 6 dynamics classes).
- Core API: `classify_dynamics`.

## Wetland Cover Type (WCT)

- Input: MNDWI + NDVI + NDTI combinations.
- Output: EMA-based Level-1 or Level-2 wetland cover classes.
- Core APIs: `classify_wct_ema`, `classify_wct_ema_level2`, `classify_wct`.

## Companion tools

- `summarize_dynamics`, `summarize_wct` for class summaries.
- Plotting helpers in `wetlandmapper.plotting` for publication-ready maps.
