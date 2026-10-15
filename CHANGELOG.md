# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased 1.2.0]

### Changed
- Terrain slope computation now detects coordinate units from CRS (or a documented heuristic) and supports explicit override via coord_units. This corrects projected-grid slope magnitudes.
- AWEInsh now uses SWIR2 in the final term to match Feyisa et al. (2014) in both local and server-side implementations.
- Climate-adaptive hydroperiod filtering now excludes cloud-masked months from the default denominator and excludes empty years from multi-year averaging.
- Climate-adaptive hydroperiod valid-mode scaling now uses climate-valid season length instead of a fixed x12 annualization. This keeps hydroperiod_months comparable to total-mode units and avoids short-season inflation.
- WCT classifiers (`classify_wct_ema`, `classify_wct_ema_level2`, `classify_wct`) now return NaN, not class 0 (Non-wetland), where MNDWI, NDVI or NDTI is missing. Output stays int8 when no pixels are missing and becomes float otherwise.
- GEE: Copernicus DEM now uses `COPERNICUS/DEM/GLO30_2024_1` (the old collection is deprecated), and Sentinel-2 cloud masking uses the `SCL` band instead of `QA60`.
- The live GEE test workflow now runs quarterly and skips when its secrets are not configured.

### Notes
- Early 1.2.0 development commits applied x12 scaling in valid mode, which was more permissive at short-season sites. Current main scales to climate-valid months.

### Added
- Optional min_support guard in dynamics classification to suppress low-support New/Lost assignments while preserving legacy behavior by default.
- Offline climate-adaptive decision-logic tests and a weekly scheduled workflow for live GEE smoke tests.

### Documentation
- Clarified climate-guided wording, scale limitations, and published/default parameter distinctions.

[Unreleased 1.2.0]: https://github.com/manudeo/wetlandmapper/compare/v1.1.11...HEAD
