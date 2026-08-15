# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased 1.2.0]

### Changed
- Terrain slope computation now detects coordinate units from CRS (or a documented heuristic) and supports explicit override via coord_units. This corrects projected-grid slope magnitudes.
- AWEInsh now uses SWIR2 in the final term to match Feyisa et al. (2014) in both local and server-side implementations.
- Climate-adaptive hydroperiod filtering now excludes cloud-masked months from the default denominator and excludes empty years from multi-year averaging.

### Added
- Optional min_support guard in dynamics classification to suppress low-support New/Lost assignments while preserving legacy behavior by default.
- Offline climate-adaptive decision-logic tests and a weekly scheduled workflow for live GEE smoke tests.

### Documentation
- Clarified climate-guided wording, scale limitations, and published/default parameter distinctions.

[Unreleased 1.2.0]: https://github.com/manudeo/wetlandmapper/compare/v1.1.11...HEAD
