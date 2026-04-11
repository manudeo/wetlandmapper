# WetlandMapper

[![PyPI](https://img.shields.io/pypi/v/wetlandmapper.svg)](https://pypi.org/project/wetlandmapper/)
[![conda-forge](https://img.shields.io/conda/vn/conda-forge/wetlandmapper)](https://anaconda.org/conda-forge/wetlandmapper)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/wetlandmapper?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/wetlandmapper)
[![Documentation](https://readthedocs.org/projects/wetlandmapper/badge/?version=latest)](https://wetlandmapper.readthedocs.io/en/latest/)
[![CI](https://github.com/manudeo/wetlandmapper/actions/workflows/ci.yml/badge.svg)](https://github.com/manudeo/wetlandmapper/actions)
[![codecov](https://codecov.io/gh/manudeo/wetlandmapper/branch/main/graph/badge.svg)](https://codecov.io/gh/manudeo/wetlandmapper)
[![DOI](https://zenodo.org/badge/1179181199.svg)](https://doi.org/10.5281/zenodo.18967176)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

**A climate-adaptive automatic wetland detection, temporal dynamics classification, and Wetland Cover Type characterisation from multispectral satellite time-series data.**

`WetlandMapper` is a Python library that operationalises two peer-reviewed remote-sensing frameworks:

| Module | Method | Output |
|--------|--------|--------|
| `classify_dynamics` | Water index time-series aggregation ([Singh & Sinha 2022, *RSL*](https://doi.org/10.1080/2150704X.2021.1980919)) | 7 classes (1 non-wetland + 6 dynamics) |
| `classify_wct` / `classify_wct_ema` | MNDWI + NDVI + NDTI combination ([Singh et al. 2022, *EMA*](https://doi.org/10.1007/s10661-022-10541-7)) | 6 classes (1 non-wetland + 5 biophysical types) |

The mapping algorithm is climate- and terrian-adaptive, based on methods developed in Singh and Tooth (in prep.). 

Both methods work on any multispectral archive (Landsat 4–9, Sentinel-2, MODIS, etc.) and require **no labelled training data**. Data can be supplied by the user or fetched directly from **Google Earth Engine** using any Landsat mission, Sentinel-2, or MODIS.

## Water Indices

WetlandMapper supports multiple water detection indices for optimal performance across different environments:

- **MNDWI** (Modified NDWI): Best for most applications, uses SWIR band
- **NDWI** (Original NDWI): Alternative using NIR band, less sensitive to built-up areas  
- **AWEIsh** (Shadow-corrected): Superior in mountainous terrain with shadows; requires SWIR2 band
- **AWEInsh** (No shadow suppression): Lighter computation without SWIR2; better in areas with atmospheric haze

Use `compute_water_indices()` to compare all indices on your data.

---

## Wetland Dynamics Classes

| Code | Class | Description |
|------|-------|-------------|
| 10 | **Persistent** | Wet ≥ threshold % of all observations |
| 6 | **Intermittent** | Wet enough, but no directional trend |
| 5 | **Intensifying** | Increasing wet frequency over time |
| 4 | **Diminishing** | Decreasing wet frequency over time |
| 3 | **Lost** | Wet historically, dry in recent period |
| 2 | **New** | Dry historically, wet in recent period |
| 0 | **Non-wetland** | Below minimum wet-frequency threshold |

## Wetland Cover Types (WCTs)

| Code | Class | Key Signal |
|------|-------|------------|
| 1 | **Open Clear Water** | High MNDWI, low NDVI, low NDTI |
| 2 | **Turbid / Sediment-laden Water** | High MNDWI, low NDVI, high NDTI |
| 3 | **Submerged Aquatic Vegetation** | High MNDWI, moderate NDVI |
| 4 | **Emergent / Floating Vegetation** | Moderate MNDWI, high NDVI |
| 5 | **Moist / Waterlogged Soil** | Low–moderate MNDWI, low NDVI |
| 0 | **Non-wetland** | Below water threshold |

## Additional Spectral Indices

WetlandMapper also provides additional water indices for enhanced wetland detection:

- **AWEIsh (Automated Water Extraction Index Shadow)**: Effective for water body extraction in various environments.
- **AWEInsh (Automated Water Extraction Index No Shadow)**: Similar to AWEIsh but without shadow consideration.

## Terrain Analysis

WetlandMapper includes topographic analysis tools for enhanced wetland mapping:

- **Slope**: Computes terrain slope in degrees from elevation data. Useful for identifying wetlands in flat areas (<5°) vs. steep terrain.
- **TPI (Topographic Position Index)**: Calculates relative topographic position to distinguish plateaus from valleys. Helps identify wetland depressions.
- **Local Range**: Computes local elevation range within a moving window. Useful for detecting micro-topographic variations in wetlands.
- **DEM Depression Mapping**: Maps closed depressions from raw vs pit-filled DEM using integer division and reclassification (depression=1, non-depression=0), with optional isolated-pixel cleanup.

## Google Earth Engine Integration

WetlandMapper can fetch satellite data directly from Google Earth Engine:

- **Supported Missions**: Landsat 4-9, Sentinel-2, MODIS
- **Cloud Masking**: Automatic cloud and shadow removal
- **DEM Masking**: Optional elevation-based masking using Copernicus GLO-30 DEM (30m resolution)
- **Custom Band Mapping**: Flexible band selection for different sensors

Use `dem_mask` parameter in `fetch()` and `fetch_xee()` functions to apply elevation thresholds and exclude high-elevation areas from analysis.

---

## Installation

**Install using conda (recommended):**
```bash
conda install -c conda-forge wetlandmapper
```

**Using pip:**
```bash
pip install wetlandmapper
```

**With optional dependencies:**
```bash
pip install "wetlandmapper[gee]"    # Google Earth Engine support
pip install "wetlandmapper[plot]"   # matplotlib, hvplot, bokeh
pip install "wetlandmapper[all]"    # everything
```

**Full development install:**
```bash
git clone https://github.com/manudeo/wetlandmapper
cd wetlandmapper
pip install -e ".[all]"
```
---
## **Documentation:** 
### https://wetlandmapper.readthedocs.io/en/latest/
---

## Quick Start

### Wetland Dynamics from your own data

```python
import xarray as xr
from wetlandmapper import compute_mndwi, classify_dynamics

# Load a multi-temporal multispectral stack (any xarray-compatible format)
ds = xr.open_dataset("landsat_timeseries.nc")

# Step 1: compute MNDWI (Landsat 8/9 bands: green=B3, swir1=B6)
mndwi = compute_mndwi(ds, green_band="B3", swir_band="B6")

# Step 2: classify into dynamics classes
dynamics = classify_dynamics(
    mndwi,
    nYear=3,               # years/scenes per temporal window
    thresholdWet=25,       # minimum wet-frequency (%) to count as wetland
    thresholdPersis=75,    # wet-frequency (%) for Persistent class
)
dynamics.rio.to_raster("wetland_dynamics.tif")
```

### Wetland Cover Types from your own data

```python
from wetlandmapper import compute_indices, classify_wct_ema

indices = compute_indices(
    ds_composite,
    green_band="B3", red_band="B4",
    nir_band="B5",   swir_band="B6",  # Landsat 8/9
    include_awei=True,  # include AWEIsh and AWEInsh indices
)
wct = classify_wct_ema(indices)
wct.rio.to_raster("wetland_cover_types.tif")
```

### Retrieve data from Google Earth Engine

**Direct download (best for small-to-medium AOIs):**

```python
from wetlandmapper.gee import fetch
from wetlandmapper import classify_dynamics, classify_wct_ema

# AOI accepts a shapefile path, GeoJSON file path, or GeoJSON dict
aoi = "study_area/chilika.shp"           # shapefile
# aoi = "study_area/chilika.geojson"     # GeoJSON file
# aoi = {"type": "Polygon", ...}         # GeoJSON dict

# Long-record annual composites — LandsatAll merges all available Landsat missions (1982–present)
mndwi = fetch(
    aoi, start="1984-01-01", end="2023-12-31",
    sensor="LandsatAll",    # auto-harmonised bands across Landsat 4, 5, 7, 8, 9
    temporal_aggregation="annual",
    use_slc_off=False,      # exclude Landsat 7 post-SLC-failure images
)
dynamics = classify_dynamics(mndwi, nYear=3, thresholdWet=25, thresholdPersis=75)

# MODIS-based analysis for coarser resolution studies (500m pixels)
mndwi_modis = fetch(
    aoi, start="2000-01-01", end="2023-12-31",
    sensor="MODISAll",      # merged Terra & Aqua (2002–present)
    temporal_aggregation="annual",
)

# Single-sensor post-monsoon WCT composite
indices = fetch(
    aoi, start="2022-10-01", end="2022-12-31",
    sensor="Landsat8", index=["MNDWI", "NDVI", "NDTI"],
)
wct = classify_wct_ema(indices.isel(time=0))
```

**Lazy Dask-backed loading (best for large AOIs & long time series):**

```python
from wetlandmapper.gee import fetch_xee

# Opens GEE collection as lazy xarray without downloading until compute()
mndwi_lazy = fetch_xee(
    aoi, start="1984-01-01", end="2023-12-31",
    sensor="LandsatAll",
    temporal_aggregation="annual",
)

# Process in memory-efficient chunks
dynamics = classify_dynamics(mndwi_lazy).compute()
```

### Temporal aggregation

```python
from wetlandmapper import aggregate_time

mndwi_annual   = aggregate_time(mndwi_ts, freq="annual",   method="median")
mndwi_seasonal = aggregate_time(mndwi_ts, freq="seasonal", method="median")
mndwi_monthly  = aggregate_time(mndwi_ts, freq="monthly",  method="median")
```

### Temporal Metrics

```python
from wetlandmapper import compute_wet_frequency

# Compute the percentage of time-steps classified as wet
wet_freq = compute_wet_frequency(mndwi, threshold=0.0)  # returns 0–100 %
```

Useful for understanding wetland persistence before classification.

### Terrain Analysis

```python
from wetlandmapper.terrain import (
    compute_slope,
    compute_tpi,
    mask_terrain_artifacts,
    map_dem_depressions,
)
import xarray as xr

# Load elevation data
dem = xr.open_dataset("elevation.nc")["elevation"]

# Compute terrain derivatives
slope = compute_slope(dem)
tpi = compute_tpi(dem, window=5)

# Mask steep/high terrain in a wetness layer (e.g., glaciers, permanent snow)
wetness_clean = mask_terrain_artifacts(mndwi, dem, max_slope=5)

# Depression protocol from raw vs pit-filled DEM (best for low-relief floodplains)
raw_dem = xr.open_dataset("dem_raw.nc")["elevation"].astype("int32")
filled_dem = xr.open_dataset("dem_filled.nc")["elevation"].astype("int32")
depression_mask = map_dem_depressions(
    raw_dem,
    filled_dem,
    apply_cleanup=True,
    cleanup_window=3,
    min_neighbours=2,
)
```

This depression workflow is most reliable in flat floodplain settings and may produce spurious isolated pixels without cleanup.
It follows the DEM depression protocol described by Sinha et al. (2017, *Current Science*): http://www.jstor.org/stable/24912702

### Visualisation

```python
from wetlandmapper.plotting import plot_dynamics, plot_wct

fig, ax = plot_dynamics(dynamics)
fig.savefig("dynamics_map.png", dpi=150, bbox_inches="tight")

fig, ax = plot_wct(wct)
fig.savefig("wct_map.png", dpi=150, bbox_inches="tight")
```

---

## Google Earth Engine — Sensor Reference

All Landsat collections are Collection 2 Level-2 surface reflectance.

| `sensor=` | GEE collection | Operational dates | Note |
|-----------|---------------|-------------------|------|
| `"Landsat4"` | LT04/C02/T1_L2 | 1982–1993 | TM |
| `"Landsat5"` | LT05/C02/T1_L2 | 1984–2013 | TM |
| `"Landsat7"` | LE07/C02/T1_L2 | 1999–2022 | ETM+; SLC failure 2003-06-01 |
| `"Landsat8"` | LC08/C02/T1_L2 | 2013–present | OLI **(default)** |
| `"Landsat9"` | LC09/C02/T1_L2 | 2021–present | OLI-2 |
| `"LandsatAll"` | all 5 merged | 1982–present | auto-harmonised band names across all missions |
| `"Sentinel2"` | S2_SR_HARMONIZED | 2015–present | MSI |
| `"MODIS_Terra"` | MOD09A1 | 2000–present | 500m resolution |
| `"MODIS_Aqua"` | MYD09A1 | 2002–present | 500m resolution |
| `"MODISAll"` | Terra + Aqua merged | 2002–present | best coverage for MODIS |

`"Landsat"` is a backward-compatible alias for `"Landsat8"`.

Use `use_slc_off=False` (default) to exclude Landsat 7 post-SLC-failure images.
Use `use_slc_off=True` to include them (covers the 2003–2012 gap before Landsat 8).

---

## Band Name Reference (for local data)

| Sensor | Green | Red | NIR | SWIR1 |
|--------|-------|-----|-----|-------|
| Landsat 4/5/7 TM/ETM+ (C02 L2) | `SR_B2` | `SR_B3` | `SR_B4` | `SR_B5` |
| Landsat 8/9 OLI (C02 L2) | `SR_B3` | `SR_B4` | `SR_B5` | `SR_B6` |
| Sentinel-2 L2A | `B3` | `B4` | `B8` | `B11` |
| MODIS MOD09GA | `sur_refl_b04` | `sur_refl_b01` | `sur_refl_b02` | `sur_refl_b06` |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
pytest --cov=wetlandmapper --cov-report=term-missing
```

---

## Citing

If you use `WetlandMapper` in your research, please cite the two underlying methods and the software:

**Dynamics classification:**
> Singh, M., & Sinha, R. (2022). A basin-scale inventory and hydrodynamics of floodplain wetlands based on time-series of remote sensing data. *Remote Sensing Letters*, 13(1), 1–13. https://doi.org/10.1080/2150704X.2021.1980919

**Wetland Cover Types:**
> Singh, M., Allaka, S., Gupta, P. K., Patel, J. G., & Sinha, R. (2022). Deriving wetland-cover types (WCTs) from integration of multispectral indices based on Earth observation data. *Environmental Monitoring and Assessment*, 194(12), 878. https://doi.org/10.1007/s10661-022-10541-7

**Topographic depression protocol for riverine wetland mapping:**
> Sinha, R., Saxena, S., & **Singh, M.** (2017). Protocols for Riverine Wetland Mapping and Classification Using Remote Sensing and GIS. *Current Science*, 112(7), 1544-1552. http://www.jstor.org/stable/24912702

**Software (JOSS paper — pending publication):**
> Singh, M. (2026). WetlandMapper: A Python package for automatic wetland mapping, dynamics classification, and cover-type characterisation. *Journal of Open Source Software*.
https://doi.org/10.5281/zenodo.18967176

# Acknowledgements

The author acknowledges the ISRO–IIT Kanpur Space Technology Cell (STC) and WWF-India for funding the original research underlying these methods, and the Google Earth Engine team for providing cloud-computing access. The author is a Newton International Fellow funded by The Royal Society, London, and gratefully acknowledges this support, as well as the facilities provided by Aberystwyth University. 

The author thanks Professor Rajiv Sinha, Principal Investigator of the STC and WWF projects and co-author of the associated algorithm papers, for valuable discussions that helped shape the original methods. Parts of the code were developed during an Alexander von Humboldt Fellowship at University of Potsdam, working with Professor Bodo Bookhagen. The Alexander von Humboldt Foundation, the University of Potsdam, and Prof. Bookhagen are gratefully acknowledged for their support.


---

## License

GPL-3.0-or-later — see [LICENSE](LICENSE). The software is provided as-is with no warranties or guarantees of any kind.

## Contact

**Manudeo Singh**  
Department of Geography and Earth Sciences, Aberystwyth University, Aberystwyth, Wales, UK  
[manudeo.singh@aber.ac.uk](mailto:manudeo.singh@aber.ac.uk) · ORCID: [0000-0002-3511-8362](https://orcid.org/0000-0002-3511-8362)
