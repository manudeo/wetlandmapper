# WetlandMapper

[![CI](https://github.com/manudeo-singh/wetlandmapper/actions/workflows/ci.yml/badge.svg)](https://github.com/manudeo-singh/wetlandmapper/actions)
[![codecov](https://codecov.io/gh/manudeo-singh/wetlandmapper/branch/main/graph/badge.svg)](https://codecov.io/gh/manudeo-singh/wetlandmapper)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/1179181199.svg)](https://doi.org/10.5281/zenodo.18967176)

**Automatic wetland detection, temporal dynamics classification, and Wetland Cover Type characterisation from multispectral satellite time-series data.**

`WetlandMapper` is a Python library that operationalises two peer-reviewed remote-sensing frameworks:

| Module | Method | Output |
|--------|--------|--------|
| `classify_dynamics` | MNDWI time-series aggregation ([Singh & Sinha 2022, *RSL*](https://doi.org/10.1080/2150704X.2021.1980919)) | 6 temporal dynamics classes |
| `classify_wct` / `classify_wct_ema` | MNDWI + NDVI + NDTI combination ([Singh et al. 2022, *EMA*](https://doi.org/10.1007/s10661-022-10541-7)) | 5 biophysical cover types |

Both methods work on any multispectral archive (Landsat 4–9, Sentinel-2, etc.) and require **no labelled training data**. Data can be supplied by the user or fetched directly from **Google Earth Engine** using any Landsat mission or Sentinel-2.

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

---

## Installation

**Minimal (core algorithms only):**
```bash
pip install wetlandmapper
```

**With plotting support:**
```bash
pip install "wetlandmapper[plot]"
```

**With Google Earth Engine support** (includes `earthengine-api`, `rasterio`, `xee`, `dask`, `geopandas`):
```bash
pip install "wetlandmapper[gee]"
earthengine authenticate  # one-time setup
```

**Full development install:**
```bash
git clone https://github.com/manudeo-singh/wetlandmapper
cd wetlandmapper
pip install -e ".[all]"
```

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
)
wct = classify_wct_ema(indices)
wct.rio.to_raster("wetland_cover_types.tif")
```

### Retrieve data from Google Earth Engine

```python
from wetlandmapper.gee import fetch
from wetlandmapper import classify_dynamics, classify_wct_ema

# AOI accepts a shapefile path, GeoJSON file path, or GeoJSON dict
aoi = "study_area/chilika.shp"           # shapefile
# aoi = "study_area/chilika.geojson"     # GeoJSON file
# aoi = {"type": "Polygon", ...}         # GeoJSON dict

# Long-record annual composites — merges all available Landsat missions
mndwi = fetch(
    aoi, start="1984-01-01", end="2023-12-31",
    sensor="LandsatAll",
    temporal_aggregation="annual",
    use_slc_off=False,     # exclude Landsat 7 post-SLC-failure images
)
dynamics = classify_dynamics(mndwi, nYear=3, thresholdWet=25, thresholdPersis=75)

# Single-sensor post-monsoon WCT composite
indices = fetch(
    aoi, start="2022-10-01", end="2022-12-31",
    sensor="Landsat8", index=["MNDWI", "NDVI", "NDTI"],
)
wct = classify_wct_ema(indices.isel(time=0))
```

### Temporal aggregation

```python
from wetlandmapper import aggregate_time

mndwi_annual   = aggregate_time(mndwi_ts, freq="annual",   method="median")
mndwi_seasonal = aggregate_time(mndwi_ts, freq="seasonal", method="median")
mndwi_monthly  = aggregate_time(mndwi_ts, freq="monthly",  method="median")
```

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
| `"LandsatAll"` | all 5 merged | 1982–present | auto-harmonised band names |
| `"Sentinel2"` | S2_SR_HARMONIZED | 2015–present | MSI |

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

**Software (JOSS paper — pending publication):**
> Singh, M. (2026). WetlandMapper: A Python package for automatic wetland mapping, dynamics classification, and cover-type characterisation. *Journal of Open Source Software*. https://doi.org/10.5281/zenodo.XXXXXXX

---

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).

## Contact

**Manudeo Singh**  
Department of Geography and Earth Science, Aberystwyth University, Aberystwyth, Wales, UK  
[manudeo.singh@aber.ac.uk](mailto:manudeo.singh@aber.ac.uk) · ORCID: [0000-0002-3511-8362](https://orcid.org/0000-0002-3511-8362)
