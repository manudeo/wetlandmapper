# Installation Guide

## Requirements

- Python 3.9 or later
- A C compiler and GDAL system libraries (required by `rasterio`; see platform notes below)

## Quick install (most users)

```bash
pip install wetlandmapper
```

This installs the core package with `numpy`, `xarray`, and `rioxarray`.

To include the visualisation utilities:

```bash
pip install "wetlandmapper[plot]"
```

To include Google Earth Engine support (includes `earthengine-api`, `rasterio`, `xee`,
`dask`, and `geopandas` for shapefile/GeoJSON AOI support):

```bash
pip install "wetlandmapper[gee]"
```

To install everything (core + plot + gee):

```bash
pip install "wetlandmapper[all]"
```

---

## Google Earth Engine setup

After installing the `[gee]` extras, authenticate once per machine:

```bash
earthengine authenticate
```

This opens a browser to complete OAuth authentication. Credentials are cached
locally and do not need to be repeated.

For accounts created after 2023, you must also provide a GEE cloud project ID:

```python
from wetlandmapper.gee import init
init(project="your-gee-project-id")
```

Or pass `project=` directly to `fetch()` / `fetch_xee()`.

---

## Using shapefiles as AOI

The `[gee]` extras include `geopandas`, which enables you to pass a shapefile path
or GeoJSON file path directly as the `aoi` argument:

```python
from wetlandmapper.gee import fetch

# Shapefile (single or multi-feature; multi-feature polygons are dissolved)
mndwi = fetch("study_area/my_wetland.shp", "2010-01-01", "2023-12-31")

# GeoJSON file
mndwi = fetch("study_area/my_wetland.geojson", "2010-01-01", "2023-12-31")

# Plain GeoJSON dict (no geopandas needed)
mndwi = fetch({"type": "Polygon", "coordinates": [...]}, "2010-01-01", "2023-12-31")
```

If you only need shapefile support without the full GEE stack:

```bash
pip install geopandas
```

---

## Platform-specific notes

### Linux (Ubuntu / Debian)

Install GDAL system libraries before `pip install`:

```bash
sudo apt-get install gdal-bin libgdal-dev python3-gdal
```

### macOS

Using [Homebrew](https://brew.sh):

```bash
brew install gdal
pip install wetlandmapper
```

### Windows

The simplest approach is to use the pre-built wheels from the
[Unofficial Windows Binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/) page,
or install via conda:

```bash
conda install -c conda-forge wetlandmapper
```

---

## Development install

```bash
git clone https://github.com/manudeo-singh/wetlandmapper
cd wetlandmapper
pip install -e ".[all]"
```

This installs the package in editable mode with all optional dependencies.

---

## Verifying the installation

```python
import wetlandmapper
print(wetlandmapper.__version__)

# Quick smoke test
from wetlandmapper import classify_dynamics, classify_wct_ema
import numpy as np, xarray as xr

t = xr.DataArray(np.random.uniform(-1, 1, (10, 5, 5)),
                 dims=["time", "y", "x"])
d = classify_dynamics(t, nYear=2)
print("Dynamics OK:", d.shape)
```

---

## Contact

**Manudeo Singh**  
Department of Geography and Earth Science, Aberystwyth University, Aberystwyth, Wales, UK  
[manudeo.singh@aber.ac.uk](mailto:manudeo.singh@aber.ac.uk)
