Workflows
=========

This page summarises the primary end-to-end workflows in WetlandMapper,
including recently added Level-2 WCT support and dedicated plotting.

Wetland Dynamics Workflow
-------------------------

.. code-block:: python

   import xarray as xr
   from wetlandmapper import compute_mndwi, classify_dynamics

   ds = xr.open_dataset("landsat_timeseries.nc")
   mndwi = compute_mndwi(ds, green_band="B3", swir_band="B6")

   dynamics = classify_dynamics(
       mndwi,
       nYear=3,
       thresholdWet=25,
       thresholdPersis=75,
   )

Wetland Cover Type (EMA Level-1)
---------------------------------

.. code-block:: python

   from wetlandmapper import compute_indices
   from wetlandmapper.wct import classify_wct_ema

   indices = compute_indices(
       ds,
       green_band="B3",
       red_band="B4",
       nir_band="B5",
       swir_band="B6",
   )

   wct_l1 = classify_wct_ema(indices)

The EMA classifier returns a Dataset with:

* wetland_cover_type
* combination_code
* wvt_code

Wetland Cover Type (EMA Level-2 Extended)
-----------------------------------------

.. code-block:: python

   from wetlandmapper import compute_indices
   from wetlandmapper.wct import classify_wct_ema_level2

   indices = compute_indices(
       ds,
       green_band="B3",
       red_band="B4",
       nir_band="B5",
       swir_band="B6",
   )

   wct_l2 = classify_wct_ema_level2(indices)
   level2 = wct_l2["wetland_cover_type_level2"]

Level-2 class legend

+------+-----------------------------------+
| Code | Class                             |
+======+===================================+
| 1    | Open Clear Water (Deep)           |
+------+-----------------------------------+
| 2    | Open Clear Water (Shallow)        |
+------+-----------------------------------+
| 3    | Highly Turbid Water               |
+------+-----------------------------------+
| 4    | Moderately Turbid Water           |
+------+-----------------------------------+
| 5    | Moist / Waterlogged Soil          |
+------+-----------------------------------+
| 6    | Submerged Aquatic Vegetation      |
+------+-----------------------------------+
| 7    | Submerged-Turbid Mixed Water      |
+------+-----------------------------------+
| 8    | Emergent / Floating Vegetation    |
+------+-----------------------------------+
| 9    | Emergent-Turbid Mixed Water       |
+------+-----------------------------------+
| 10   | Moist Vegetated Fringe            |
+------+-----------------------------------+
| 11   | Saturated Sediment Fringe         |
+------+-----------------------------------+
| 12   | Vegetation-masked Water Fringe    |
+------+-----------------------------------+
| 0    | Non-wetland / Dry                 |
+------+-----------------------------------+

Dedicated Plotting Workflows
----------------------------

.. code-block:: python

   from wetlandmapper.plotting import (
       plot_dynamics,
       plot_wct,
       plot_wct_level2,
       plot_ema_codes,
   )

   fig, ax = plot_dynamics(dynamics)
   fig, ax = plot_wct(wct_l1)
   fig, ax = plot_wct_level2(wct_l2)
   fig, ax = plot_ema_codes(wct_l1["wvt_code"])

Google Earth Engine Workflows
-----------------------------

Direct download for small and medium AOIs:

.. code-block:: python

   from wetlandmapper.gee import fetch

   mndwi = fetch(
       aoi="study_area/chilika.shp",
       start="1984-01-01",
       end="2023-12-31",
       sensor="LandsatAll",
       temporal_aggregation="annual",
   )

Lazy xarray loading for large AOIs:

.. code-block:: python

   from wetlandmapper.gee import fetch_xee

   mndwi_lazy = fetch_xee(
       aoi="study_area/chilika.shp",
       start="1984-01-01",
       end="2023-12-31",
       sensor="LandsatAll",
       temporal_aggregation="annual",
   )

Supported composite sensors include LandsatAll and MODISAll.

Terrain and DEM Depression Workflow
-----------------------------------

.. code-block:: python

   import xarray as xr
   from wetlandmapper.terrain import (
       compute_slope,
       compute_tpi,
       mask_terrain_artifacts,
       map_dem_depressions,
   )

   dem = xr.open_dataset("elevation.nc")["elevation"]
   slope = compute_slope(dem)
   tpi = compute_tpi(dem, window=5)
   wetness_clean = mask_terrain_artifacts(mndwi, dem, max_slope=5)

   raw_dem = xr.open_dataset("dem_raw.nc")["elevation"].astype("int32")
   filled_dem = xr.open_dataset("dem_filled.nc")["elevation"].astype("int32")
   depression_mask = map_dem_depressions(raw_dem, filled_dem, apply_cleanup=True)

End-to-End Trend and Change Summary Workflow
--------------------------------------------

This workflow combines three utilities for analysis-ready products:

* ``trend_products`` for slope, p-value, and directional trend classes
* ``class_area_timeseries`` for annual class area accounting
* ``class_transition_matrix`` for start-end class transitions

.. code-block:: python

   import xarray as xr
   from wetlandmapper import class_area_timeseries, class_transition_matrix
   from wetlandmapper import compute_mndwi, trend_products

   ds = xr.open_dataset("landsat_timeseries.nc")
   mndwi = compute_mndwi(ds, green_band="B3", swir_band="B6")

   # 1) Pixelwise trend products from the continuous index stack
   trend = trend_products(
       mndwi,
       alpha=0.05,
       stable_epsilon=0.001,
   )

   # 2) Class time-series summary
   # classes_ts should be a class-coded DataArray with dimensions (time, y, x)
   classes_ts = xr.open_dataarray("dynamics_timeseries.nc")
   area_by_time = class_area_timeseries(
       classes_ts,
       pixel_area=900.0,
       area_unit="km2",
   )

   # 3) Start-end class transition matrix (example: first to last timestamp)
   transition = class_transition_matrix(
       classes_ts,
       start_time=classes_ts.time.values[0],
       end_time=classes_ts.time.values[-1],
       pixel_area=900.0,
       area_unit="km2",
   )

   trend.to_netcdf("trend_products.nc")
   area_by_time.to_netcdf("dynamics_area_timeseries.nc")
   transition.to_netcdf("dynamics_transition_matrix.nc")
