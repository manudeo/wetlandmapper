Installation
============

Conda-forge (recommended)
-------------------------

.. code-block:: bash

   conda install -c conda-forge wetlandmapper

PyPI
------------------

.. code-block:: bash

   pip install wetlandmapper

With optional dependencies:

.. code-block:: bash

   pip install "wetlandmapper[gee]"    # Google Earth Engine support
   pip install "wetlandmapper[plot]"   # matplotlib, hvplot, bokeh
   pip install "wetlandmapper[all]"    # everything

Requirements
------------

* Python >= 3.9
* numpy >= 1.24
* xarray >= 2023.1
* rioxarray >= 0.15


Live Google Earth Engine Tests (local opt-in)
----------------------------------------------

Live tests are skipped by default and only run when enabled with environment
variables. This avoids requiring Earth Engine credentials in regular CI runs.

.. code-block:: powershell

   $env:WETLANDMAPPER_RUN_GEE_LIVE='1'
   $env:WETLANDMAPPER_GEE_PROJECT='your-gee-project-id'
   $env:WETLANDMAPPER_GEE_AOI='notebooks/chilika_north/Chilika_north.shp'
   pytest tests/test_gee_live.py -m live_gee

Live smoke coverage includes:

* Sentinel-2 reducers (mean, percentile)
* LandsatAll and individual Landsat missions (4, 5, 7, 8, 9)
* MODISAll and individual MODIS missions (Terra, Aqua)
