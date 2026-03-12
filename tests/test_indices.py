"""Tests for wetlandmapper.indices"""

import numpy as np
import pytest
import xarray as xr

from wetlandmapper import compute_mndwi, compute_ndvi, compute_ndti, compute_indices


class TestComputeMNDWI:
    def test_positive_values_for_water(self, multispectral_ds):
        """Open-water pixels (high green, low swir) should yield positive MNDWI."""
        mndwi = compute_mndwi(multispectral_ds, green_band="green", swir_band="swir")
        # Zone 1 (rows 0-4) is open water → MNDWI should be positive
        assert (mndwi.isel(y=slice(0, 5)) > 0).all()

    def test_negative_values_for_dry(self, multispectral_ds):
        """Non-wetland pixels (low green, high swir) should yield negative MNDWI."""
        mndwi = compute_mndwi(multispectral_ds, green_band="green", swir_band="swir")
        assert (mndwi.isel(y=slice(15, None)) < 0).all()

    def test_output_range(self, multispectral_ds):
        mndwi = compute_mndwi(multispectral_ds, green_band="green", swir_band="swir")
        assert float(mndwi.max()) <= 1.0
        assert float(mndwi.min()) >= -1.0

    def test_output_name(self, multispectral_ds):
        mndwi = compute_mndwi(multispectral_ds)
        assert mndwi.name == "MNDWI"

    def test_missing_band_raises(self, multispectral_ds):
        with pytest.raises(KeyError):
            compute_mndwi(multispectral_ds, green_band="B3_nonexistent", swir_band="swir")

    def test_zero_denominator_returns_nan(self):
        """When green == swir == 0, denominator (green+swir) is zero → NaN."""
        ds = xr.Dataset({
            "green": xr.DataArray([[0.0]], dims=["y", "x"]),
            "swir":  xr.DataArray([[0.0]], dims=["y", "x"]),
        })
        mndwi = compute_mndwi(ds)
        assert np.isnan(mndwi.values).all()

    def test_dataarray_input_with_band_coord(self):
        """compute_mndwi should accept a DataArray with a 'band' coordinate."""
        data = np.array([[[0.15]], [[0.04]]])   # green, swir
        da = xr.DataArray(
            data,
            dims=["band", "y", "x"],
            coords={"band": ["green", "swir"]},
        )
        mndwi = compute_mndwi(da, green_band="green", swir_band="swir")
        assert (mndwi.values > 0).all()


class TestComputeNDVI:
    def test_positive_for_vegetation(self, multispectral_ds):
        """Vegetated pixels (high NIR, low red) should have positive NDVI."""
        ndvi = compute_ndvi(multispectral_ds, nir_band="nir", red_band="red")
        # Zone 3 (rows 10-14) has high NIR
        assert (ndvi.isel(y=slice(10, 15)) > 0.3).all()

    def test_output_range(self, multispectral_ds):
        ndvi = compute_ndvi(multispectral_ds)
        assert float(ndvi.max()) <= 1.0
        assert float(ndvi.min()) >= -1.0

    def test_output_name(self, multispectral_ds):
        ndvi = compute_ndvi(multispectral_ds)
        assert ndvi.name == "NDVI"


class TestComputeNDTI:
    def test_positive_for_turbid_water(self, multispectral_ds):
        """Turbid water (high red relative to green) should give positive NDTI."""
        ndti = compute_ndti(multispectral_ds, red_band="red", green_band="green")
        # Zone 2 (rows 5-9) has high red
        assert (ndti.isel(y=slice(5, 10)) > 0).all()

    def test_output_name(self, multispectral_ds):
        ndti = compute_ndti(multispectral_ds)
        assert ndti.name == "NDTI"


class TestComputeIndices:
    def test_returns_dataset(self, multispectral_ds):
        idx = compute_indices(multispectral_ds)
        assert isinstance(idx, xr.Dataset)

    def test_has_all_three_variables(self, multispectral_ds):
        idx = compute_indices(multispectral_ds)
        assert "MNDWI" in idx
        assert "NDVI"  in idx
        assert "NDTI"  in idx

    def test_spatial_dims_preserved(self, multispectral_ds):
        idx = compute_indices(multispectral_ds)
        assert idx["MNDWI"].dims == ("y", "x")

