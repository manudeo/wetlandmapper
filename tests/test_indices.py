"""Tests for wetlandmapper.indices"""

import numpy as np
import pytest
import xarray as xr

from wetlandmapper import (
    compute_aweinsh,
    compute_aweish,
    compute_indices,
    compute_mndwi,
    compute_ndti,
    compute_ndvi,
    compute_ndwi,
    compute_water_indices,
)


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
        ds = xr.Dataset(
            {
                "green": xr.DataArray([[0.0]], dims=["y", "x"]),
                "swir": xr.DataArray([[0.0]], dims=["y", "x"]),
            }
        )
        mndwi = compute_mndwi(ds)
        assert np.isnan(mndwi.values).all()

    def test_dataarray_input_with_band_coord(self):
        """compute_mndwi should accept a DataArray with a 'band' coordinate."""
        data = np.array([[[0.15]], [[0.04]]])  # green, swir
        da = xr.DataArray(
            data,
            dims=["band", "y", "x"],
            coords={"band": ["green", "swir"]},
        )
        mndwi = compute_mndwi(da, green_band="green", swir_band="swir")
        assert (mndwi.values > 0).all()


class TestComputeNDWI:
    def test_positive_values_for_water(self, multispectral_ds):
        """Open-water pixels (high green, low nir) should yield positive NDWI."""
        ndwi = compute_ndwi(multispectral_ds, green_band="green", nir_band="nir")
        # Zone 1 (rows 0-4) is open water → NDWI should be positive
        assert (ndwi.isel(y=slice(0, 5)) > 0).all()

    def test_negative_values_for_vegetation(self, multispectral_ds):
        """Vegetated pixels (low green, high nir) should yield negative NDWI."""
        ndwi = compute_ndwi(multispectral_ds, green_band="green", nir_band="nir")
        # Zone 3 (rows 10-14) has high NIR, low green
        assert (ndwi.isel(y=slice(10, 15)) < 0).all()

    def test_output_range(self, multispectral_ds):
        ndwi = compute_ndwi(multispectral_ds, green_band="green", nir_band="nir")
        assert float(ndwi.max()) <= 1.0
        assert float(ndwi.min()) >= -1.0

    def test_output_name(self, multispectral_ds):
        ndwi = compute_ndwi(multispectral_ds)
        assert ndwi.name == "NDWI"


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
        assert "NDVI" in idx
        assert "NDTI" in idx

    def test_spatial_dims_preserved(self, multispectral_ds):
        idx = compute_indices(multispectral_ds)
        assert idx["MNDWI"].dims == ("y", "x")


class TestComputeAWEIsh:
    """Test Automated Water Extraction Index with shadow suppression."""

    def test_output_is_dataarray(self):
        """compute_aweish should return a DataArray."""
        ds = xr.Dataset(
            {
                "blue": xr.DataArray([[0.1]], dims=["y", "x"]),
                "green": xr.DataArray([[0.15]], dims=["y", "x"]),
                "nir": xr.DataArray([[0.05]], dims=["y", "x"]),
                "swir": xr.DataArray([[0.04]], dims=["y", "x"]),
                "swir2": xr.DataArray([[0.03]], dims=["y", "x"]),
            }
        )
        result = compute_aweish(
            ds,
            blue_band="blue",
            green_band="green",
            nir_band="nir",
            swir_band="swir",
            swir2_band="swir2",
        )
        assert isinstance(result, xr.DataArray)

    def test_output_name(self):
        """AWEIsh output should be named 'AWEIsh'."""
        ds = xr.Dataset(
            {
                "blue": xr.DataArray([[0.1]], dims=["y", "x"]),
                "green": xr.DataArray([[0.15]], dims=["y", "x"]),
                "nir": xr.DataArray([[0.05]], dims=["y", "x"]),
                "swir": xr.DataArray([[0.04]], dims=["y", "x"]),
                "swir2": xr.DataArray([[0.03]], dims=["y", "x"]),
            }
        )
        result = compute_aweish(ds)
        assert result.name == "AWEIsh"

    def test_water_threshold_in_attrs(self):
        """AWEIsh should have water_threshold=0.0 in attrs."""
        ds = xr.Dataset(
            {
                "blue": xr.DataArray([[0.1]], dims=["y", "x"]),
                "green": xr.DataArray([[0.15]], dims=["y", "x"]),
                "nir": xr.DataArray([[0.05]], dims=["y", "x"]),
                "swir": xr.DataArray([[0.04]], dims=["y", "x"]),
                "swir2": xr.DataArray([[0.03]], dims=["y", "x"]),
            }
        )
        result = compute_aweish(ds)
        assert "water_threshold" in result.attrs
        assert result.attrs["water_threshold"] == 0.0

    def test_output_spatial_dims(self):
        """Output should preserve spatial dimensions (y, x)."""
        ds = xr.Dataset(
            {
                "blue": xr.DataArray([[0.1, 0.1]], dims=["y", "x"]),
                "green": xr.DataArray([[0.15, 0.15]], dims=["y", "x"]),
                "nir": xr.DataArray([[0.05, 0.05]], dims=["y", "x"]),
                "swir": xr.DataArray([[0.04, 0.04]], dims=["y", "x"]),
                "swir2": xr.DataArray([[0.03, 0.03]], dims=["y", "x"]),
            }
        )
        result = compute_aweish(ds)
        assert result.dims == ("y", "x")

    def test_missing_band_raises(self, multispectral_ds):
        """Missing band should raise KeyError."""
        with pytest.raises(KeyError):
            compute_aweish(
                multispectral_ds,
                blue_band="blue_nonexistent",
                green_band="green",
            )


class TestComputeAWEInsh:
    """Test Automated Water Extraction Index without shadow suppression."""

    def test_output_is_dataarray(self, multispectral_ds):
        """compute_aweinsh should return a DataArray."""
        result = compute_aweinsh(
            multispectral_ds,
            green_band="green",
            nir_band="nir",
            swir_band="swir",
        )
        assert isinstance(result, xr.DataArray)

    def test_output_name(self, multispectral_ds):
        """AWEInsh output should be named 'AWEInsh'."""
        result = compute_aweinsh(multispectral_ds)
        assert result.name == "AWEInsh"

    def test_water_threshold_in_attrs(self, multispectral_ds):
        """AWEInsh should have water_threshold=0.0 in attrs."""
        result = compute_aweinsh(multispectral_ds)
        assert "water_threshold" in result.attrs
        assert result.attrs["water_threshold"] == 0.0

    def test_output_spatial_dims(self, multispectral_ds):
        """Output should preserve spatial dimensions (y, x)."""
        result = compute_aweinsh(multispectral_ds)
        assert result.dims == ("y", "x")

    def test_positive_for_water(self, multispectral_ds):
        """Open water should yield positive AWEInsh."""
        aweinsh = compute_aweinsh(multispectral_ds)
        # Zone 1 (open water) should be positive
        assert (aweinsh.isel(y=slice(0, 5)) > 0).any()

    def test_negative_for_nonwater(self, multispectral_ds):
        """Non-water areas should yield negative AWEInsh."""
        aweinsh = compute_aweinsh(multispectral_ds)
        # Zone 3 (vegetation) should be negative
        assert (aweinsh.isel(y=slice(10, 15)) < 0).any()

    def test_missing_band_raises(self, multispectral_ds):
        """Missing band should raise KeyError."""
        with pytest.raises(KeyError):
            compute_aweinsh(
                multispectral_ds,
                green_band="green_nonexistent",
                nir_band="nir",
            )


class TestComputeWaterIndices:
    """Test compute_water_indices combining all water indices."""

    def test_returns_dataset(self):
        """compute_water_indices should return a Dataset."""
        ds = xr.Dataset(
            {
                "blue": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "green": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "red": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "nir": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "swir": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "swir2": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
            }
        )
        result = compute_water_indices(ds)
        assert isinstance(result, xr.Dataset)

    def test_has_all_water_indices(self):
        """Result should have MNDWI, NDWI, AWEIsh, AWEInsh variables."""
        ds = xr.Dataset(
            {
                "blue": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "green": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "red": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "nir": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "swir": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "swir2": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
            }
        )
        result = compute_water_indices(ds)
        assert "MNDWI" in result
        assert "NDWI" in result
        assert "AWEIsh" in result
        assert "AWEInsh" in result

    def test_spatial_dims_preserved(self):
        """All indices should preserve spatial dimensions (y, x)."""
        ds = xr.Dataset(
            {
                "blue": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "green": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "red": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "nir": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "swir": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "swir2": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
            }
        )
        result = compute_water_indices(ds)
        for var_name in ["MNDWI", "NDWI", "AWEIsh", "AWEInsh"]:
            assert result[var_name].dims == ("y", "x")


class TestComputeIndicesWithAWEI:
    """Test compute_indices with optional AWEI indices."""

    def test_include_awei_false_excludes_awei(self, multispectral_ds):
        """With include_awei=False, result should not have AWEI bands."""
        result = compute_indices(multispectral_ds, include_awei=False)
        assert "MNDWI" in result
        assert "NDVI" in result
        assert "NDTI" in result
        assert "AWEIsh" not in result
        assert "AWEInsh" not in result

    def test_include_awei_true_includes_awei(self):
        """With include_awei=True, result should have AWEI bands."""
        ds = xr.Dataset(
            {
                "blue": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "green": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "red": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "nir": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "swir": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
                "swir2": xr.DataArray(np.random.rand(5, 5), dims=["y", "x"]),
            }
        )
        result = compute_indices(ds, include_awei=True)
        assert "MNDWI" in result
        assert "NDVI" in result
        assert "NDTI" in result
        assert "AWEIsh" in result
        assert "AWEInsh" in result
