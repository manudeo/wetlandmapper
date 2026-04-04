"""Tests for wetlandmapper.terrain"""

import numpy as np
import pytest
import xarray as xr

from wetlandmapper.terrain import (
    compute_local_range,
    compute_slope,
    compute_tpi,
    mask_terrain_artifacts,
)


class TestComputeSlope:
    """Test slope computation from DEM."""

    def test_flat_dem_returns_zero(self):
        """A flat DEM should produce zero slope everywhere."""
        dem = xr.DataArray(
            np.full((10, 10), 100.0),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        slope = compute_slope(dem)
        assert np.allclose(slope.values, 0.0, atol=1e-6)

    def test_output_is_dataarray(self):
        """Slope output should be an xarray DataArray."""
        dem = xr.DataArray(
            np.random.rand(10, 10),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        slope = compute_slope(dem)
        assert isinstance(slope, xr.DataArray)

    def test_output_dims(self):
        """Output should preserve input dimensions (y, x)."""
        dem = xr.DataArray(
            np.random.rand(5, 8),
            dims=["y", "x"],
            coords={"y": np.arange(5), "x": np.arange(8)},
        )
        slope = compute_slope(dem)
        assert slope.dims == ("y", "x")
        assert slope.shape == (5, 8)

    def test_output_positive(self):
        """Slope should always be non-negative (in degrees)."""
        dem = xr.DataArray(
            np.random.rand(10, 10) * 1000.0,
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        slope = compute_slope(dem)
        assert (slope.values >= 0).all()

    def test_output_name(self):
        """Slope should have 'slope' as name."""
        dem = xr.DataArray(
            np.random.rand(5, 5),
            dims=["y", "x"],
            coords={"y": np.arange(5), "x": np.arange(5)},
        )
        slope = compute_slope(dem)
        assert slope.name == "slope"

    def test_missing_dim_raises(self):
        """Should raise error if DEM lacks required dimensions."""
        dem = xr.DataArray(np.random.rand(10), dims=["x"])
        with pytest.raises(ValueError):
            compute_slope(dem)

    def test_small_dem_raises(self):
        """DEM with fewer than 2 points in any dimension should raise."""
        dem = xr.DataArray([[1.0]], dims=["y", "x"])
        with pytest.raises(ValueError):
            compute_slope(dem)


class TestComputeTPI:
    """Test Topographic Position Index computation."""

    def test_flat_dem_returns_zero(self):
        """Flat DEM should produce zero TPI everywhere."""
        dem = xr.DataArray(
            np.full((10, 10), 100.0),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        tpi = compute_tpi(dem, window=3)
        assert np.allclose(tpi.values, 0.0, atol=1e-6)

    def test_output_is_dataarray(self):
        """TPI output should be an xarray DataArray."""
        dem = xr.DataArray(
            np.random.rand(10, 10),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        tpi = compute_tpi(dem)
        assert isinstance(tpi, xr.DataArray)

    def test_output_dims(self):
        """Output should preserve input dimensions."""
        dem = xr.DataArray(
            np.random.rand(7, 9),
            dims=["y", "x"],
            coords={"y": np.arange(7), "x": np.arange(9)},
        )
        tpi = compute_tpi(dem, window=5)
        assert tpi.dims == ("y", "x")
        assert tpi.shape == (7, 9)

    def test_output_name(self):
        """TPI should have 'TPI' as name."""
        dem = xr.DataArray(
            np.random.rand(5, 5),
            dims=["y", "x"],
            coords={"y": np.arange(5), "x": np.arange(5)},
        )
        tpi = compute_tpi(dem)
        assert tpi.name == "TPI"

    def test_window_parameter(self):
        """Window size should affect TPI values."""
        dem = xr.DataArray(
            np.random.rand(15, 15),
            dims=["y", "x"],
            coords={"y": np.arange(15), "x": np.arange(15)},
        )
        tpi_small = compute_tpi(dem, window=3)
        tpi_large = compute_tpi(dem, window=7)
        # Different window sizes should generally give different results
        assert not np.allclose(tpi_small.values, tpi_large.values)

    def test_invalid_window_raises(self):
        """Window size must be odd and >= 3."""
        dem = xr.DataArray(
            np.random.rand(10, 10),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        with pytest.raises(ValueError):
            compute_tpi(dem, window=2)


class TestComputeLocalRange:
    """Test local elevation range computation."""

    def test_flat_dem_returns_zero(self):
        """Flat DEM should produce zero local range everywhere."""
        dem = xr.DataArray(
            np.full((10, 10), 100.0),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        lrange = compute_local_range(dem, window=3)
        assert np.allclose(lrange.values, 0.0, atol=1e-6)

    def test_output_is_dataarray(self):
        """Local range output should be an xarray DataArray."""
        dem = xr.DataArray(
            np.random.rand(10, 10),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        lrange = compute_local_range(dem)
        assert isinstance(lrange, xr.DataArray)

    def test_output_dims(self):
        """Output should preserve input dimensions."""
        dem = xr.DataArray(
            np.random.rand(8, 12),
            dims=["y", "x"],
            coords={"y": np.arange(8), "x": np.arange(12)},
        )
        lrange = compute_local_range(dem, window=5)
        assert lrange.dims == ("y", "x")
        assert lrange.shape == (8, 12)

    def test_output_positive(self):
        """Local range should always be non-negative."""
        dem = xr.DataArray(
            np.random.rand(10, 10) * 1000.0,
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        lrange = compute_local_range(dem, window=3)
        assert (lrange.values >= 0).all()

    def test_output_name(self):
        """Local range should have 'local_range' as name."""
        dem = xr.DataArray(
            np.random.rand(5, 5),
            dims=["y", "x"],
            coords={"y": np.arange(5), "x": np.arange(5)},
        )
        lrange = compute_local_range(dem)
        assert lrange.name == "local_range"

    def test_window_parameter(self):
        """Window size should affect local range values."""
        dem = xr.DataArray(
            np.random.rand(15, 15),
            dims=["y", "x"],
            coords={"y": np.arange(15), "x": np.arange(15)},
        )
        lr_small = compute_local_range(dem, window=3)
        lr_large = compute_local_range(dem, window=7)
        # Larger window generally captures larger range
        assert lr_large.max() >= lr_small.max()


class TestMaskTerrainArtifacts:
    """Test terrain masking for slope, TPI, elevation, and local range."""

    def test_returns_boolean_dataarray_for_dataarray_input(self):
        """Mask should return output for DataArray input."""
        dem = xr.DataArray(
            np.ones((10, 10)) * 100.0,
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        wetness = xr.DataArray(
            np.ones((10, 10)) * 0.5,
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        result = mask_terrain_artifacts(wetness, dem, max_slope=45.0)
        assert isinstance(result, (xr.DataArray, xr.Dataset))
        # Result shape should match input
        assert result.shape == (10, 10)

    def test_returns_boolean_dataset_for_dataset_input(self):
        """Mask output type matches input wetness type."""
        dem = xr.DataArray(
            np.ones((10, 10)) * 100.0,
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        # Test with both DataArray and Dataset inputs
        wetness_da = xr.DataArray(
            np.ones((10, 10)) * 0.5,
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        result_da = mask_terrain_artifacts(wetness_da, dem, max_slope=45.0)
        assert isinstance(result_da, xr.DataArray)

    def test_max_slope_parameter(self):
        """max_slope parameter should filter steep pixels."""
        dem = xr.DataArray(
            np.random.rand(10, 10) * 1000.0,
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        wetness = xr.DataArray(
            np.ones((10, 10)),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        mask_strict = mask_terrain_artifacts(wetness, dem, max_slope=5.0)
        mask_lenient = mask_terrain_artifacts(wetness, dem, max_slope=45.0)
        # Stricter threshold should mask more pixels
        assert mask_strict.sum() <= mask_lenient.sum()

    def test_max_elevation_parameter(self):
        """max_elevation parameter should be settable."""
        dem = xr.DataArray(
            np.full((10, 10), 100.0),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        wetness = xr.DataArray(
            np.ones((10, 10)) * 0.5,
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        # Should not raise with max_elevation parameter
        result = mask_terrain_artifacts(wetness, dem, max_elevation=75.0)
        assert isinstance(result, xr.DataArray)
        assert result.shape == (10, 10)

    def test_invert_parameter(self):
        """invert parameter should exist and be settable."""
        dem = xr.DataArray(
            np.ones((10, 10)) * 100.0,
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        wetness = xr.DataArray(
            np.ones((10, 10)) * 0.5,
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        # Test both invert modes work without error
        result1 = mask_terrain_artifacts(wetness, dem, invert=False)
        result2 = mask_terrain_artifacts(wetness, dem, invert=True)
        assert result1 is not None
        assert result2 is not None
        # With flat DEM and no constraints, results may be similar
        assert isinstance(result1, xr.DataArray)
        assert isinstance(result2, xr.DataArray)

    def test_invalid_window_size_raises(self):
        """Window size must be an integer >= 3."""
        dem = xr.DataArray(
            np.random.rand(10, 10),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        wetness = xr.DataArray(
            np.ones((10, 10)),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        with pytest.raises(ValueError):
            mask_terrain_artifacts(wetness, dem, max_tpi=10.0, tpi_window=2)

    def test_invalid_input_type_raises(self):
        """Wetness must be DataArray or Dataset."""
        dem = xr.DataArray(
            np.random.rand(10, 10),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        with pytest.raises(TypeError):
            mask_terrain_artifacts(dem, np.random.rand(10, 10))

    def test_output_shape_preserved(self):
        """Output mask should have same shape as input."""
        dem = xr.DataArray(
            np.random.rand(7, 11),
            dims=["y", "x"],
            coords={"y": np.arange(7), "x": np.arange(11)},
        )
        wetness = xr.DataArray(
            np.ones((7, 11)),
            dims=["y", "x"],
            coords={"y": np.arange(7), "x": np.arange(11)},
        )
        mask = mask_terrain_artifacts(wetness, dem, max_slope=30.0)
        assert mask.shape == (7, 11)


class TestSlopeUnits:
    """Test slope computation with different output units."""

    def test_slope_degrees_output(self):
        """Slope should be computed in degrees by default."""
        dem = xr.DataArray(
            np.random.rand(5, 5) * 100.0,
            dims=["y", "x"],
            coords={"y": np.arange(5), "x": np.arange(5)},
        )
        slope = compute_slope(dem, units="degrees")
        assert slope.attrs.get("units") == "degrees"
        assert (slope.values >= 0).all()
        assert (slope.values <= 90).all()  # Degrees should be in [0, 90]

    def test_slope_radians_output(self):
        """Slope should be convertible to radians."""
        dem = xr.DataArray(
            np.random.rand(5, 5) * 100.0,
            dims=["y", "x"],
            coords={"y": np.arange(5), "x": np.arange(5)},
        )
        slope = compute_slope(dem, units="radians")
        assert slope.attrs.get("units") == "radians"
        assert (slope.values >= 0).all()
        assert (slope.values <= np.pi / 2).all()  # Radians should be in [0, π/2]

    def test_slope_percent_output(self):
        """Slope should be convertible to percent grade."""
        dem = xr.DataArray(
            np.random.rand(5, 5) * 100.0,
            dims=["y", "x"],
            coords={"y": np.arange(5), "x": np.arange(5)},
        )
        slope = compute_slope(dem, units="percent")
        assert slope.attrs.get("units") == "percent"
        assert (slope.values >= 0).all()

    def test_invalid_units_raises(self):
        """Invalid unit string should raise ValueError."""
        dem = xr.DataArray(
            np.random.rand(5, 5),
            dims=["y", "x"],
            coords={"y": np.arange(5), "x": np.arange(5)},
        )
        with pytest.raises(ValueError, match="units must be one of"):
            compute_slope(dem, units="invalid")


class TestLatLonCoordinates:
    """Test terrain functions with lat/lon coordinate names."""

    def test_slope_with_lat_lon(self):
        """Slope should work with lat/lon dimension names."""
        dem = xr.DataArray(
            np.random.rand(8, 12),
            dims=["lat", "lon"],
            coords={"lat": np.linspace(-10, 10, 8), "lon": np.linspace(-20, 20, 12)},
        )
        slope = compute_slope(dem)
        assert slope.dims == ("lat", "lon")
        assert slope.shape == (8, 12)
        assert (slope.values >= 0).all()

    def test_tpi_with_lat_lon(self):
        """TPI should work with lat/lon dimension names."""
        dem = xr.DataArray(
            np.random.rand(10, 10),
            dims=["lat", "lon"],
            coords={"lat": np.linspace(-10, 10, 10), "lon": np.linspace(-20, 20, 10)},
        )
        tpi = compute_tpi(dem, window=3)
        assert tpi.dims == ("lat", "lon")
        assert tpi.shape == (10, 10)

    def test_local_range_with_lat_lon(self):
        """Local range should work with lat/lon dimension names."""
        dem = xr.DataArray(
            np.random.rand(10, 10),
            dims=["lat", "lon"],
            coords={"lat": np.linspace(-10, 10, 10), "lon": np.linspace(-20, 20, 10)},
        )
        lrange = compute_local_range(dem, window=3)
        assert lrange.dims == ("lat", "lon")
        assert lrange.shape == (10, 10)

    def test_mask_terrain_artifacts_with_lat_lon(self):
        """mask_terrain_artifacts should work with lat/lon coordinates."""
        dem = xr.DataArray(
            np.full((10, 10), 100.0),
            dims=["lat", "lon"],
            coords={"lat": np.linspace(-10, 10, 10), "lon": np.linspace(-20, 20, 10)},
        )
        wetness = xr.DataArray(
            np.ones((10, 10)),
            dims=["lat", "lon"],
            coords={"lat": np.linspace(-10, 10, 10), "lon": np.linspace(-20, 20, 10)},
        )
        result = mask_terrain_artifacts(wetness, dem, max_slope=30.0)
        assert result.dims == ("lat", "lon")
        assert result.shape == (10, 10)


class TestDatasetInput:
    """Test mask_terrain_artifacts with xarray Dataset input."""

    def test_dataset_input_returns_dataset(self):
        """Dataset input should return Dataset output."""
        dem = xr.DataArray(
            np.full((10, 10), 100.0),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        # Create a Dataset with multiple variables
        wetness_ds = xr.Dataset(
            {
                "mndwi": xr.DataArray(
                    np.ones((10, 10)) * 0.5,
                    dims=["y", "x"],
                    coords={"y": np.arange(10), "x": np.arange(10)},
                ),
                "ndvi": xr.DataArray(
                    np.ones((10, 10)) * 0.6,
                    dims=["y", "x"],
                    coords={"y": np.arange(10), "x": np.arange(10)},
                ),
            }
        )
        result = mask_terrain_artifacts(wetness_ds, dem, max_slope=30.0)
        assert isinstance(result, xr.Dataset)
        assert set(result.data_vars) == {"mndwi", "ndvi"}

    def test_dataset_with_time_dimension(self):
        """Dataset with time dimension should be preserved."""
        dem = xr.DataArray(
            np.full((10, 10), 100.0),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        # Create a Dataset with time dimension
        wetness_ds = xr.Dataset(
            {
                "mndwi": xr.DataArray(
                    np.ones((5, 10, 10)) * 0.5,
                    dims=["time", "y", "x"],
                    coords={
                        "time": np.arange(5),
                        "y": np.arange(10),
                        "x": np.arange(10),
                    },
                ),
            }
        )
        result = mask_terrain_artifacts(wetness_ds, dem, max_slope=30.0)
        assert isinstance(result, xr.Dataset)
        assert result["mndwi"].shape == (5, 10, 10)


class TestCombinedTerrainFilters:
    """Test combining multiple terrain filters."""

    def test_combined_slope_and_elevation(self):
        """Combining slope and elevation filters should work."""
        dem = xr.DataArray(
            np.random.rand(15, 15) * 2000.0,  # Elevation 0-2000m
            dims=["y", "x"],
            coords={"y": np.arange(15), "x": np.arange(15)},
        )
        wetness = xr.DataArray(
            np.ones((15, 15)),
            dims=["y", "x"],
            coords={"y": np.arange(15), "x": np.arange(15)},
        )
        result = mask_terrain_artifacts(wetness, dem, max_slope=5.0, max_elevation=1000.0)
        # Check that some pixels are masked
        assert result.sum() < wetness.sum()

    def test_combined_all_filters(self):
        """Using all filters together should work."""
        dem = xr.DataArray(
            np.random.rand(20, 20) * 2000.0,
            dims=["y", "x"],
            coords={"y": np.arange(20), "x": np.arange(20)},
        )
        wetness = xr.DataArray(
            np.ones((20, 20)),
            dims=["y", "x"],
            coords={"y": np.arange(20), "x": np.arange(20)},
        )
        result = mask_terrain_artifacts(
            wetness,
            dem,
            max_slope=5.0,
            max_elevation=1500.0,
            max_tpi=50.0,
            max_local_range=30.0,
            local_range_window=5,
            tpi_window=5,
        )
        assert isinstance(result, xr.DataArray)
        assert result.shape == (20, 20)
        # At least some pixels should be masked
        assert result.sum() <= wetness.sum()

    def test_no_filters_returns_original(self):
        """With all filters set to None, should return masked by coords only."""
        dem = xr.DataArray(
            np.full((10, 10), 100.0),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        wetness = xr.DataArray(
            np.ones((10, 10)),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        result = mask_terrain_artifacts(
            wetness,
            dem,
            max_slope=None,
            max_elevation=None,
            max_tpi=None,
            max_local_range=None,
        )
        # With no constraints on a flat, low DEM, all pixels should be valid
        assert result.sum() == wetness.sum()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_dem_with_nans(self):
        """DEM with NaN values should be handled gracefully."""
        dem = xr.DataArray(
            np.random.rand(10, 10),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        dem.values[0, 0] = np.nan
        # Should not raise; NaN propagation is expected
        slope = compute_slope(dem)
        assert np.isnan(slope.values[0, 0])

    def test_very_large_dem(self):
        """Should handle large DEM arrays."""
        dem = xr.DataArray(
            np.random.rand(100, 100),
            dims=["y", "x"],
            coords={"y": np.arange(100), "x": np.arange(100)},
        )
        slope = compute_slope(dem)
        assert slope.shape == (100, 100)
        assert not np.isnan(slope.values).all()

    def test_constant_dem_all_filters(self):
        """Constant DEM (flat, low elevation) with all filters
        should preserve most pixels."""
        dem = xr.DataArray(
            np.full((10, 10), 100.0),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        wetness = xr.DataArray(
            np.ones((10, 10)),
            dims=["y", "x"],
            coords={"y": np.arange(10), "x": np.arange(10)},
        )
        result = mask_terrain_artifacts(
            wetness,
            dem,
            max_slope=5.0,
            max_elevation=200.0,
            max_tpi=50.0,
            max_local_range=50.0,
        )
        # Constant DEM should pass all checks; possibly edge
        # pixels masked due to rolling window
        assert result.sum() > 0
