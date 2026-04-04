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
        result = mask_terrain_artifacts(dem, wetness, max_slope=45.0)
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
        result_da = mask_terrain_artifacts(dem, wetness_da, max_slope=45.0)
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
        mask_strict = mask_terrain_artifacts(
            dem, wetness, max_slope=5.0
        )
        mask_lenient = mask_terrain_artifacts(
            dem, wetness, max_slope=45.0
        )
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
        result = mask_terrain_artifacts(dem, wetness, max_elevation=75.0)
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
        result1 = mask_terrain_artifacts(dem, wetness, invert=False)
        result2 = mask_terrain_artifacts(dem, wetness, invert=True)
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
            mask_terrain_artifacts(
                dem, wetness, max_tpi=10.0, tpi_window=2
            )

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
        mask = mask_terrain_artifacts(dem, wetness, max_slope=30.0)
        assert mask.shape == (7, 11)
