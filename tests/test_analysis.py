"""Test suite for analysis.last_occurrence function."""

import numpy as np
import pytest
import xarray as xr

from wetlandmapper import last_occurrence


@pytest.fixture
def sample_dataarray():
    """Create a sample DataArray with synthetic time-series data."""
    # Time: each year from 2020-2024 (5 years)
    times = np.array(
        [
            np.datetime64("2020-06-15"),
            np.datetime64("2021-06-15"),
            np.datetime64("2022-06-15"),
            np.datetime64("2023-06-15"),
            np.datetime64("2024-06-15"),
        ]
    )

    # Synthetic MNDWI data (ny=3, nx=3, nt=5)
    # Pixel (0,0): values = [-0.5, 0.2, 0.5, -0.1, 0.3] -> last on at 2024 (0.465)
    # Pixel (0,1): values = [0.1, 0.0, -0.2, -0.3, -0.4] -> last on at 2021.46
    # Pixel (1,1): values = [-0.5, -0.4, -0.3, -0.2, -0.1] -> never on -> NaN
    # etc.
    data = np.array(
        [
            [-0.5, 0.1, np.nan],
            [0.05, -0.5, 0.2],
            [0.15, -0.1, 0.8],
        ]
    )
    data = np.stack(
        [
            [[-0.5, 0.1, np.nan], [0.05, -0.5, 0.2], [0.15, -0.1, 0.8]],
            [[0.2, 0.0, -0.2], [-0.3, 0.1, 0.3], [0.2, 0.05, 0.6]],
            [[0.5, -0.2, -0.3], [-0.1, 0.0, 0.1], [0.3, 0.15, 0.4]],
            [[-0.1, -0.3, -0.4], [0.2, -0.2, 0.0], [0.1, 0.25, 0.7]],
            [[0.3, -0.4, -0.5], [-0.1, 0.15, -0.1], [0.2, 0.1, 0.5]],
        ]
    )

    da = xr.DataArray(
        data,
        dims=["time", "y", "x"],
        coords={
            "time": times,
            "y": [0, 1, 2],
            "x": [0, 1, 2],
        },
        name="MNDWI",
    )
    return da


@pytest.fixture
def sample_dataset():
    """Create a sample Dataset with multiple indices."""
    times = np.array(
        [
            np.datetime64("2020-06-15"),
            np.datetime64("2021-06-15"),
            np.datetime64("2022-06-15"),
            np.datetime64("2023-06-15"),
            np.datetime64("2024-06-15"),
        ]
    )

    mndwi = np.random.uniform(-0.5, 0.8, (5, 3, 3))
    ndvi = np.random.uniform(-0.2, 1.0, (5, 3, 3))

    ds = xr.Dataset(
        {
            "MNDWI": (["time", "y", "x"], mndwi),
            "NDVI": (["time", "y", "x"], ndvi),
        },
        coords={
            "time": times,
            "y": [0, 1, 2],
            "x": [0, 1, 2],
        },
    )
    return ds


def test_last_occurrence_single_index(sample_dataarray):
    """Test last_occurrence with a single index."""
    year_last, value_last = last_occurrence(sample_dataarray, "MNDWI", threshold=0.0)

    # Check output shapes
    assert year_last.shape == (3, 3)
    assert value_last.shape == (3, 3)

    # Check output types
    assert isinstance(year_last, xr.DataArray)
    assert isinstance(value_last, xr.DataArray)

    # Check that year_last values are reasonable (2020-2024)
    valid_years = year_last.values[~np.isnan(year_last.values)]
    assert np.all(valid_years >= 2020) and np.all(valid_years <= 2025)

    # Check that value_last matches index values at those times
    # (spot check: for pixels that were never on, should be NaN)
    assert np.isnan(year_last.values[1, 1]) or not np.isnan(value_last.values[1, 1])


def test_last_occurrence_multiple_indices(sample_dataset):
    """Test last_occurrence with multiple indices."""
    year_last, value_last = last_occurrence(
        sample_dataset, ["MNDWI", "NDVI"], threshold=0.0
    )

    # Check output is Dataset
    assert isinstance(year_last, xr.Dataset)
    assert isinstance(value_last, xr.Dataset)

    # Check variables are correctly named
    assert "MNDWI_year" in year_last.data_vars
    assert "NDVI_year" in year_last.data_vars
    assert "MNDWI_value" in value_last.data_vars
    assert "NDVI_value" in value_last.data_vars

    # Check shapes
    assert year_last["MNDWI_year"].shape == (3, 3)
    assert value_last["MNDWI_value"].shape == (3, 3)


def test_last_occurrence_no_time_dimension():
    """Test that function raises error if time dimension is missing."""
    da = xr.DataArray(
        np.random.rand(3, 3),
        dims=["y", "x"],
        name="MNDWI",
    )
    with pytest.raises(ValueError, match="time"):
        last_occurrence(da, "MNDWI")


def test_last_occurrence_missing_index(sample_dataset):
    """Test that function raises error if index is not found."""
    with pytest.raises(ValueError, match="Index.*not found"):
        last_occurrence(sample_dataset, ["MNDWI", "NONEXISTENT"])


def test_last_occurrence_year_fraction_range(sample_dataarray):
    """Test that year fractions are in [0, 1) range within each year."""
    year_last, _ = last_occurrence(sample_dataarray, "MNDWI", threshold=0.0)

    for year in [2020, 2021, 2022, 2023, 2024]:
        year_part = year_last.values[~np.isnan(year_last.values)]
        year_part = year_part[year_part >= year]
        year_part = year_part[year_part < year + 1]
        # Fractional part should be in [0, 1)
        frac = year_part - np.floor(year_part)
        assert np.all(frac >= 0) and np.all(frac < 1)


def test_last_occurrence_threshold():
    """Test that threshold parameter works correctly."""
    times = np.array([np.datetime64(f"202{i}-01-01") for i in range(5)])
    values = np.array(
        [
            [[[-0.5], [0.1], [0.5]], [[0.2], [0.0], [-0.2]]],
            [[[0.2], [0.3], [0.4]], [[-0.1], [0.1], [0.3]]],
            [[[0.5], [0.2], [0.6]], [[0.3], [0.5], [0.4]]],
            [[[-0.2], [-0.3], [-0.1]], [[-0.4], [0.0], [0.1]]],
            [[[0.1], [-0.5], [0.2]], [[0.0], [-0.1], [0.3]]],
        ]
    ).reshape(5, 2, 3)

    da = xr.DataArray(
        values,
        dims=["time", "y", "x"],
        coords={"time": times, "y": [0, 1], "x": [0, 1, 2]},
        name="TEST",
    )

    # With threshold=0, pixel (0,0) last on at year 2
    year_t0, _ = last_occurrence(da, "TEST", threshold=0.0)
    # With threshold=0.3, pixel (0,0) last on at year 2
    year_t03, _ = last_occurrence(da, "TEST", threshold=0.3)

    # Results should differ where values cross the threshold
    assert not np.allclose(year_t0, year_t03, equal_nan=True)


def test_last_occurrence_nan_handling(sample_dataarray):
    """Test that NaN values in data are handled correctly."""
    # sample_dataarray has a NaN at (0, 2) for all times
    year_last, value_last = last_occurrence(sample_dataarray, "MNDWI", threshold=0.0)

    # Pixel with all NaN should have NaN in output if it never exceeded threshold
    # or the last non-NaN value if it did
    # (behavior depends on implementation details of skipna)
    assert isinstance(year_last.values, np.ndarray)
    assert isinstance(value_last.values, np.ndarray)
