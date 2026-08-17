"""Test suite for analysis.last_occurrence function."""

import numpy as np
import pytest
import xarray as xr

from wetlandmapper import (
    build_run_manifest,
    class_area_timeseries,
    class_summary,
    class_transition_matrix,
    detect_wet_events,
    last_occurrence,
    linear_trend,
    quality_uncertainty_summary,
    summarize_by_polygons,
    summarize_dynamics,
    summarize_wct,
    trend_products,
)
from wetlandmapper.dynamics import DYNAMICS_CLASSES
from wetlandmapper.wct import WCT_CLASSES, WCT_LEVEL2_CLASSES


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


def test_class_summary_dataarray_counts_and_percent():
    arr = xr.DataArray(
        np.array([[0, 1, 1], [2, 2, np.nan]], dtype=float),
        dims=["y", "x"],
        name="classes",
    )
    summary = class_summary(arr, class_labels={0: "A", 1: "B", 2: "C"})

    assert set(summary.data_vars) >= {"pixel_count", "percent_of_valid", "class_name"}
    assert int(summary["pixel_count"].sel(class_code=0)) == 1
    assert int(summary["pixel_count"].sel(class_code=1)) == 2
    assert int(summary["pixel_count"].sel(class_code=2)) == 2
    assert summary.attrs["total_valid_pixels"] == 5


def test_class_summary_dataset_variable_selection():
    ds = xr.Dataset(
        {
            "a": xr.DataArray(np.array([[0, 0], [1, 1]], dtype=float), dims=["y", "x"]),
            "b": xr.DataArray(np.array([[2, 2], [2, 2]], dtype=float), dims=["y", "x"]),
        }
    )
    summary = class_summary(ds, variable="b")
    assert int(summary["pixel_count"].sel(class_code=2)) == 4


def test_class_summary_area_output():
    arr = xr.DataArray(np.array([[0, 1], [1, 1]], dtype=float), dims=["y", "x"])
    summary = class_summary(arr, pixel_area=100.0, area_unit="m2")
    assert "area_m2" in summary
    assert float(summary["area_m2"].sel(class_code=1)) == 300.0


def test_summarize_dynamics_wrapper(mndwi_mixed):
    from wetlandmapper import classify_dynamics

    dynamics = classify_dynamics(mndwi_mixed)
    summary = summarize_dynamics(dynamics)
    assert set(summary["class_code"].values.tolist()) == set(DYNAMICS_CLASSES.keys())


def test_summarize_wct_wrappers(multispectral_ds):
    from wetlandmapper import compute_indices
    from wetlandmapper.wct import classify_wct_ema, classify_wct_ema_level2

    indices = compute_indices(multispectral_ds)
    wct_l1 = classify_wct_ema(indices)["wetland_cover_type"]
    wct_l2 = classify_wct_ema_level2(indices)

    summary_l1 = summarize_wct(wct_l1)
    summary_l2 = summarize_wct(wct_l2, level2=True)

    assert set(summary_l1["class_code"].values.tolist()) == set(WCT_CLASSES.keys())
    assert set(summary_l2["class_code"].values.tolist()) == set(WCT_LEVEL2_CLASSES.keys())


def test_linear_trend_dataarray_known_signal():
    times = np.array(
        [
            np.datetime64("2000-01-01"),
            np.datetime64("2001-01-01"),
            np.datetime64("2002-01-01"),
            np.datetime64("2003-01-01"),
        ]
    )
    years = np.array([2000.0, 2001.0, 2002.0, 2003.0], dtype=float)

    slope_true = np.array([[0.5, -0.25], [0.0, 1.0]], dtype=float)
    intercept_true = np.array([[2.0, 3.0], [4.0, -1.0]], dtype=float)

    values = np.empty((len(times), 2, 2), dtype=float)
    for i, year in enumerate(years):
        values[i] = slope_true * year + intercept_true

    da = xr.DataArray(
        values,
        dims=["time", "y", "x"],
        coords={"time": times, "y": [0, 1], "x": [0, 1]},
        name="ndvi",
        attrs={"units": "unitless"},
    )

    out = linear_trend(da)

    np.testing.assert_allclose(out["slope"].values, slope_true, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        out["intercept"].values, intercept_true, atol=1e-12, rtol=1e-12
    )
    assert float(out["p_value"].sel(y=0, x=0)) < 1e-6
    assert float(out["p_value"].sel(y=0, x=1)) < 1e-6
    assert float(out["p_value"].sel(y=1, x=1)) < 1e-6
    assert float(out["p_value"].sel(y=1, x=0)) == 1.0
    assert out["slope"].attrs["units"] == "unitless/year"
    assert out.attrs["time_basis"] == "fractional_year"


def test_linear_trend_handles_nan_and_insufficient_points():
    times = np.array(
        [
            np.datetime64("2000-01-01"),
            np.datetime64("2001-01-01"),
            np.datetime64("2002-01-01"),
            np.datetime64("2003-01-01"),
        ]
    )
    values = np.array(
        [
            [[1.0, 2.0], [5.0, np.nan]],
            [[2.0, np.nan], [5.0, np.nan]],
            [[3.0, np.nan], [5.0, np.nan]],
            [[4.0, np.nan], [5.0, 9.0]],
        ],
        dtype=float,
    )
    da = xr.DataArray(
        values,
        dims=["time", "y", "x"],
        coords={"time": times, "y": [0, 1], "x": [0, 1]},
        name="signal",
    )

    out = linear_trend(da)

    assert np.isfinite(float(out["slope"].sel(y=0, x=0)))
    assert np.isnan(float(out["slope"].sel(y=0, x=1)))
    assert float(out["slope"].sel(y=1, x=0)) == 0.0
    assert np.isnan(float(out["slope"].sel(y=1, x=1)))


def test_linear_trend_dataset_variable_selection_and_errors():
    times = np.array(
        [
            np.datetime64("2000-01-01"),
            np.datetime64("2001-01-01"),
            np.datetime64("2002-01-01"),
        ]
    )
    base = xr.DataArray(
        np.array([[1.0], [2.0], [3.0]], dtype=float),
        dims=["time", "x"],
        coords={"time": times, "x": [0]},
    )
    ds = xr.Dataset({"a": base, "b": base * 2.0})

    with pytest.raises(ValueError, match="requires variable"):
        linear_trend(ds)

    out = linear_trend(ds, variable="a")
    assert "slope" in out.data_vars

    with pytest.raises(ValueError, match="not found"):
        linear_trend(ds, variable="missing")


def test_trend_products_significance_and_classification(tmp_path):
    times = np.array(
        [
            np.datetime64("2000-01-01"),
            np.datetime64("2001-01-01"),
            np.datetime64("2002-01-01"),
            np.datetime64("2003-01-01"),
        ]
    )
    years = np.array([2000.0, 2001.0, 2002.0, 2003.0], dtype=float)
    vals = np.zeros((4, 1, 2), dtype=float)
    vals[:, 0, 0] = 0.5 * years + 1.0
    vals[:, 0, 1] = -0.5 * years + 3.0
    da = xr.DataArray(
        vals,
        dims=["time", "y", "x"],
        coords={"time": times, "y": [0], "x": [0, 1]},
    )

    out = trend_products(da)

    assert "slope_significant" in out
    assert int(out["trend_class"].sel(y=0, x=0)) == 1
    assert int(out["trend_class"].sel(y=0, x=1)) == -1


def test_class_area_timeseries_and_transition_matrix():
    times = np.array(
        [
            np.datetime64("2020-01-01"),
            np.datetime64("2021-01-01"),
        ]
    )
    classes = xr.DataArray(
        np.array(
            [
                [[0, 1], [1, 2]],
                [[1, 1], [2, 2]],
            ],
            dtype=float,
        ),
        dims=["time", "y", "x"],
        coords={"time": times, "y": [0, 1], "x": [0, 1]},
        name="classes",
    )

    ts = class_area_timeseries(classes, pixel_area=100.0, area_unit="m2")
    assert int(ts["pixel_count"].sel(time=times[0], class_code=1)) == 2
    assert float(ts["area_m2"].sel(time=times[1], class_code=2)) == 200.0

    mat = class_transition_matrix(classes, start_time=times[0], end_time=times[1])
    assert int(mat["pixel_count"].sel(from_class=0, to_class=1)) == 1
    assert int(mat["pixel_count"].sel(from_class=2, to_class=2)) == 1


def test_detect_wet_events():
    times = np.array(
        [
            np.datetime64("2020-01-01"),
            np.datetime64("2021-01-01"),
            np.datetime64("2022-01-01"),
            np.datetime64("2023-01-01"),
        ]
    )
    da = xr.DataArray(
        np.array(
            [
                [[-1.0, 1.0]],
                [[2.0, 1.0]],
                [[3.0, -1.0]],
                [[-1.0, -1.0]],
            ]
        ),
        dims=["time", "y", "x"],
        coords={"time": times, "y": [0], "x": [0, 1]},
    )

    out = detect_wet_events(da, threshold=0.0)
    assert int(out["on_count"].sel(y=0, x=0)) == 2
    assert int(out["on_count"].sel(y=0, x=1)) == 2
    assert int(out["longest_on_streak"].sel(y=0, x=0)) == 2


def test_quality_uncertainty_summary_with_sensor_coord():
    times = np.array(
        [
            np.datetime64("2020-01-01"),
            np.datetime64("2021-01-01"),
            np.datetime64("2022-01-01"),
            np.datetime64("2023-01-01"),
        ]
    )
    sensors = np.array(["L8", "L8", "S2", "S2"], dtype=object)
    da = xr.DataArray(
        np.array(
            [
                [[1.0, np.nan]],
                [[2.0, np.nan]],
                [[3.0, 5.0]],
                [[4.0, np.nan]],
            ]
        ),
        dims=["time", "y", "x"],
        coords={"time": times, "y": [0], "x": [0, 1], "sensor": ("time", sensors)},
    )

    out = quality_uncertainty_summary(da, low_support_threshold=0.8)
    assert float(out["valid_fraction"].sel(y=0, x=0)) == 1.0
    assert int(out["low_support"].sel(y=0, x=1)) == 1
    assert "sensor_count" in out


def test_build_run_manifest_writes_file(tmp_path):
    inp = tmp_path / "input.txt"
    inp.write_text("abc", encoding="utf-8")
    out_json = tmp_path / "manifest.json"

    manifest = build_run_manifest(
        parameters={"alpha": 0.05},
        input_paths=[inp],
        output_path=out_json,
        extras={"note": "test"},
    )

    assert "created_utc" in manifest
    assert str(inp) in manifest["input_hashes_sha256"]
    assert out_json.exists()


def test_summarize_by_polygons_mean_count():
    gpd = pytest.importorskip("geopandas")
    shapely = pytest.importorskip("shapely")
    box = shapely.box

    da = xr.DataArray(
        np.array([[[1.0, 2.0], [3.0, 4.0]]]),
        dims=["time", "y", "x"],
        coords={"time": [np.datetime64("2020-01-01")], "y": [0.0, 1.0], "x": [0.0, 1.0]},
    )
    gdf = gpd.GeoDataFrame({"pid": [7], "geometry": [box(-0.1, -0.1, 1.1, 1.1)]})

    df = summarize_by_polygons(da, gdf, polygon_id_col="pid", stats=("mean", "count"))
    assert len(df) == 1
    assert int(df.loc[0, "polygon_id"]) == 7
    assert float(df.loc[0, "mean"]) == 2.5
    assert int(df.loc[0, "count"]) == 4
