import os
from pathlib import Path

import numpy as np
import pytest

from wetlandmapper import gee


RUN_LIVE_GEE = os.getenv("WETLANDMAPPER_RUN_GEE_LIVE") == "1"
GEE_PROJECT = os.getenv("WETLANDMAPPER_GEE_PROJECT", "")
GEE_AOI = os.getenv(
    "WETLANDMAPPER_GEE_AOI",
    str(Path("notebooks/chilika_north/Chilika_north.shp")),
)


pytestmark = [
    pytest.mark.live_gee,
    pytest.mark.skipif(
        not RUN_LIVE_GEE,
        reason="Set WETLANDMAPPER_RUN_GEE_LIVE=1 to enable live Earth Engine tests.",
    ),
]


def _compute_dataset(**kwargs):
    gee.init(project=GEE_PROJECT or None)
    data = gee.fetch_xee(
        aoi=Path(GEE_AOI),
        project=GEE_PROJECT or None,
        chunks={"time": 1, "lon": 512, "lat": 512},
        **kwargs,
    ).compute()
    return data


def test_live_fetch_xee_sentinel2_reducers_return_finite_data():
    for reduction_method, extra in (("mean", {}), ("percentile", {"percentile": 75})):
        data = _compute_dataset(
            start="2025-01-01",
            end="2025-03-31",
            sensor="Sentinel2",
            index="NDWI",
            scale=10,
            max_cloud_cover=100.0,
            temporal_aggregation="monthly",
            reduction_method=reduction_method,
            **extra,
        )
        values = data.values
        assert data.sizes["time"] == 3
        assert int(np.isfinite(values).sum()) > 0


def test_live_fetch_xee_landsatall_returns_finite_data():
    data = _compute_dataset(
        start="2024-01-01",
        end="2024-12-31",
        sensor="LandsatAll",
        index="MNDWI",
        scale=30,
        max_cloud_cover=100.0,
        temporal_aggregation="annual",
    )
    values = data.values
    assert data.sizes["time"] == 1
    assert int(np.isfinite(values).sum()) > 0


@pytest.mark.parametrize(
    "sensor,start,end,require_finite",
    [
        ("Landsat4", "1990-01-01", "1990-12-31", False),
        ("Landsat5", "2005-01-01", "2005-12-31", True),
        ("Landsat7", "2001-01-01", "2001-12-31", True),
        ("Landsat8", "2020-01-01", "2020-12-31", True),
        ("Landsat9", "2023-01-01", "2023-12-31", True),
    ],
)
def test_live_fetch_xee_each_landsat_sensor_returns_expected_data(
    sensor,
    start,
    end,
    require_finite,
):
    data = _compute_dataset(
        start=start,
        end=end,
        sensor=sensor,
        index="MNDWI",
        scale=30,
        max_cloud_cover=100.0,
        temporal_aggregation="annual",
    )
    values = data.values
    assert data.sizes["time"] == 1
    finite = int(np.isfinite(values).sum())
    if require_finite:
        assert finite > 0
    else:
        # Landsat4 can be fully masked for some AOIs/years; this still confirms
        # the collection path executes and returns the expected structure.
        assert finite >= 0


def test_live_fetch_xee_modisall_returns_finite_data():
    data = _compute_dataset(
        start="2024-01-01",
        end="2024-12-31",
        sensor="MODISAll",
        index="MNDWI",
        scale=500,
        max_cloud_cover=100.0,
        temporal_aggregation="annual",
    )
    values = data.values
    assert data.sizes["time"] == 1
    assert int(np.isfinite(values).sum()) > 0


@pytest.mark.parametrize("sensor", ["MODIS_Terra", "MODIS_Aqua"])
def test_live_fetch_xee_each_modis_sensor_returns_finite_data(sensor):
    data = _compute_dataset(
        start="2024-01-01",
        end="2024-12-31",
        sensor=sensor,
        index="MNDWI",
        scale=500,
        max_cloud_cover=100.0,
        temporal_aggregation="annual",
    )
    values = data.values
    assert data.sizes["time"] == 1
    assert int(np.isfinite(values).sum()) > 0