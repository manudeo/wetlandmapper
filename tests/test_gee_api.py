import inspect
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from wetlandmapper import gee
from wetlandmapper.indices import compute_aweinsh, compute_aweish


def test_fetch_xee_exposes_fetch_parameters_plus_chunks():
    """fetch_xee should mirror fetch options and add xee-specific chunks."""
    fetch_sig = inspect.signature(gee.fetch)
    fetch_xee_sig = inspect.signature(gee.fetch_xee)

    fetch_names = list(fetch_sig.parameters)
    fetch_xee_names = list(fetch_xee_sig.parameters)

    for name in fetch_names:
        assert name in fetch_xee_names, f"Missing parameter in fetch_xee: {name}"

    extra = set(fetch_xee_names) - set(fetch_names)
    assert extra == {"chunks"}


def test_fetch_xee_shared_defaults_match_fetch():
    """Shared parameters should keep identical defaults between APIs."""
    fetch_sig = inspect.signature(gee.fetch)
    fetch_xee_sig = inspect.signature(gee.fetch_xee)

    for name, fetch_param in fetch_sig.parameters.items():
        xee_param = fetch_xee_sig.parameters[name]
        assert xee_param.default == fetch_param.default, (
            f"Default mismatch for parameter '{name}': "
            f"fetch={fetch_param.default!r}, fetch_xee={xee_param.default!r}"
        )


def test_normalize_reduction_method_accepts_supported_values():
    assert gee._normalize_reduction_method("median") == "median"
    assert gee._normalize_reduction_method("MEAN") == "mean"
    assert gee._normalize_reduction_method("percentile") == "percentile"


def test_normalize_reduction_method_rejects_unknown_values():
    with pytest.raises(ValueError, match="reduction_method"):
        gee._normalize_reduction_method("sum")


def test_validate_percentile_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="percentile"):
        gee._validate_percentile(-1)

    with pytest.raises(ValueError, match="percentile"):
        gee._validate_percentile(101)


def test_format_percentile_token_handles_integer_and_fractional_values():
    assert gee._format_percentile_token(50.0) == "50"
    assert gee._format_percentile_token(33.3) == "33_3"


def test_gee_valid_indices_match_indices_module_support():
    """GEE fetch validators should include all index names provided by indices.py."""
    expected = {"MNDWI", "NDWI", "NDVI", "NDTI", "AWEIsh", "AWEInsh"}
    assert gee._VALID_INDICES == expected


def _build_processed_collection_kwargs(**overrides):
    """Minimal required kwargs for _build_processed_collection, patchable per test."""
    kwargs = dict(
        aoi={"type": "Point", "coordinates": [0.0, 0.0]},
        start="2020-01-01",
        end="2020-12-31",
        sensor="Landsat8",
        index="MNDWI",
        custom_indices=None,
        max_cloud_cover=20.0,
        temporal_aggregation="annual",
        use_slc_off=False,
        climate_adaptive=False,
        min_precip_mm=20.0,
        min_temp_c=5.0,
        hydroperiod_months=1,
        hydroperiod_nan_policy="valid",
        wetness_index="MNDWI",
        wetness_threshold=0.0,
        dem_mask=False,
        max_slope_deg=5.0,
        max_tpi_m=None,
        tpi_window_px=5,
        max_local_range_m=None,
        local_range_window_px=5,
        max_elevation_m=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_return_scene_count_rejects_temporal_aggregation_all():
    """A scene count is undefined for 'all' — each image is already one scene."""
    with pytest.raises(ValueError, match="return_scene_count"):
        gee._build_processed_collection(
            **_build_processed_collection_kwargs(
                temporal_aggregation="all",
                track_count=True,
            )
        )


def test_return_scene_count_rejects_climate_adaptive():
    """Climate-adaptive compositing picks one best month, not a reduction."""
    with pytest.raises(ValueError, match="return_scene_count"):
        gee._build_processed_collection(
            **_build_processed_collection_kwargs(
                climate_adaptive=True,
                track_count=True,
            )
        )


def test_return_scene_count_present_in_fetch_and_fetch_xee_signatures():
    """return_scene_count should be a shared, opt-in, default-False parameter."""
    for func in (gee.fetch, gee.fetch_xee):
        sig = inspect.signature(func)
        assert "return_scene_count" in sig.parameters
        assert sig.parameters["return_scene_count"].default is False


def test_hydroperiod_equivalent_months_valid_policy_ignores_masked_months():
    """Wet in all valid months should remain fully wet despite many masked months."""
    wet = np.array([5.0])
    observed = np.array([5.0])
    climate_valid = np.array([5.0])
    equiv = gee._hydroperiod_equivalent_months_numpy(
        wet,
        observed,
        climate_valid,
        hydroperiod_nan_policy="valid",
    )
    assert float(equiv[0]) == pytest.approx(5.0)


def test_hydroperiod_equivalent_months_total_policy_counts_masked_as_dry():
    """Total policy should preserve raw wet-month counts."""
    wet = np.array([5.0])
    observed = np.array([5.0])
    climate_valid = np.array([5.0])
    equiv = gee._hydroperiod_equivalent_months_numpy(
        wet,
        observed,
        climate_valid,
        hydroperiod_nan_policy="total",
    )
    assert float(equiv[0]) == pytest.approx(5.0)


def test_hydroperiod_mean_excludes_empty_years_from_average():
    """Years with zero valid months should not pull means toward zero."""
    yearly_equiv = np.array([[[12.0]], [[0.0]]])
    yearly_valid = np.array([[[4.0]], [[0.0]]])
    mean_equiv = gee._mean_hydroperiod_over_nonempty_years_numpy(
        yearly_equiv,
        yearly_valid,
    )
    assert float(mean_equiv[0, 0]) == pytest.approx(12.0)


def test_hydroperiod_nan_policy_rejects_invalid_value():
    with pytest.raises(ValueError, match="hydroperiod_nan_policy"):
        gee._normalize_hydroperiod_nan_policy("bad_mode")


def test_sentinel2_cloud_mask_uses_scl_cloud_shadow_and_snow_classes():
    class FakeMask:
        def __init__(self, excluded_classes=()):
            self.excluded_classes = set(excluded_classes)

        def And(self, other):
            return FakeMask(self.excluded_classes | other.excluded_classes)

    class FakeScl:
        def neq(self, value):
            return FakeMask({value})

    class FakeImage:
        def __init__(self):
            self.selected_band = None
            self.applied_mask = None

        def select(self, band_name):
            self.selected_band = band_name
            return FakeScl()

        def updateMask(self, mask):
            self.applied_mask = mask
            return self

    image = FakeImage()

    result = gee._mask_sentinel2_clouds(image)

    assert result is image
    assert image.selected_band == "SCL"
    assert image.applied_mask.excluded_classes == {3, 8, 9, 10, 11}
    assert gee._BAND_MAP["Sentinel2"]["qa"] == "SCL"


def test_dem_mask_uses_current_copernicus_collection_and_native_projection(monkeypatch):
    collection = Mock()
    collection.filterBounds.return_value = collection
    collection.select.return_value = collection
    projection = object()
    collection.first.return_value.projection.return_value = projection
    dem = Mock()
    collection.mosaic.return_value = dem
    dem.setDefaultProjection.return_value = dem
    terrain_mask = Mock()
    terrain_mask.And.return_value = terrain_mask
    terrain_mask.rename.return_value = terrain_mask
    slope = Mock()
    slope.lte.return_value = Mock()
    image_api = SimpleNamespace(constant=Mock(return_value=terrain_mask))
    ee_mock = SimpleNamespace(
        ImageCollection=Mock(return_value=collection),
        Image=image_api,
        Terrain=SimpleNamespace(slope=Mock(return_value=slope)),
    )
    monkeypatch.setattr(gee, "ee", ee_mock)

    gee._build_dem_mask(
        ee_geom=object(),
        max_slope_deg=5.0,
        max_tpi_m=None,
        max_local_range_m=None,
        max_elevation_m=None,
    )

    ee_mock.ImageCollection.assert_called_once_with("COPERNICUS/DEM/GLO30_2024_1")
    collection.mosaic.assert_called_once_with()
    dem.setDefaultProjection.assert_called_once_with(projection)
    ee_mock.Terrain.slope.assert_called_once_with(dem)


@dataclass
class _FakeScalarImage:
    value: float
    band_name: str | None = None

    def add(self, other):
        return _FakeScalarImage(self.value + other.value, self.band_name)

    def subtract(self, other):
        return _FakeScalarImage(self.value - other.value, self.band_name)

    def multiply(self, factor):
        if isinstance(factor, _FakeScalarImage):
            return _FakeScalarImage(self.value * factor.value, self.band_name)
        return _FakeScalarImage(self.value * float(factor), self.band_name)

    def rename(self, name):
        return _FakeScalarImage(self.value, name)


class _FakeImage:
    def __init__(self, bands):
        self._bands = dict(bands)

    def select(self, band_name):
        return _FakeScalarImage(self._bands[band_name], band_name)

    def normalizedDifference(self, band_names):
        a = self._bands[band_names[0]]
        b = self._bands[band_names[1]]
        return _FakeScalarImage((a - b) / (a + b))

    def addBands(self, derived):
        return _FakeImage(self._bands | derived._bands)


class _FakeCatImage:
    def __init__(self, images):
        self._bands = {img.band_name: img.value for img in images}


class _FakeEeImage:
    @staticmethod
    def cat(images):
        return _FakeCatImage(images)


class _FakeEeModule:
    Image = _FakeEeImage


def test_add_indices_awei_formulas_match_local_implementations(monkeypatch):
    """Server-side AWEI formulas should match indices.py exactly."""
    monkeypatch.setattr(gee, "ee", _FakeEeModule())

    vals = {
        "blue": 0.12,
        "green": 0.23,
        "red": 0.09,
        "nir": 0.15,
        "swir": 0.05,
        "swir2": 0.31,
        "qa": 1.0,
    }
    fake_img = _FakeImage(vals)
    bands = {
        "blue": "blue",
        "green": "green",
        "red": "red",
        "nir": "nir",
        "swir": "swir",
        "swir2": "swir2",
        "qa": "qa",
    }

    out = gee._add_indices(fake_img, bands)

    import xarray as xr

    ds = xr.Dataset({k: xr.DataArray([[v]], dims=["y", "x"]) for k, v in vals.items()})
    expected_aweish = float(compute_aweish(ds).values[0, 0])
    expected_aweinsh = float(compute_aweinsh(ds).values[0, 0])

    assert out._bands["AWEIsh"] == pytest.approx(expected_aweish, abs=1e-12)
    assert out._bands["AWEInsh"] == pytest.approx(expected_aweinsh, abs=1e-12)
