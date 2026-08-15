import inspect
from dataclasses import dataclass

import pytest

from wetlandmapper.indices import compute_aweish, compute_aweinsh
from wetlandmapper import gee


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
