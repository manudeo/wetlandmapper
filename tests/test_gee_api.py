import inspect

import pytest

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
