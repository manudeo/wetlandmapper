import inspect

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
