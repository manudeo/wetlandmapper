"""Offline tests of the inter-sensor calibration (no Earth Engine calls)."""

import inspect
import json
import warnings

import numpy as np
import pytest

from wetlandmapper import gee
from wetlandmapper import sensor_harmonize as sh


# --------------------------------------------------------------------------- chain arithmetic


def test_chain_corrections_bring_every_sensor_onto_landsat8():
    corr = sh.chain_corrections({
        ("Landsat5", "Landsat7"): -0.03,
        ("Landsat7", "Landsat8"): -0.04,
        ("Landsat8", "Landsat9"): -0.006,
    })
    assert corr["Landsat8"] == 0.0
    assert corr["Landsat9"] == pytest.approx(0.006)      # L9 = L8 - 0.006 -> add +0.006
    assert corr["Landsat7"] == pytest.approx(-0.04)      # L8 = L7 - 0.04  -> add -0.04
    assert corr["Landsat5"] == pytest.approx(-0.07)      # chain 5 -> 7 -> 8
    assert corr["Landsat4"] == corr["Landsat5"]


def test_chain_corrections_leaves_out_sensors_with_a_broken_chain():
    corr = sh.chain_corrections({("Landsat5", "Landsat7"): -0.03})
    assert "Landsat5" not in corr and "Landsat7" not in corr
    assert corr == {"Landsat8": 0.0}


def test_correction_makes_the_sensors_agree():
    """A synthetic world with known offsets: correcting by the chained values equalises the sensors."""
    truth = np.linspace(-0.5, 0.6, 50)
    o57, o78, o89 = -0.03, -0.04, 0.01
    l8 = truth
    l7 = l8 - o78
    l5 = l7 - o57
    l9 = l8 + o89
    corr = sh.chain_corrections({("Landsat5", "Landsat7"): o57, ("Landsat7", "Landsat8"): o78,
                                 ("Landsat8", "Landsat9"): o89})
    for sensor_values, name in [(l5, "Landsat5"), (l7, "Landsat7"), (l9, "Landsat9")]:
        assert sensor_values + corr[name] == pytest.approx(l8)


def test_needed_pairs():
    assert sh.needed_pairs({"Landsat8"}) == []
    assert sh.needed_pairs({"Landsat9"}) == [("Landsat8", "Landsat9")]
    assert sh.needed_pairs({"Landsat7"}) == [("Landsat7", "Landsat8")]
    assert sh.needed_pairs({"Landsat5", "Landsat8"}) == [("Landsat5", "Landsat7"), ("Landsat7", "Landsat8")]
    assert ("Landsat8", "Landsat9") in sh.needed_pairs({"Landsat5", "Landsat9"})


def test_missions_in_window_and_landsat7_cut():
    all_ms = sh.missions_in_window("1994-01-01", "2023-12-31")
    assert {"Landsat5", "Landsat7", "Landsat8", "Landsat9"} <= all_ms
    cut = sh.missions_in_window("2014-01-01", "2023-12-31", landsat7_end="2013-03-17")
    assert "Landsat7" not in cut and "Landsat5" not in cut and {"Landsat8", "Landsat9"} <= cut
    assert "Landsat7" in sh.missions_in_window("2005-01-01", "2010-12-31", landsat7_end="2013-03-17")


# --------------------------------------------------------------------------- quality gates and fallback


def _info(n_years=4, count=5000, median=-0.05):
    return {"median": median, "count": count, "years": list(range(2013, 2013 + n_years)),
            "yearly": [median + d for d in (-0.02, 0.0, 0.01, 0.03)[:n_years]]}


def test_pair_row_local_estimate_with_interval():
    row = sh._pair_row("Landsat7", "Landsat8", _info(), 3, 300, -0.04, "global")
    assert row["source"] == "local" and row["offset"] == pytest.approx(-0.05)
    assert row["ci_low"] <= row["offset"] <= row["ci_high"] or np.isfinite(row["ci_low"])


def test_pair_row_falls_back_to_global_with_a_warning():
    with pytest.warns(UserWarning, match="global fallback"):
        row = sh._pair_row("Landsat7", "Landsat8", _info(n_years=2), 3, 300, -0.04, "global")
    assert row["source"] == "global" and row["offset"] == -0.04
    with pytest.warns(UserWarning):
        row = sh._pair_row("Landsat7", "Landsat8", _info(count=10), 3, 300, -0.04, "global")
    assert row["source"] == "global"
    with pytest.warns(UserWarning):
        row = sh._pair_row("Landsat7", "Landsat8", None, 3, 300, -0.04, "global")
    assert row["source"] == "global"


def test_pair_row_raises_without_fallback():
    with pytest.raises(RuntimeError, match="Cannot estimate"):
        sh._pair_row("Landsat7", "Landsat8", _info(n_years=1), 3, 300, None, "global")
    with pytest.raises(RuntimeError, match="Cannot estimate"):
        sh._pair_row("Landsat7", "Landsat8", _info(n_years=1), 3, 300, -0.04, "error")


def test_bootstrap_interval_brackets_the_median():
    lo, hi = sh._bootstrap_median_ci([-0.08, -0.05, -0.04, -0.03, -0.01], seed=1)
    assert lo <= -0.04 <= hi
    assert sh._bootstrap_median_ci([0.1]) == (0.1, 0.1)
    assert all(np.isnan(sh._bootstrap_median_ci([])))


def test_global_fallback_table_has_expected_pairs():
    g = sh.GLOBAL_PAIR_OFFSETS["MNDWI"]
    assert set(g) == {("Landsat5", "Landsat7"), ("Landsat7", "Landsat8"), ("Landsat8", "Landsat9")}
    assert all(-0.1 < v < 0.05 for v in g.values())


def test_estimate_rejects_bad_arguments_before_touching_earth_engine():
    with pytest.raises(ValueError, match="fallback"):
        sh.estimate_sensor_offsets({}, fallback="maybe")
    with pytest.raises(ValueError, match="built-in"):
        sh.estimate_sensor_offsets({}, index="MY_INDEX")


# --------------------------------------------------------------------------- resolver in gee.py


def test_resolve_off_returns_none():
    for flag in (False, None, "off"):
        assert gee._resolve_harmonization(flag, {}, ["MNDWI"], "2000-01-01", "2010-01-01", None, 20.0,
                                          None, 0.0, None, None) is None


def test_resolve_dict_forms_and_validation():
    info: dict = {}
    out = gee._resolve_harmonization({"Landsat7": -0.04, "Landsat5": -0.07}, {}, ["MNDWI"],
                                     "2000-01-01", "2010-01-01", None, 20.0, None, 0.0, None, info)
    assert out == {"MNDWI": {"Landsat7": -0.04, "Landsat5": -0.07}} and info["MNDWI"]["source"] == "user"
    with pytest.raises(ValueError, match="ambiguous"):
        gee._resolve_harmonization({"Landsat7": -0.04}, {}, ["MNDWI", "NDVI"], "2000-01-01", "2010-01-01",
                                   None, 20.0, None, 0.0, None, None)
    nested = gee._resolve_harmonization({"NDVI": {"Landsat7": 0.01}}, {}, ["MNDWI", "NDVI"],
                                        "2000-01-01", "2010-01-01", None, 20.0, None, 0.0, None, None)
    assert nested == {"NDVI": {"Landsat7": 0.01}}
    with pytest.raises(ValueError, match="harmonize must be"):
        gee._resolve_harmonization("sometimes", {}, ["MNDWI"], "2000-01-01", "2010-01-01", None, 20.0,
                                   None, 0.0, None, None)


def test_resolve_global_uses_table_and_skips_indices_without_one():
    info: dict = {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = gee._resolve_harmonization("global", {}, ["MNDWI", "NDVI"], "1994-01-01", "2023-12-31",
                                         [9, 10, 11], 20.0, None, 0.0, None, info)
    assert set(out) == {"MNDWI"}
    assert out["MNDWI"]["Landsat7"] == pytest.approx(-0.04)
    assert out["MNDWI"]["Landsat5"] == pytest.approx(-0.07)
    assert any("NDVI" in str(x.message) for x in w)
    assert info["MNDWI"]["source"] == "global"


def test_resolve_accepts_sensor_offsets_object():
    import pandas as pd

    so = sh.SensorOffsets(index="MNDWI", table=pd.DataFrame(), correction={"Landsat8": 0.0, "Landsat7": -0.05})
    out = gee._resolve_harmonization(so, {}, ["MNDWI"], "2000-01-01", "2010-01-01", None, 20.0, None, 0.0,
                                     None, {})
    assert out == {"MNDWI": {"Landsat8": 0.0, "Landsat7": -0.05}}
    assert json.loads(so.to_json())["reference"] == "Landsat8"


# --------------------------------------------------------------------------- server-side application (fake ee)


class _Img:
    def __init__(self, bands):
        self.bands = dict(bands)

    def select(self, name):
        return _Img({name: self.bands[name]})

    def add(self, v):
        return _Img({k: x + v for k, x in self.bands.items()})

    def toFloat(self):
        return _Img({k: float(x) for k, x in self.bands.items()})

    def rename(self, name):
        (v,) = self.bands.values()
        return _Img({name: v})

    def addBands(self, other, overwrite=False):
        out = dict(self.bands)
        for k, v in other.bands.items():
            assert overwrite or k not in out
            out[k] = v
        return _Img(out)


class _Col:
    def __init__(self, imgs):
        self.imgs = imgs

    def map(self, fn):
        return _Col([fn(i) for i in self.imgs])


def test_apply_index_offsets_shifts_only_named_bands():
    col = _Col([_Img({"MNDWI": 0.10, "NDVI": 0.5}), _Img({"MNDWI": -0.30, "NDVI": 0.2})])
    out = gee._apply_index_offsets(col, {"MNDWI": -0.04})
    assert [i.bands["MNDWI"] for i in out.imgs] == pytest.approx([0.06, -0.34])
    assert [i.bands["NDVI"] for i in out.imgs] == [0.5, 0.2]
    assert gee._apply_index_offsets(col, None) is col
    # a zero offset still re-casts the band, so that all missions share one band type at merge
    zero = gee._apply_index_offsets(col, {"MNDWI": 0.0})
    assert [i.bands["MNDWI"] for i in zero.imgs] == pytest.approx([0.10, -0.30])


# --------------------------------------------------------------------------- public signature


def test_fetch_functions_expose_harmonization_options_with_defaults():
    for fn in (gee.fetch, gee.fetch_xee):
        p = inspect.signature(fn).parameters
        assert p["harmonize"].default == "auto"
        assert p["harmonize_buffer_m"].default == 0.0
        assert p["landsat7_end"].default is None
