"""Tests for wetlandmapper.wct"""

import numpy as np
import pytest
import xarray as xr

from wetlandmapper import classify_wct, compute_indices, WCT_CLASSES


class TestClassifyWCT:

    # ------------------------------------------------------------------
    # Basic output structure
    # ------------------------------------------------------------------

    def test_returns_dataarray(self, multispectral_ds):
        indices = compute_indices(multispectral_ds)
        result = classify_wct(indices)
        assert isinstance(result, xr.DataArray)

    def test_output_dtype(self, multispectral_ds):
        indices = compute_indices(multispectral_ds)
        result = classify_wct(indices)
        assert result.dtype == np.int8

    def test_output_dims(self, multispectral_ds):
        indices = compute_indices(multispectral_ds)
        result = classify_wct(indices)
        assert result.dims == ("y", "x")

    def test_output_name(self, multispectral_ds):
        indices = compute_indices(multispectral_ds)
        result = classify_wct(indices)
        assert result.name == "wetland_cover_type"

    def test_output_codes_valid(self, multispectral_ds):
        indices = compute_indices(multispectral_ds)
        result = classify_wct(indices)
        valid = set(WCT_CLASSES.keys())
        actual = set(int(v) for v in np.unique(result.values))
        assert actual.issubset(valid), f"Unexpected codes: {actual - valid}"

    # ------------------------------------------------------------------
    # Classification correctness against synthetic zones
    # ------------------------------------------------------------------

    def test_open_water_zone_classified_correctly(self, multispectral_ds):
        """Zone 1 (rows 0-4): open clear water → WCT 1."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct(indices)
        zone1 = result.isel(y=slice(0, 5))
        assert (zone1 == 1).all(), \
            f"Zone 1 (open water) expected WCT 1, got: {np.unique(zone1.values)}"

    def test_turbid_zone_classified_correctly(self, multispectral_ds):
        """Zone 2 (rows 5-9): turbid water → WCT 2."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct(indices)
        zone2 = result.isel(y=slice(5, 10))
        assert (zone2 == 2).all(), \
            f"Zone 2 (turbid) expected WCT 2, got: {np.unique(zone2.values)}"

    def test_vegetation_zone_classified_correctly(self, multispectral_ds):
        """Zone 3 (rows 10-14): emergent vegetation → WCT 4."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct(indices)
        zone3 = result.isel(y=slice(10, 15))
        assert (zone3 == 4).all(), \
            f"Zone 3 (emergent veg) expected WCT 4, got: {np.unique(zone3.values)}"

    def test_nonwetland_zone_is_zero(self, multispectral_ds):
        """Zone 4 (rows 15-19): non-wetland → WCT 0."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct(indices)
        zone4 = result.isel(y=slice(15, None))
        assert (zone4 == 0).all(), \
            f"Zone 4 (non-wetland) expected 0, got: {np.unique(zone4.values)}"

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_missing_variable_raises(self):
        """Dataset missing one of the required indices should raise KeyError."""
        ds_incomplete = xr.Dataset({
            "MNDWI": xr.DataArray(np.zeros((5, 5)), dims=["y", "x"]),
            "NDVI":  xr.DataArray(np.zeros((5, 5)), dims=["y", "x"]),
            # NDTI missing
        })
        with pytest.raises(KeyError, match="NDTI"):
            classify_wct(ds_incomplete)

    def test_unknown_threshold_key_warns(self, multispectral_ds):
        """Passing an unknown threshold key should emit a UserWarning."""
        indices = compute_indices(multispectral_ds)
        with pytest.warns(UserWarning, match="Unknown threshold"):
            classify_wct(indices, thresholds={"not_a_key": 0.5})

    # ------------------------------------------------------------------
    # Custom thresholds
    # ------------------------------------------------------------------

    def test_custom_thresholds_accepted(self, multispectral_ds):
        """classify_wct should run without error when thresholds are customised."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct(
            indices,
            thresholds={"mndwi_water": 0.05, "ndvi_veg_high": 0.3},
        )
        assert isinstance(result, xr.DataArray)

    def test_very_high_mndwi_threshold_leaves_nothing_as_water(self, multispectral_ds):
        """With a very high MNDWI threshold, no pixel should be water (WCT 1 or 2)."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct(indices, thresholds={"mndwi_water": 0.99})
        assert not (result == 1).any()
        assert not (result == 2).any()
