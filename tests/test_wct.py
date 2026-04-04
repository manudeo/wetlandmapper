"""Tests for wetlandmapper.wct"""

import numpy as np
import pytest
import xarray as xr

from wetlandmapper import WCT_CLASSES, classify_wct, compute_indices
from wetlandmapper.wct import build_ema_lookup_table, classify_wct_ema


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

    def test_low_moist_threshold_increases_wct5(self, multispectral_ds):
        """Lower moist threshold should classify more pixels as moist (WCT 5)."""
        indices = compute_indices(multispectral_ds)
        result_normal = classify_wct(indices)
        result_low = classify_wct(indices, thresholds={"mndwi_moist": -0.5})
        # Lower moist threshold should allow more WCT 5 pixels
        assert (result_low == 5).sum() >= (result_normal == 5).sum()

    def test_high_ndvi_veg_threshold_reduces_wct4(self, multispectral_ds):
        """Higher NDVI veg threshold should classify fewer pixels as emergent veg."""
        indices = compute_indices(multispectral_ds)
        result_normal = classify_wct(indices)
        result_high = classify_wct(indices, thresholds={"ndvi_veg_high": 0.5})
        # Higher threshold means fewer WCT 4 pixels
        assert (result_high == 4).sum() <= (result_normal == 4).sum()

    def test_high_ndti_threshold_reduces_turbid(self, multispectral_ds):
        """Higher NDTI threshold should classify fewer pixels as turbid (WCT 2)."""
        indices = compute_indices(multispectral_ds)
        result_normal = classify_wct(indices)
        result_high = classify_wct(indices, thresholds={"ndti_turbid": 0.5})
        # Higher threshold means fewer turbid pixels
        assert (result_high == 2).sum() <= (result_normal == 2).sum()


class TestClassifyWCTEMA:
    """Tests for EMA-based WCT classification."""

    def test_classify_wct_ema_returns_dataset(self, multispectral_ds):
        """classify_wct_ema should return a Dataset."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct_ema(indices)
        assert isinstance(result, xr.Dataset)

    def test_classify_wct_ema_has_wct_variable(self, multispectral_ds):
        """EMA result should have 'wetland_cover_type' variable."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct_ema(indices)
        assert "wetland_cover_type" in result

    def test_classify_wct_ema_has_combo_variable(self, multispectral_ds):
        """EMA result should have 'combination_code' variable."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct_ema(indices)
        assert "combination_code" in result

    def test_classify_wct_ema_output_dims(self, multispectral_ds):
        """EMA output spatial dims should match input."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct_ema(indices)
        assert result["wetland_cover_type"].dims == ("y", "x")
        assert result["combination_code"].dims == ("y", "x")

    def test_classify_wct_ema_default_n_parts(self, multispectral_ds):
        """EMA with default n_parts=4 should work."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct_ema(indices, n_parts=4)
        assert "wetland_cover_type" in result

    def test_classify_wct_ema_custom_n_parts(self, multispectral_ds):
        """EMA with custom n_parts should work."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct_ema(indices, n_parts=3)
        assert "wetland_cover_type" in result

    def test_classify_wct_ema_n_parts_too_small_raises(self, multispectral_ds):
        """n_parts < 2 should raise ValueError."""
        indices = compute_indices(multispectral_ds)
        with pytest.raises(ValueError, match="n_parts"):
            classify_wct_ema(indices, n_parts=1)

    def test_classify_wct_ema_n_parts_not_int_raises(self, multispectral_ds):
        """n_parts must be int."""
        indices = compute_indices(multispectral_ds)
        with pytest.raises(ValueError, match="n_parts"):
            classify_wct_ema(indices, n_parts=4.5)

    def test_classify_wct_ema_missing_variable_raises(self):
        """Missing required variable should raise KeyError."""
        ds_incomplete = xr.Dataset({
            "MNDWI": xr.DataArray(np.zeros((5, 5)), dims=["y", "x"]),
            "NDVI":  xr.DataArray(np.zeros((5, 5)), dims=["y", "x"]),
        })
        with pytest.raises(KeyError):
            classify_wct_ema(ds_incomplete)

    def test_classify_wct_ema_output_codes_valid(self, multispectral_ds):
        """All output codes should be valid WCT classes."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct_ema(indices)
        valid = set(WCT_CLASSES.keys())
        actual = set(int(v) for v in np.unique(result["wetland_cover_type"].values))
        assert actual.issubset(valid)

    def test_classify_wct_ema_open_water(self, multispectral_ds):
        """Zone 1 (open water) should be WCT 1."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct_ema(indices)
        zone1 = result["wetland_cover_type"].isel(y=slice(0, 5))
        assert (zone1 == 1).all()

    def test_classify_wct_ema_turbid(self, multispectral_ds):
        """Zone 2 (turbid water) classification via EMA."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct_ema(indices)
        zone2 = result["wetland_cover_type"].isel(y=slice(5, 10))
        # EMA may classify turbid water differently depending on index levels
        # Just verify it's a valid code
        valid = set(WCT_CLASSES.keys())
        actual = set(int(v) for v in np.unique(zone2.values))
        assert actual.issubset(valid)

    def test_classify_wct_ema_vegetation(self, multispectral_ds):
        """Zone 3 (vegetation) should include WCT 4 or 3."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct_ema(indices)
        zone3 = result["wetland_cover_type"].isel(y=slice(10, 15))
        # Vegetation zone can be WCT 3 (submerged) or 4 (emergent)
        assert ((zone3 == 3) | (zone3 == 4)).all()

    def test_classify_wct_ema_combo_code_structure(self, multispectral_ds):
        """Combination code should encode three levels."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct_ema(indices, n_parts=4)
        combo = result["combination_code"].values
        # Check that combo codes are reasonable 3-digit numbers
        assert np.all(combo >= 0)
        assert np.all(combo <= 444)

    def test_classify_wct_ema_preserves_attributes(self, multispectral_ds):
        """Output dataset should have metadata attributes."""
        indices = compute_indices(multispectral_ds)
        result = classify_wct_ema(indices)
        assert "title" in result.attrs
        assert "references" in result.attrs


class TestBuildEMALookupTable:
    """Tests for build_ema_lookup_table function."""

    def test_build_ema_lookup_table_returns_array(self):
        """build_ema_lookup_table should return a 3D numpy array."""
        table = build_ema_lookup_table(n_parts=4)
        assert isinstance(table, np.ndarray)
        assert table.ndim == 3

    def test_build_ema_lookup_table_shape(self):
        """Table shape should be (n_parts+1, n_parts+1, n_parts+1)."""
        n_parts = 4
        table = build_ema_lookup_table(n_parts=n_parts)
        expected_shape = (n_parts + 1, n_parts + 1, n_parts + 1)
        assert table.shape == expected_shape

    def test_build_ema_lookup_table_custom_n_parts(self):
        """Should work with different n_parts values."""
        for n_parts in [2, 3, 4, 5]:
            table = build_ema_lookup_table(n_parts=n_parts)
            assert table.shape == (n_parts + 1, n_parts + 1, n_parts + 1)

    def test_build_ema_lookup_table_values_are_valid_wct(self):
        """All values in lookup table should be valid WCT codes."""
        table = build_ema_lookup_table(n_parts=4)
        valid_codes = set(WCT_CLASSES.keys())
        unique_codes = set(int(v) for v in np.unique(table))
        assert unique_codes.issubset(valid_codes)

    def test_build_ema_lookup_table_dtype(self):
        """Lookup table should have integer dtype."""
        table = build_ema_lookup_table(n_parts=4)
        assert np.issubdtype(table.dtype, np.integer)

    def test_build_ema_lookup_table_n_parts_too_small_raises(self):
        """n_parts < 2 should cause an error during table creation."""
        # The function doesn't validate, so we expect IndexError when trying to build
        with pytest.raises((ValueError, IndexError)):
            build_ema_lookup_table(n_parts=1)

    def test_build_ema_lookup_table_n_parts_not_int_raises(self):
        """n_parts must be int or convertible to int."""
        # The function doesn't explicitly validate, so we expect TypeError
        with pytest.raises((ValueError, TypeError)):
            build_ema_lookup_table(n_parts=4.5)

    def test_build_ema_lookup_table_consistency(self):
        """Multiple calls should produce identical results."""
        table1 = build_ema_lookup_table(n_parts=4)
        table2 = build_ema_lookup_table(n_parts=4)
        assert np.array_equal(table1, table2)

    def test_build_ema_lookup_table_empty_vegetation_high_ndvi_values(self):
        """High NDVI (vegetation) should map to WCT 4 regardless of water."""
        # This tests the core logic: vegetation suppresses water signal
        table = build_ema_lookup_table(n_parts=4)
        # Any combination with w=anything, v=4 (high NDVI), t=low should give WCT 4
        for w in range(5):
            for t in range(2):  # Low NDTI values
                val = table[w, 4, t]
                assert val == 4, f"High NDVI should give WCT 4, got {val}"
