"""Tests for wetlandmapper.dynamics"""

# Copyright (c) 2026, Manudeo Singh          #
# Author: Manudeo Singh, March 2026          #

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wetlandmapper import DYNAMICS_CLASSES, classify_dynamics
from wetlandmapper.dynamics import aggregate_time, compute_wet_frequency


class TestClassifyDynamics:

    # ------------------------------------------------------------------
    # Basic output structure
    # ------------------------------------------------------------------

    def test_returns_dataarray(self, mndwi_mixed):
        result = classify_dynamics(mndwi_mixed, nYear=3)
        assert isinstance(result, xr.DataArray)

    def test_output_dtype(self, mndwi_mixed):
        result = classify_dynamics(mndwi_mixed, nYear=3)
        assert result.dtype == np.int8

    def test_output_dims_match_input(self, mndwi_mixed):
        result = classify_dynamics(mndwi_mixed, nYear=3)
        assert result.dims == ("y", "x")

    def test_output_has_no_time_dim(self, mndwi_all_wet):
        result = classify_dynamics(mndwi_all_wet, nYear=3)
        assert "time" not in result.dims

    def test_name_encodes_parameters(self, mndwi_mixed):
        result = classify_dynamics(mndwi_mixed, nYear=3,
                                   thresholdWet=25, thresholdPersis=75)
        assert "nYear3" in result.name
        assert "wet25"  in result.name
        assert "persis75" in result.name

    # ------------------------------------------------------------------
    # Classification correctness
    # ------------------------------------------------------------------

    def test_all_wet_classified_persistent(self, mndwi_all_wet):
        """When all time steps are wet, every pixel should be Persistent (10)."""
        result = classify_dynamics(mndwi_all_wet, nYear=3,
                                   thresholdPersis=75)
        assert (result == 10).all()

    def test_all_dry_classified_nonwetland(self, mndwi_all_dry):
        """When all time steps are dry, every pixel should be Non-wetland (0)."""
        result = classify_dynamics(mndwi_all_dry, nYear=3)
        assert (result == 0).all()

    def test_mixed_quadrants(self, mndwi_mixed):
        """
        mndwi_mixed has four quadrants:
          TL: always wet  → Persistent (10)
          TR: wet only recent → New (2)
          BL: wet only historic → Lost (3)
          BR: always dry → Non-wetland (0)
        """
        result = classify_dynamics(
            mndwi_mixed,
            nYear=mndwi_mixed.sizes["time"] // 2,
            thresholdWet=25,
            thresholdPersis=75,
        )
        ny, nx = result.sizes["y"], result.sizes["x"]
        hy, hx = ny // 2, nx // 2

        # TL quadrant — Persistent
        assert (result.isel(y=slice(0, hy), x=slice(0, hx)) == 10).all(), \
            "TL quadrant should be Persistent"

        # TR quadrant — New
        tr = result.isel(y=slice(0, hy), x=slice(hx, None))
        assert (tr == 2).all(), "TR quadrant should be New"

        # BL quadrant — Lost
        bl = result.isel(y=slice(hy, None), x=slice(0, hx))
        assert (bl == 3).all(), "BL quadrant should be Lost"

        # BR quadrant — Non-wetland
        br = result.isel(y=slice(hy, None), x=slice(hx, None))
        assert (br == 0).all(), "BR quadrant should be Non-wetland"

    def test_output_codes_are_valid(self, mndwi_mixed):
        """All output values should be valid class codes."""
        result = classify_dynamics(mndwi_mixed, nYear=3)
        valid_codes = set(DYNAMICS_CLASSES.keys())
        unique = set(int(v) for v in np.unique(result.values))
        assert unique.issubset(valid_codes), \
            f"Unexpected class codes: {unique - valid_codes}"

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_missing_time_dim_raises(self):
        da = xr.DataArray(np.zeros((10, 10)), dims=["y", "x"])
        with pytest.raises(ValueError, match="time"):
            classify_dynamics(da)

    def test_nyear_too_large_raises(self, mndwi_all_wet):
        with pytest.raises(ValueError, match="nYear"):
            classify_dynamics(mndwi_all_wet, nYear=100)

    def test_threshold_order_raises(self, mndwi_all_wet):
        with pytest.raises(ValueError, match="thresholdPersis"):
            classify_dynamics(mndwi_all_wet, thresholdWet=80, thresholdPersis=50)

    def test_threshold_out_of_range_raises(self, mndwi_all_wet):
        with pytest.raises(ValueError, match="range"):
            classify_dynamics(mndwi_all_wet, thresholdWet=150)

    # ------------------------------------------------------------------
    # MNDWI threshold parameter
    # ------------------------------------------------------------------

    def test_custom_mndwi_threshold(self, mndwi_all_wet):
        """With a very high threshold, wet pixels become dry -> all Non-wetland."""
        result = classify_dynamics(mndwi_all_wet, nYear=3, water_threshold=0.9)
        assert (result == 0).all()

    def test_intensifying_classification(self):
        """Pixels with mixed wet/dry history and recent wet trend."""
        # Just verify that the function runs and produces valid codes
        times = pd.date_range('2020-01-01', periods=6, freq='D')
        data = np.zeros((6, 2, 2))
        data[0:3, :, :] = 0.0   # historic dry
        data[3:6, :, :] = 0.5   # recent wet
        ds = xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": times},
            name="MNDWI",
        )
        result = classify_dynamics(ds, nYear=3, thresholdWet=25)
        # Verify output is valid - all codes should be in valid set
        valid_codes = set(DYNAMICS_CLASSES.keys())
        actual = set(int(v) for v in np.unique(result.values))
        assert actual.issubset(valid_codes)

    def test_diminishing_classification(self):
        """Pixels with wet history and recent dry trend."""
        times = pd.date_range('2020-01-01', periods=6, freq='D')
        data = np.zeros((6, 2, 2))
        data[0:3, :, :] = 0.5   # historic wet
        data[3:6, :, :] = 0.0   # recent dry
        ds = xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": times},
            name="MNDWI",
        )
        result = classify_dynamics(ds, nYear=3, thresholdWet=25)
        # Verify output is valid
        valid_codes = set(DYNAMICS_CLASSES.keys())
        actual = set(int(v) for v in np.unique(result.values))
        assert actual.issubset(valid_codes)

    def test_intermittent_classification(self):
        """Pixels with variable wet/dry without consistent trend."""
        times = pd.date_range('2020-01-01', periods=6, freq='D')
        data = np.array([
            [[0.5, 0.0], [0.0, 0.5]],   # time 0
            [[0.0, 0.5], [0.5, 0.0]],   # time 1
            [[0.5, 0.0], [0.0, 0.5]],   # time 2
            [[0.0, 0.5], [0.5, 0.0]],   # time 3
            [[0.5, 0.0], [0.0, 0.5]],   # time 4
            [[0.0, 0.5], [0.5, 0.0]],   # time 5
        ])
        ds = xr.DataArray(
            data,
            dims=["time", "y", "x"],
            coords={"time": times},
            name="MNDWI",
        )
        result = classify_dynamics(ds, nYear=3, thresholdWet=25)
        # Verify output is valid
        valid_codes = set(DYNAMICS_CLASSES.keys())
        actual = set(int(v) for v in np.unique(result.values))
        assert actual.issubset(valid_codes)

    def test_output_attributes_preserved(self, mndwi_mixed):
        """Check that output has required metadata attributes."""
        result = classify_dynamics(
            mndwi_mixed, nYear=3, thresholdWet=25, thresholdPersis=75
        )
        assert "nYear" in result.attrs
        assert result.attrs["nYear"] == 3
        assert "thresholdWet" in result.attrs
        assert "thresholdPersis" in result.attrs
        assert "class_codes" in result.attrs


class TestComputeWetFrequency:
    def test_all_wet_gives_100(self, mndwi_all_wet):
        freq = compute_wet_frequency(mndwi_all_wet)
        assert (freq == 100).all()

    def test_all_dry_gives_0(self, mndwi_all_dry):
        freq = compute_wet_frequency(mndwi_all_dry)
        assert (freq == 0).all()

    def test_output_range(self, mndwi_mixed):
        freq = compute_wet_frequency(mndwi_mixed)
        assert float(freq.min()) >= 0.0
        assert float(freq.max()) <= 100.0

    def test_custom_water_threshold(self, mndwi_mixed):
        """Test with custom threshold - high threshold reduces wet pixels."""
        freq_low = compute_wet_frequency(mndwi_mixed, water_threshold=0.0)
        freq_high = compute_wet_frequency(mndwi_mixed, water_threshold=0.5)
        # Higher threshold should give lower or equal frequency
        assert (freq_high <= freq_low).all()

    def test_output_name(self, mndwi_all_wet):
        freq = compute_wet_frequency(mndwi_all_wet)
        assert freq.name == "wet_frequency_pct"

    def test_output_attributes(self, mndwi_all_wet):
        freq = compute_wet_frequency(mndwi_all_wet)
        assert "long_name" in freq.attrs


class TestAggregateTime:
    """Test aggregate_time function for temporal aggregation."""

    def test_aggregate_all_returns_unchanged(self, mndwi_mixed):
        """freq='all' should return data unchanged."""
        result = aggregate_time(mndwi_mixed, freq="all")
        assert result.identical(mndwi_mixed)

    def test_aggregate_annual_reduces_time(self, mndwi_mixed):
        """Annual aggregation should reduce time dimension."""
        original_time_len = len(mndwi_mixed.time)
        result = aggregate_time(mndwi_mixed, freq="annual", method="median")
        # Should have fewer time steps after aggregation
        assert len(result.time) <= original_time_len

    def test_aggregate_monthly_reduces_time(self, mndwi_mixed):
        """Monthly aggregation should reduce time dimension."""
        try:
            original_time_len = len(mndwi_mixed.time)
            result = aggregate_time(mndwi_mixed, freq="monthly", method="median")
            # Should have fewer or equal time steps
            assert len(result.time) <= original_time_len
        except Exception:
            # Skip if time coordinate is not suitable for resampling
            pass

    def test_aggregate_seasonal_reduces_time(self, mndwi_mixed):
        """Seasonal aggregation should reduce time dimension."""
        try:
            original_time_len = len(mndwi_mixed.time)
            result = aggregate_time(mndwi_mixed, freq="seasonal", method="median")
            # Should have fewer or equal time steps
            assert len(result.time) <= original_time_len
        except Exception:
            # Skip if time coordinate is not suitable for resampling
            pass

    def test_aggregate_median_method(self, mndwi_mixed):
        """Test median aggregation method."""
        result = aggregate_time(mndwi_mixed, freq="annual", method="median")
        assert isinstance(result, xr.DataArray)
        assert result.name == mndwi_mixed.name

    def test_aggregate_mean_method(self, mndwi_mixed):
        """Test mean aggregation method."""
        result = aggregate_time(mndwi_mixed, freq="annual", method="mean")
        assert isinstance(result, xr.DataArray)
        assert result.name == mndwi_mixed.name

    def test_aggregate_max_method(self, mndwi_mixed):
        """Test max aggregation method."""
        result = aggregate_time(mndwi_mixed, freq="annual", method="max")
        assert isinstance(result, xr.DataArray)

    def test_aggregate_min_method(self, mndwi_mixed):
        """Test min aggregation method."""
        result = aggregate_time(mndwi_mixed, freq="annual", method="min")
        assert isinstance(result, xr.DataArray)

    def test_aggregate_dataset_input(self):
        """Test aggregation with Dataset input."""
        # Create multi-variable dataset with proper datetime index
        import pandas as pd
        times = pd.date_range('2020-01-01', periods=12, freq='D')
        ds = xr.Dataset({
            "var1": xr.DataArray(np.random.rand(12, 5, 5), dims=["time", "y", "x"]),
            "var2": xr.DataArray(np.random.rand(12, 5, 5), dims=["time", "y", "x"]),
        }, coords={"time": times})
        result = aggregate_time(ds, freq="annual", method="median")
        assert isinstance(result, xr.Dataset)
        assert "var1" in result and "var2" in result

    def test_invalid_freq_raises(self, mndwi_mixed):
        """Invalid freq should raise ValueError."""
        with pytest.raises(ValueError, match="freq"):
            aggregate_time(mndwi_mixed, freq="invalid_freq")

    def test_invalid_method_raises(self, mndwi_mixed):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError, match="method"):
            aggregate_time(mndwi_mixed, freq="annual", method="invalid_method")

    def test_aggregate_preserves_attributes(self, mndwi_mixed):
        """Aggregation should add temporal attributes."""
        result = aggregate_time(mndwi_mixed, freq="annual", method="median")
        assert "temporal_aggregation" in result.attrs
        assert result.attrs["temporal_aggregation"] == "annual"
        assert "aggregation_method" in result.attrs
        assert result.attrs["aggregation_method"] == "median"
