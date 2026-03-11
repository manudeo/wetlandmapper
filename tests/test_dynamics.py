"""Tests for wetlandmapper.dynamics"""

import numpy as np
import pytest
import xarray as xr

from wetlandmapper import classify_dynamics, DYNAMICS_CLASSES
from wetlandmapper.dynamics import compute_wet_frequency


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
        """With a very high MNDWI threshold, wet pixels become dry → all Non-wetland."""
        result = classify_dynamics(mndwi_all_wet, nYear=3, mndwi_threshold=0.9)
        assert (result == 0).all()


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
