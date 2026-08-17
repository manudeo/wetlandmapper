"""Offline tests for climate-adaptive compositing decision logic."""

import numpy as np
import pytest

from wetlandmapper import gee


def test_month_validity_respects_precip_and_temp_thresholds():
    precip = np.array([10.0, 25.0, 30.0, 40.0])
    temp = np.array([12.0, 3.0, 8.0, 6.0])

    valid = gee._climate_valid_month_mask_numpy(
        precip,
        temp,
        min_precip_mm=20.0,
        min_temp_c=5.0,
    )

    assert valid.dtype == np.bool_
    assert valid.shape == (4,)
    assert valid.tolist() == [False, False, True, True]


def test_wettest_selection_uses_wettest_valid_month_not_global_wettest():
    values = np.array([[0.1], [0.7], [0.5], [0.2]])
    precip = np.array([[5.0], [100.0], [40.0], [30.0]])
    valid = np.array([[False], [False], [True], [True]])

    selected = gee._select_wettest_valid_month_numpy(values, precip, valid)

    assert selected.shape == (1,)
    assert float(selected[0]) == pytest.approx(0.5)
    assert float(selected[0]) != pytest.approx(0.7)


def test_wettest_selection_returns_nan_when_no_valid_month_exists():
    values = np.array([[0.1, 0.2], [0.3, 0.4]])
    precip = np.array([[10.0, 20.0], [15.0, 25.0]])
    valid = np.zeros_like(values, dtype=bool)

    selected = gee._select_wettest_valid_month_numpy(values, precip, valid)

    assert selected.shape == (2,)
    assert np.isnan(selected[0])
    assert np.isnan(selected[1])


def test_hydroperiod_valid_policy_is_invariant_to_cloud_fraction():
    wet_months = np.array([1.0, 2.0])
    observed_months = np.array([1.0, 2.0])
    climate_valid_months = np.array([2.0, 2.0])

    equiv = gee._hydroperiod_equivalent_months_numpy(
        wet_months,
        observed_months,
        climate_valid_months,
        hydroperiod_nan_policy="valid",
    )

    assert float(equiv[0]) == pytest.approx(2.0)
    assert float(equiv[1]) == pytest.approx(2.0)
    assert bool(equiv[0] >= 2.0)
    assert bool(equiv[1] >= 2.0)


def test_hydroperiod_total_policy_is_cloud_fraction_sensitive():
    wet_months = np.array([1.0, 2.0])
    observed_months = np.array([1.0, 2.0])
    climate_valid_months = np.array([2.0, 2.0])

    equiv = gee._hydroperiod_equivalent_months_numpy(
        wet_months,
        observed_months,
        climate_valid_months,
        hydroperiod_nan_policy="total",
    )

    assert float(equiv[0]) == pytest.approx(1.0)
    assert float(equiv[1]) == pytest.approx(2.0)
    assert bool(equiv[0] <= equiv[1])


def test_hydroperiod_scales_to_climate_valid_season_not_calendar_year():
    wet_months = np.array([1.0, 1.0])
    observed_months = np.array([1.0, 1.0])
    climate_valid_months = np.array([3.0, 6.0])

    equiv = gee._hydroperiod_equivalent_months_numpy(
        wet_months,
        observed_months,
        climate_valid_months,
        hydroperiod_nan_policy="valid",
    )

    assert float(equiv[0]) == pytest.approx(3.0)
    assert float(equiv[1]) == pytest.approx(6.0)


def test_valid_and_total_agree_when_no_cloud_loss():
    wet_months = np.array([1.0, 2.0, 4.0])
    observed_months = np.array([3.0, 5.0, 6.0])
    climate_valid_months = observed_months.copy()

    equiv_valid = gee._hydroperiod_equivalent_months_numpy(
        wet_months,
        observed_months,
        climate_valid_months,
        hydroperiod_nan_policy="valid",
    )
    equiv_total = gee._hydroperiod_equivalent_months_numpy(
        wet_months,
        observed_months,
        climate_valid_months,
        hydroperiod_nan_policy="total",
    )

    assert np.allclose(equiv_valid, equiv_total)


def test_mean_hydroperiod_excludes_years_with_zero_valid_months():
    yearly_equiv = np.array(
        [
            [[8.0, 10.0]],
            [[0.0, 0.0]],
            [[12.0, 6.0]],
        ]
    )
    yearly_valid = np.array(
        [
            [[4.0, 4.0]],
            [[0.0, 0.0]],
            [[6.0, 3.0]],
        ]
    )

    mean_equiv = gee._mean_hydroperiod_over_nonempty_years_numpy(
        yearly_equiv,
        yearly_valid,
    )

    assert mean_equiv.shape == (1, 2)
    assert float(mean_equiv[0, 0]) == pytest.approx(10.0)
    assert float(mean_equiv[0, 1]) == pytest.approx(8.0)


def test_validate_climate_adaptive_params_rejects_bad_inputs():
    with pytest.raises(ValueError, match="min_precip_mm"):
        gee._validate_climate_adaptive_params(
            min_precip_mm=-1.0,
            hydroperiod_months=1,
            hydroperiod_nan_policy="valid",
        )

    with pytest.raises(ValueError, match="hydroperiod_months"):
        gee._validate_climate_adaptive_params(
            min_precip_mm=0.0,
            hydroperiod_months=1.5,
            hydroperiod_nan_policy="valid",
        )

    with pytest.raises(ValueError, match="hydroperiod_nan_policy"):
        gee._validate_climate_adaptive_params(
            min_precip_mm=0.0,
            hydroperiod_months=1,
            hydroperiod_nan_policy="bad",
        )


def test_era5_unit_conversions_match_expected_scaling():
    precip_mm, temp_c = gee._era5_unit_conversions_numpy(
        np.array([0.001, 0.02]),
        np.array([273.15, 293.15]),
    )
    assert np.allclose(precip_mm, [1.0, 20.0])
    assert np.allclose(temp_c, [0.0, 20.0])


def test_year_month_join_key_uses_zero_padded_month():
    assert gee._format_year_month_key(2024, 1) == "2024_01"
    assert gee._format_year_month_key(2024, 11) == "2024_11"
