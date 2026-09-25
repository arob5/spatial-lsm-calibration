"""The attribute-carrying arithmetic in ``sipnet_calibration.observation.units``."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pysipnet import niwot_reference_output
from pysipnet.units import conversion_factor, convert_dataarray_units

from sipnet_calibration.fields import label_run
from sipnet_calibration.observation import add, divide, multiply, step_length, subtract


@pytest.fixture(scope="module")
def niwot():
    """Niwot output labeled as one run, with the variables these tests combine."""
    out = niwot_reference_output()
    return label_run(
        out.select(["leaf_carbon", "net_ecosystem_exchange", "wood_carbon", "soil_wetness_fraction"]),
        site=1,
        member=0,
        site_table=_site_table(),
    )


def _site_table():
    import pandas as pd

    return pd.DataFrame({"site_id": [1], "lon": [-105.5], "lat": [40.0]}).set_index("site_id", drop=False)


def _parameter(value, units, constituent="C", name="leaf_carbon_per_area"):
    attrs = {"units": units, "long_name": name}
    if constituent:
        attrs["constituent"] = constituent
    return xr.DataArray(value, name=name, attrs=attrs)


class TestStepLength:
    def test_is_the_declared_length_in_days(self, niwot):
        nee = niwot["net_ecosystem_exchange"]
        days = step_length(nee)
        assert days.attrs["units"] == "d"
        expected = nee["time_step_length"].values.astype("timedelta64[ns]").astype("int64") / 86_400e9
        np.testing.assert_allclose(days.values, expected)

    def test_other_units(self, niwot):
        nee = niwot["net_ecosystem_exchange"]
        assert np.allclose(step_length(nee, "h").values, step_length(nee).values * 24)
        assert np.allclose(step_length(nee, "s").values, step_length(nee).values * 86_400)

    def test_refuses_an_array_with_no_declared_lengths(self):
        array = xr.DataArray([1.0, 2.0], dims="time", attrs={"units": "g m-2"})
        with pytest.raises(ValueError, match="time_step_length"):
            step_length(array)

    def test_refuses_an_unknown_unit(self, niwot):
        with pytest.raises(ValueError, match="units must be one of"):
            step_length(niwot["net_ecosystem_exchange"], "fortnight")


class TestDivide:
    def test_a_total_over_the_step_length_is_a_daily_rate(self, niwot):
        nee = niwot["net_ecosystem_exchange"]
        rate = divide(nee, step_length(nee))
        assert rate.attrs["units"] == "g m-2 d-1"
        assert rate.attrs["constituent"] == "C"
        assert rate.attrs["kind"] == "daily_rate"
        np.testing.assert_allclose(rate.values, nee.values / step_length(nee).values)

    def test_the_rate_converts_to_the_observed_nee_units(self, niwot):
        nee = niwot["net_ecosystem_exchange"]
        rate = divide(nee, step_length(nee))
        observed = convert_dataarray_units(rate, to_units="umol m-2 s-1", to_constituent="CO2")
        factor = conversion_factor(
            units="g m-2 d-1", constituent="C", to_units="umol m-2 s-1", to_constituent="CO2"
        )
        np.testing.assert_allclose(observed.values, rate.values * factor)
        assert observed.attrs["units"] == "umol m-2 s-1"
        assert observed.attrs["constituent"] == "CO2"

    def test_leaf_carbon_over_leaf_carbon_per_area_is_dimensionless_with_no_constituent(self, niwot):
        lai = divide(niwot["leaf_carbon"], _parameter(270.0, "g m-2"))
        assert lai.attrs["units"] == "1"
        assert "constituent" not in lai.attrs
        assert lai.attrs["kind"] == "timestep_end_state"
        np.testing.assert_allclose(lai.values, niwot["leaf_carbon"].values / 270.0)
        # and it converts to the observation's unit by a factor of one
        as_observed = convert_dataarray_units(lai, to_units="m2 m-2")
        np.testing.assert_allclose(as_observed.values, lai.values)

    def test_a_per_site_parameter_broadcasts(self, niwot):
        stacked = xr.concat([niwot["leaf_carbon"].assign_coords(site=1), niwot["leaf_carbon"].assign_coords(site=2)], dim="site")
        stacked.attrs = niwot["leaf_carbon"].attrs
        per_area = xr.DataArray([270.0, 135.0], dims="site", coords={"site": [1, 2]}, attrs={"units": "g m-2", "constituent": "C"})
        lai = divide(stacked, per_area)
        np.testing.assert_allclose(lai.sel(site=2).values, 2 * lai.sel(site=1).values)

    def test_refuses_a_denominator_with_a_constituent_the_numerator_lacks(self, niwot):
        with pytest.raises(ValueError, match="no constituent"):
            divide(niwot["soil_wetness_fraction"], _parameter(1.0, "g m-2"))

    def test_refuses_two_different_constituents(self, niwot):
        with pytest.raises(ValueError, match="convert one of"):
            divide(niwot["leaf_carbon"], _parameter(1.0, "kg m-2", constituent="N"))

    def test_refuses_two_model_variables(self, niwot):
        with pytest.raises(ValueError, match="only one operand"):
            divide(niwot["leaf_carbon"], niwot["wood_carbon"])

    def test_refuses_an_operand_without_units(self, niwot):
        with pytest.raises(ValueError, match="no 'units' attribute"):
            divide(niwot["leaf_carbon"], xr.DataArray(2.0))


class TestMultiply:
    def test_a_rate_times_the_step_length_is_a_total_again(self, niwot):
        nee = niwot["net_ecosystem_exchange"]
        back = multiply(divide(nee, step_length(nee)), step_length(nee))
        assert back.attrs["units"] == "g m-2"
        assert back.attrs["kind"] == "timestep_total"
        np.testing.assert_allclose(back.values, nee.values)

    def test_a_plain_number_is_a_dimensionless_factor(self, niwot):
        doubled = multiply(niwot["wood_carbon"], 2)
        assert doubled.attrs["units"] == "g m-2"
        assert doubled.attrs["constituent"] == "C"
        assert doubled.attrs["kind"] == "timestep_end_state"
        np.testing.assert_allclose(doubled.values, 2 * niwot["wood_carbon"].values)

    def test_refuses_two_constituents(self, niwot):
        with pytest.raises(ValueError, match="at most one operand"):
            multiply(niwot["wood_carbon"], _parameter(1.0, "1", constituent="N"))

    def test_refuses_a_boolean_operand(self, niwot):
        with pytest.raises(TypeError):
            multiply(niwot["wood_carbon"], True)


class TestAddSubtract:
    def test_two_pools_add(self, niwot):
        total = add(niwot["wood_carbon"], niwot["leaf_carbon"])
        assert total.attrs["units"] == "g m-2"
        assert total.attrs["constituent"] == "C"
        assert total.attrs["kind"] == "timestep_end_state"
        np.testing.assert_allclose(total.values, niwot["wood_carbon"].values + niwot["leaf_carbon"].values)

    def test_subtract(self, niwot):
        diff = subtract(niwot["wood_carbon"], niwot["leaf_carbon"])
        np.testing.assert_allclose(diff.values, niwot["wood_carbon"].values - niwot["leaf_carbon"].values)

    def test_refuses_different_kinds(self, niwot):
        with pytest.raises(ValueError, match="agree in units, constituent and kind"):
            add(niwot["wood_carbon"], niwot["net_ecosystem_exchange"])

    def test_refuses_different_units(self, niwot):
        with pytest.raises(ValueError, match="agree in units"):
            add(niwot["wood_carbon"], _parameter(1.0, "kg m-2"))


class TestUnitStrings:
    def test_exponents_combine_by_symbol(self):
        a = xr.DataArray(1.0, attrs={"units": "g m-2"})
        b = xr.DataArray(1.0, attrs={"units": "m2 s-1"})
        assert multiply(a, b).attrs["units"] == "g s-1"
        assert divide(a, b).attrs["units"] == "g m-4 s"

    def test_identical_units_cancel_to_one(self):
        a = xr.DataArray(2.0, attrs={"units": "kg m-2"})
        assert divide(a, a).attrs["units"] == "1"

    def test_the_result_carries_no_stale_attributes(self, niwot):
        rate = divide(niwot["net_ecosystem_exchange"], step_length(niwot["net_ecosystem_exchange"]))
        assert "output_decimals" not in rate.attrs
        assert "sipnet_name" not in rate.attrs
        assert rate.attrs["derivation"] == "net_ecosystem_exchange / time_step_length"


class TestKindAlgebra:
    def test_dividing_by_a_model_variable_is_refused(self, niwot):
        with pytest.raises(ValueError, match="cannot divide by a model variable"):
            divide(_parameter(1.0, "1", constituent=""), niwot["net_ecosystem_exchange"])

    def test_a_pool_times_a_per_day_parameter_is_refused(self, niwot):
        turnover = _parameter(0.01, "d-1", constituent="", name="leaf_turnover_rate")
        with pytest.raises(ValueError, match="no pySIPNET kind names the result"):
            multiply(niwot["wood_carbon"], turnover)

    def test_a_total_times_a_per_day_is_a_rate_and_back(self, niwot):
        nee = niwot["net_ecosystem_exchange"]
        per_day = _parameter(1.0, "d-1", constituent="")
        rate = multiply(nee, per_day)
        assert rate.attrs["kind"] == "daily_rate" and rate.attrs["units"] == "g m-2 d-1"
        total = divide(rate, per_day)
        assert total.attrs["kind"] == "timestep_total" and total.attrs["units"] == "g m-2"

    def test_a_rate_carries_the_rates_time_reference(self, niwot):
        nee = niwot["net_ecosystem_exchange"]
        rate = divide(nee, step_length(nee))
        assert rate.attrs["time_reference"] != nee.attrs["time_reference"]
        assert rate.attrs["cell_methods"] == "time: mean"
        assert rate.attrs["sign_convention"] == nee.attrs["sign_convention"]

    def test_a_missing_step_length_is_missing(self, niwot):
        nee = niwot["net_ecosystem_exchange"].copy()
        lengths = nee["time_step_length"].values.copy()
        lengths[5] = np.timedelta64("NaT", "ns")
        nee = nee.assign_coords(time_step_length=("time", lengths))
        days = step_length(nee)
        assert np.isnan(days.values[5]) and np.isfinite(days.values[4])


class TestConstituentLeadsTheUnits:
    """pySIPNET reads a constituent as qualifying the first unit token."""

    def test_a_product_puts_the_constituents_unit_first_in_either_order(self, niwot):
        nee = niwot["net_ecosystem_exchange"]
        per_day = _parameter(1.0, "d-1", constituent="")
        for product in (multiply(nee, per_day), multiply(per_day, nee)):
            assert product.attrs["units"] == "g m-2 d-1"
            converted = convert_dataarray_units(product, to_units="umol m-2 s-1", to_constituent="CO2")
            assert converted.attrs["constituent"] == "CO2"
