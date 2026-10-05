"""The observation vector as components of a model: the sources' standard
deviations, observation dims, constants and prediction names, the observed
values on those dims and back to fields, and ``observation.model``'s
components, bound and conditioned on in a small model.
"""

from __future__ import annotations

import pickle
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats
from tensorflow_probability.substrates import jax as tfp

from conftest import dated_observed_values, static_observed_values, windowed_observed_values
from sipnet_calibration.observation import (
    CALENDAR_YEAR,
    OBSERVED,
    STANDARD_DEVIATION,
    TIME_SINCE_EPOCH,
    WINDOW_LENGTH,
    YEAR,
    ObservationSource,
    ObservationVector,
    ReduceOverRun,
    ReduceOverWindows,
    SelectTimestep,
)
from sipnet_calibration.observation.model import observed_components, prediction_components
from sipnet_calibration.probability import REAL, ArraySpec, DeterministicSpec, FactorSpec, condition_on, joint
from sipnet_calibration.probability.labels import as_constants, as_coords

tfd = tfp.distributions

TIMES = pd.date_range("2012-12-30", periods=3, freq="D")
#: Site 2 observes the last two times, site 1 the first and the last.
LAI = [[1.0, np.nan, 2.0], [np.nan, 3.0, 4.0]]


@pytest.fixture
def lai() -> xr.DataArray:
    return windowed_observed_values([2, 1], TIMES, values=LAI[::-1], name="lai")


@pytest.fixture
def lai_standard_deviation(lai) -> xr.DataArray:
    return lai.copy(data=np.array([[0.4, 0.5, 0.6], [0.1, 0.2, 0.3]]))


@pytest.fixture
def soil() -> xr.DataArray:
    return static_observed_values([1, 2], values=[5.0, np.nan], name="soil")


@pytest.fixture
def vector(lai, lai_standard_deviation, soil) -> ObservationVector:
    return ObservationVector(observation_sources=[
        ObservationSource(observation_source_name="lai", observed_values=lai,
                          standard_deviation=lai_standard_deviation, operator=ReduceOverWindows("leaf_carbon", "mean")),
        ObservationSource(observation_source_name="soil", observed_values=soil,
                          standard_deviation=soil.copy(data=[0.5, np.nan]), operator=ReduceOverRun("soil_carbon", "mean")),
    ])


def source_with(observed: xr.DataArray, standard_deviation) -> ObservationSource:
    return ObservationSource(observation_source_name="lai", observed_values=observed,
                             standard_deviation=standard_deviation, operator=SelectTimestep("leaf_carbon"))


# ── the standard deviation ────────────────────────────────────────────────────


class TestStandardDeviation:
    def test_is_none_by_default(self, lai):
        assert ObservationSource(observation_source_name="lai", observed_values=lai,
                                 operator=SelectTimestep("leaf_carbon")).standard_deviation is None

    def test_is_on_the_observed_grid_and_missing_where_they_are(self, vector, lai):
        held = vector["lai"].standard_deviation
        observed = vector["lai"].observed_values
        assert held.dims == observed.dims
        np.testing.assert_array_equal(held["site"].values, [1, 2])
        np.testing.assert_array_equal(held.isnull().values, observed.isnull().values)
        assert float(held.sel(site=1, time=TIMES[0])) == 0.1
        assert held.attrs["units"] == "m2 m-2"

    def test_keeps_its_own_attributes(self, lai, lai_standard_deviation):
        source = source_with(lai, lai_standard_deviation.assign_attrs(long_name="reported deviation"))
        assert source.standard_deviation.attrs["long_name"] == "reported deviation"
        assert "long_name" not in source.observed_values.attrs

    def test_is_read_by_label_from_a_larger_grid(self, lai):
        larger = dated_observed_values([1, 2, 3], TIMES.append(pd.DatetimeIndex(["2013-06-01"])), name="lai")
        larger = larger.copy(data=np.arange(12.0).reshape(3, 4))
        held = source_with(lai, larger).standard_deviation
        assert float(held.sel(site=2, time=TIMES[2])) == 6.0

    def test_a_zero_is_allowed(self, lai, lai_standard_deviation):
        source = source_with(lai, lai_standard_deviation * 0.0)
        assert float(source.standard_deviation.max()) == 0.0

    def test_an_array_of_another_type_is_refused(self, lai):
        with pytest.raises(TypeError, match="give a DataArray"):
            source_with(lai, np.ones((2, 3)))

    def test_other_dims_are_refused(self, lai):
        with pytest.raises(ValueError, match="one standard deviation per observation"):
            source_with(lai, static_observed_values([1, 2], units="m2 m-2", constituent="", name="lai"))

    def test_other_units_are_refused(self, lai, lai_standard_deviation):
        other = lai_standard_deviation.assign_attrs(units="cm2 m-2")
        with pytest.raises(ValueError, match="convert it to the observed values' units"):
            source_with(lai, other)

    def test_units_equal_by_a_factor_of_one_are_accepted(self, lai, lai_standard_deviation):
        source_with(lai, lai_standard_deviation.assign_attrs(units="1"))

    def test_a_missing_label_is_refused(self, lai, lai_standard_deviation):
        with pytest.raises(KeyError, match="lacks the time label"):
            source_with(lai, lai_standard_deviation.isel(time=[0, 1]))

    def test_a_gap_at_an_observation_is_refused(self, lai, lai_standard_deviation):
        gappy = lai_standard_deviation.copy(data=np.where(np.isnan(LAI[::-1]), 0.1, np.nan))
        with pytest.raises(ValueError, match="missing at 4 observation"):
            source_with(lai, gappy)

    def test_a_gap_where_nothing_is_observed_is_allowed(self, lai, lai_standard_deviation):
        source_with(lai, lai_standard_deviation.where(lai.notnull()))

    def test_a_negative_value_is_refused(self, lai, lai_standard_deviation):
        with pytest.raises(ValueError, match="negative at 1 observation"):
            source_with(lai, lai_standard_deviation.where(lai_standard_deviation != 0.6, -0.6))

    def test_is_handed_out_read_only(self, vector):
        with pytest.raises(ValueError, match="read-only"):
            vector["lai"].standard_deviation.values[0, 0] = 9.0

    def test_follows_a_selection(self, vector):
        selected = vector.select(sites=[2], time=slice("2012-12-31", None))
        held = selected["lai"].standard_deviation
        np.testing.assert_array_equal(held.values, [[0.5, 0.6]])
        assert "soil" not in selected
        assert vector.restrict_to_sites([1, 99])["soil"].standard_deviation.values.tolist() == [0.5]

    def test_survives_pickling(self, vector):
        restored = pickle.loads(pickle.dumps(vector))
        xr.testing.assert_identical(restored["lai"].standard_deviation, vector["lai"].standard_deviation)


# ── observation dims and their labels ─────────────────────────────────────────


def test_observation_labels_are_sorted_site_time_pairs(vector):
    labels = vector["lai"].observation_labels
    assert list(labels.names) == ["site", "time"]
    assert labels.tolist() == [(1, TIMES[0]), (1, TIMES[2]), (2, TIMES[1]), (2, TIMES[2])]
    assert labels.levels[0].dtype == np.int32
    assert labels.levels[1].dtype == "datetime64[ns]"


def test_a_static_sources_labels_have_one_level(vector):
    labels = vector["soil"].observation_labels
    assert list(labels.names) == ["site"]
    assert labels.tolist() == [(1,)]


def test_coords_are_one_stacked_dim_per_source(vector):
    coords = vector.coords
    assert list(coords) == ["lai_observation", "soil_observation"]
    assert coords["lai_observation"].equals(vector["lai"].observation_labels)
    as_coords(coords)


def test_coords_hand_out_copies(vector):
    vector.coords["lai_observation"].names = ["a", "b"]
    assert list(vector.coords["lai_observation"].names) == ["site", "time"]


def test_dim_and_prediction_names_are_derived(vector):
    assert vector.observation_dim_name("soil") == "soil_observation"
    assert vector.prediction_name("lai") == "predicted_lai"
    for method in (vector.observation_dim_name, vector.prediction_name, vector.constants, vector.year_label_map):
        with pytest.raises(KeyError, match="no observation source 'nee'"):
            method("nee")


# ── constants ─────────────────────────────────────────────────────────────────


def test_a_dated_windowed_sources_constants(vector):
    constants = vector.constants("lai")
    assert list(constants) == [OBSERVED, STANDARD_DEVIATION, TIME_SINCE_EPOCH, CALENDAR_YEAR, WINDOW_LENGTH]
    for constant in constants.values():
        assert constant.dims == ("lai_observation",)
        assert constant.dtype == np.float64
        assert constant.indexes["lai_observation"].equals(vector.coords["lai_observation"])
    np.testing.assert_array_equal(constants[OBSERVED].values, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(constants[STANDARD_DEVIATION].values, [0.1, 0.3, 0.5, 0.6])
    times = vector.coords["lai_observation"].get_level_values("time")
    np.testing.assert_array_equal(constants[TIME_SINCE_EPOCH].values, [t.timestamp() for t in times])
    np.testing.assert_array_equal(constants[CALENDAR_YEAR].values, [2012.0, 2013.0, 2012.0, 2013.0])
    np.testing.assert_array_equal(constants[WINDOW_LENGTH].values, [86_400.0] * 4)
    assert constants[OBSERVED].attrs["units"] == "m2 m-2"
    assert constants[TIME_SINCE_EPOCH].attrs["units"] == "s"
    assert constants[STANDARD_DEVIATION].attrs == vector["lai"].standard_deviation.attrs
    as_constants(constants, message_name="constants")


def test_window_lengths_follow_each_observation(lai):
    starts = lai["window_end"].values - np.array([1, 2, 3], dtype="timedelta64[D]")
    unequal = lai.assign_coords(window_start=("time", starts))
    vector = ObservationVector(observation_sources=[ObservationSource(
        observation_source_name="lai", observed_values=unequal, operator=SelectTimestep("leaf_carbon"))])
    # Observations (1, t0), (1, t2), (2, t1), (2, t2): windows of 1, 3, 2 and 3 days.
    np.testing.assert_array_equal(vector.constants("lai")[WINDOW_LENGTH].values / 86_400.0, [1.0, 3.0, 2.0, 3.0])


def test_the_observed_constant_agrees_with_y_by_label(vector):
    observed = vector.constants("lai")[OBSERVED]
    index = vector.index
    for (site, time), value in zip(observed.indexes["lai_observation"], observed.values):
        assert vector.y[index.get_loc((site, "lai", time))] == value


def test_a_static_source_has_no_time_constants(vector):
    constants = vector.constants("soil")
    assert list(constants) == [OBSERVED, STANDARD_DEVIATION]
    assert constants[OBSERVED].values.tolist() == [5.0]


def test_a_source_without_windows_or_deviations_has_neither(lai):
    vector = ObservationVector(observation_sources=[ObservationSource(
        observation_source_name="lai", observed_values=lai.drop_vars(["window_start", "window_end"]),
        operator=SelectTimestep("leaf_carbon"))])
    assert list(vector.constants("lai")) == [OBSERVED, TIME_SINCE_EPOCH, CALENDAR_YEAR]


def test_the_year_label_map_holds_each_observations_year(vector):
    years = vector.year_label_map("lai")
    assert years.name == YEAR
    assert years.dims == ("lai_observation",)
    assert years.values.tolist() == ["2012", "2013", "2012", "2013"]
    with pytest.raises(ValueError, match="soil is static"):
        vector.year_label_map("soil")


# ── observed values on their dims, and back ───────────────────────────────────


def test_observed_values_by_component(vector):
    observed = vector.observed_values_by_component()
    assert list(observed) == ["lai", "soil"]
    assert observed["lai"].name == "lai"
    xr.testing.assert_identical(observed["lai"], vector.constants("lai")[OBSERVED].rename("lai"))


def test_to_fields_inverts_observed_values_by_component(vector):
    fields = vector.to_fields(vector.observed_values_by_component())
    for name in ("lai", "soil"):
        xr.testing.assert_identical(fields[name], vector[name].observed_values)


def test_to_fields_unstacks_a_batch(vector):
    observed = vector.observed_values_by_component()["lai"]
    batch = xr.concat([observed, 2 * observed], dim=pd.Index([0, 1], name="sample")).rename("predicted_lai")
    labeled = {"predicted_lai": batch, "theta": xr.DataArray(np.zeros((2, 3)), dims=("sample", "theta_entry"))}
    fields = vector.to_fields(labeled)
    assert list(fields) == ["predicted_lai"]
    field = fields["predicted_lai"]
    assert field.dims == ("sample", "site", "time")
    assert float(field.sel(sample=1, site=2, time=TIMES[2])) == 8.0
    assert np.isnan(float(field.sel(sample=1, site=1, time=TIMES[1])))
    assert field["sample"].attrs == vector.fields(np.zeros((1, vector.dimension)))["lai"]["sample"].attrs


def test_to_fields_fills_what_an_array_lacks(vector):
    observed = vector.observed_values_by_component()["lai"].isel(lai_observation=[0, 3])
    field = vector.to_fields({"lai": observed})["lai"]
    assert int(field.notnull().sum()) == 2


def test_to_fields_refuses_a_label_that_is_no_observation(vector):
    observed = vector.observed_values_by_component()["lai"]
    labels = pd.MultiIndex.from_tuples([(1, TIMES[1])], names=["site", "time"])
    stray = xr.DataArray([1.0], coords=xr.Coordinates.from_pandas_multiindex(labels, "lai_observation"),
                         dims="lai_observation")
    with pytest.raises(KeyError, match="which are not observations of"):
        vector.to_fields({"lai": xr.concat([observed, stray], dim="lai_observation")})


def test_to_fields_refuses_an_observation_given_twice(vector):
    observed = vector.observed_values_by_component()["lai"].isel(lai_observation=[0, 0, 1])
    with pytest.raises(ValueError, match="more than once"):
        vector.to_fields({"lai": observed})


def test_to_fields_refuses_a_dim_that_is_no_batch_dim(vector):
    observed = vector.observed_values_by_component()["lai"].expand_dims(part=["a"])
    with pytest.raises(ValueError, match="neither its observation dim nor a batch dim"):
        vector.to_fields({"lai": observed})


def test_to_fields_refuses_other_levels(vector):
    observed = vector.observed_values_by_component()["lai"]
    swapped = observed.copy().assign_coords(
        xr.Coordinates.from_pandas_multiindex(vector.coords["lai_observation"].swaplevel(), "lai_observation"))
    with pytest.raises(ValueError, match="label it with the vector's coords"):
        vector.to_fields({"lai": swapped})


def test_to_fields_refuses_an_array_on_two_observation_dims(vector):
    # Labeled, two observation dims would share their site level, which
    # xarray refuses; unlabeled, the array is still refused.
    both = xr.DataArray(np.zeros((4, 1)), dims=("lai_observation", "soil_observation"))
    with pytest.raises(ValueError, match="on its own dim alone"):
        vector.to_fields({"both": both})


def test_to_fields_refuses_an_unlabeled_observation_dim(vector):
    with pytest.raises(ValueError, match="a plain index"):
        vector.to_fields({"lai": xr.DataArray(np.zeros(4), dims="lai_observation")})


def test_to_fields_refuses_what_is_not_an_array(vector):
    with pytest.raises(TypeError, match="labeled must be a mapping"):
        vector.to_fields([1.0])
    with pytest.raises(TypeError, match="expected a DataArray"):
        vector.to_fields({"lai": [1.0]})


# ── other observed values ─────────────────────────────────────────────────────


def test_with_observed_values_replaces_the_values_and_keeps_the_rest(vector, lai):
    twin = vector.with_observed_values({"lai": lai * 10.0})
    np.testing.assert_array_equal(twin.constants("lai")[OBSERVED].values, [10.0, 20.0, 30.0, 40.0])
    xr.testing.assert_identical(twin["lai"].standard_deviation, vector["lai"].standard_deviation)
    assert twin["lai"].operator is vector["lai"].operator
    assert twin["soil"] is vector["soil"]
    assert twin.coords["lai_observation"].equals(vector.coords["lai_observation"])


def test_with_observed_values_refuses_other_observations(vector, lai):
    with pytest.raises(ValueError, match="at the same observations"):
        vector.with_observed_values({"lai": lai.fillna(7.0)})


def test_with_observed_values_refuses_an_unknown_source_and_a_non_mapping(vector, lai):
    with pytest.raises(KeyError, match="no observation source 'nee'"):
        vector.with_observed_values({"nee": lai})
    with pytest.raises(TypeError, match="values must be a mapping"):
        vector.with_observed_values([lai])


@pytest.mark.parametrize(
    ("change", "what"),
    [
        (lambda a: a.drop_vars(["window_start", "window_end"]), "window_end"),
        (lambda a: a.assign_coords(window_start=a["window_start"] - pd.Timedelta("7D")), "window_start"),
        (lambda a: a.assign_coords(lon=a["lon"] + 3.0), "lon"),
        (lambda a: a.assign_attrs(units="cm2 m-2"), "units"),
    ],
)
def test_with_observed_values_refuses_other_coordinates_or_units(vector, lai, change, what):
    with pytest.raises(ValueError, match=f"differ from the source's in .*{what}"):
        vector.with_observed_values({"lai": change(lai)})


# ── observation.model ─────────────────────────────────────────────────────────


def test_the_observed_and_prediction_components(vector):
    observed = observed_components(vector)
    predicted = prediction_components(vector)
    assert [spec.name for spec in observed] == ["lai", "soil"]
    assert [spec.name for spec in predicted] == ["predicted_lai", "predicted_soil"]
    for spec, units, dim in zip(observed, ("m2 m-2", "Mg ha-1"), ("lai_observation", "soil_observation")):
        assert (spec.units, spec.support, spec.indexed_by, spec.shape) == (units, REAL, (dim,), ())
    assert predicted[1].indexed_by == ("soil_observation",)


def test_the_observation_package_does_not_import_its_model_module():
    code = (
        "import sys; import sipnet_calibration.observation; "
        "print('sipnet_calibration.observation.model' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


def test_a_model_over_the_components_binds_and_conditions(vector):
    """A Gaussian noise factor on the observed LAI, centered on a prediction
    computed from one parameter, reads the vector's constants and observed
    values, and its density is the closed form's."""
    (observed_lai, _), (predicted_lai, _) = observed_components(vector), prediction_components(vector)
    constants = vector.constants("lai")
    model = joint(
        FactorSpec(observed_lai, law=lambda predicted_lai, standard_deviation: tfd.Independent(
            tfd.Normal(predicted_lai, standard_deviation), reinterpreted_batch_ndims=1),
            constants={STANDARD_DEVIATION: constants[STANDARD_DEVIATION]}),
        DeterministicSpec(predicted_lai, function=lambda offset, calendar_year: offset + 0.0 * calendar_year,
                          constants={CALENDAR_YEAR: constants[CALENDAR_YEAR]}),
        FactorSpec(ArraySpec("offset", units="m2 m-2"), law=tfd.Normal(jnp.float64(2.0), jnp.float64(1.0))),
    ).bind(coords=vector.coords)
    posterior = condition_on(model, {"lai": vector.observed_values_by_component()["lai"]})
    assert posterior.parameter_names == ("offset",)
    theta = jnp.array([[0.5], [2.5]])
    expected = [
        stats.norm(2.0, 1.0).logpdf(x) + stats.norm(x, [0.1, 0.3, 0.5, 0.6]).logpdf([1.0, 2.0, 3.0, 4.0]).sum()
        for x in (0.5, 2.5)
    ]
    np.testing.assert_allclose(posterior.log_density(theta), expected, rtol=1e-12)
