"""Tests for the variable registry.

The registry is a literal mapping, so what is worth testing about it is not
that it can be read but that it agrees with everything else that states the
same facts. Two of its fields are duplicated by the readers -- ``drivers``
writes an ``aggregation`` and a ``units`` attribute onto every driver field,
and ``constraints`` a ``units`` attribute onto every constraint field -- and
the whole reason ``aggregate_time`` reads the registry rather than those
attributes is that one of them could otherwise drift from the other silently.
The agreement tests are what makes the duplication safe: neither side can be
changed alone.

The rest is the invariants the consumers rely on: that every ``agg`` is a rule
that ``aggregate_time`` recognizes, that every ``cmap`` is a colormap
matplotlib has, and that a center is set exactly where a zero means
something.
"""

from __future__ import annotations

import dataclasses
import re

import matplotlib
import pytest

from sipnet_calibration.constraints import CONSTRAINT_VARIABLE_ATTRS
from sipnet_calibration.drivers import DRIVER_VARIABLE_ATTRS
from sipnet_calibration.observation_operators import AGGREGATION_METHODS
from sipnet_calibration.variable_registry import (
    AGGREGATION_RULES,
    GRAMS_CARBON_PER_MICROMOLE_CO2,
    INSTANTANEOUS,
    VARIABLES,
    VarSpec,
    variable_spec,
)

#: Variables whose zero is a real boundary rather than an arbitrary origin, so
#: a diverging colormap centered there is correctness and not decoration.
#: Written out here rather than derived from the registry, so that the test
#: below compares two independent statements.
CENTERED_ON_ZERO = {"nee", "air_temperature", "soil_temperature"}

#: The processed naming convention: lower case, digits and underscores.
PROCESSED_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class TestTheEntries:
    def test_every_agg_is_a_known_rule(self):
        """An unknown rule would raise only when someone aggregated it."""
        for name, spec in VARIABLES.items():
            assert spec.agg in AGGREGATION_RULES, name

    def test_every_rule_is_either_a_method_or_the_refusal(self):
        """``AGGREGATION_METHODS`` is derived from ``AGGREGATION_RULES``, so
        this pins that the vocabulary is one and not two."""
        assert set(AGGREGATION_METHODS) | {INSTANTANEOUS} == set(AGGREGATION_RULES)
        assert INSTANTANEOUS not in AGGREGATION_METHODS

    def test_every_entry_has_a_nonempty_label_and_units(self):
        """An empty unit string would pass a naive validate_field check."""
        for name, spec in VARIABLES.items():
            assert isinstance(spec.label, str) and spec.label.strip(), name
            assert isinstance(spec.units, str) and spec.units.strip(), name

    def test_every_cmap_names_a_colormap_matplotlib_has(self):
        """A typo would raise inside a map panel rather than here."""
        for name, spec in VARIABLES.items():
            assert spec.cmap in matplotlib.colormaps, f"{name}: {spec.cmap}"

    def test_every_transform_is_none_or_a_scale_matplotlib_has(self):
        """``transform`` is ``None`` throughout today; the check outlives
        that."""
        import matplotlib.scale

        allowed = set(matplotlib.scale.get_scale_names())
        for name, spec in VARIABLES.items():
            assert spec.transform is None or spec.transform in allowed, name

    def test_a_spec_cannot_be_mutated(self):
        """Frozen, so a consumer holding an entry cannot edit the registry."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            VARIABLES["nee"].agg = "mean"

    def test_the_names_follow_the_processed_naming_convention(self):
        """Lower case with underscores: the registry is keyed on processed
        names, which is what makes it usable against what an adapter
        produces."""
        for name in VARIABLES:
            assert PROCESSED_NAME.match(name), name

    def test_the_registry_holds_every_variable_that_has_a_reader(self):
        """A registry that lost its entries would make every other test here
        vacuously true, since they all loop over it."""
        assert set(VARIABLES) >= (
            set(DRIVER_VARIABLE_ATTRS) | set(CONSTRAINT_VARIABLE_ATTRS) | {"nee"}
        )


class TestCentersAndSigns:
    def test_nee_is_centered_on_zero_and_records_its_sign(self):
        """A sequential colormap on a signed flux is a misleading map, and the
        sign convention is not inferable from the values."""
        nee = VARIABLES["nee"]
        assert nee.center == 0.0
        assert nee.sign == "+ to atmosphere"
        assert nee.agg == "sum"

    def test_a_center_is_set_exactly_where_zero_is_meaningful(self):
        """NEE, whose sign is the thing being read, and the two temperatures,
        where zero is the phase boundary rather than an arbitrary origin.
        Compared as a set against an independent list, so adding a signed
        variable without a center, or a center on a quantity whose zero means
        nothing, fails here rather than in a figure."""
        centered = {name for name, spec in VARIABLES.items() if spec.center is not None}
        assert centered == CENTERED_ON_ZERO

    def test_a_center_is_paired_with_a_diverging_colormap(self):
        """A center on a sequential colormap does nothing, which is the same
        misleading map with the intention recorded."""
        diverging = {"RdBu_r", "RdBu", "BrBG", "PuOr", "coolwarm", "bwr", "seismic"}
        for name in CENTERED_ON_ZERO:
            assert VARIABLES[name].cmap in diverging, name


class TestAgreementWithTheReaders:
    """The duplication guard. Each of these fails if either side moves."""

    def test_every_driver_variable_is_registered(self):
        """A reader producing a field the registry does not know would make
        ``aggregate_time`` raise on data the project already loads."""
        assert not sorted(set(DRIVER_VARIABLE_ATTRS) - set(VARIABLES))

    def test_driver_aggregation_attributes_match_the_registry(self):
        """``DRIVER_VARIABLE_ATTRS[name]["aggregation"]`` is the attribute
        written onto the field; the registry is what ``aggregate_time`` reads.
        They must say the same thing."""
        for name, attrs in DRIVER_VARIABLE_ATTRS.items():
            assert VARIABLES[name].agg == attrs["aggregation"], name

    def test_driver_units_match_the_registry(self):
        """The readers already emit canonical units, so the registry's unit is
        theirs; a disagreement means one of the two is not canonical."""
        for name, attrs in DRIVER_VARIABLE_ATTRS.items():
            assert VARIABLES[name].units == attrs["units"], name

    def test_the_drivers_are_not_all_under_one_rule(self):
        """The agreement test above passes trivially if both sides say
        ``mean`` everywhere, so pin that the drivers exercise both rules."""
        assert {VARIABLES[name].agg for name in DRIVER_VARIABLE_ATTRS} == {
            "sum",
            "mean",
        }

    def test_every_constraint_variable_is_registered(self):
        """As for the drivers."""
        assert not sorted(set(CONSTRAINT_VARIABLE_ATTRS) - set(VARIABLES))

    def test_constraint_units_match_the_registry(self):
        """The constraint units are unconfirmed (README note 9) but they are
        what the product carries, so the canonical unit has to be the same."""
        for name, attrs in CONSTRAINT_VARIABLE_ATTRS.items():
            assert VARIABLES[name].units == attrs["units"], name

    def test_the_constraint_variables_are_stocks(self):
        """All four are levels at an instant, so all four refuse aggregation.
        Written as an assertion rather than left to the table's appearance."""
        for name in CONSTRAINT_VARIABLE_ATTRS:
            assert VARIABLES[name].agg == INSTANTANEOUS, name

    def test_the_units_are_not_all_one_string(self):
        """The two units tests above would both pass against a registry whose
        units were all the same; this fails on one that was."""
        assert len({spec.units for spec in VARIABLES.values()}) > 1


class TestVariableSpec:
    def test_returns_the_registered_spec(self):
        """The same object the mapping holds, not a copy."""
        assert variable_spec("nee") is VARIABLES["nee"]
        assert isinstance(variable_spec("par"), VarSpec)

    def test_raises_on_an_unregistered_name(self):
        """Naming the registry and the known names, since inferring a rule for
        an unknown variable is what this module exists to prevent."""
        with pytest.raises(ValueError, match="not in the variable registry"):
            variable_spec("gross_primary_productivity")

    def test_the_message_names_the_registry_and_the_known_variables(self):
        """So the reader knows where to add the entry without going looking."""
        with pytest.raises(ValueError) as raised:
            variable_spec("gross_primary_productivity")
        message = str(raised.value)
        assert "variable_registry" in message
        assert "nee" in message

    @pytest.mark.parametrize("name", [None, 3, ("nee",)])
    def test_raises_on_a_name_that_is_not_a_string(self, name):
        """``None`` is the case that matters: it is what an ``xarray``
        operation leaves behind when it drops a field's name."""
        with pytest.raises(ValueError, match="must be a string"):
            variable_spec(name)


class TestTheCarbonConversionFactor:
    def test_agrees_with_the_factor_the_producer_documents(self):
        """The producer gives ``kg C m-2 s-1 = umol CO2 m-2 s-1 * 12e-9``, so
        the constant is that factor scaled to grams, to the precision the
        rounder figure implies."""
        assert GRAMS_CARBON_PER_MICROMOLE_CO2 == pytest.approx(12e-9 * 1e3, rel=1e-3)

    def test_is_the_molar_mass_of_carbon_in_grams_per_micromole(self):
        """One micromole of CO2 carries one micromole of carbon, so the factor
        is 12.011 g/mol scaled by 1e-6. Pinned tightly, so a factor wrong by a
        thousand cannot pass the looser test above."""
        assert GRAMS_CARBON_PER_MICROMOLE_CO2 == pytest.approx(12.011e-6, rel=1e-12)

    def test_converts_the_observed_nee_scale_to_a_plausible_total(self):
        """A round trip on one value at the ``.clim`` timestep, so that a
        factor wrong by 1000 or by the timestep is caught here rather than in
        a figure that is off by orders of magnitude."""
        from sipnet_calibration.drivers import CLIM_FILE_CONSTANTS

        seconds = CLIM_FILE_CONSTANTS["length"] * 86400
        assert seconds == pytest.approx(3 * 3600)

        # A strong midday uptake, in the observed product's own units.
        rate = -20.0
        total = rate * GRAMS_CARBON_PER_MICROMOLE_CO2 * seconds
        # About -2.6 g C m-2 over three hours: the right order for a forest at
        # peak photosynthesis, and three orders from what a missing 1e-6 or a
        # missing timestep would give.
        assert total == pytest.approx(-2.594, abs=0.01)

    def test_the_canonical_nee_unit_is_the_per_timestep_total(self):
        """Recorded as an assertion, since the choice between the total and
        the rate is what fixes every NEE conversion in the project."""
        assert VARIABLES["nee"].units == "g C m-2"
