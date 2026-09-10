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

import pytest

from sipnet_calibration.variable_registry import (
    AGGREGATION_RULES,
    GRAMS_CARBON_PER_MICROMOLE_CO2,
    INSTANTANEOUS,
    VARIABLES,
    VarSpec,
    variable_spec,
)

pytestmark = pytest.mark.skip(reason="issue #6: not implemented yet")


class TestTheEntries:
    def test_every_agg_is_a_known_rule(self):
        """An unknown rule would raise only when someone aggregated it."""

    def test_every_entry_has_a_nonempty_label_and_units(self):
        """An empty unit string would pass a naive validate_field check."""

    def test_every_cmap_names_a_colormap_matplotlib_has(self):
        """A typo would raise inside a map panel rather than here."""

    def test_every_transform_is_none_or_a_scale_matplotlib_has(self):
        """``transform`` is ``None`` throughout today; the check outlives that."""

    def test_a_spec_cannot_be_mutated(self):
        """Frozen, so a consumer holding an entry cannot edit the registry."""

    def test_the_names_follow_the_processed_naming_convention(self):
        """Lower case with underscores: the registry is keyed on processed
        names, which is what makes it usable against what an adapter emits."""


class TestCentersAndSigns:
    def test_nee_is_centered_on_zero_and_records_its_sign(self):
        """A sequential colormap on a signed flux is a misleading map, and the
        sign convention is not inferable from the values."""

    def test_a_center_is_set_exactly_where_zero_is_meaningful(self):
        """NEE, whose sign is the thing being read, and the two temperatures,
        where zero is the phase boundary rather than an arbitrary origin.
        Pinned as a set, so adding a variable with a signed quantity but no
        center, or a center on a quantity whose zero means nothing, fails here
        rather than in a figure."""

    def test_a_center_is_paired_with_a_diverging_colormap(self):
        """A center on a sequential colormap does nothing, which is the same
        misleading map with the intention recorded."""


class TestAgreementWithTheReaders:
    """The duplication guard. Each of these fails if either side moves."""

    def test_every_driver_variable_is_registered(self):
        """A reader producing a field the registry does not know would make
        ``aggregate_time`` raise on data the project already loads."""

    def test_driver_aggregation_attributes_match_the_registry(self):
        """``drivers.DRIVER_VARIABLE_ATTRS[name]["aggregation"]`` is the
        attribute written onto the field; the registry is what
        ``aggregate_time`` reads. They must say the same thing."""

    def test_driver_units_match_the_registry(self):
        """The readers already emit canonical units, so the registry's unit is
        theirs; a disagreement means one of the two is not canonical."""

    def test_every_constraint_variable_is_registered(self):
        """As for the drivers."""

    def test_constraint_units_match_the_registry(self):
        """The constraint units are unconfirmed (README note 9) but they are
        what the product carries, so the canonical unit has to be the same."""

    def test_the_constraint_variables_are_stocks(self):
        """All four are levels at an instant, so all four refuse aggregation.
        Written as an assertion rather than left to the table's appearance."""

    def test_the_registry_holds_nothing_the_readers_contradict(self):
        """The complement of the two agreement tests: a registered name that
        a reader also produces must match on every shared field, so a variable
        added here later cannot quietly disagree with an existing product."""


class TestVariableSpec:
    def test_returns_the_registered_spec(self):
        """The same object the mapping holds, not a copy."""

    def test_raises_on_an_unregistered_name(self):
        """Naming the registry and the known names, since inferring a rule for
        an unknown variable is what this module exists to prevent."""

    def test_raises_on_a_name_that_is_not_a_string(self):
        """``None`` is the case that matters: it is what an ``xarray``
        operation leaves behind when it drops a field's name."""


class TestTheCarbonConversionFactor:
    def test_agrees_with_the_factor_the_producer_documents(self):
        """The producer gives ``kg C m-2 s-1 = umol CO2 m-2 s-1 * 12e-9``, so
        the constant is that factor scaled to grams, to the precision the
        rounder figure implies."""

    def test_converts_the_observed_nee_scale_to_a_plausible_total(self):
        """A round trip on one value at the ``.clim`` timestep, so that a
        factor wrong by 1000 or by the timestep is caught here rather than in
        a figure that is off by orders of magnitude."""
