"""Tests for :mod:`sipnet_calibration.plotting.facet`.

Skipped until the module is implemented; collected so that the intended
assertions are on the record and reviewable.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="facet.py is a contract skeleton; not yet implemented")


# ── the grid ──────────────────────────────────────────────────────────────────


def test_the_grid_shape_follows_ncol():
    """Seven items at ``ncol=3`` give three rows."""
    raise NotImplementedError


def test_unused_axes_are_hidden():
    """The padding axes of the last row are not visible."""
    raise NotImplementedError


def test_the_returned_axes_align_with_the_items():
    """``len(axes) == len(items)``, and hidden axes are not among them."""
    raise NotImplementedError


def test_the_figure_size_follows_panel_size_and_the_grid():
    """A six-panel grid is not the size of a twenty-panel one."""
    raise NotImplementedError


def test_the_callback_is_called_once_per_item_in_order():
    """``panel_fn(ax, item)`` sees the items in the order given."""
    raise NotImplementedError


def test_the_callback_return_is_ignored():
    """A callback returning an ``Axes`` -- as the panel functions do -- is fine."""
    raise NotImplementedError


# ── sharing ───────────────────────────────────────────────────────────────────


def test_share_y_gives_one_common_y_limit():
    """With ``share="y"`` every panel ends with the same y limits."""
    raise NotImplementedError


def test_share_none_leaves_the_limits_independent():
    """Panels with different data ranges keep different limits."""
    raise NotImplementedError


def test_an_unknown_share_is_rejected():
    """The message lists :data:`SHARE_MODES`."""
    raise NotImplementedError


# ── titles ────────────────────────────────────────────────────────────────────


def test_labels_as_a_sequence_title_the_panels():
    """Each panel's title is the corresponding element."""
    raise NotImplementedError


def test_labels_as_a_callable_title_the_panels():
    """The callable is applied to each item."""
    raise NotImplementedError


def test_a_labels_sequence_of_the_wrong_length_is_rejected():
    """A mismatch raises rather than titling some panels."""
    raise NotImplementedError


# ── legend ────────────────────────────────────────────────────────────────────


def test_dedup_keeps_one_entry_per_label():
    """Panels drawing the same three roles give a legend with three entries."""
    raise NotImplementedError


def test_dedup_keeps_first_seen_order():
    """The legend order is the order the labels first appeared."""
    raise NotImplementedError


def test_legend_none_draws_no_legend():
    """No figure legend and no panel legends."""
    raise NotImplementedError


def test_legend_each_gives_every_panel_its_own():
    """Each panel carries a legend and the figure carries none."""
    raise NotImplementedError


# ── input validation ──────────────────────────────────────────────────────────


def test_empty_items_is_rejected():
    """An empty sequence raises rather than returning a blank figure."""
    raise NotImplementedError


def test_a_non_positive_ncol_is_rejected():
    """``ncol=0`` raises."""
    raise NotImplementedError


# ── by_site ───────────────────────────────────────────────────────────────────


def test_by_site_draws_one_panel_per_site(field_member_site_time):
    """One panel per site, and each panel's data are that site's slice."""
    raise NotImplementedError


def test_by_site_titles_the_panels_with_the_site_ids(field_member_site_time):
    """The default labels name the site."""
    raise NotImplementedError


def test_by_site_selects_and_orders_by_the_sites_given(field_member_site_time):
    """``sites=`` picks a subset, in the order given."""
    raise NotImplementedError


def test_by_site_accepts_a_bound_panel_composer(field_member_site_time):
    """``functools.partial(series_panel, show="spaghetti")`` reaches the panel."""
    raise NotImplementedError


def test_by_site_rejects_a_field_without_a_site_dim(field_member_time):
    """A field with no ``site`` dim raises."""
    raise NotImplementedError


def test_by_site_rejects_a_site_that_is_not_on_the_field(field_member_site_time):
    """An unknown site id raises rather than yielding an empty panel."""
    raise NotImplementedError


# ── by_variable ───────────────────────────────────────────────────────────────


def test_by_variable_draws_one_panel_per_variable(field_time):
    """One panel per entry of the mapping, in its order."""
    raise NotImplementedError


def test_by_variable_titles_the_panels_with_the_long_names(field_time):
    """The default labels come from each field's ``long_name``."""
    raise NotImplementedError


def test_by_variable_accepts_fields_on_different_time_axes(field_time):
    """Annual and 3-hourly fields in one call, which is why it is a mapping."""
    raise NotImplementedError


def test_by_variable_rejects_an_empty_mapping():
    """An empty mapping raises."""
    raise NotImplementedError
