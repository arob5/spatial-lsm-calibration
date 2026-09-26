"""Tests for :mod:`sipnet_calibration.plotting.maps` and the map grids in facet.

Two properties matter most here, because a map that breaks them still looks
plausible: that no renderer but :class:`Triangles` draws a value away from
the site it belongs to, and that panels asked to share a color scale really
do. Assertions are on artist data, never on rendered images.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.colors import LogNorm, Normalize, to_hex
from matplotlib.image import AxesImage

from sipnet_calibration.plotting import primitives
from sipnet_calibration.plotting.basemap import BASEMAP_ZORDER, MAX_ANGULAR_DISTANCE
from sipnet_calibration.plotting.facet import (
    plot_map_by,
    plot_map_grid,
    plot_map_quantiles,
)
from sipnet_calibration.plotting.maps import (
    Cells,
    ProjectedBounds,
    Triangles,
    animate_map,
    color_scale,
    coordinate_label,
    map_bounds,
    plot_map,
    quantile_label,
    summarize_batch,
)
from sipnet_calibration.plotting.style import axis_label, category_colors
from sipnet_calibration.projection import SITE_PROJECTION
from sipnet_calibration.sites import EXTENTS

# An animation that is never rendered warns when collected, which the
# animation tests do on purpose: they step it by hand.
pytestmark = pytest.mark.filterwarnings("ignore:Animation was deleted without rendering")

ATTRS = {"units": "kg m-2", "long_name": "Wood carbon"}


def site_field(lon, lat, values, **attrs) -> xr.DataArray:
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    return xr.DataArray(
        np.asarray(values),
        dims="site",
        coords={
            "site": np.arange(1, lon.size + 1, dtype=np.int32),
            "lon": ("site", lon),
            "lat": ("site", lat),
        },
        attrs={**ATTRS, **attrs},
        name="wood_carbon",
    )


@pytest.fixture
def dense() -> xr.DataArray:
    """A one-degree lattice of sites over the central US, valued by longitude."""
    lon, lat = np.meshgrid(np.arange(-110.0, -80.0), np.arange(30.0, 48.0))
    return site_field(lon.ravel(), lat.ravel(), lon.ravel() + 120.0)


@pytest.fixture
def ensemble(dense) -> xr.DataArray:
    rng = np.random.default_rng(0)
    samples = dense.values[None, :] + rng.normal(size=(20, dense.sizes["site"]))
    return xr.DataArray(
        samples, dims=("sample", "site"),
        coords={"sample": np.arange(20), **{k: v for k, v in dense.coords.items()}},
        attrs=dict(ATTRS), name="wood_carbon",
    )


@pytest.fixture
def categorical() -> xr.DataArray:
    lon = np.array([-100.0, -99.0, -98.0, -97.0])
    field = site_field(lon, np.full(4, 40.0), np.array([0, 2, 2, 1], dtype=np.int8))
    field.attrs = {
        "long_name": "Plant functional type",
        "flag_values": np.array([0, 1, 2], dtype=np.int8),
        "flag_meanings": "conifer deciduous grass",
    }
    return field


def data_artist(ax):
    """The one artist on *ax* that holds the mapped values."""
    found = [a for a in ax.collections + ax.images if hasattr(a, "get_array") and a.get_array() is not None
             and getattr(a, "zorder", 0) < BASEMAP_ZORDER]
    assert len(found) == 1, found
    return found[0]


def pixel_centers(image: AxesImage) -> np.ndarray:
    x_min, x_max, y_min, y_max = image.get_extent()
    n_y, n_x = image.site_index.shape
    xs = x_min + (x_max - x_min) * (np.arange(n_x) + 0.5) / n_x
    ys = y_min + (y_max - y_min) * (np.arange(n_y) + 0.5) / n_y
    return np.stack(np.meshgrid(xs, ys), axis=-1)


# ── the L1 primitives ─────────────────────────────────────────────────────────


def test_site_points_drops_sites_whose_value_is_missing(ax):
    drawn = primitives.site_points(ax, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [1.0, np.nan, 3.0])
    np.testing.assert_array_equal(drawn.get_offsets(), [[0, 0], [2, 2]])
    np.testing.assert_array_equal(drawn.get_array(), [1.0, 3.0])


def test_site_cells_never_colors_a_pixel_beyond_the_radius_of_its_site(ax):
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 1e6, 40), rng.uniform(0, 1e6, 40)
    image = primitives.site_cells(
        ax, x, y, np.arange(40.0), radius=5e4, bounds=(0, 0, 1e6, 1e6), pixels=200
    )
    centers, index = pixel_centers(image), image.site_index
    owned = index >= 0
    distance = np.hypot(centers[..., 0] - x[index.clip(0)], centers[..., 1] - y[index.clip(0)])
    assert np.all(distance[owned] <= 5e4)
    # And every pixel within the radius of some site is colored, by the nearest.
    nearest = np.min(np.hypot(centers[..., 0, None] - x, centers[..., 1, None] - y), axis=-1)
    np.testing.assert_array_equal(owned, nearest <= 5e4)
    np.testing.assert_array_equal(image.get_array()[owned], index[owned].astype(float))


def test_a_sparse_set_of_cells_colors_little_of_the_frame(ax):
    image = primitives.site_cells(
        ax, [1e5, 5e5, 9e5], [1e5, 5e5, 9e5], [1.0, 2.0, 3.0],
        radius=5e4, bounds=(0, 0, 1e6, 1e6), pixels=200,
    )
    assert np.mean(image.site_index >= 0) < 0.03


def test_a_site_with_a_missing_value_leaves_a_hole_rather_than_a_neighbors_value(ax):
    image = primitives.site_cells(
        ax, [2e5, 4e5], [5e5, 5e5], [1.0, np.nan], radius=1.5e5, bounds=(0, 0, 1e6, 1e6), pixels=100
    )
    owned_by_second = image.site_index == 1
    assert owned_by_second.any()
    assert np.all(np.isnan(np.asarray(image.get_array())[owned_by_second]))


def test_cells_from_index_puts_each_sites_value_in_its_cells():
    index = np.array([[0, -1], [1, 1]])
    np.testing.assert_array_equal(
        primitives.cells_from_index(index, [5.0, 7.0]), [[5.0, np.nan], [7.0, 7.0]]
    )


def test_site_triangles_masks_long_edges_and_missing_corners(ax):
    x = np.array([0.0, 1.0, 0.0, 1.0, 10.0])
    y = np.array([0.0, 0.0, 1.0, 1.0, 10.0])
    drawn = primitives.site_triangles(ax, x, y, np.array([1.0, 2, 3, 4, 5]), max_edge=2.0)
    mask = drawn._triangulation.mask
    kept = drawn._triangulation.triangles[~mask]
    assert 4 not in kept and len(kept) == 2
    drawn = primitives.site_triangles(ax, x, y, np.array([1.0, np.nan, 3, 4, 5]), max_edge=2.0)
    assert 1 not in drawn._triangulation.triangles[~drawn._triangulation.mask]


def test_site_triangles_needs_three_sites(ax):
    with pytest.raises(ValueError, match="three sites"):
        primitives.site_triangles(ax, [0, 1], [0, 1], [1, 2], max_edge=5.0)


def test_raster_refuses_corners_that_do_not_bound_the_values(ax):
    with pytest.raises(ValueError, match="corners of shape"):
        primitives.raster(ax, np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)))


# ── plot_map: kinds, renderers, refusals ──────────────────────────────────────


def test_points_are_the_default_and_draw_every_site_with_a_value(ax, dense):
    plot_map(dense, ax)
    artist = data_artist(ax)
    assert isinstance(artist, PathCollection)
    x, y = SITE_PROJECTION.forward(dense.lon.values, dense.lat.values)
    np.testing.assert_allclose(artist.get_offsets(), np.column_stack([x, y]))


def test_render_chooses_cells_and_a_renderer_instance_carries_its_settings(ax, dense):
    plot_map(dense, ax, render=Cells(radius_km=20.0, pixels=300))
    image = data_artist(ax)
    assert isinstance(image, AxesImage) and image.site_index.shape[1] == 300


def test_a_batch_dim_is_refused_and_the_message_names_the_alternatives(ax, ensemble):
    with pytest.raises(ValueError, match="'sample'.*plot_map_by.*plot_map_quantiles.*summarize_batch"):
        plot_map(ensemble, ax)
    renamed = ensemble.rename(sample="initial_condition_member")
    with pytest.raises(ValueError, match="batch_dim='initial_condition_member'"):
        plot_map(renamed, ax)


def test_a_field_is_validated_before_it_is_mapped(ax, dense):
    with pytest.raises(ValueError, match="site ids are int32"):
        plot_map(dense.assign_coords(site=dense["site"].astype(np.int64)), ax)


def test_a_time_dimension_is_refused_and_the_message_names_the_alternatives(ax, dense):
    timed = dense.expand_dims(time=np.array(["2012-01-01"], dtype="datetime64[ns]")).transpose(
        "site", "time"
    )
    with pytest.raises(ValueError, match="aggregate_time.*animate_map"):
        plot_map(timed, ax)


def test_a_site_field_without_coordinates_is_refused(ax, dense):
    with pytest.raises(ValueError, match="no 'lon' coordinate"):
        plot_map(dense.drop_vars(["lon", "lat"]), ax)


@pytest.mark.parametrize("render, match", [("smooth", "render must be one of"), (object(), "SiteRenderer")])
def test_an_unknown_renderer_is_refused(ax, dense, render, match):
    with pytest.raises(ValueError, match=match):
        plot_map(dense, ax, render=render)


def test_triangles_refuse_a_categorical_field(ax, categorical):
    with pytest.raises(ValueError, match="cannot be interpolated"):
        plot_map(categorical, ax, render="triangles")


def test_triangles_draw_a_continuous_field(ax, dense):
    plot_map(dense, ax, render=Triangles(max_edge_km=300.0))
    assert data_artist(ax).get_array() is not None


def test_the_basemap_can_be_left_off(ax, dense):
    plot_map(dense, ax, basemap=False, graticule=False)
    assert not [c for c in ax.collections if c.zorder >= BASEMAP_ZORDER]
    plot_map(dense, ax, basemap=("coastline",))


# ── the frame ─────────────────────────────────────────────────────────────────


def test_a_fitted_frame_contains_every_site_and_has_an_equal_aspect(ax, dense):
    plot_map(dense, ax)
    x, y = SITE_PROJECTION.forward(dense.lon.values, dense.lat.values)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    assert x0 < x.min() and x.max() < x1 and y0 < y.min() and y.max() < y1
    assert ax.get_aspect() == 1.0


def test_named_boxed_and_projected_extents_resolve_to_projected_bounds(dense):
    conus = SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"])
    assert tuple(map_bounds([dense], "CONUS")) == conus
    box = (-100.0, 35.0, -90.0, 45.0)
    assert tuple(map_bounds([dense], box)) == SITE_PROJECTION.projected_bounds(box)
    frame = ProjectedBounds(0.0, 0.0, 1.0, 1.0)
    assert map_bounds([dense], frame) is frame
    with pytest.raises(ValueError, match="unknown extent"):
        map_bounds([dense], "EUROPE")


# ── the color scale ───────────────────────────────────────────────────────────


def test_limits_come_from_the_values_inside_the_frame(ax):
    field = site_field([-100.0, -95.0, -150.0], [40.0, 40.0, 64.0], [1.0, 3.0, 99.0])
    plot_map(field, ax, extent="CONUS")
    norm = data_artist(ax).norm
    assert (norm.vmin, norm.vmax) == (1.0, 3.0)


def test_center_makes_the_limits_symmetric(ax, dense):
    plot_map(dense - 15.0, ax, center=0.0)
    norm = data_artist(ax).norm
    assert norm.vmin == -norm.vmax and norm.vmax > 0
    assert data_artist(ax).cmap.name == "RdBu_r"


def test_a_log_scale_refuses_nonpositive_values_and_a_center(ax, dense):
    plot_map(dense, ax, log=True)
    assert isinstance(data_artist(ax).norm, LogNorm)
    with pytest.raises(ValueError, match="positive values"):
        plot_map(dense - 20.0, ax, log=True)
    with pytest.raises(ValueError, match="no center"):
        plot_map(dense, ax, log=True, center=1.0)


def test_robust_limits_sit_inside_the_range_and_an_explicit_norm_wins(ax, dense):
    spread = dense.copy(data=np.random.default_rng(2).normal(size=dense.sizes["site"]))
    scale = color_scale([spread], bounds=map_bounds([spread]), robust=True)
    assert spread.min() < scale.norm.vmin < scale.norm.vmax < spread.max()
    given = Normalize(-5, 5)
    plot_map(dense, ax, norm=given, vmin=0.0)
    assert data_artist(ax).norm is given


# ── categorical fields ────────────────────────────────────────────────────────


def test_a_class_keeps_its_color_in_a_subset(ax, categorical):
    only_grass = categorical.isel(site=[1, 2])
    plot_map(only_grass, ax)
    artist = data_artist(ax)
    face = artist.to_rgba(artist.get_array())
    assert {to_hex(c) for c in face} == {category_colors(3)[2].lower()}


def test_the_legend_lists_the_classes_present_in_order(ax, categorical):
    plot_map(categorical.isel(site=[0, 1, 2]), ax)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["conifer", "grass"]


def test_the_legend_shows_display_names_while_colors_stay_keyed_on_classes(ax, categorical):
    named = categorical.assign_attrs(flag_display_names=("Conifer forest", "Broadleaf forest", "Grassland"))
    plot_map(named.isel(site=[0, 1, 2]), ax, colors={"grass": "#123456"})
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == ["Conifer forest", "Grassland"]
    assert to_hex(legend.legend_handles[1].get_facecolor()) == "#123456"


def test_display_names_that_do_not_pair_with_the_classes_are_refused(ax, categorical):
    with pytest.raises(ValueError, match="flag_display_names has 2 names"):
        plot_map(categorical.assign_attrs(flag_display_names=("A", "B")), ax)


def test_string_valued_classes_and_color_overrides(ax):
    field = site_field([-100.0, -99.0], [40.0, 40.0], np.array(["b", "a"], dtype=object))
    plot_map(field, ax, colors={"b": "#123456"})
    artist = data_artist(ax)
    colors = [to_hex(c) for c in artist.to_rgba(artist.get_array())]
    assert colors[0] == "#123456"
    with pytest.raises(ValueError, match="not a class"):
        plot_map(field, ax, colors={"c": "red"})


def test_an_undeclared_code_is_refused(ax, categorical):
    bad = categorical.copy(data=np.array([0, 7, 2, 1], dtype=np.int8))
    with pytest.raises(ValueError, match="not among flag_values"):
        plot_map(bad, ax)


# ── rasters ───────────────────────────────────────────────────────────────────


def raster(lat, lon) -> xr.DataArray:
    values = np.add.outer(np.asarray(lat, float), np.asarray(lon, float))
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon}, attrs=dict(ATTRS))


def test_a_raster_is_drawn_as_a_mesh_and_render_is_refused(ax):
    field = raster(np.arange(30.0, 50.0), np.arange(-110.0, -80.0))
    plot_map(field, ax)
    mesh = data_artist(ax)
    assert isinstance(mesh, QuadMesh) and mesh.get_array().shape == (20, 30)
    with pytest.raises(ValueError, match="render applies to site fields"):
        plot_map(field, ax, render="cells")


def test_raster_cells_too_far_from_the_center_are_not_drawn(ax):
    field = raster(np.arange(-40.0, 80.0, 5.0), np.arange(-180.0, 20.0, 5.0))
    plot_map(field, ax)
    drawn = ~np.ma.getmaskarray(data_artist(ax).get_array())
    lon, lat = np.meshgrid(field.lon, field.lat)
    far = SITE_PROJECTION.angular_distance(lon, lat) > MAX_ANGULAR_DISTANCE
    assert far.any() and not np.any(drawn & far)


def test_a_categorical_raster_is_drawn_by_class(ax):
    lat, lon = np.arange(30.0, 34.0), np.arange(-100.0, -96.0)
    field = xr.DataArray(
        np.array([[0, 1, 1, 0]] * 4, dtype=np.int8), dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        attrs={"long_name": "Class", "flag_values": np.array([0, 1]), "flag_meanings": "a b"},
    )
    plot_map(field, ax)
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["a", "b"]


def test_a_raster_reaching_the_antipode_or_out_of_order_is_refused(ax):
    with pytest.raises(ValueError, match="antipode"):
        plot_map(raster(np.arange(-89.0, 90.0, 2.0), np.arange(-180.0, 180.0, 2.0)), ax)
    with pytest.raises(ValueError, match="strictly monotonic"):
        plot_map(raster(np.array([30.0, 32.0, 31.0]), np.arange(-100.0, -90.0)), ax)


# ── summarize_batch and the labels ────────────────────────────────────────────


def test_summarize_batch_keeps_units_and_says_what_it_took(ensemble):
    summary = summarize_batch(ensemble, 0.05)
    assert summary.dims == ("site",) and summary.attrs["units"] == "kg m-2"
    assert summary.attrs["long_name"] == "Wood carbon, 5th percentile over sample"
    np.testing.assert_allclose(summary.values, np.quantile(ensemble.values, 0.05, axis=0))
    assert "standard deviation" in summarize_batch(ensemble, "standard_deviation").attrs["long_name"]


def test_summarize_batch_reduces_the_batch_dim_named(ensemble):
    renamed = ensemble.rename(sample="initial_condition_member")
    summary = summarize_batch(renamed, "mean", batch_dim="initial_condition_member")
    np.testing.assert_allclose(summary.values, ensemble.mean("sample").values)


@pytest.mark.parametrize("stat", ["sd", 1.5, 0.0])
def test_summarize_batch_refuses_an_unknown_statistic(ensemble, stat):
    with pytest.raises(ValueError):
        summarize_batch(ensemble, stat)


def test_summarize_batch_refuses_classes_and_a_missing_or_non_batch_dim(categorical, dense):
    with pytest.raises(ValueError, match="categorical"):
        summarize_batch(categorical.expand_dims(sample=[0, 1]), "mean")
    with pytest.raises(ValueError, match="'sample' is not one of the field's"):
        summarize_batch(dense, "mean")
    with pytest.raises(ValueError, match="'site' is not one of the field's"):
        summarize_batch(dense, "mean", batch_dim="site")


@pytest.mark.parametrize(
    "q, label", [(0.5, "median"), (0.05, "5th percentile"), (0.01, "1st percentile"),
                 (0.12, "12th percentile"), (0.975, "97.5th percentile"), (0.22, "22nd percentile")]
)
def test_quantile_label(q, label):
    assert quantile_label(q) == label


def test_coordinate_label():
    assert coordinate_label("sample", 3) == "sample 3"
    assert coordinate_label("time", np.datetime64("2012-07-01")) == "2012-07-01"
    assert coordinate_label("time", np.datetime64("2012-07-01T03:00")) == "2012-07-01 03:00"


# ── animation ─────────────────────────────────────────────────────────────────


def frames(dense) -> xr.DataArray:
    stack = xr.concat([dense, dense * 2.0, dense.where(dense.lon < -95)], dim="time")
    stack = stack.assign_coords(time=np.array(["2012-01", "2012-02", "2012-03"], dtype="datetime64[ns]"))
    return stack.transpose("site", "time")


@pytest.mark.parametrize("render", ["points", "cells", "triangles"])
def test_an_animation_keeps_one_scale_and_shows_each_frames_values(dense, render):
    field = frames(dense)
    animation = animate_map(field, render=render)
    ax = animation._fig.axes[0]
    drawn = []
    for position in range(3):
        (artist,) = animation._func(position)
        assert (artist.norm.vmin, artist.norm.vmax) == (float(dense.min()), float(dense.max() * 2))
        assert ax.get_title() == coordinate_label("time", field.time.values[position])
        drawn.append(_drawn_count(artist))
    # The last frame is missing the eastern sites, so less of it is drawn.
    assert drawn[2] < drawn[0] == drawn[1]


def _drawn_count(artist) -> int:
    if hasattr(artist, "_triangulation"):
        return int((~artist._triangulation.mask).sum())
    return int(np.isfinite(np.ma.filled(np.asarray(artist.get_array(), dtype=float), np.nan)).sum())


def test_an_animated_points_frame_places_exactly_the_sites_with_values(dense):
    animation = animate_map(frames(dense))
    (artist,) = animation._func(2)
    assert len(artist.get_offsets()) == int((dense.lon < -95).sum())


# ── grids of maps ─────────────────────────────────────────────────────────────


def test_a_shared_grid_puts_every_panel_on_one_scale_with_one_colorbar(dense):
    figure, axes = plot_map_grid({"a": dense, "b": dense * 3.0}, scale="shared")
    norms = [data_artist(ax).norm for ax in axes]
    assert norms[0] is norms[1]
    assert (norms[0].vmin, norms[0].vmax) == (float(dense.min()), float(dense.max() * 3))
    assert len(figure.axes) == 3  # two panels, one colorbar
    assert axes[0].get_xlim() == axes[1].get_xlim()


def test_an_each_grid_gives_every_panel_its_own_scale(dense):
    figure, axes = plot_map_grid({"a": dense, "b": dense * 3.0})
    assert data_artist(axes[0]).norm.vmax != data_artist(axes[1]).norm.vmax
    assert [ax.get_title() for ax in axes] == ["a", "b"]


def test_plot_map_by_thins_to_n_max_and_titles_by_value(ensemble):
    figure, axes = plot_map_by(ensemble, "sample", n_max=4)
    assert [ax.get_title() for ax in axes] == ["sample 0", "sample 6", "sample 13", "sample 19"]
    figure, axes = plot_map_by(ensemble, "sample", values=[2, 5])
    assert len(axes) == 2
    with pytest.raises(ValueError, match="no such sample"):
        plot_map_by(ensemble, "sample", values=[99])


def test_plot_map_quantiles_titles_and_labels_on_the_original_quantity(ensemble):
    figure, axes = plot_map_quantiles(ensemble)
    assert [ax.get_title() for ax in axes] == ["5th percentile", "median", "95th percentile"]
    colorbar = [ax for ax in figure.axes if ax not in list(axes)][0]
    assert colorbar.get_ylabel() == axis_label(ensemble)


def test_a_shared_categorical_grid_has_one_figure_legend(categorical):
    figure, axes = plot_map_grid({"all": categorical, "some": categorical.isel(site=[0])}, scale="shared")
    assert len(figure.legends) == 1
    assert [t.get_text() for t in figure.legends[0].get_texts()] == ["conifer", "deciduous", "grass"]


def test_a_shared_scale_refuses_fields_whose_display_names_differ(categorical):
    named = categorical.assign_attrs(flag_display_names=("A", "B", "C"))
    with pytest.raises(ValueError, match="same flag_display_names"):
        plot_map_grid({"named": named, "plain": categorical}, scale="shared")


def test_a_shared_scale_refuses_categorical_beside_continuous(categorical, dense):
    with pytest.raises(ValueError, match="categorical and continuous"):
        plot_map_grid({"a": categorical, "b": dense}, scale="shared")


# ── real data ─────────────────────────────────────────────────────────────────


def test_the_site_pool_maps_by_class(real_site_table):
    from sipnet_calibration.site_labels import site_labels_field

    try:
        field = site_labels_field("reanalysis_3pft", site_table=real_site_table)
    except FileNotFoundError as error:
        pytest.skip(str(error))
    ax = plot_map(field)
    assert len(data_artist(ax).get_offsets()) == field.sizes["site"]
    assert len(ax.get_legend().get_texts()) == 3


def test_the_site_pool_maps_by_sixteen_classes_with_display_names(real_site_table):
    from sipnet_calibration.site_labels import resolve_site_labels, site_labels_field

    try:
        field = site_labels_field("pft_16class", site_table=real_site_table)
    except FileNotFoundError as error:
        pytest.skip(str(error))
    spec = resolve_site_labels("pft_16class")
    ax = plot_map(field)
    legend = ax.get_legend()
    assert [t.get_text() for t in legend.get_texts()] == [spec.display_names[label] for label in spec.labels]
    colors = [to_hex(handle.get_facecolor()) for handle in legend.legend_handles]
    assert len(set(colors)) == len(spec.labels)


def test_initial_wood_carbon_quantiles_over_conus():
    from sipnet_calibration.initial_conditions import initial_condition_fields

    try:
        wood = initial_condition_fields(["initial_wood_carbon"])["initial_wood_carbon"]
    except FileNotFoundError as error:
        pytest.skip(str(error))
    figure, axes = plot_map_quantiles(
        wood, batch_dim="initial_condition_member", extent="CONUS", robust=True
    )
    assert len(axes) == 3


# ── one time of a field that carries interval coordinates ────────────────────


@pytest.fixture
def model_wood():
    """Real Niwot wood carbon stacked over (sample, site), with its timestep coordinates."""
    from conftest import niwot_stack_of

    return niwot_stack_of(["wood_carbon"], sites=(1, 27, 865), n_samples=3)["wood_carbon"]


def test_model_output_at_one_time_is_mapped(ax, model_wood):
    plot_map(model_wood.isel(sample=0, time=-1), ax=ax)
    assert data_artist(ax).get_array().size == 3


def test_model_output_is_mapped_by_time_and_by_quantile_at_one_time(model_wood):
    figure, axes = plot_map_by(model_wood.isel(sample=0, time=slice(0, 2)), "time")
    assert len(axes) == 2
    figure, axes = plot_map_quantiles(model_wood.isel(time=-1))
    assert len(axes) == 3


def test_model_output_is_animated_over_time(model_wood):
    animation = animate_map(model_wood.isel(sample=0, time=slice(0, 3)))
    (artist,) = animation._func(2)
    assert artist.get_array().size == 3


def test_an_annual_field_with_windows_is_animated(dense):
    annual = frames(dense).assign_coords(
        window_start=("time", np.array(["2011-12", "2012-01", "2012-02"], dtype="datetime64[ns]")),
        window_end=("time", np.array(["2012-01", "2012-02", "2012-03"], dtype="datetime64[ns]")),
    )
    animation = animate_map(annual)
    animation._func(1)


def test_an_animation_refuses_a_batch_dim_with_advice(dense):
    field = xr.concat([frames(dense), frames(dense)], dim="driver_member").assign_coords(
        driver_member=[0, 1]
    )
    with pytest.raises(ValueError, match="driver_member.*summarize_batch"):
        animate_map(field)


def test_plot_map_quantiles_refuses_the_retired_dim_keyword_naming_batch_dim(ensemble):
    """It fell through to matplotlib as an unknown artist property."""
    with pytest.raises(TypeError, match="batch_dim='sample'"):
        plot_map_quantiles(ensemble, dim="sample")


# ── the entry points validate first ───────────────────────────────────────────


def _without_units(field):
    bare = field.copy()
    bare.attrs = {}
    return bare


def test_a_table_is_refused_as_a_type_error(ax):
    import pandas as pd

    with pytest.raises(TypeError, match="expected an xarray.DataArray"):
        plot_map(pd.DataFrame({"x": [1.0]}), ax=ax)


def test_summarize_batch_refuses_a_field_without_units(ensemble):
    """With its validate_field removed, a field without units was summarized."""
    with pytest.raises(ValueError, match="attrs\\['units'\\]"):
        summarize_batch(_without_units(ensemble), "mean")


def test_plot_map_by_refuses_a_non_field_in_the_fields_words(ensemble):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    with pytest.raises(ValueError, match="attrs\\['units'\\]; set them"):
        plot_map_by(_without_units(ensemble), "sample")
    assert plt.get_fignums() == before


def test_animate_map_refuses_a_non_field_in_the_fields_words(dense):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    with pytest.raises(ValueError, match="attrs\\['units'\\]; set them"):
        animate_map(_without_units(frames(dense)))
    assert plt.get_fignums() == before


def test_plot_map_by_refuses_a_field_stored_out_of_order(ensemble):
    """Each panel alone would be a map; the field given is not a field."""
    with pytest.raises(ValueError, match="not in the order"):
        plot_map_by(ensemble.transpose("site", "sample"), "sample")


def test_animate_map_refuses_a_field_stored_out_of_order(dense):
    with pytest.raises(ValueError, match="not in the order"):
        animate_map(frames(dense).transpose("time", "site"))


def test_quantile_maps_take_the_batch_dim_named(ensemble):
    renamed = ensemble.rename(sample="initial_condition_member")
    figure, axes = plot_map_quantiles(renamed, batch_dim="initial_condition_member")
    assert len(np.ravel(axes)) == 3


# ── a second batch dim is refused before a shared scale is computed ───────────


def _two_batch_dims(ensemble):
    return ensemble.expand_dims(driver_member=[0, 1]).transpose("sample", "driver_member", "site")


@pytest.mark.parametrize("scale", ["shared", "each"])
def test_plot_map_by_refuses_a_second_batch_dim_with_advice(ensemble, scale):
    """With a shared scale it crashed with a raw IndexError."""
    with pytest.raises(ValueError, match="'driver_member'.*plot_map_quantiles"):
        plot_map_by(_two_batch_dims(ensemble), "sample", scale=scale)


def test_plot_map_quantiles_refuses_a_second_batch_dim_with_advice(ensemble):
    with pytest.raises(ValueError, match="'driver_member'.*summarize_batch"):
        plot_map_quantiles(_two_batch_dims(ensemble))


def test_plot_map_quantiles_names_the_batch_dim_the_field_has(ensemble):
    """A field on driver_member alone, with the default batch_dim, was told it
    "also has" driver_member, as if the quantiles were taken over sample."""
    on_members = ensemble.rename(sample="driver_member")
    with pytest.raises(ValueError, match="'sample' is not a batch dim of the field .*'driver_member'"):
        plot_map_quantiles(on_members)


def test_a_shared_map_grid_checks_every_panel_first(ensemble):
    with pytest.raises(ValueError, match="'sample'.*plot_map_by"):
        plot_map_grid({"a": ensemble, "b": ensemble}, scale="shared")


def test_animating_a_batch_dim_of_a_field_with_time_is_refused(ensemble):
    moving = ensemble.expand_dims(time=pd_dates(2)).transpose("sample", "site", "time")
    with pytest.raises(ValueError, match="for 'time'"):
        animate_map(moving, "sample")


def test_animating_a_zero_length_dim_is_refused_in_the_modules_words(dense):
    """It raised a raw IndexError from the first of no frames."""
    with pytest.raises(ValueError, match="no 'time' steps to play"):
        animate_map(frames(dense).isel(time=slice(0, 0)))


def test_animate_map_checks_a_frame_is_a_map_before_its_scale_is_read():
    """Without the check before the scale, a field with no spatial dim reached
    map_bounds and failed with xarray's own error about lat and lon."""
    from conftest import make_field

    with pytest.raises(ValueError, match="a map needs a 'site' dimension"):
        animate_map(make_field(("time",), n_time=3))


def pd_dates(n):
    import pandas as pd

    return pd.date_range("2012-01-01", periods=n)


# ── one definition of categorical ─────────────────────────────────────────────


def test_a_boolean_field_is_mapped_as_classes(ax, dense):
    """``run_succeeded`` and ``driver_present`` validate, and were refused for units."""
    flags = (dense > dense.median()).rename("run_succeeded")
    flags.attrs = {"long_name": "Whether the run succeeded"}
    plot_map(flags, ax=ax)
    assert data_artist(ax).get_array().size == dense.sizes["site"]


def test_flag_values_alone_are_mapped_as_classes(ax, categorical):
    codes = categorical.copy()
    codes.attrs = {"long_name": "Class", "flag_values": np.array([0, 1, 2], dtype=np.int8)}
    plot_map(codes, ax=ax)
    np.testing.assert_array_equal(data_artist(ax).get_array(), [0.0, 2.0, 2.0, 1.0])


def test_run_succeeded_is_mapped(ax):
    from conftest import make_field

    field = make_field(("sample", "site"))
    succeeded = (field > -10).rename("run_succeeded")
    succeeded.attrs = {"long_name": "Whether the run succeeded"}
    plot_map(succeeded.isel(sample=0), ax=ax)
