"""Tests for :mod:`sipnet_calibration.plotting.basemap` and the scripts behind it.

The tracked basemap is checked against a rebuild from the tracked Natural
Earth archives, so a hand-edit or a stale build is a failure here rather than
a quietly wrong coastline.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import load_script

from sipnet_calibration.conventions import data_root
from sipnet_calibration.io import file_md5
from sipnet_calibration.plotting import basemap
from sipnet_calibration.plotting.basemap import (
    BASEMAP_LAYERS,
    BASEMAP_ZORDER,
    MAX_ANGULAR_DISTANCE,
    clip_to_drawable,
    draw_basemap,
    draw_graticule,
    graticule_spacing,
    load_basemap,
)
from sipnet_calibration.projection import SITE_PROJECTION
from sipnet_calibration.sites import EXTENTS

RAW_DIR = data_root() / "raw" / "natural_earth"


# ── the tracked file ──────────────────────────────────────────────────────────


def test_every_layer_is_present_and_every_vertex_is_drawable():
    layers = load_basemap()
    assert set(layers) == set(BASEMAP_LAYERS)
    for name, parts in layers.items():
        assert parts, f"layer {name} is empty"
        vertices = np.concatenate(parts)
        distance = SITE_PROJECTION.angular_distance(vertices[:, 0], vertices[:, 1])
        assert distance.max() <= MAX_ANGULAR_DISTANCE + 1e-4
        x, y = SITE_PROJECTION.forward(vertices[:, 0], vertices[:, 1])
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert all(len(part) >= 2 for part in parts)


def test_the_tracked_file_is_what_the_tracked_archives_build_to():
    if not all((RAW_DIR / layer.source_file).is_file() for layer in BASEMAP_LAYERS.values()):
        pytest.skip("the Natural Earth archives are not in this working copy")
    build = load_script("scripts/build_basemap.py")
    parts, source_md5 = build.build(RAW_DIR)
    stored = load_basemap()
    for name in BASEMAP_LAYERS:
        assert len(parts[name]) == len(stored[name])
        for built, kept in zip(parts[name], stored[name]):
            np.testing.assert_allclose(built, kept, atol=1e-5)
    with np.load(basemap.basemap_path()) as archive:
        for name in BASEMAP_LAYERS:
            assert str(archive[f"{name}_source_md5"]) == source_md5[name]


def test_the_archives_are_the_ones_the_download_script_records():
    download = load_script("scripts/raw_sources/download_natural_earth.py")
    recorded = {source.file_name: source.md5 for source in download.SOURCES}
    assert set(recorded) == {layer.source_file for layer in BASEMAP_LAYERS.values()}
    for file_name, md5 in recorded.items():
        path = RAW_DIR / file_name
        if not path.is_file():
            pytest.skip(f"{path} is not in this working copy")
        assert file_md5(path) == md5


def test_the_download_refuses_an_archive_with_the_wrong_md5():
    download = load_script("scripts/raw_sources/download_natural_earth.py")
    with pytest.raises(download.DownloadError, match="new release"):
        download.check_md5_matches(download.SOURCES[0], "0" * 32)


def test_a_file_built_for_another_center_is_refused(tmp_path):
    with np.load(basemap.basemap_path()) as archive:
        arrays = dict(archive)
    arrays["center"] = np.array([-90.0, 45.0])
    path = tmp_path / "moved.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="Rebuild it"):
        load_basemap(path)


def test_write_then_load_round_trips(tmp_path):
    part = np.array([[-100.0, 40.0], [-99.0, 41.0], [-98.0, 40.5]])
    parts = {name: [part] for name in BASEMAP_LAYERS}
    path = tmp_path / "basemap.npz"
    basemap.write_basemap(parts, {name: "x" * 32 for name in BASEMAP_LAYERS}, path)
    for name, loaded in load_basemap(path).items():
        np.testing.assert_allclose(loaded[0], part, atol=1e-5)


# ── clipping ──────────────────────────────────────────────────────────────────


def test_clipping_splits_a_line_where_it_leaves_the_drawable_region():
    # Along the equator from 100 W: in range near the center meridian, out of
    # range beyond it, and back.
    lon = np.array([-100.0, -90.0, 60.0, 70.0, -120.0, -110.0])
    lat = np.zeros_like(lon)
    runs = clip_to_drawable(lon, lat)
    assert [len(run) for run in runs] == [2, 2]
    assert clip_to_drawable(np.array([80.0, 81.0]), np.array([-50.0, -50.0])) == []


def test_angular_distance_is_zero_at_the_center_and_180_at_the_antipode():
    assert SITE_PROJECTION.angular_distance(-100.0, 50.0) == pytest.approx(0.0, abs=1e-9)
    assert SITE_PROJECTION.angular_distance(*SITE_PROJECTION.antipode) == pytest.approx(180.0)
    assert SITE_PROJECTION.angular_distance(-100.0, 90.0) == pytest.approx(40.0)


# ── drawing ───────────────────────────────────────────────────────────────────


def test_draw_basemap_adds_one_collection_per_layer_above_the_data(ax):
    drawn = draw_basemap(ax, layers=("coastline", "states"))
    assert len(drawn) == 2 and all(c.zorder == BASEMAP_ZORDER for c in drawn)
    with pytest.raises(ValueError, match="unknown basemap layer"):
        draw_basemap(ax, layers=("rivers",))


def test_the_graticule_labels_meridians_on_the_bottom_and_parallels_on_the_left(ax):
    x0, y0, x1, y1 = SITE_PROJECTION.projected_bounds(EXTENTS["CONUS"])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    draw_graticule(ax)
    labels = [text.get_text() for text in ax.texts]
    assert "100°W" in labels and "30°N" in labels
    with pytest.raises(ValueError, match="spacing"):
        draw_graticule(ax, spacing=0.0)


@pytest.mark.parametrize("size, spacing", [(1.6e7, 20.0), (5e6, 10.0), (2e6, 5.0), (5e5, 2.0), (1e5, 1.0)])
def test_graticule_spacing_shrinks_with_the_frame(size, spacing):
    assert graticule_spacing(size) == spacing
