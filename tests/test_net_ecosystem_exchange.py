"""Tests for the net ecosystem exchange package and its three scripts.

Every test runs on a synthetic AmeriFlux download written into a temporary
directory -- six towers, chosen so that each matching basis, the shared cell,
the exclusions and both resolutions occur -- so nothing here needs the SCC or
the real raw files. The chain is exercised end to end: FULLSET CSVs and their
zips, ``convert_ameriflux_nee.py``, ``build_ameriflux_towers.py``,
``ingest_net_ecosystem_exchange.py``, and the readers.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from sipnet_calibration.net_ecosystem_exchange import (
    HALF_HOURLY,
    HOURLY,
    NET_ECOSYSTEM_EXCHANGE,
    NET_ECOSYSTEM_EXCHANGE_NAMES,
    SOURCE,
    build_net_ecosystem_exchange,
    load_net_ecosystem_exchange,
    net_ecosystem_exchange_quality_flags,
    net_ecosystem_exchange_values,
    parse_file_name,
    raw_path,
    read_raw,
    read_source_file,
    read_tower_table,
    recover_utc_offset,
    resolve_net_ecosystem_exchange,
    shortwave_lag_steps,
)
from sipnet_calibration.net_ecosystem_exchange.towers import _top_of_atmosphere_shortwave
from sipnet_calibration.obs_ops import aggregate_time
from sipnet_calibration.sites import SITE_COLUMNS, SITE_GRID, load_sites

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(relative: str, name: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


convert = _load_script("scripts/raw_sources/convert_ameriflux_nee.py", "convert_ameriflux_nee")
build_towers = _load_script("scripts/raw_sources/build_ameriflux_towers.py", "build_ameriflux_towers")
ingest = _load_script("scripts/ingest_net_ecosystem_exchange.py", "ingest_net_ecosystem_exchange")


# ── the synthetic download ────────────────────────────────────────────────────

#: tower: (resolution, first year, last year, UTC offset, pool site or None,
#: cell offset from that site's cell, whether the file carries CUT columns,
#: measured-shortwave shift in steps)
TOWERS = {
    "US-Aaa": (HALF_HOURLY, 2012, 2013, -5.0, 101, (0, 0), True, 0),
    "US-Ddd": (HALF_HOURLY, 2013, 2013, -5.0, 101, (0, 0), True, 0),
    "US-Bbb": (HALF_HOURLY, 2012, 2012, -8.0, 102, (0, 0), False, 0),
    "US-Ccc": (HOURLY, 2013, 2014, -6.0, 103, (0, 0), True, 0),
    "US-Eee": (HALF_HOURLY, 2012, 2012, -7.0, None, (0, 0), True, 0),
    "US-Fff": (HALF_HOURLY, 2012, 2012, -5.0, 104, (0, 0), True, 2),
}

#: site_id: (lon_index, lat_index, site_name)
SITES = {
    101: (11400, 4200, "Oak ridge (US-Aaa)"),
    102: (7000, 3500, "ameriflux"),
    103: (9000, 3900, "weighted_sample"),
    104: (12000, 4100, "Test bog (us-Fff)"),
    105: (5000, 6000, "weighted_sample"),
}

#: Where the tower that is in no pool cell stands.
LONELY_CELL = (8000, 3000)


def _cell_corner(lon_index: int, lat_index: int) -> tuple[float, float]:
    """The real south-west corner of a cell."""
    step = 1 / SITE_GRID.cells_per_degree
    return (
        SITE_GRID.west + SITE_GRID.edge_shift_lon + lon_index * step,
        SITE_GRID.south + SITE_GRID.edge_shift_lat + lat_index * step,
    )


def _tower_lonlat(tower: str) -> tuple[float, float]:
    site = TOWERS[tower][4]
    j, k = (SITES[site][0], SITES[site][1]) if site is not None else LONELY_CELL
    lon, lat = _cell_corner(j, k)
    step = 1 / SITE_GRID.cells_per_degree
    return lon + 0.3 * step, lat + 0.6 * step


def _fullset_frame(tower: str) -> pd.DataFrame:
    resolution, first, last, offset, _, _, has_cut, shift = TOWERS[tower]
    starts = pd.date_range(f"{first}-01-01", f"{last + 1}-01-01", freq=resolution.step, inclusive="left")
    ends = starts + resolution.step
    lon, lat = _tower_lonlat(tower)
    middle_utc = starts + resolution.step / 2 - pd.Timedelta(hours=offset)
    potential = np.round(_top_of_atmosphere_shortwave(pd.DatetimeIndex(middle_utc), lat, lon), 3)
    n = len(starts)
    step = np.arange(n)
    nee = np.round(2.0 * np.sin(step / 7.0) + 0.001 * step % 5, 4)
    qc = (step % 3 == 0).astype(float) * 2
    gap = step % 97 == 0
    frame = {
        "TIMESTAMP_START": starts.strftime("%Y%m%d%H%M").astype(np.int64),
        "TIMESTAMP_END": ends.strftime("%Y%m%d%H%M").astype(np.int64),
        "SW_IN_POT": potential,
        "SW_IN_F": np.round(0.7 * np.roll(potential, shift), 3),
        "SW_IN_F_QC": np.zeros(n),
        "NIGHT": (potential == 0).astype(float),
    }
    estimates = ["VUT_REF", "VUT_USTAR50"] + (["CUT_REF", "CUT_USTAR50"] if has_cut else [])
    for i, estimate in enumerate(estimates):
        frame[f"NEE_{estimate}"] = np.where(gap, -9999.0, nee + i)
        frame[f"NEE_{estimate}_QC"] = np.where(gap, -9999.0, qc)
        frame[f"NEE_{estimate}_RANDUNC"] = np.where(qc == 0, 0.5, -9999.0)
        frame[f"NEE_{estimate}_JOINTUNC"] = np.where(gap, -9999.0, 0.75)
    frame["TA_F"] = np.full(n, 10.5)  # a column the conversion does not keep
    return pd.DataFrame(frame)


def _file_name(tower: str, *, with_code: bool = True) -> str:
    resolution, first, last = TOWERS[tower][0], TOWERS[tower][1], TOWERS[tower][2]
    code = f"{resolution.source_code}_" if with_code else ""
    return f"AMF_{tower}_FLUXNET_FULLSET_{code}{first}-{last}_4-7.csv"


def _write_download(root: Path) -> None:
    root.mkdir(parents=True)
    for tower in TOWERS:
        frame = _fullset_frame(tower)
        csv = root / _file_name(tower)
        frame.to_csv(csv, index=False)
        archive = root / _file_name(tower, with_code=False).replace(".csv", ".zip")
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.write(csv, arcname=csv.name)
    (root / "AMF_US-Aaa_FLUXNET_FULLSET_DD_2012-2013_4-7.csv").write_text("not read\n")


def _write_site_list(path: Path) -> None:
    rows = []
    for tower in TOWERS:
        lon, lat = _tower_lonlat(tower)
        rows.append(
            {
                "Site ID": tower,
                "Name": f"Tower {tower}",
                "Principal Investigator": "Someone (someone@example.org)",
                "Latitude (degrees)": repr(lat),
                "Longitude (degrees)": repr(lon),
                "Vegetation Abbreviation (IGBP)": "DBF",
                "AmeriFlux FLUXNET DOI": f"https://doi.org/10.17190/AMF/{tower}",
            }
        )
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _write_pool_input_list(path: Path) -> None:
    lon, lat = _tower_lonlat("US-Bbb")
    lines = [
        '"","Site.ID","Latitude..degrees.","Longitude..degrees.","Elevation..m."',
        f'"1","US-Bbb",{lat!r},{lon!r},10',
        f'"2","US-Zzz",{lat!r},{lon!r},10',  # a later tower in the same cell owns nothing
    ]
    path.write_text("\n".join(lines) + "\n")


def _write_site_table(path: Path) -> None:
    rows = []
    for site, (j, k, name) in SITES.items():
        lon, lat = SITE_GRID.index_to_lonlat(j, k)
        rows.append(
            {
                "site_id": site,
                "lon": lon + SITE_GRID.edge_shift_lon,
                "lat": lat + SITE_GRID.edge_shift_lat,
                "lon_index": j,
                "lat_index": k,
                "site_name": name,
                "site_order": 0 if name == "weighted_sample" else site,
                "cluster": 1,
                "landcover": 1,
                "ameriflux_site_id": "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(SITE_COLUMNS)).to_csv(path, index=False)


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """The synthetic download run through all three scripts."""
    base = tmp_path_factory.mktemp("nee")
    download = base / "fluxnet"
    raw_dir = base / "raw"
    raw_dir.mkdir()
    out_dir = base / "processed"
    sites = base / "sites" / "sites.csv"
    _write_download(download)
    _write_site_list(raw_dir / "ameri_sites.tsv")
    _write_pool_input_list(raw_dir / "Unmatched_Sites.csv")
    _write_site_table(sites)

    assert convert.main(["--root", str(download), "--out-dir", str(raw_dir), "--jobs", "1"]) == 0
    assert build_towers.main(["--raw-dir", str(raw_dir), "--sites", str(sites), "--out", str(raw_dir / "ameriflux_towers.csv")]) == 0
    assert ingest.main(["--raw-dir", str(raw_dir), "--sites", str(sites), "--out-dir", str(out_dir)]) == 0
    return {"download": download, "raw_dir": raw_dir, "out_dir": out_dir, "sites": sites}


# ── the source files ──────────────────────────────────────────────────────────


class TestSourceFiles:
    def test_a_file_name_decodes(self):
        tower, resolution, first, last, version = parse_file_name("AMF_CA-ARB_FLUXNET_FULLSET_HH_2011-2015_5-7.csv")
        assert (tower, resolution, first, last, version) == ("CA-ARB", HALF_HOURLY, 2011, 2015, "5-7")

    def test_values_are_the_source_text_and_the_fill_becomes_missing(self, pipeline):
        path = pipeline["download"] / _file_name("US-Aaa")
        parsed = read_source_file(path)
        frame = pd.read_csv(path)
        index = ((pd.to_datetime(frame["TIMESTAMP_START"].astype(str), format="%Y%m%d%H%M") - pd.Timestamp("2011-12-31")) // HALF_HOURLY.step).to_numpy()
        source = frame["NEE_VUT_REF"].to_numpy()
        got = parsed.values["NEE_VUT_REF"][index]
        assert np.array_equal(got[source != -9999], source[source != -9999])
        assert np.isnan(got[source == -9999]).all()
        assert (parsed.values["NEE_VUT_REF_QC"][index][source == -9999] == -1).all()

    def test_a_file_without_cut_columns_names_them_absent(self, pipeline):
        parsed = read_source_file(pipeline["download"] / _file_name("US-Bbb"))
        assert set(parsed.absent_columns) == SOURCE.optional_columns
        assert np.isnan(parsed.values["NEE_CUT_USTAR50"]).all()

    def test_a_skipped_stamp_is_refused(self, tmp_path):
        frame = _fullset_frame("US-Bbb").drop(index=500)
        path = tmp_path / _file_name("US-Bbb")
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="not contiguous"):
            read_source_file(path)

    def test_a_flag_outside_its_vocabulary_is_refused(self, tmp_path):
        frame = _fullset_frame("US-Bbb")
        frame.loc[3, "NEE_VUT_REF_QC"] = 7
        path = tmp_path / _file_name("US-Bbb")
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="outside its vocabulary"):
            read_source_file(path)

    def test_a_partial_cut_group_is_refused(self, tmp_path):
        frame = _fullset_frame("US-Aaa").drop(columns=["NEE_CUT_REF_JOINTUNC"])
        path = tmp_path / _file_name("US-Aaa")
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="present whole or not at all"):
            read_source_file(path)

    def test_a_fill_like_value_is_refused(self, tmp_path):
        frame = _fullset_frame("US-Bbb")
        frame.loc[5, "SW_IN_F"] = -9998.0
        path = tmp_path / _file_name("US-Bbb")
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match="fill-like"):
            read_source_file(path)


# ── the conversion ────────────────────────────────────────────────────────────


class TestConversion:
    def test_one_raw_file_per_resolution_with_every_tower(self, pipeline):
        with read_raw(raw_path(HALF_HOURLY, pipeline["raw_dir"])) as half_hourly:
            assert half_hourly["tower"].values.tolist() == sorted(t for t, v in TOWERS.items() if v[0] == HALF_HOURLY)
        with read_raw(raw_path(HOURLY, pipeline["raw_dir"])) as hourly:
            assert hourly["tower"].values.tolist() == ["US-Ccc"]

    def test_the_raw_axis_is_local_standard_time_unshifted(self, pipeline):
        with read_raw(raw_path(HALF_HOURLY, pipeline["raw_dir"])) as raw:
            assert raw["TIMESTAMP_START"].values[0] == 201112310000
            first = int(np.flatnonzero(np.isfinite(raw["SW_IN_POT"].sel(tower="US-Aaa").values))[0])
            assert raw["TIMESTAMP_START"].values[first] == 201201010000

    def test_a_csv_changed_after_the_download_is_refused(self, pipeline, tmp_path):
        copy = tmp_path / "fluxnet"
        copy.mkdir()
        for path in pipeline["download"].iterdir():
            (copy / path.name).write_bytes(path.read_bytes())
        edited = copy / _file_name("US-Bbb")
        edited.write_text(edited.read_text().replace(",10.5\n", ",10.6\n", 1))
        status = convert.main(["--root", str(copy), "--out-dir", str(tmp_path / "raw"), "--jobs", "1"])
        assert status == 1

    def test_a_trial_run_needs_an_explicit_out_dir(self, pipeline):
        assert convert.main(["--root", str(pipeline["download"]), "--towers", "US-Aaa"]) == 1


# ── the tower table ───────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def table(pipeline):
    return read_tower_table(pipeline["raw_dir"] / "ameriflux_towers.csv").set_index("tower")


class TestTowerTable:
    def test_each_basis_is_used_where_it_applies(self, table):
        assert table.at["US-Aaa", "match_basis"] == "named_in_pool"
        assert table.at["US-Bbb", "match_basis"] == "pool_input_list"
        assert table.at["US-Ccc", "match_basis"] == "same_cell"
        assert table.at["US-Ddd", "match_basis"] == "same_cell"
        assert table.at["US-Fff", "match_basis"] == "named_in_pool"  # the name's case differs

    def test_the_named_tower_is_primary_where_two_share_a_site(self, table):
        assert table.at["US-Aaa", "site_id"] == table.at["US-Ddd", "site_id"] == 101
        assert table.at["US-Aaa", "primary"] and not table.at["US-Ddd", "primary"]
        assert "US-Ddd" in table.at["US-Aaa", "primary_reason"]

    def test_a_tower_in_no_pool_cell_is_excluded(self, table):
        assert pd.isna(table.at["US-Eee", "site_id"])
        assert table.at["US-Eee", "excluded_reason"].startswith("no pool site")

    def test_a_tower_whose_clock_disagrees_is_excluded(self, table):
        assert table.at["US-Fff", "site_id"] == 104
        assert not table.at["US-Fff", "primary"]
        assert table.at["US-Fff", "excluded_reason"].startswith("clock")

    def test_an_excluded_clock_is_not_also_noted_as_tolerated(self, table):
        assert "tolerance" not in table.at["US-Fff", "comment"]

    def test_offsets_are_recovered(self, table):
        for tower, row in TOWERS.items():
            assert table.at[tower, "utc_offset_hours"] == row[3]

    def test_the_table_never_carries_contact_details(self, pipeline):
        text = (pipeline["raw_dir"] / "ameriflux_towers.csv").read_text()
        assert "example.org" not in text


class TestClock:
    @pytest.mark.parametrize("offset", [-9.0, -5.0, -3.5, 1.0])
    def test_the_offset_is_recovered_from_potential_shortwave(self, offset):
        starts = pd.date_range("2013-01-01", "2013-12-31", freq="30min")
        potential = _top_of_atmosphere_shortwave(
            pd.DatetimeIndex(starts + pd.Timedelta(minutes=15) - pd.Timedelta(hours=offset)), 45.0, -80.0
        )
        fit = recover_utc_offset(starts, 30, potential, 45.0, -80.0)
        assert fit.offset_hours == offset
        assert fit.separation > 10

    def test_measured_shortwave_on_the_same_clock_has_no_lag(self):
        starts = pd.date_range("2013-01-01", "2013-12-31", freq="30min")
        potential = _top_of_atmosphere_shortwave(pd.DatetimeIndex(starts + pd.Timedelta(hours=5)), 45.0, -80.0)
        flag = np.zeros(len(starts), dtype=np.int8)
        assert shortwave_lag_steps(starts, 30, 0.7 * potential, flag, potential) == 0
        assert shortwave_lag_steps(starts, 30, 0.7 * np.roll(potential, 2), flag, potential) == 2

    def test_too_little_measured_shortwave_cannot_be_checked(self):
        starts = pd.date_range("2013-01-01", "2013-01-10", freq="30min")
        potential = np.ones(len(starts))
        assert shortwave_lag_steps(starts, 30, potential, np.zeros(len(starts), np.int8), potential) is None


# ── the products ──────────────────────────────────────────────────────────────


class TestProducts:
    def test_every_spec_has_a_product(self, pipeline):
        for name in NET_ECOSYSTEM_EXCHANGE_NAMES:
            with load_net_ecosystem_exchange(name, pipeline["out_dir"] / f"{name}.nc") as product:
                assert product.attrs["product_name"] == name

    def test_sites_are_the_primary_towers_carrying_the_series(self, pipeline):
        values = net_ecosystem_exchange_values(
            ["ameriflux_nee_half_hourly_ustar_variable", "ameriflux_nee_half_hourly_ustar_constant"],
            directory=pipeline["out_dir"],
        )
        assert values["ameriflux_nee_half_hourly_ustar_variable"]["site"].values.tolist() == [101, 102]
        # US-Bbb's file has no CUT columns, so site 102 is absent from the CUT product.
        assert values["ameriflux_nee_half_hourly_ustar_constant"]["site"].values.tolist() == [101]
        assert values["ameriflux_nee_half_hourly_ustar_variable"]["ameriflux_site_id"].values.tolist() == ["US-Aaa", "US-Bbb"]

    def test_a_value_is_labeled_with_its_utc_step_end(self, pipeline):
        name = "ameriflux_nee_half_hourly_ustar_variable"
        value = net_ecosystem_exchange_values(name, sites=102, directory=pipeline["out_dir"])[name]
        frame = _fullset_frame("US-Bbb")
        # US-Bbb is on UTC-8: the step ending 2012-06-01 12:00 UTC ends at 04:00 local standard time.
        row = frame.loc[frame["TIMESTAMP_END"] == 201206010400].iloc[0]
        assert value.sel(site=102, time="2012-06-01T12:00").item() == row["NEE_VUT_REF"]

    def test_no_step_is_lost_or_doubled_around_daylight_saving(self, pipeline):
        name = "ameriflux_nee_half_hourly_ustar_variable"
        value = net_ecosystem_exchange_values(name, sites=101, directory=pipeline["out_dir"])[name]
        frame = _fullset_frame("US-Aaa")
        source = frame["NEE_VUT_REF"].to_numpy()
        # Every source value inside the UTC window is present once; US-Aaa starts
        # 2012-01-01 00:00 local, which is 05:00 UTC, so nothing is cut at the start.
        assert int(np.isfinite(value.values).sum()) == int((source != -9999).sum())

    def test_the_time_coordinates_are_pysipnet_s(self, pipeline):
        name = "ameriflux_nee_hourly_ustar_variable"
        with load_net_ecosystem_exchange(name, pipeline["out_dir"] / f"{name}.nc") as product:
            assert product["time"].values[0] == np.datetime64("2012-01-01T01:00")
            assert (product["time_step_length"].values == np.timedelta64(60, "m")).all()
            assert np.array_equal(product["time_bounds"].values[:, 1], product["time"].values)
            assert product["value"].attrs["kind"] == "timestep_mean"
            assert product["value"].attrs["units"] == "umol m-2 s-1"

    def test_aggregation_averages_onto_utc_three_hour_cells(self, pipeline):
        name = "ameriflux_nee_half_hourly_ustar_variable"
        value = net_ecosystem_exchange_values(name, sites=101, directory=pipeline["out_dir"])[name]
        # Whole cells only: aggregate_time labels a partly covered cell by the
        # last step end it holds. Step i ends (i + 1) half hours after midnight.
        three_hourly = aggregate_time(value.isel(time=slice(1998, 2598)), "3h")
        hours = pd.DatetimeIndex(three_hourly["time"].values)
        assert (hours.hour % 3 == 0).all() and (hours.minute == 0).all()
        assert three_hourly.attrs["kind"] == "timestep_mean"

    def test_quality_flags_come_with_the_values(self, pipeline):
        name = "ameriflux_nee_half_hourly_ustar_variable"
        value = net_ecosystem_exchange_values(name, directory=pipeline["out_dir"])[name]
        flag = net_ecosystem_exchange_quality_flags(name, directory=pipeline["out_dir"])[name]
        assert ((flag >= 0) == np.isfinite(value)).all()

    def test_sites_must_be_whole_numbers_and_present(self, pipeline):
        name = "ameriflux_nee_half_hourly_ustar_variable"
        with pytest.raises(TypeError):
            net_ecosystem_exchange_values(name, sites="101", directory=pipeline["out_dir"])
        with pytest.raises(ValueError, match="not in"):
            net_ecosystem_exchange_values(name, sites=[105], directory=pipeline["out_dir"])

    def test_an_ensemble_source_passes_through_with_a_member_dimension(self, pipeline):
        spec = resolve_net_ecosystem_exchange("ameriflux_nee_half_hourly_ustar_variable")
        table = read_tower_table(pipeline["raw_dir"] / "ameriflux_towers.csv")
        n = len(HALF_HOURLY.product_step_starts())
        ends = HALF_HOURLY.product_step_starts() + HALF_HOURLY.step
        series = xr.Dataset(
            {"value": (("member", "tower", "time"), np.zeros((3, 2, n)))},
            coords={
                "member": np.arange(3),
                "tower": ["US-Aaa", "US-Bbb"],
                "time": ends.as_unit("ns").to_numpy(),
                "utc_offset": ("tower", [-5.0, -8.0]),
            },
        )
        product = build_net_ecosystem_exchange(spec, series, table, load_sites(pipeline["sites"]))
        assert product["value"].dims == ("member", "site", "time")


class TestSpecs:
    def test_names_are_the_raw_stem_and_a_series(self):
        for spec in NET_ECOSYSTEM_EXCHANGE:
            assert spec.name.startswith(spec.raw_file.removesuffix(".nc") + "_")

    def test_describe_option_prints_every_spec(self, capsys):
        assert ingest.main(["--describe"]) == 0
        printed = capsys.readouterr().out
        assert all(name in printed for name in NET_ECOSYSTEM_EXCHANGE_NAMES)


# ── edge cases found in review ────────────────────────────────────────────────


def _site_table_frame() -> pd.DataFrame:
    rows = []
    for site, (j, k, name) in SITES.items():
        lon, lat = SITE_GRID.index_to_lonlat(j, k)
        rows.append({"site_id": site, "lon": lon, "lat": lat, "lon_index": j, "lat_index": k, "site_name": name})
    return pd.DataFrame(rows)


def _summary(tower, *, minutes=30, steps=1000, offset=-5.0, separation=10.0, lag=0):
    return {
        "tower": tower, "resolution_minutes": minutes, "source_file": f"{tower}.csv", "site_version": "4-7",
        "record_steps": steps, "utc_offset_hours": offset, "utc_offset_separation": separation,
        "shortwave_lag_steps": lag,
    }


def _site_list(towers: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"tower": t, "tower_lat": lat, "tower_lon": lon, "igbp": "DBF", "doi": ""} for t, (lon, lat) in towers.items()]
    )


def _empty_pool_input_list() -> pd.DataFrame:
    return pd.DataFrame(columns=["input_order", "tower", "input_lat", "input_lon"])


def _build(summaries, towers):
    from sipnet_calibration.net_ecosystem_exchange import build_tower_table

    frame = pd.DataFrame(summaries)
    frame["shortwave_lag_steps"] = frame["shortwave_lag_steps"].astype("Int32")
    return build_tower_table(frame, _site_list(towers), _empty_pool_input_list(), _site_table_frame()).set_index("tower")


class TestTowerTableEdgeCases:
    def test_the_longer_record_in_time_wins_across_resolutions(self):
        cell = _tower_lonlat("US-Ccc")
        table = _build(
            [_summary("US-Hhh", minutes=30, steps=17000), _summary("US-Rrr", minutes=60, steps=17000, offset=-6.0)],
            {"US-Hhh": cell, "US-Rrr": cell},
        )
        # 17000 hours outlast 17000 half-hours.
        assert table.at["US-Rrr", "primary"] and not table.at["US-Hhh", "primary"]

    def test_a_tower_with_no_potential_shortwave_is_excluded_not_fatal(self):
        table = _build(
            [_summary("US-Old", offset=math.nan, separation=math.nan, lag=None)], {"US-Old": _tower_lonlat("US-Ccc")}
        )
        assert table.at["US-Old", "excluded_reason"].startswith("clock: no SW_IN_POT")
        assert not table.at["US-Old", "primary"]

    def test_an_hourly_tower_on_a_half_hour_offset_is_excluded(self):
        table = _build([_summary("US-Nfl", minutes=60, offset=-3.5)], {"US-Nfl": _tower_lonlat("US-Ccc")})
        assert "not a whole number" in table.at["US-Nfl", "excluded_reason"]

    def test_several_naming_sites_none_in_the_cell_fall_back_to_the_cell(self):
        from sipnet_calibration.net_ecosystem_exchange import match_towers

        sites = _site_table_frame()
        sites.loc[sites.site_id == 101, "site_name"] = "A (US-Two)"
        sites.loc[sites.site_id == 102, "site_name"] = "B (US-Two)"
        lon, lat = _tower_lonlat("US-Ccc")  # site 103's cell
        matched = match_towers(pd.DataFrame({"tower": ["US-Two"], "tower_lon": [lon], "tower_lat": [lat]}), sites, _empty_pool_input_list())
        assert matched.at[0, "site_id"] == 103 and matched.at[0, "match_basis"] == "same_cell"
        assert "several pool sites" in matched.at[0, "comment"]

    @pytest.mark.parametrize("column, text", [("site_id", "101.9"), ("record_steps", "3000000000"), ("utc_offset_hours", "inf")])
    def test_the_table_reader_refuses_malformed_numbers(self, pipeline, tmp_path, column, text):
        frame = pd.read_csv(pipeline["raw_dir"] / "ameriflux_towers.csv", dtype=str, keep_default_na=False)
        frame.loc[frame["tower"] == "US-Aaa", column] = text
        path = tmp_path / "towers.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError):
            read_tower_table(path)


class TestScriptGuards:
    def test_the_tower_table_is_written_beside_the_raw_files_by_default(self, pipeline, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        for path in pipeline["raw_dir"].iterdir():
            if path.suffix in (".nc", ".tsv") or path.name == "Unmatched_Sites.csv":
                (raw / path.name).symlink_to(path)
        assert build_towers.main(["--raw-dir", str(raw), "--sites", str(pipeline["sites"])]) == 0
        assert (raw / "ameriflux_towers.csv").is_file()

    def test_another_download_needs_an_explicit_out_dir(self, pipeline):
        assert convert.main(["--root", str(pipeline["download"])]) == 1

    def test_the_ingest_refuses_a_primary_tower_the_raw_file_lacks(self, pipeline, tmp_path):
        frame = pd.read_csv(pipeline["raw_dir"] / "ameriflux_towers.csv", dtype=str, keep_default_na=False)
        ghost = frame[frame["tower"] == "US-Aaa"].assign(tower="US-Ghs", site_id="105")
        table = tmp_path / "towers.csv"
        pd.concat([frame, ghost]).sort_values("tower").to_csv(table, index=False)
        status = ingest.main(
            ["--raw-dir", str(pipeline["raw_dir"]), "--tower-table", str(table), "--sites", str(pipeline["sites"]),
             "--out-dir", str(tmp_path / "out"), "--product", "ameriflux_nee_half_hourly_ustar_variable"]
        )
        assert status == 1


class TestReaderArguments:
    def test_read_raw_takes_a_string_path(self, pipeline):
        with read_raw(str(raw_path(HOURLY, pipeline["raw_dir"]))) as raw:
            assert raw.attrs["resolution"] == "hourly"

    @pytest.mark.parametrize("sites", [True, [float("nan")], [None]])
    def test_sites_that_are_not_identifiers_are_refused(self, pipeline, sites):
        with pytest.raises(TypeError):
            net_ecosystem_exchange_values("ameriflux_nee_half_hourly_ustar_variable", sites=sites, directory=pipeline["out_dir"])

    def test_cell_of_broadcasts(self):
        j, k = SITE_GRID.cell_of(np.array([-100.0, -90.0]), 40.0)
        assert j.shape == k.shape == (2,)


# ── mutation-testing survivors ────────────────────────────────────────────────


def _listed(*towers_and_cells) -> pd.DataFrame:
    rows = [
        {"input_order": i + 1, "tower": tower, "input_lon": lon, "input_lat": lat}
        for i, (tower, (lon, lat)) in enumerate(towers_and_cells)
    ]
    return pd.DataFrame(rows)


def _match(tower_cells: dict, sites: pd.DataFrame | None = None, pool_input_list: pd.DataFrame | None = None):
    from sipnet_calibration.net_ecosystem_exchange import match_towers

    towers = pd.DataFrame(
        [{"tower": t, "tower_lon": lon, "tower_lat": lat} for t, (lon, lat) in tower_cells.items()]
    )
    matched = match_towers(
        towers,
        sites if sites is not None else _site_table_frame(),
        pool_input_list if pool_input_list is not None else _empty_pool_input_list(),
    )
    return matched.set_index("tower")


class TestMatchingRule:
    def test_only_the_first_listed_tower_in_a_cell_owns_it(self):
        cell = _tower_lonlat("US-Bbb")  # site 102, labeled ameriflux
        matched = _match({"US-One": cell, "US-Two": cell}, pool_input_list=_listed(("US-One", cell), ("US-Two", cell)))
        assert matched.at["US-One", "match_basis"] == "pool_input_list"
        assert matched.at["US-Two", "match_basis"] == "same_cell"

    def test_the_pool_list_owns_only_cells_labeled_ameriflux(self):
        cell = _tower_lonlat("US-Ccc")  # site 103, weighted_sample
        matched = _match({"US-One": cell}, pool_input_list=_listed(("US-One", cell)))
        assert matched.at["US-One", "match_basis"] == "same_cell"

    def test_a_name_wins_over_the_pool_list(self):
        sites = _site_table_frame()
        sites.loc[sites.site_id == 101, "site_name"] = "Named (US-One)"
        cell = _tower_lonlat("US-Bbb")  # the tower stands in 102's cell and is listed there
        matched = _match({"US-One": cell}, sites=sites, pool_input_list=_listed(("US-One", cell)))
        assert matched.at["US-One", "site_id"] == 101 and matched.at["US-One", "match_basis"] == "named_in_pool"

    def test_several_naming_sites_take_the_one_in_the_tower_s_cell(self):
        sites = _site_table_frame()
        sites.loc[sites.site_id.isin([101, 103]), "site_name"] = "(US-Two)"
        matched = _match({"US-Two": _tower_lonlat("US-Ccc")}, sites=sites)
        assert matched.at["US-Two", "site_id"] == 103 and matched.at["US-Two", "match_basis"] == "named_in_pool"


class TestPrimaryAndClock:
    def test_the_first_identifier_breaks_a_full_tie(self):
        cell = _tower_lonlat("US-Ccc")
        table = _build([_summary("US-Bee"), _summary("US-Aye")], {"US-Aye": cell, "US-Bee": cell})
        assert table.at["US-Aye", "primary"] and not table.at["US-Bee", "primary"]
        assert "the first identifier" in table.at["US-Aye", "primary_reason"]

    def test_the_longer_record_wins_at_the_same_basis(self):
        cell = _tower_lonlat("US-Ccc")
        table = _build([_summary("US-Aye", steps=10), _summary("US-Bee", steps=20)], {"US-Aye": cell, "US-Bee": cell})
        assert table.at["US-Bee", "primary"] and "the longer record" in table.at["US-Bee", "primary_reason"]

    def test_an_unclear_offset_is_excluded(self):
        table = _build([_summary("US-Amb", separation=1.5)], {"US-Amb": _tower_lonlat("US-Ccc")})
        assert table.at["US-Amb", "excluded_reason"].startswith("clock: the UTC offset is not clearly recovered")

    def test_an_unchecked_clock_is_excluded(self):
        table = _build([_summary("US-Unc", lag=None)], {"US-Unc": _tower_lonlat("US-Ccc")})
        assert table.at["US-Unc", "excluded_reason"].startswith("clock: too little measured shortwave")
        assert not table.at["US-Unc", "primary"]

    def test_a_one_step_lag_is_kept_and_noted(self):
        table = _build([_summary("US-Lag", lag=1)], {"US-Lag": _tower_lonlat("US-Ccc")})
        assert table.at["US-Lag", "excluded_reason"] == "" and table.at["US-Lag", "primary"]
        assert "within the check's tolerance" in table.at["US-Lag", "comment"]


class TestTowerTableReader:
    @pytest.fixture()
    def frame(self, pipeline):
        return pd.read_csv(pipeline["raw_dir"] / "ameriflux_towers.csv", dtype=str, keep_default_na=False)

    def _refused(self, frame, tmp_path, match):
        path = tmp_path / "towers.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(ValueError, match=match):
            read_tower_table(path)

    def test_two_primaries_at_one_site_are_refused(self, frame, tmp_path):
        frame.loc[frame.tower == "US-Ddd", "primary"] = "True"
        self._refused(frame, tmp_path, "two primary towers")

    def test_an_excluded_primary_is_refused(self, frame, tmp_path):
        frame.loc[frame.tower == "US-Fff", "primary"] = "True"
        self._refused(frame, tmp_path, "unmatched or excluded")


class TestSourceFileRefusals:
    def _written(self, tmp_path, frame, tower="US-Bbb"):
        path = tmp_path / _file_name(tower)
        frame.to_csv(path, index=False)
        return path

    def test_a_value_missing_where_its_flag_is_set_is_refused(self, tmp_path):
        frame = _fullset_frame("US-Bbb")
        frame.loc[10, "NEE_VUT_REF"] = -9999.0
        frame.loc[10, "NEE_VUT_REF_QC"] = 0
        with pytest.raises(ValueError, match="where NEE_VUT_REF_QC is set"):
            read_source_file(self._written(tmp_path, frame))

    def test_a_record_short_of_its_declared_years_is_refused(self, tmp_path):
        frame = _fullset_frame("US-Bbb").iloc[:-48]
        with pytest.raises(ValueError, match="whole years"):
            read_source_file(self._written(tmp_path, frame))

    def test_a_non_finite_value_is_refused(self, tmp_path):
        frame = _fullset_frame("US-Bbb")
        frame.loc[7, "SW_IN_F"] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            read_source_file(self._written(tmp_path, frame))


class TestWritesArePublishedAtomically:
    def test_the_ingest_leaves_nothing_behind_when_its_round_trip_fails(self, pipeline, tmp_path, monkeypatch):
        spec = resolve_net_ecosystem_exchange("ameriflux_nee_hourly_ustar_variable")
        dataset = load_net_ecosystem_exchange(spec.name, pipeline["out_dir"] / f"{spec.name}.nc").load()
        monkeypatch.setattr(ingest, "check_round_trip", lambda *a: (_ for _ in ()).throw(ingest.IngestError("no")))
        out = tmp_path / f"{spec.name}.nc"
        with pytest.raises(ingest.IngestError):
            ingest.write_product(dataset, spec, out)
        assert not out.exists() and not out.with_suffix(".nc.partial").exists()

    def test_the_conversion_leaves_nothing_behind_when_its_round_trip_fails(self, pipeline, tmp_path, monkeypatch):
        with read_raw(raw_path(HOURLY, pipeline["raw_dir"])) as raw:
            dataset = raw.load()
        monkeypatch.setattr(convert, "check_round_trip", lambda *a: (_ for _ in ()).throw(convert.ConversionError("no")))
        out = tmp_path / "ameriflux_nee_hourly.nc"
        with pytest.raises(convert.ConversionError):
            convert.write_raw(dataset, out)
        assert not out.exists() and not out.with_suffix(".nc.partial").exists()


class TestIngestRefusals:
    def _run(self, pipeline, tmp_path, frame):
        table = tmp_path / "towers.csv"
        frame.to_csv(table, index=False)
        return ingest.main(
            ["--raw-dir", str(pipeline["raw_dir"]), "--tower-table", str(table), "--sites", str(pipeline["sites"]),
             "--out-dir", str(tmp_path / "out"), "--product", "ameriflux_nee_half_hourly_ustar_variable"]
        )

    def test_a_raw_tower_missing_from_the_table_is_refused(self, pipeline, tmp_path, capsys):
        frame = pd.read_csv(pipeline["raw_dir"] / "ameriflux_towers.csv", dtype=str, keep_default_na=False)
        assert self._run(pipeline, tmp_path, frame[frame.tower != "US-Eee"]) == 1
        assert "not in the tower table" in capsys.readouterr().err

    def test_a_resolution_disagreement_is_refused(self, pipeline, tmp_path):
        frame = pd.read_csv(pipeline["raw_dir"] / "ameriflux_towers.csv", dtype=str, keep_default_na=False)
        frame.loc[frame.tower == "US-Eee", "resolution_minutes"] = "60"
        assert self._run(pipeline, tmp_path, frame) == 1

    def test_a_value_without_a_quality_flag_is_refused(self, pipeline):
        spec = resolve_net_ecosystem_exchange("ameriflux_nee_hourly_ustar_variable")
        dataset = load_net_ecosystem_exchange(spec.name, pipeline["out_dir"] / f"{spec.name}.nc").load()
        where = np.argwhere(np.isfinite(dataset["value"].values))[0]
        dataset["quality_flag"].values[tuple(where)] = -1
        with pytest.raises(ingest.IngestError, match="no quality flag"):
            ingest.check_product(dataset, spec)


class TestBuildProduct:
    def _series(self, towers, offsets):
        n = len(HALF_HOURLY.product_step_starts())
        ends = HALF_HOURLY.product_step_starts() + HALF_HOURLY.step
        return xr.Dataset(
            {"value": (("tower", "time"), np.arange(len(towers) * n, dtype=float).reshape(len(towers), n))},
            coords={"tower": towers, "time": ends.as_unit("ns").to_numpy(), "utc_offset": ("tower", offsets)},
        )

    def test_sites_come_out_ascending_whatever_the_tower_order(self, pipeline):
        spec = resolve_net_ecosystem_exchange("ameriflux_nee_half_hourly_ustar_variable")
        table = read_tower_table(pipeline["raw_dir"] / "ameriflux_towers.csv")
        # US-Bbb (site 102) before US-Aaa (site 101).
        product = build_net_ecosystem_exchange(spec, self._series(["US-Bbb", "US-Aaa"], [-8.0, -5.0]), table, load_sites(pipeline["sites"]))
        assert product["site"].values.tolist() == [101, 102]
        assert product["ameriflux_site_id"].values.tolist() == ["US-Aaa", "US-Bbb"]

    def test_a_tower_that_is_not_primary_is_refused(self, pipeline):
        spec = resolve_net_ecosystem_exchange("ameriflux_nee_half_hourly_ustar_variable")
        table = read_tower_table(pipeline["raw_dir"] / "ameriflux_towers.csv")
        with pytest.raises(ValueError, match="not primary"):
            build_net_ecosystem_exchange(spec, self._series(["US-Ddd"], [-5.0]), table, load_sites(pipeline["sites"]))


class TestRawReaderRefusals:
    def test_a_flag_outside_its_vocabulary_is_refused(self, pipeline, tmp_path):
        from sipnet_calibration.net_ecosystem_exchange import raw_encoding

        with read_raw(raw_path(HOURLY, pipeline["raw_dir"])) as raw:
            dataset = raw.load()
        dataset["NEE_VUT_REF_QC"].values[0, 0] = 5
        path = tmp_path / "bad.nc"
        dataset.to_netcdf(path, engine="h5netcdf", encoding=raw_encoding(dataset))
        with pytest.raises(ValueError, match="outside its vocabulary"):
            read_raw(path)


def test_cell_of_refuses_a_point_east_of_the_grid():
    with pytest.raises(ValueError, match="longitude outside"):
        SITE_GRID.cell_of(SITE_GRID.east + 1e-9, 40.0)
