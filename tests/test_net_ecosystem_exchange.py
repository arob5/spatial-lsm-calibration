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
