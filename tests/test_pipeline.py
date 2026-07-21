from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from shrecc.pipeline import NewDatabase


def _canonical_results(year):
    times = pd.date_range(f"{year}-06-01 10:00", periods=2, freq="h")
    consumption_mix = xr.DataArray(
        np.ones((2, 1, 1, 1)),
        dims=("time", "consumer_country", "source_country", "technology"),
        coords={
            "time": times,
            "consumer_country": ["FR"],
            "source_country": ["FR"],
            "technology": ["Wind"],
        },
        name="consumption_mix",
    )
    consumption_volume = xr.DataArray(
        [[100.0], [120.0]],
        dims=("time", "consumer_country"),
        coords={"time": times, "consumer_country": ["FR"]},
        name="consumption_volume",
    )
    return xr.Dataset(
        {
            "consumption_mix": consumption_mix,
            "consumption_volume": consumption_volume,
        },
        attrs={"volume_unit": "MWh"},
    )


def _activity_mix(year):
    times = pd.date_range(f"{year}-06-01 10:00", periods=2, freq="h")
    return xr.DataArray(
        np.ones((2, 1, 1)),
        dims=("time", "consumer_country", "activity"),
        coords={
            "time": times,
            "consumer_country": ["FR"],
            "activity": [0],
            "geography": ("activity", ["FR"]),
            "activity_name": ("activity", ["wind activity"]),
            "product": ("activity", ["electricity, high voltage"]),
            "unit": ("activity", ["kWh"]),
        },
        name="activity_mix",
    )


def _premise_activity_mix(year):
    times = pd.date_range(f"{year}-06-01 10:00", periods=2, freq="h")
    return xr.DataArray(
        np.ones((2, 1, 1, 1)),
        dims=(
            "time",
            "consumer_country",
            "source_country",
            "premise_activity",
        ),
        coords={
            "time": times,
            "consumer_country": ["FR"],
            "source_country": ["FR"],
            "premise_activity": ["wind activity"],
        },
        name="consumption_mix",
    )


def _database_table():
    index = pd.MultiIndex.from_tuples(
        [("FR", "wind activity", "electricity, high voltage", "kWh")],
        names=["geography", "activityName", "product", "unit"],
    )
    return pd.DataFrame({"FR": [1.0]}, index=index)


def _energy_charts_data():
    columns = pd.MultiIndex.from_tuples(
        [
            ("A", "production mix", "Solar"),
            ("A", "trade", "B"),
            ("A", "load", "load"),
            ("B", "production mix", "Wind"),
            ("B", "trade", "A"),
            ("B", "load", "load"),
        ],
        names=["country", "type", "source"],
    )
    return pd.DataFrame(
        [[100.0, -40.0, 60.0, 60.0, 40.0, 100.0]],
        index=pd.to_datetime(["2025-06-01 10:00"]),
        columns=columns,
    )


def test_historical_create_uses_canonical_cache_and_retains_volume(monkeypatch):
    results = _canonical_results(2025)
    activity_mix = _activity_mix(2025)
    table = _database_table()
    monkeypatch.setattr("shrecc.pipeline.get_energy_charts_data", MagicMock())
    process = MagicMock(return_value="cache")
    monkeypatch.setattr("shrecc.pipeline.process_energy_charts_data", process)
    monkeypatch.setattr(
        "shrecc.pipeline.load_consumption_result_cache",
        MagicMock(return_value=results),
    )
    monkeypatch.setattr("shrecc.pipeline.load_mapping_data", MagicMock())
    monkeypatch.setattr(
        "shrecc.pipeline.map_consumption_mix_to_ecoinvent_activities",
        MagicMock(return_value=activity_mix),
    )
    monkeypatch.setattr(
        "shrecc.pipeline.activity_mix_to_database_table",
        MagicMock(return_value=table),
    )

    database = NewDatabase(
        years=2025,
        countries=["FR"],
        project_name="project",
        premise_db="ecoinvent-3.11-cutoff",
        my_db_name="shrecc_FR_2025",
        general_range=["2025-06-01 10:00", "2025-06-01 11:00"],
        cutoff=0,
        include_cutoff=False,
        data_dir="data",
    ).create()

    assert database.sources == {2025: "energy_charts"}
    assert database.table().equals(table)
    assert "consumption_mix_volume" in database.results()
    xr.testing.assert_allclose(
        database.results()["consumption_mix_volume"].sum(
            ["source_country", "technology"]
        ),
        database.results()["consumption_volume"],
    )
    process.assert_called_once()


def test_historical_create_runs_real_canonical_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "shrecc.pipeline.get_energy_charts_data",
        MagicMock(return_value=_energy_charts_data()),
    )

    database = NewDatabase(
        years=2025,
        countries=["A", "B"],
        project_name="project",
        premise_db="ecoinvent",
        my_db_name="shrecc_AB_2025",
        times=["2025-06-01 10:00"],
        cutoff=0,
        include_cutoff=False,
        data_dir=tmp_path,
    ).create()

    np.testing.assert_allclose(database.table().sum(), [1, 1])
    assert database.results().attrs["solver"] == "country_trade_block"
    assert (tmp_path / "2025" / "consumption_results_v1" / "manifest.json").is_file()


def test_historical_create_repairs_cache_missing_required_country(
    monkeypatch,
    tmp_path,
):
    cache_dir = tmp_path / "2025" / "consumption_results_v1"
    cache_dir.mkdir(parents=True)
    (cache_dir / "manifest.json").write_text("{}", encoding="utf-8")
    pd.to_pickle(
        {"fr": {"production mix": pd.DataFrame()}},
        tmp_path / "2025" / "prod_and_trade_data_2025.pkl",
    )

    complete = _canonical_results(2025).assign_coords(
        consumer_country=["NL"],
        source_country=["NL"],
    )
    # NL appears in the solved coordinates as a trade partner despite having
    # no raw production/load response in the partial API cache.
    load_results = MagicMock(side_effect=[complete, complete])
    download = MagicMock(return_value=_energy_charts_data())
    process = MagicMock(return_value=cache_dir)
    monkeypatch.setattr(
        "shrecc.pipeline.load_consumption_result_cache",
        load_results,
    )
    monkeypatch.setattr("shrecc.pipeline.get_energy_charts_data", download)
    monkeypatch.setattr("shrecc.pipeline.process_energy_charts_data", process)
    monkeypatch.setattr("shrecc.pipeline.load_mapping_data", MagicMock())
    monkeypatch.setattr(
        "shrecc.pipeline.map_consumption_mix_to_ecoinvent_activities",
        MagicMock(return_value=_activity_mix(2025)),
    )
    monkeypatch.setattr(
        "shrecc.pipeline.activity_mix_to_database_table",
        MagicMock(return_value=_database_table()),
    )

    database = NewDatabase(
        years=2025,
        countries=["NL"],
        project_name="project",
        premise_db="ecoinvent",
        my_db_name="shrecc_NL_2025",
        times=["2025-06-01 10:00"],
        data_dir=tmp_path,
    ).create()

    download.assert_called_once_with(
        2025,
        path_to_data=tmp_path,
        required_countries=("NL",),
    )
    process.assert_called_once()
    assert "NL" in database.results()["consumer_country"]


def test_prospective_create_maps_with_premise_and_write_is_separate(monkeypatch):
    year = 2040
    times = pd.date_range(f"{year}-06-01 10:00", periods=2, freq="h")
    Z_gross = pd.DataFrame({"value": [1.0, 2.0]}, index=times)
    results = _canonical_results(year)
    activity_mix = _premise_activity_mix(year)
    exchange_map = pd.DataFrame(
        {"FR": ["FR"]},
        index=pd.Index(["wind activity"], name="premise_activity"),
    )
    table = _database_table()

    monkeypatch.setattr(
        "shrecc.pipeline.build_z_gross_from_tyndp_scenario",
        MagicMock(return_value=Z_gross),
    )
    monkeypatch.setattr(
        "shrecc.pipeline.consumption_results_from_z_gross",
        MagicMock(return_value=results),
    )
    mapper = MagicMock()
    mapper.map_technologies.return_value = activity_mix
    mapper.build_exchange_geography_map.return_value = exchange_map
    monkeypatch.setattr(
        "shrecc.pipeline.PremiseConsumptionMixMapper",
        MagicMock(return_value=mapper),
    )
    monkeypatch.setattr(
        "shrecc.pipeline.premise_activity_mix_to_database_table",
        MagicMock(return_value=table),
    )
    write_database = MagicMock()
    monkeypatch.setattr("shrecc.pipeline.create_database", write_database)

    database = NewDatabase(
        scenario="DE",
        years=[year],
        climate_year=2009,
        premise_db="premise-remind-eu-2040",
        my_db_name="shrecc_DE_2040",
        countries=["FR"],
        general_range=["2040-06-01 10:00", "2040-06-01 11:00"],
        project_name="project",
        cutoff=0,
        include_cutoff=False,
    ).create()

    assert database.sources == {year: "tyndp"}
    assert database.exchange_geography_maps[year].equals(exchange_map)
    write_database.assert_not_called()

    database.write()

    write_database.assert_called_once_with(
        dataframe_filt=database.database_tables[year],
        project_name="project",
        db_name="shrecc_DE_2040",
        eidb_name="premise-remind-eu-2040",
        network=True,
        strict=False,
    )


def test_multi_year_names_and_time_ranges_are_expanded():
    database = NewDatabase(
        scenario="DE",
        years=[2035, 2040],
        climate_year=2009,
        premise_db="premise-remind-eu-{year}",
        my_db_name="shrecc_DE",
        countries=["FR"],
        general_range=["2040-06-01 10:00", "2040-06-30 14:00"],
        project_name="project",
    )

    assert database.background_databases == {
        2035: "premise-remind-eu-2035",
        2040: "premise-remind-eu-2040",
    }
    assert database.database_names == {
        2035: "shrecc_DE_2035",
        2040: "shrecc_DE_2040",
    }
    assert database._selection_for_year(2035)[0][0].year == 2035


def test_multiple_years_require_explicit_background_database_names():
    with pytest.raises(ValueError, match="premise_db must be a year mapping"):
        NewDatabase(
            scenario="DE",
            years=[2035, 2040],
            climate_year=2009,
            premise_db="one-background-database",
            my_db_name="shrecc_DE",
            countries=["FR"],
            general_range=["2040-06-01", "2040-06-30"],
            project_name="project",
        )


def test_tyndp_scenario_year_is_validated_during_initialization():
    with pytest.raises(ValueError, match="scenario year 2035"):
        NewDatabase(
            scenario="NT",
            years=2035,
            climate_year=2009,
            premise_db="premise-remind-eu-2035",
            my_db_name="shrecc_NT_2035",
            countries=["FR"],
            general_range=["2035-06-01", "2035-06-30"],
            project_name="project",
        )
