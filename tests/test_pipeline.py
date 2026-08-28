from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from shrecc.pipeline import (
    NewDatabase,
    _adapt_profile_to_tyndp_times,
)


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
    results = _canonical_results(2025).reindex(
        consumer_country=["FR", "DE"],
        fill_value=1.0,
    )
    activity_mix = _activity_mix(2025)
    table = _database_table()
    monkeypatch.setattr("shrecc.pipeline.get_energy_charts_data", MagicMock())
    process = MagicMock(return_value="cache")
    monkeypatch.setattr("shrecc.pipeline.process_energy_charts_data", process)
    monkeypatch.setattr(
        "shrecc.pipeline.load_consumption_result_cache",
        MagicMock(return_value=results),
    )
    monkeypatch.setattr("shrecc.pipeline.load_ecoinvent_mapping", MagicMock())
    map_activities = MagicMock(
        return_value=(
            activity_mix,
            xr.zeros_like(
                results["consumption_mix"]
                .sel(consumer_country=["FR"])
                .mean("time")
            ),
        )
    )
    monkeypatch.setattr(
        "shrecc.pipeline.map_consumption_mix_to_ecoinvent_activities",
        map_activities,
    )
    monkeypatch.setattr(
        "shrecc.pipeline.activity_mix_to_database_table",
        MagicMock(return_value=table),
    )

    database = NewDatabase(
        years=2025,
        countries=["FR"],
        project_name="project",
        bg_db_name="ecoinvent-3.11-cutoff",
        my_db_name="shrecc_FR_2025",
        time_range=["2025-06-01 10:00", "2025-06-01 11:00"],
        cutoff=0,
        include_cutoff=False,
        data_dir="data",
    ).create()

    assert database.sources == {2025: "energy_charts"}
    assert database.table().equals(table)
    assert database.results()["consumer_country"].to_numpy().tolist() == [
        "FR",
        "DE",
    ]
    assert (
        map_activities.call_args.args[0]["consumer_country"].to_numpy().tolist()
        == ["FR"]
    )
    assert "time" not in map_activities.call_args.args[0].dims
    assert "consumption_mix_volume" in database.results()
    assert database.mapping_report().empty
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
        bg_db_name="ecoinvent",
        my_db_name="shrecc_AB_2025",
        times=["2025-06-01 10:00"],
        cutoff=0,
        include_cutoff=False,
        data_dir=tmp_path,
    ).create()

    np.testing.assert_allclose(database.table().sum(), [1, 1])
    assert database.results().attrs["solver"] == "country_trade_block"
    assert (tmp_path / "2025" / "consumption_results_v1" / "manifest.json").is_file()


def test_historical_create_applies_national_demand_before_activity_mapping(
    monkeypatch,
):
    year = 2025
    times = pd.date_range(f"{year}-06-01 10:00", periods=2, freq="h")
    consumption_mix = xr.DataArray(
        [[[[1.0], [0.0]]], [[[0.0], [1.0]]]],
        dims=("time", "consumer_country", "source_country", "technology"),
        coords={
            "time": times,
            "consumer_country": ["FR"],
            "source_country": ["FR", "DE"],
            "technology": ["Wind"],
        },
        name="consumption_mix",
    )
    consumption_volume = xr.DataArray(
        [[1.0], [3.0]],
        dims=("time", "consumer_country"),
        coords={"time": times, "consumer_country": ["FR"]},
        name="consumption_volume",
    )
    results = xr.Dataset(
        {
            "consumption_mix": consumption_mix,
            "consumption_volume": consumption_volume,
        },
        attrs={"volume_unit": "MWh"},
    )
    mapped_activity_mix = _activity_mix(year).mean("time")
    map_activities = MagicMock(
        return_value=(
            mapped_activity_mix,
            xr.zeros_like(consumption_mix.mean("time")),
        )
    )

    monkeypatch.setattr("shrecc.pipeline.get_energy_charts_data", MagicMock())
    monkeypatch.setattr(
        "shrecc.pipeline.process_energy_charts_data",
        MagicMock(return_value="cache"),
    )
    monkeypatch.setattr(
        "shrecc.pipeline.load_consumption_result_cache",
        MagicMock(return_value=results),
    )
    monkeypatch.setattr("shrecc.pipeline.load_ecoinvent_mapping", MagicMock())
    monkeypatch.setattr(
        "shrecc.pipeline.map_consumption_mix_to_ecoinvent_activities",
        map_activities,
    )
    monkeypatch.setattr(
        "shrecc.pipeline.activity_mix_to_database_table",
        MagicMock(return_value=_database_table()),
    )

    NewDatabase(
        years=year,
        countries=["FR"],
        project_name="project",
        bg_db_name="ecoinvent",
        my_db_name="shrecc_FR_2025",
        time_range=[times[0], times[-1]],
        consumption_profile="national_demand",
        data_dir="data",
    ).create()

    weighted_mix = map_activities.call_args.args[0]
    assert "time" not in weighted_mix.dims
    np.testing.assert_allclose(
        weighted_mix.to_numpy().ravel(),
        [0.25, 0.75],
    )


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
    monkeypatch.setattr("shrecc.pipeline.load_ecoinvent_mapping", MagicMock())
    monkeypatch.setattr(
        "shrecc.pipeline.map_consumption_mix_to_ecoinvent_activities",
        MagicMock(
            return_value=(
                _activity_mix(2025),
                xr.zeros_like(complete["consumption_mix"]),
            )
        ),
    )
    monkeypatch.setattr(
        "shrecc.pipeline.activity_mix_to_database_table",
        MagicMock(return_value=_database_table()),
    )

    database = NewDatabase(
        years=2025,
        countries=["NL"],
        project_name="project",
        bg_db_name="ecoinvent",
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
        bg_db_name="premise-remind-eu-2040",
        my_db_name="shrecc_DE_2040",
        countries=["FR"],
        time_range=["2040-06-01 10:00", "2040-06-01 11:00"],
        project_name="project",
        cutoff=0,
        include_cutoff=False,
    ).create()

    assert database.sources == {year: "tyndp"}
    assert database.exchange_geography_maps[year].equals(exchange_map)
    assert "time" not in mapper.map_technologies.call_args.args[0].dims
    write_database.assert_not_called()

    database.write()

    write_database.assert_called_once_with(
        dataframe_filt=database.database_tables[year],
        project_name="project",
        db_name="shrecc_DE_2040",
        eidb_name="premise-remind-eu-2040",
        network=True,
        strict=False,
        year=year,
        consumption_profile="flat",
        inventory_resolution="annual",
    )


def test_monthly_resolution_reaches_mapping_table_and_writer(monkeypatch):
    year = 2040
    times = pd.date_range(f"{year}-06-01 10:00", periods=2, freq="h")
    results = _canonical_results(year)
    mapper = MagicMock()
    mapper.map_technologies.return_value = _premise_activity_mix(year)
    mapper.build_exchange_geography_map.return_value = pd.DataFrame(
        {"FR": ["FR"]},
        index=pd.Index(["wind activity"], name="premise_activity"),
    )
    table_builder = MagicMock(return_value=_database_table())
    write_database = MagicMock()

    monkeypatch.setattr(
        "shrecc.pipeline.build_z_gross_from_tyndp_scenario",
        MagicMock(return_value=pd.DataFrame({"value": [1.0, 2.0]}, index=times)),
    )
    monkeypatch.setattr(
        "shrecc.pipeline.consumption_results_from_z_gross",
        MagicMock(return_value=results),
    )
    monkeypatch.setattr(
        "shrecc.pipeline.PremiseConsumptionMixMapper",
        MagicMock(return_value=mapper),
    )
    monkeypatch.setattr(
        "shrecc.pipeline.premise_activity_mix_to_database_table",
        table_builder,
    )
    monkeypatch.setattr("shrecc.pipeline.create_database", write_database)

    database = NewDatabase(
        scenario="DE",
        years=year,
        climate_year=2009,
        bg_db_name="premise-remind-eu-2040",
        my_db_name="shrecc_monthly",
        countries=["FR"],
        time_range=[times[0], times[-1]],
        project_name="project",
        inventory_resolution="monthly",
        cutoff=0,
        include_cutoff=False,
    ).create()

    mapped_mix = mapper.map_technologies.call_args.args[0]
    assert mapped_mix["time"].to_index().equals(pd.to_datetime(["2040-06-01"]))
    assert mapped_mix.attrs["inventory_resolution"] == "monthly"
    assert table_builder.call_args.kwargs["inventory_resolution"] == "monthly"

    database.write()

    assert write_database.call_args.kwargs["inventory_resolution"] == "monthly"


def test_create_can_release_hourly_results_after_aggregation(monkeypatch):
    year = 2040
    times = pd.date_range(f"{year}-06-01 10:00", periods=2, freq="h")
    results = _canonical_results(year)
    solve = MagicMock(return_value=results)
    mapper = MagicMock()
    mapper.map_technologies.return_value = _premise_activity_mix(year)
    mapper.build_exchange_geography_map.return_value = pd.DataFrame(
        {"FR": ["FR"]},
        index=pd.Index(["wind activity"], name="premise_activity"),
    )

    monkeypatch.setattr(
        "shrecc.pipeline.build_z_gross_from_tyndp_scenario",
        MagicMock(return_value=pd.DataFrame({"value": [1.0, 2.0]}, index=times)),
    )
    monkeypatch.setattr(
        "shrecc.pipeline.consumption_results_from_z_gross",
        solve,
    )
    monkeypatch.setattr(
        "shrecc.pipeline.PremiseConsumptionMixMapper",
        MagicMock(return_value=mapper),
    )
    monkeypatch.setattr(
        "shrecc.pipeline.premise_activity_mix_to_database_table",
        MagicMock(return_value=_database_table()),
    )

    database = NewDatabase(
        scenario="DE",
        years=year,
        climate_year=2009,
        bg_db_name="premise-remind-eu-2040",
        my_db_name="shrecc_DE_2040",
        countries=["FR"],
        time_range=[times[0], times[-1]],
        project_name="project",
        include_cutoff=False,
        include_consumption_mix_volume=False,
        retain_hourly_results=False,
    ).create()

    assert database.consumption_results == {}
    assert database.table().equals(_database_table())
    assert solve.call_args.kwargs["include_consumption_mix_volume"] is False
    with pytest.raises(RuntimeError, match="were not retained"):
        database.results()


def test_lcia_uses_retained_hourly_results_without_writing(monkeypatch, tmp_path):
    year = 2025
    results = _canonical_results(year)
    table = pd.DataFrame(
        [[1.0, 1.0]],
        index=pd.MultiIndex.from_tuples(
            [("FR", "wind", "electricity", "kWh")],
            names=["geography", "activityName", "product", "unit"],
        ),
        columns=pd.MultiIndex.from_product(
            [results["time"].to_index(), ["FR"]],
            names=["time", "country"],
        ),
    )
    method = ("ecoinvent", "EF v3.1", "climate change", "GWP100")
    intensity = xr.DataArray(
        np.array([[[10.0]], [[20.0]]]),
        dims=("time", "consumer_country", "impact_category"),
        coords={
            "time": results["time"],
            "consumer_country": ["FR"],
            "impact_category": ["climate"],
        },
    )
    basis = object()
    database = NewDatabase(
        years=year,
        source="energy_charts",
        bg_db_name="ecoinvent",
        my_db_name="shrecc",
        countries=["FR"],
        project_name="project",
        times=results["time"].to_index(),
        data_dir=tmp_path,
    )
    database.consumption_results[year] = results
    monkeypatch.setattr(
        database,
        "_hourly_database_table",
        MagicMock(return_value=table),
    )
    monkeypatch.setattr(
        "shrecc.pipeline.resolve_lcia_methods",
        MagicMock(return_value=(method,)),
    )
    build_basis = MagicMock(return_value=basis)
    monkeypatch.setattr(
        "shrecc.pipeline.build_resolved_inventory_basis",
        build_basis,
    )
    source_score_cache = {}

    def calculate_scores(*args, source_score_cache, **kwargs):
        source_score_cache[("ecoinvent", "wind")] = np.array([10.0])
        return intensity

    calculate = MagicMock(side_effect=calculate_scores)
    monkeypatch.setattr("shrecc.pipeline.calculate_lcia", calculate)
    cache_path = tmp_path / "source-scores.npz"
    load_cache = MagicMock(return_value=(cache_path, source_score_cache))
    write_cache = MagicMock()
    monkeypatch.setattr("shrecc.pipeline.load_source_score_cache", load_cache)
    monkeypatch.setattr("shrecc.pipeline.write_source_score_cache", write_cache)
    monkeypatch.setattr("shrecc.pipeline.bd.projects.set_current", MagicMock())
    background_index = object()
    build_background_index = MagicMock(return_value=background_index)
    monkeypatch.setattr(
        "shrecc.pipeline.build_background_activity_index",
        build_background_index,
    )

    lcia_results = database.lcia()

    assert database.lcia_results is lcia_results
    assert lcia_results.annual().sel(
        year=year,
        consumer_country="FR",
        impact_category="climate",
    ).item() == pytest.approx(15.0)
    build_basis.assert_called_once_with(
        table,
        year=year,
        background_database="ecoinvent",
        include_network=True,
        strict=False,
        background_index=background_index,
    )
    assert calculate.call_args.args == (basis, (method,))
    assert calculate.call_args.kwargs == {
        "engine": "linear",
        "batch_size": 1000,
        "source_score_cache": source_score_cache,
    }
    load_cache.assert_called_once_with(tmp_path, "ecoinvent", (method,))
    write_cache.assert_called_once_with(
        cache_path,
        source_score_cache,
        (method,),
    )
    build_background_index.assert_called_once_with("ecoinvent")


def test_lcia_requires_retained_hourly_results():
    database = NewDatabase(
        years=2025,
        source="energy_charts",
        bg_db_name="ecoinvent",
        my_db_name="shrecc",
        countries=["FR"],
        project_name="project",
        retain_hourly_results=False,
        times=["2025-06-01 10:00"],
    )

    with pytest.raises(RuntimeError, match="retain_hourly_results=True"):
        database.lcia()


def test_prospective_create_reports_countries_absent_from_tyndp_before_solving(
    monkeypatch,
):
    year = 2035
    times = pd.date_range(f"{year}-01-01", periods=2, freq="h")
    columns = pd.MultiIndex.from_tuples(
        [
            ("production mix", "FR", "FR", "Wind"),
            ("trade", "FR", "DE", "electricity"),
        ],
        names=["type", "country from", "country to", "source"],
    )
    Z_gross = pd.DataFrame(1.0, index=times, columns=columns)
    solve = MagicMock()
    monkeypatch.setattr(
        "shrecc.pipeline.build_z_gross_from_tyndp_scenario",
        MagicMock(return_value=Z_gross),
    )
    monkeypatch.setattr(
        "shrecc.pipeline.consumption_results_from_z_gross",
        solve,
    )

    database = NewDatabase(
        scenario="DE",
        years=year,
        climate_year=2009,
        bg_db_name="premise-remind-eu-2035",
        my_db_name="shrecc_DE_2035",
        countries=["FR", "MD", "UA"],
        time_range=[f"{year}-01-01 00:00", f"{year}-01-01 01:00"],
        project_name="project",
    )

    with pytest.raises(ValueError, match=r"requested countries: MD, UA"):
        database.create()
    solve.assert_not_called()


def test_multi_year_names_and_time_ranges_are_expanded():
    database = NewDatabase(
        scenario="DE",
        years=[2035, 2040],
        climate_year=2009,
        bg_db_name="premise-remind-eu-{year}",
        my_db_name="shrecc_DE",
        countries=["FR"],
        time_range=["2040-06-01 10:00", "2040-06-30 14:00"],
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


def test_missing_time_selection_defaults_to_each_complete_year():
    database = NewDatabase(
        years=[2025, 2026],
        source="energy_charts",
        bg_db_name="ecoinvent-{year}",
        my_db_name="shrecc_annual",
        countries=["FR"],
        project_name="project",
    )

    assert database.time_range == [
        "2025-01-01 00:00:00",
        "2025-12-31 23:00:00",
    ]
    range_2025, _ = database._selection_for_year(2025)
    range_2026, _ = database._selection_for_year(2026)
    assert range_2025 == list(
        pd.to_datetime(["2025-01-01 00:00:00", "2025-12-31 23:00:00"])
    )
    assert range_2026 == list(
        pd.to_datetime(["2026-01-01 00:00:00", "2026-12-31 23:00:00"])
    )


def test_hour_range_uses_default_complete_year():
    database = NewDatabase(
        years=2025,
        source="energy_charts",
        bg_db_name="ecoinvent",
        my_db_name="shrecc_daytime",
        countries=["FR"],
        project_name="project",
        hour_range=[10, 14],
    )

    assert database.time_range == [
        "2025-01-01 00:00:00",
        "2025-12-31 23:00:00",
    ]
    assert database.hour_range == [10, 14]


def test_february_time_range_clips_leap_day_for_non_leap_model_years():
    database = NewDatabase(
        years=[2035, 2040, 2050],
        source="energy_charts",
        bg_db_name="ecoinvent-{year}",
        my_db_name="shrecc_february",
        countries=["FR"],
        time_range=["2040-02-01 00:00", "2040-02-29 23:00"],
        project_name="project",
    )

    with pytest.warns(UserWarning, match="Clipped 1 February 29"):
        range_2035, _ = database._selection_for_year(2035)
    range_2040, _ = database._selection_for_year(2040)
    with pytest.warns(UserWarning, match="Clipped 1 February 29"):
        range_2050, _ = database._selection_for_year(2050)

    assert range_2035 == list(
        pd.to_datetime(["2035-02-01 00:00", "2035-02-28 23:00"])
    )
    assert range_2040 == list(
        pd.to_datetime(["2040-02-01 00:00", "2040-02-29 23:00"])
    )
    assert range_2050 == list(
        pd.to_datetime(["2050-02-01 00:00", "2050-02-28 23:00"])
    )


def test_explicit_february_29_time_is_dropped_for_non_leap_model_year():
    database = NewDatabase(
        years=[2035, 2040],
        source="energy_charts",
        bg_db_name="ecoinvent-{year}",
        my_db_name="shrecc_explicit_times",
        countries=["FR"],
        times=["2040-02-28 12:00", "2040-02-29 12:00"],
        project_name="project",
    )

    with pytest.warns(UserWarning, match="Dropped 1 explicit February 29"):
        _, times_2035 = database._selection_for_year(2035)

    assert times_2035.equals(pd.to_datetime(["2035-02-28 12:00"]))


def test_single_year_custom_profile_is_a_partial_reusable_template():
    profile = pd.Series(
        [1.0, 2.0],
        index=pd.to_datetime(
            ["2024-06-01 10:00", "2024-06-02 14:00"]
        ),
        name="office",
    )

    database = NewDatabase(
        years=[2025, 2026],
        source="energy_charts",
        bg_db_name="ecoinvent-{year}",
        my_db_name="shrecc_profile",
        countries=["FR"],
        project_name="project",
        consumption_profile=profile,
    )

    assert database._selection_for_year(2025)[1].equals(
        pd.to_datetime(["2025-06-01 10:00", "2025-06-02 14:00"])
    )
    assert database._selection_for_year(2026)[1].equals(
        pd.to_datetime(["2026-06-01 10:00", "2026-06-02 14:00"])
    )


def test_custom_profile_warns_and_ignores_explicit_time_selection():
    profile = pd.Series(
        [1.0],
        index=pd.to_datetime(["2025-06-01 10:00"]),
    )

    with pytest.warns(UserWarning, match="Ignoring: times, time_range, hour_range"):
        database = NewDatabase(
            years=2025,
            source="energy_charts",
            bg_db_name="ecoinvent",
            my_db_name="shrecc_profile",
            countries=["FR"],
            project_name="project",
            consumption_profile=profile,
            times=["2025-07-01 10:00"],
            time_range=["2025-08-01", "2025-08-02"],
            hour_range=[10, 14],
        )

    assert database.times is None
    assert database.time_range is None
    assert database.hour_range is None
    assert database._selection_for_year(2025)[1].equals(profile.index)


def test_hourly_inventory_resolution_points_to_lcia():
    with pytest.raises(ValueError, match=r"Use NewDatabase\.lcia\(\)"):
        NewDatabase(
            years=2025,
            source="energy_charts",
            bg_db_name="ecoinvent",
            my_db_name="shrecc_hourly",
            countries=["FR"],
            project_name="project",
            inventory_resolution="hourly",
        )


def test_yearly_inventory_resolution_is_normalized_to_annual():
    database = NewDatabase(
        years=2025,
        source="energy_charts",
        bg_db_name="ecoinvent",
        my_db_name="shrecc_annual",
        countries=["FR"],
        project_name="project",
        inventory_resolution="yearly",
        times=["2025-06-01 10:00"],
    )

    assert database.inventory_resolution == "annual"


def test_february_29_is_dropped_with_warning_for_non_leap_model_year():
    profile = pd.Series(
        [1.0, 2.0],
        index=pd.to_datetime(["2024-02-28 12:00", "2024-02-29 12:00"]),
    )

    with pytest.warns(UserWarning, match="Dropped 1 February 29"):
        database = NewDatabase(
            years=[2024, 2025],
            source="energy_charts",
            bg_db_name="ecoinvent-{year}",
            my_db_name="shrecc_profile",
            countries=["FR"],
            project_name="project",
            consumption_profile=profile,
        )

    assert database._selection_for_year(2024)[1].equals(profile.index)
    assert database._selection_for_year(2025)[1].equals(
        pd.to_datetime(["2025-02-28 12:00"])
    )


def test_tyndp_december_31_profile_times_are_dropped_with_warning():
    profile = pd.Series(
        [1.0, 2.0],
        index=pd.to_datetime(["2040-12-30 12:00", "2040-12-31 12:00"]),
    )
    available = pd.to_datetime(["2040-12-30 12:00"])

    with pytest.warns(UserWarning, match="do not include December 31"):
        adapted = _adapt_profile_to_tyndp_times(
            profile,
            available,
            year=2040,
        )

    assert adapted.index.equals(available)
    np.testing.assert_allclose(adapted.to_numpy(), [1.0])


def test_tyndp_other_missing_positive_profile_times_raise():
    profile = pd.Series(
        [1.0],
        index=pd.to_datetime(["2040-06-01 10:30"]),
    )

    with pytest.raises(ValueError, match="positive-weight custom"):
        _adapt_profile_to_tyndp_times(
            profile,
            pd.to_datetime(["2040-06-01 10:00"]),
            year=2040,
        )


def test_zero_weight_custom_profile_times_do_not_select_source_data():
    profile = pd.Series(
        [0.0, 2.0],
        index=pd.to_datetime(["2025-06-01 10:00", "2025-06-01 11:00"]),
    )
    database = NewDatabase(
        years=2025,
        source="energy_charts",
        bg_db_name="ecoinvent",
        my_db_name="shrecc_profile",
        countries=["FR"],
        project_name="project",
        consumption_profile=profile,
    )

    assert database._selection_for_year(2025)[1].equals(profile.index[1:])


def test_hour_range_is_inclusive_and_hourly_by_definition():
    database = NewDatabase(
        years=2025,
        bg_db_name="ecoinvent",
        my_db_name="shrecc_FR_2025",
        countries=["FR"],
        time_range=["2025-06-01", "2025-06-30 23:00"],
        hour_range=[10, 14],
        project_name="project",
    )

    assert database.hour_range == [10, 14]


@pytest.mark.parametrize("hour_range", ([10], [-1, 10], [14, 10], [10, 24]))
def test_hour_range_rejects_invalid_bounds(hour_range):
    with pytest.raises(ValueError, match="hour_range"):
        NewDatabase(
            years=2025,
            bg_db_name="ecoinvent",
            my_db_name="shrecc_FR_2025",
            countries=["FR"],
            time_range=["2025-06-01", "2025-06-30 23:00"],
            hour_range=hour_range,
            project_name="project",
        )


def test_multiple_years_require_explicit_background_database_names():
    with pytest.raises(ValueError, match="bg_db_name must be a year mapping"):
        NewDatabase(
            scenario="DE",
            years=[2035, 2040],
            climate_year=2009,
            bg_db_name="one-background-database",
            my_db_name="shrecc_DE",
            countries=["FR"],
            time_range=["2040-06-01", "2040-06-30"],
            project_name="project",
        )


def test_tyndp_scenario_year_is_validated_during_initialization():
    with pytest.raises(ValueError, match="scenario year 2035"):
        NewDatabase(
            scenario="NT",
            years=2035,
            climate_year=2009,
            bg_db_name="premise-remind-eu-2035",
            my_db_name="shrecc_NT_2035",
            countries=["FR"],
            time_range=["2035-06-01", "2035-06-30"],
            project_name="project",
        )
