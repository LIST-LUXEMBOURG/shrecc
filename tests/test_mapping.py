import numpy as np
import pandas as pd
import pytest
import xarray as xr

from shrecc.mapping import (
    activity_mix_to_database_table,
    aggregate_consumption_mix,
    load_ecoinvent_mapping,
    map_consumption_mix_to_ecoinvent_activities,
    mapping_gap_to_report,
    validate_consumption_profile,
    validate_inventory_resolution,
)


def test_load_ecoinvent_mapping_copies_dataframes():
    mapping = pd.DataFrame({"value": [1.0]})

    loaded = load_ecoinvent_mapping(mapping)

    pd.testing.assert_frame_equal(loaded, mapping)
    assert loaded is not mapping


def _consumption_mix():
    return xr.DataArray(
        [[[[0.6, 0.4]]], [[[0.2, 0.8]]]],
        dims=("time", "consumer_country", "source_country", "technology"),
        coords={
            "time": pd.to_datetime(["2040-01-01 10:00", "2040-01-01 11:00"]),
            "consumer_country": ["FR"],
            "source_country": ["DE"],
            "technology": ["Solar", "Unknown"],
        },
        name="consumption_mix",
    )


def _activity_mapping():
    index = pd.MultiIndex.from_tuples(
        [
            (
                "DE",
                "electricity production, photovoltaic, commercial",
                "electricity, low voltage",
                "kWh",
            ),
            (
                "DE",
                "electricity production, photovoltaic, residential",
                "electricity, low voltage",
                "kWh",
            ),
        ],
        names=["geography_source", "activityName", "product", "unitName"],
    )
    columns = pd.MultiIndex.from_tuples(
        [("DE", "Solar")],
        names=["country", "technology"],
    )
    return pd.DataFrame([[0.75], [0.25]], index=index, columns=columns)


def _profile_inputs():
    times = pd.date_range("2040-01-01", periods=2, freq="h")
    consumption_mix = xr.DataArray(
        [
            [
                [[1.0], [0.0]],
                [[0.2], [0.8]],
            ],
            [
                [[0.0], [1.0]],
                [[0.6], [0.4]],
            ],
        ],
        dims=("time", "consumer_country", "source_country", "technology"),
        coords={
            "time": times,
            "consumer_country": ["FR", "DE"],
            "source_country": ["domestic", "imported"],
            "technology": ["electricity"],
        },
        name="consumption_mix",
    )
    consumption_volume = xr.DataArray(
        [[1.0, 3.0], [3.0, 1.0]],
        dims=("time", "consumer_country"),
        coords={
            "time": times,
            "consumer_country": ["FR", "DE"],
        },
        name="consumption_volume",
    )
    return consumption_mix, consumption_volume


def test_flat_profile_is_the_arithmetic_mean_of_hourly_mixes():
    consumption_mix, consumption_volume = _profile_inputs()

    aggregated = aggregate_consumption_mix(
        consumption_mix,
        consumption_profile="flat",
        consumption_volume=consumption_volume,
    )

    xr.testing.assert_allclose(aggregated, consumption_mix.mean("time"))
    assert "time" not in aggregated.dims
    assert aggregated.attrs["consumption_profile"] == "flat"


def test_national_demand_profile_weights_each_consumer_country_separately():
    consumption_mix, consumption_volume = _profile_inputs()

    aggregated = aggregate_consumption_mix(
        consumption_mix,
        consumption_profile="national_demand",
        consumption_volume=consumption_volume,
    )

    np.testing.assert_allclose(
        aggregated.sel(consumer_country="FR").to_numpy().ravel(),
        [0.25, 0.75],
    )
    np.testing.assert_allclose(
        aggregated.sel(consumer_country="DE").to_numpy().ravel(),
        [0.3, 0.7],
    )
    np.testing.assert_allclose(
        aggregated.sum(["source_country", "technology"]),
        1,
    )


def test_monthly_resolution_applies_profile_within_each_month():
    times = pd.to_datetime(
        [
            "2040-01-01 00:00",
            "2040-01-02 00:00",
            "2040-02-01 00:00",
            "2040-02-02 00:00",
        ]
    )
    consumption_mix = xr.DataArray(
        [[[[1.0], [0.0]]], [[[0.0], [1.0]]], [[[0.2], [0.8]]], [[[0.6], [0.4]]]],
        dims=("time", "consumer_country", "source_country", "technology"),
        coords={
            "time": times,
            "consumer_country": ["FR"],
            "source_country": ["domestic", "imported"],
            "technology": ["electricity"],
        },
        name="consumption_mix",
    )
    consumption_volume = xr.DataArray(
        [[1.0], [3.0], [2.0], [2.0]],
        dims=("time", "consumer_country"),
        coords={"time": times, "consumer_country": ["FR"]},
    )

    aggregated = aggregate_consumption_mix(
        consumption_mix,
        consumption_profile="national_demand",
        consumption_volume=consumption_volume,
        inventory_resolution="monthly",
    )

    assert (
        aggregated["time"]
        .to_index()
        .equals(pd.to_datetime(["2040-01-01", "2040-02-01"]))
    )
    np.testing.assert_allclose(
        aggregated.sel(source_country="domestic").to_numpy().ravel(),
        [0.25, 0.4],
    )
    np.testing.assert_allclose(
        aggregated.sum(["source_country", "technology"]),
        1,
    )
    assert aggregated.attrs["inventory_resolution"] == "monthly"


def test_hourly_resolution_preserves_selected_mix_and_ignores_profile():
    consumption_mix, consumption_volume = _profile_inputs()

    with pytest.warns(UserWarning, match="consumption_profile is ignored"):
        resolved = aggregate_consumption_mix(
            consumption_mix,
            consumption_profile="national_demand",
            consumption_volume=consumption_volume,
            inventory_resolution="hourly",
        )

    xr.testing.assert_allclose(resolved, consumption_mix)
    assert resolved.attrs["inventory_resolution"] == "hourly"
    assert resolved.attrs["consumption_profile"] == "not_applicable"


def test_yearly_inventory_resolution_is_an_annual_alias():
    assert validate_inventory_resolution("yearly") == "annual"


def test_custom_profile_is_shared_between_countries_and_scale_invariant():
    consumption_mix, consumption_volume = _profile_inputs()
    profile = pd.Series(
        [1.0, 3.0],
        index=consumption_mix["time"].to_index(),
        name="factory load",
    )

    aggregated = aggregate_consumption_mix(
        consumption_mix,
        consumption_profile=profile,
        consumption_volume=consumption_volume,
    )
    scaled = aggregate_consumption_mix(
        consumption_mix,
        consumption_profile=profile * 1000,
        consumption_volume=consumption_volume,
    )

    xr.testing.assert_allclose(aggregated, scaled)
    np.testing.assert_allclose(
        aggregated.sel(consumer_country="FR").to_numpy().ravel(),
        [0.25, 0.75],
    )
    np.testing.assert_allclose(
        aggregated.sel(consumer_country="DE").to_numpy().ravel(),
        [0.5, 0.5],
    )
    assert aggregated.attrs["consumption_profile"] == "custom"
    assert aggregated.attrs["consumption_profile_name"] == "factory load"


def test_custom_profile_requires_every_selected_timestamp():
    consumption_mix, _ = _profile_inputs()
    profile = pd.Series(
        [1.0],
        index=consumption_mix["time"].to_index()[:1],
    )

    with pytest.raises(ValueError, match="missing selected timestamps"):
        aggregate_consumption_mix(
            consumption_mix,
            consumption_profile=profile,
        )


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        ("unsupported", "must be 'flat'"),
        ([1.0, 2.0], "must be 'flat'"),
        (pd.Series([], index=pd.DatetimeIndex([]), dtype=float), "cannot be empty"),
        (pd.Series([1.0], index=["2040-01-01"]), "DatetimeIndex"),
        (
            pd.Series(
                [1.0, 2.0],
                index=pd.to_datetime(["2040-01-01", "2040-01-01"]),
            ),
            "duplicate timestamps",
        ),
        (
            pd.Series([-1.0], index=pd.to_datetime(["2040-01-01"])),
            "negative",
        ),
        (
            pd.Series([np.nan], index=pd.to_datetime(["2040-01-01"])),
            "missing or non-finite",
        ),
    ],
)
def test_consumption_profile_validation_rejects_invalid_values(profile, message):
    error = ValueError if isinstance(profile, (str, pd.Series)) else TypeError
    with pytest.raises(error, match=message):
        validate_consumption_profile(profile)


def test_national_demand_profile_rejects_zero_country_total():
    consumption_mix, consumption_volume = _profile_inputs()
    consumption_volume.loc[{"consumer_country": "DE"}] = 0

    with pytest.raises(ValueError, match="more than zero: DE"):
        aggregate_consumption_mix(
            consumption_mix,
            consumption_profile="national_demand",
            consumption_volume=consumption_volume,
        )


def test_activity_mapping_allocates_known_shares_and_preserves_fallback():
    activity_mix = map_consumption_mix_to_ecoinvent_activities(
        _consumption_mix(),
        _activity_mapping(),
    )

    np.testing.assert_allclose(activity_mix.sum("activity"), 1)
    np.testing.assert_allclose(activity_mix.isel(time=0, activity=0), 0.45)
    np.testing.assert_allclose(activity_mix.isel(time=0, activity=1), 0.15)
    np.testing.assert_allclose(activity_mix.isel(time=0, activity=2), 0.4)
    assert (
        activity_mix["activity_name"].isel(activity=2).item()
        == "electricity, high voltage, production mix"
    )
    assert activity_mix["geography"].isel(activity=2).item() == "DE"


def test_activity_mix_table_reuses_range_selection_and_averages_time():
    activity_mix = map_consumption_mix_to_ecoinvent_activities(
        _consumption_mix(),
        _activity_mapping(),
    )

    table = activity_mix_to_database_table(
        activity_mix,
        countries=["FR"],
        general_range=["2040-01-01 10:00", "2040-01-01 11:00"],
        refined_range=[10, 10],
        freq="h",
    )

    np.testing.assert_allclose(table["FR"].sum(), 1)
    np.testing.assert_allclose(
        table.loc[
            (
                "DE",
                "electricity, high voltage, production mix",
                "electricity, high voltage",
                "kWh",
            ),
            "FR",
        ],
        0.4,
    )


def test_activity_mix_table_preserves_hourly_inventory_columns():
    activity_mix = map_consumption_mix_to_ecoinvent_activities(
        _consumption_mix(),
        _activity_mapping(),
    )

    table = activity_mix_to_database_table(
        activity_mix,
        countries=["FR"],
        inventory_resolution="hourly",
    )

    assert table.columns.names == ["time", "country"]
    assert (
        table.columns.get_level_values("time")
        .unique()
        .equals(_consumption_mix()["time"].to_index())
    )
    np.testing.assert_allclose(table.sum(axis=0), 1)


def test_mapping_gaps_are_reported_by_source_technology_and_consumer():
    _, mapping_gap = map_consumption_mix_to_ecoinvent_activities(
        _consumption_mix(),
        _activity_mapping(),
        return_mapping_gaps=True,
    )

    report = mapping_gap_to_report(
        mapping_gap,
        general_range=["2040-01-01 10:00", "2040-01-01 11:00"],
    )

    assert list(report.index) == [("DE", "Unknown")]
    np.testing.assert_allclose(report.loc[("DE", "Unknown"), "FR"], 0.6)


def test_explicit_fallback_geography_remains_supported():
    activity_mix = map_consumption_mix_to_ecoinvent_activities(
        _consumption_mix(),
        _activity_mapping(),
        fallback_activity=(
            "RER",
            "electricity, high voltage, European attribute mix",
            "electricity, high voltage",
            "kWh",
        ),
    )

    assert activity_mix["geography"].isel(activity=-1).item() == "RER"
    np.testing.assert_allclose(activity_mix.isel(activity=-1), [[0.4], [0.8]])


def test_sparse_activity_mapping_matches_dense_multi_country_reference():
    rng = np.random.default_rng(42)
    consumption_mix = xr.DataArray(
        rng.random((3, 2, 3, 3)),
        dims=("time", "consumer_country", "source_country", "technology"),
        coords={
            "time": pd.date_range("2025-01-01", periods=3, freq="h"),
            "consumer_country": ["FR", "IT"],
            "source_country": ["DE", "UK", "ZZ"],
            "technology": ["Solar", "Wind", "Unknown"],
        },
        name="consumption_mix",
    )
    consumption_mix /= consumption_mix.sum(["source_country", "technology"])

    activity_index = pd.MultiIndex.from_tuples(
        [
            ("DE", "solar A", "electricity", "kWh"),
            ("DE", "solar B", "electricity", "kWh"),
            ("RER", "wind", "electricity", "kWh"),
        ],
        names=["geography_source", "activityName", "product", "unitName"],
    )
    mapping_columns = pd.MultiIndex.from_tuples(
        [
            ("DE", "Solar"),
            ("DE", "Wind"),
            ("GB", "Solar"),
            ("GB", "Wind"),
        ],
        names=["country", "technology"],
    )
    activity_mapping = pd.DataFrame(
        [
            [0.7, 0.0, 1.0, 0.0],
            [0.3, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        index=activity_index,
        columns=mapping_columns,
    )

    actual, actual_gaps = map_consumption_mix_to_ecoinvent_activities(
        consumption_mix,
        activity_mapping,
        return_mapping_gaps=True,
        time_chunk_size=1,
    )

    source_countries = consumption_mix["source_country"].to_index()
    technologies = consumption_mix["technology"].to_index()
    dense_weights = np.zeros((3, 3, 3), dtype=float)
    for country_idx, source_country in enumerate(source_countries):
        mapping_country = {"UK": "GB"}.get(source_country, source_country)
        if mapping_country not in mapping_columns.get_level_values("country"):
            continue
        dense_weights[country_idx] = (
            activity_mapping.xs(mapping_country, axis=1, level="country")
            .reindex(columns=technologies, fill_value=0)
            .to_numpy(dtype=float)
            .T
        )

    direct = np.einsum(
        "tcsk,ska->tca",
        consumption_mix.to_numpy(),
        dense_weights,
    )
    coverage = dense_weights.sum(axis=2)
    expected_gaps = consumption_mix.to_numpy() * np.clip(
        1 - coverage,
        a_min=0,
        a_max=None,
    )
    fallback = expected_gaps.sum(axis=3)
    expected = np.concatenate([direct, fallback], axis=2)

    np.testing.assert_allclose(actual.to_numpy(), expected)
    np.testing.assert_allclose(actual_gaps.to_numpy(), expected_gaps)
    assert actual.dims == ("time", "consumer_country", "activity")
    assert actual["geography"].to_numpy().tolist()[-3:] == ["DE", "GB", "ZZ"]
    np.testing.assert_allclose(actual.sum("activity"), 1)
