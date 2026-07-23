import numpy as np
import pandas as pd
import xarray as xr

from shrecc.mapping import (
    activity_mix_to_database_table,
    load_ecoinvent_mapping,
    map_consumption_mix_to_ecoinvent_activities,
    mapping_gap_to_report,
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
