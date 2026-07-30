"""Tests for premise-specific technology and geography mapping."""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import xarray as xr

from shrecc.premise_mapping import (
    PremiseConsumptionMixMapper,
    build_country_activity_technology_map,
    build_country_activity_presence_from_ecoinvent_mapping,
    build_country_activity_shares_from_ecoinvent_mapping,
    build_country_specific_premise_technology_map,
    build_premise_exchange_geography_map,
    map_consumption_mix_regions_xr,
    map_consumption_mix_technologies_xr,
    premise_activity_mix_to_database_table,
)


def test_build_country_activity_technology_map_uses_market_shares_and_unique_fallback():
    concordance = pd.DataFrame(
        {
            "wind": [1, 1, 0],
            "gas with CCS": [0, 0, 1],
        },
        index=pd.Index(
            ["wind small", "wind large", "new gas CCS"],
            name="premise_activity",
        ),
    )
    activity_shares = pd.DataFrame(
        {
            "FR": [0.1, 0.3],
            "DE": [0.4, 0.1],
        },
        index=["wind small", "wind large"],
    )

    result = build_country_activity_technology_map(concordance, activity_shares)

    np.testing.assert_allclose(
        result.loc[
            ["wind small", "wind large"],
            [("FR", "wind"), ("DE", "wind")],
        ],
        [[0.25, 0.8], [0.75, 0.2]],
    )
    assert result.loc["new gas CCS", ("FR", "gas with CCS")] == 1
    assert result.loc["new gas CCS", ("DE", "gas with CCS")] == 1
    np.testing.assert_allclose(result.sum(axis=0), 1)


def test_build_country_activity_technology_map_rejects_ambiguous_zero_shares():
    concordance = pd.DataFrame(
        {"new technology": [1, 1]},
        index=["new activity A", "new activity B"],
    )
    activity_shares = pd.DataFrame({"FR": []})

    with np.testing.assert_raises_regex(
        ValueError,
        "FR / new technology",
    ):
        build_country_activity_technology_map(concordance, activity_shares)


def test_build_country_activity_technology_map_can_fallback_to_concordance():
    concordance = pd.DataFrame(
        {"wind": [0.25, 0.75]},
        index=pd.Index(["wind small", "wind large"], name="premise_activity"),
    )
    activity_shares = pd.DataFrame(
        {"FR": [0.1, 0.3], "DE": [0.0, 0.0]},
        index=["wind small", "wind large"],
    )

    result = build_country_activity_technology_map(
        concordance,
        activity_shares,
        fallback_activity_concordance=True,
    )

    np.testing.assert_allclose(
        result.loc[["wind small", "wind large"], ("FR", "wind")],
        [0.25, 0.75],
    )
    np.testing.assert_allclose(
        result.loc[["wind small", "wind large"], ("DE", "wind")],
        [0.25, 0.75],
    )


def test_build_country_activity_shares_from_ecoinvent_mapping_collapses_mapping():
    index = pd.MultiIndex.from_tuples(
        [
            ("FR", "wind small", "electricity", "kWh"),
            ("FR", "wind large", "electricity", "kWh"),
            ("DE", "wind small", "electricity", "kWh"),
        ],
        names=["geography_source", "activityName", "product", "unitName"],
    )
    columns = pd.MultiIndex.from_tuples(
        [
            ("FR", "Wind Onshore"),
            ("FR", "Wind Offshore"),
            ("DE", "Wind Onshore"),
        ],
    )
    ecoinvent_mapping = pd.DataFrame(
        [
            [0.2, 0.1, 0.4],
            [0.8, 0.3, 0.0],
            [0.0, 0.0, 0.6],
        ],
        index=index,
        columns=columns,
    )

    shares = build_country_activity_shares_from_ecoinvent_mapping(
        ecoinvent_mapping,
        premise_activities=["wind small", "wind large", "new activity"],
    )

    expected = pd.DataFrame(
        {
            "DE": [1.0, 0.0, 0.0],
            "FR": [0.3, 1.1, 0.0],
        },
        index=pd.Index(
            ["wind small", "wind large", "new activity"],
            name="premise_activity",
        ),
    )
    pdt.assert_frame_equal(shares, expected)


def test_country_activity_shares_alias_uk_to_gb():
    index = pd.MultiIndex.from_tuples(
        [
            ("GB", "wind small", "electricity", "kWh"),
            ("GB", "wind large", "electricity", "kWh"),
        ],
        names=["geography_source", "activityName", "product", "unitName"],
    )
    columns = pd.MultiIndex.from_tuples([("GB", "Wind Onshore")])
    ecoinvent_mapping = pd.DataFrame(
        [[0.25], [0.75]],
        index=index,
        columns=columns,
    )

    shares = build_country_activity_shares_from_ecoinvent_mapping(
        ecoinvent_mapping,
        premise_activities=["wind small", "wind large"],
        source_countries=["UK"],
    )

    expected = pd.DataFrame(
        {"UK": [0.25, 0.75]},
        index=pd.Index(["wind small", "wind large"], name="premise_activity"),
    )
    pdt.assert_frame_equal(shares, expected)


def test_build_premise_exchange_geography_map_uses_country_then_region():
    index = pd.MultiIndex.from_tuples(
        [
            ("FR", "wind small", "electricity", "kWh"),
            ("DE", "wind large", "electricity", "kWh"),
        ],
        names=["geography_source", "activityName", "product", "unitName"],
    )
    columns = pd.MultiIndex.from_tuples(
        [
            ("FR", "Wind Onshore"),
            ("DE", "Wind Onshore"),
        ],
    )
    ecoinvent_mapping = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=index,
        columns=columns,
    )
    premise_region_map = pd.DataFrame(
        {"remind-eu": ["FRA", "DEU"]},
        index=pd.Index(["FR", "DE"], name="country"),
    )

    presence = build_country_activity_presence_from_ecoinvent_mapping(
        ecoinvent_mapping,
        premise_activities=["wind small", "new ccs"],
        source_countries=["FR", "DE"],
    )
    exchange_map = build_premise_exchange_geography_map(
        ecoinvent_mapping,
        premise_region_map,
        iam_model="remind-eu",
        premise_activities=["wind small", "new ccs"],
        source_countries=["FR", "DE"],
    )

    assert presence.loc["wind small", "FR"]
    assert not presence.loc["wind small", "DE"]
    assert exchange_map.loc["wind small", "FR"] == "FR"
    assert exchange_map.loc["wind small", "DE"] == "DEU"
    assert exchange_map.loc["new ccs", "FR"] == "FRA"


def test_build_premise_exchange_geography_map_aliases_uk_existing_activity_to_gb():
    index = pd.MultiIndex.from_tuples(
        [
            ("GB", "wind small", "electricity", "kWh"),
        ],
        names=["geography_source", "activityName", "product", "unitName"],
    )
    columns = pd.MultiIndex.from_tuples([("GB", "Wind Onshore")])
    ecoinvent_mapping = pd.DataFrame([[1.0]], index=index, columns=columns)
    premise_region_map = pd.DataFrame(
        {"remind-eu": ["UKI"]},
        index=pd.Index(["UK"], name="country"),
    )

    exchange_map = build_premise_exchange_geography_map(
        ecoinvent_mapping,
        premise_region_map,
        iam_model="remind-eu",
        premise_activities=["wind small", "new ccs"],
        source_countries=["UK"],
    )

    assert exchange_map.loc["wind small", "UK"] == "GB"
    assert exchange_map.loc["new ccs", "UK"] == "UKI"


def test_build_premise_exchange_geography_map_keeps_country_specific_pv_activity():
    index = pd.MultiIndex.from_tuples(
        [
            ("ES", "wind small", "electricity", "kWh"),
        ],
        names=["geography_source", "activityName", "product", "unitName"],
    )
    columns = pd.MultiIndex.from_tuples([("ES", "Wind Onshore")])
    ecoinvent_mapping = pd.DataFrame([[1.0]], index=index, columns=columns)
    premise_region_map = pd.DataFrame(
        {"remind-eu": ["ESW"]},
        index=pd.Index(["ES"], name="country"),
    )

    exchange_map = build_premise_exchange_geography_map(
        ecoinvent_mapping,
        premise_region_map,
        iam_model="remind-eu",
        premise_activities=[
            "electricity production, photovoltaic, commercial",
            "electricity production, photovoltaic, residential",
        ],
        source_countries=["ES"],
    )

    assert (
        exchange_map.loc["electricity production, photovoltaic, commercial", "ES"]
        == "ES"
    )
    assert (
        exchange_map.loc["electricity production, photovoltaic, residential", "ES"]
        == "ESW"
    )


def test_build_country_specific_premise_technology_map_uses_ecoinvent_shares():
    technology_mapping = pd.DataFrame(
        {
            "wind": [0.5, 0.5],
            "future ccs": [0.25, 0.75],
        },
        index=pd.Index(["wind small", "wind large"], name="premise_activity"),
    )
    index = pd.MultiIndex.from_tuples(
        [
            ("FR", "wind small", "electricity", "kWh"),
            ("FR", "wind large", "electricity", "kWh"),
        ],
        names=["geography_source", "activityName", "product", "unitName"],
    )
    columns = pd.MultiIndex.from_tuples(
        [
            ("FR", "Wind Onshore"),
            ("DE", "Wind Onshore"),
        ],
    )
    ecoinvent_mapping = pd.DataFrame(
        [
            [0.1, 0.0],
            [0.3, 0.0],
        ],
        index=index,
        columns=columns,
    )

    result = build_country_specific_premise_technology_map(
        technology_mapping,
        ecoinvent_mapping,
    )

    np.testing.assert_allclose(
        result.loc[["wind small", "wind large"], ("FR", "wind")],
        [0.25, 0.75],
    )
    np.testing.assert_allclose(
        result.loc[["wind small", "wind large"], ("DE", "wind")],
        [0.5, 0.5],
    )
    np.testing.assert_allclose(
        result.loc[["wind small", "wind large"], ("FR", "future ccs")],
        [0.25, 0.75],
    )


def test_premise_consumption_mix_mapper_uses_ecoinvent_mapping_for_technologies():
    consumption_mix_xr = xr.DataArray(
        np.array([[[[0.4, 0.1], [0.2, 0.3]]]]),
        dims=("time", "consumer_country", "source_country", "technology"),
        coords={
            "time": pd.to_datetime(["2050-01-01 00:00"]),
            "consumer_country": ["LU"],
            "source_country": ["FR", "DE"],
            "technology": ["wind", "future ccs"],
        },
        name="consumption_mix",
    )
    technology_mapping = pd.DataFrame(
        {
            "wind": [1.0, 1.0, 0.0],
            "future ccs": [0.0, 0.0, 1.0],
        },
        index=pd.Index(
            ["wind small", "wind large", "new ccs"],
            name="premise_activity",
        ),
    )
    index = pd.MultiIndex.from_tuples(
        [
            ("FR", "wind small", "electricity", "kWh"),
            ("FR", "wind large", "electricity", "kWh"),
            ("DE", "wind small", "electricity", "kWh"),
            ("DE", "wind large", "electricity", "kWh"),
        ],
        names=["geography_source", "activityName", "product", "unitName"],
    )
    columns = pd.MultiIndex.from_tuples(
        [
            ("FR", "Wind Onshore"),
            ("DE", "Wind Onshore"),
        ],
    )
    ecoinvent_mapping = pd.DataFrame(
        [
            [0.1, 0.0],
            [0.3, 0.0],
            [0.0, 0.4],
            [0.0, 0.1],
        ],
        index=index,
        columns=columns,
    )
    mapper = PremiseConsumptionMixMapper(
        technology_mapping=technology_mapping,
        ecoinvent_mapping=ecoinvent_mapping,
        iam_model="remind-eu",
    )

    mapped = mapper.map_technologies(consumption_mix_xr)

    expected = xr.DataArray(
        np.array([[[[0.1, 0.3, 0.1], [0.16, 0.04, 0.3]]]]),
        dims=("time", "consumer_country", "source_country", "premise_activity"),
        coords={
            "time": consumption_mix_xr.time,
            "consumer_country": ["LU"],
            "source_country": ["FR", "DE"],
            "premise_activity": ["wind small", "wind large", "new ccs"],
        },
        name="consumption_mix",
    )
    xr.testing.assert_allclose(mapped, expected)


def test_premise_consumption_mix_mapper_aliases_uk_to_gb_mapping():
    consumption_mix_xr = xr.DataArray(
        np.array([[[[0.5, 0.5]]]]),
        dims=("time", "consumer_country", "source_country", "technology"),
        coords={
            "time": pd.to_datetime(["2050-01-01 00:00"]),
            "consumer_country": ["LU"],
            "source_country": ["UK"],
            "technology": ["wind", "future ccs"],
        },
        name="consumption_mix",
    )
    technology_mapping = pd.DataFrame(
        {
            "wind": [1.0, 1.0, 0.0],
            "future ccs": [0.0, 0.0, 1.0],
        },
        index=pd.Index(
            ["wind small", "wind large", "new ccs"],
            name="premise_activity",
        ),
    )
    index = pd.MultiIndex.from_tuples(
        [
            ("GB", "wind small", "electricity", "kWh"),
            ("GB", "wind large", "electricity", "kWh"),
        ],
        names=["geography_source", "activityName", "product", "unitName"],
    )
    columns = pd.MultiIndex.from_tuples([("GB", "Wind Onshore")])
    ecoinvent_mapping = pd.DataFrame(
        [[0.1], [0.3]],
        index=index,
        columns=columns,
    )
    mapper = PremiseConsumptionMixMapper(
        technology_mapping=technology_mapping,
        ecoinvent_mapping=ecoinvent_mapping,
        iam_model="remind-eu",
    )

    mapped = mapper.map_technologies(consumption_mix_xr)

    expected = xr.DataArray(
        np.array([[[[0.125, 0.375, 0.5]]]]),
        dims=("time", "consumer_country", "source_country", "premise_activity"),
        coords={
            "time": consumption_mix_xr.time,
            "consumer_country": ["LU"],
            "source_country": ["UK"],
            "premise_activity": ["wind small", "wind large", "new ccs"],
        },
        name="consumption_mix",
    )
    xr.testing.assert_allclose(mapped, expected)


def test_premise_activity_mix_to_database_table_uses_exchange_geography():
    premise_activity_mix_xr = xr.DataArray(
        np.array([[[[0.2, 0.3], [0.1, 0.4]]]]),
        dims=("time", "consumer_country", "source_country", "premise_activity"),
        coords={
            "time": pd.to_datetime(["2050-01-01 00:00"]),
            "consumer_country": ["LU"],
            "source_country": ["FR", "DE"],
            "premise_activity": ["wind small", "new ccs"],
        },
        name="consumption_mix",
    )
    exchange_geography_map = pd.DataFrame(
        {"FR": ["FR", "FRA"], "DE": ["DE", "DEU"]},
        index=pd.Index(["wind small", "new ccs"], name="premise_activity"),
    )
    exchange_geography_map.columns.name = "source_country"

    table = premise_activity_mix_to_database_table(
        premise_activity_mix_xr,
        exchange_geography_map,
        countries=["LU"],
    )

    assert (
        table.loc[("FR", "wind small", "electricity, high voltage", "kWh"), "LU"] == 0.2
    )
    assert (
        table.loc[("DE", "wind small", "electricity, high voltage", "kWh"), "LU"] == 0.1
    )
    assert (
        table.loc[("FRA", "new ccs", "electricity, high voltage", "kWh"), "LU"] == 0.3
    )
    assert (
        table.loc[("DEU", "new ccs", "electricity, high voltage", "kWh"), "LU"] == 0.4
    )
    np.testing.assert_allclose(table.sum(), 1)


def test_premise_activity_mix_table_preserves_monthly_columns():
    times = pd.to_datetime(["2050-01-01", "2050-02-01"])
    mix, exchange_map = _premise_activity_time_series(
        times,
        [0.2, 0.8],
    )

    table = premise_activity_mix_to_database_table(
        mix,
        exchange_map,
        inventory_resolution="monthly",
    )

    assert table.columns.names == ["time", "country"]
    assert table.columns.get_level_values("time").unique().equals(times)
    np.testing.assert_allclose(table.sum(axis=0), 1)


def _premise_activity_time_series(times, first_activity_shares):
    first_activity_shares = np.asarray(first_activity_shares)
    values = np.column_stack(
        [first_activity_shares, 1 - first_activity_shares]
    )[:, None, None, :]
    mix = xr.DataArray(
        values,
        dims=("time", "consumer_country", "source_country", "premise_activity"),
        coords={
            "time": pd.to_datetime(times),
            "consumer_country": ["ES"],
            "source_country": ["FR"],
            "premise_activity": ["wind", "solar"],
        },
        name="consumption_mix",
    )
    exchange_map = pd.DataFrame(
        {"FR": ["FR", "FR"]},
        index=pd.Index(["wind", "solar"], name="premise_activity"),
    )
    exchange_map.columns.name = "source_country"
    return mix, exchange_map


def test_premise_activity_mix_to_database_table_selects_explicit_times():
    times = pd.date_range("2050-01-01", periods=4, freq="h")
    mix, exchange_map = _premise_activity_time_series(
        times,
        [0.1, 0.2, 0.3, 0.4],
    )

    table = premise_activity_mix_to_database_table(
        mix,
        exchange_map,
        times=[times[0], times[2]],
    )

    assert (
        table.loc[("FR", "wind", "electricity, high voltage", "kWh"), "ES"]
        == 0.2
    )


def test_premise_activity_mix_to_database_table_refines_hours_and_frequency():
    times = pd.date_range("2050-01-01", periods=4, freq="h")
    mix, exchange_map = _premise_activity_time_series(
        times,
        [0.1, 0.2, 0.3, 0.4],
    )

    table = premise_activity_mix_to_database_table(
        mix,
        exchange_map,
        general_range=[times[0], times[-1]],
        refined_range=[1, 2],
        freq="h",
    )

    assert (
        table.loc[("FR", "wind", "electricity, high voltage", "kWh"), "ES"]
        == 0.25
    )


def test_premise_activity_mix_to_database_table_supports_monthly_frequency():
    times = pd.date_range("2050-01-01", "2050-03-01", freq="D")
    shares = np.arange(len(times)) / 100
    mix, exchange_map = _premise_activity_time_series(times, shares)

    table = premise_activity_mix_to_database_table(
        mix,
        exchange_map,
        general_range=[times[0], times[-1]],
        refined_range=[0],
        freq="MS",
    )

    expected = shares[[0, 31, 59]].mean()
    actual = table.loc[
        ("FR", "wind", "electricity, high voltage", "kWh"),
        "ES",
    ]
    assert actual == expected


def test_premise_activity_mix_to_database_table_validates_refined_range():
    times = pd.date_range("2050-01-01", periods=2, freq="h")
    mix, exchange_map = _premise_activity_time_series(times, [0.4, 0.6])

    with pytest.raises(ValueError, match="general_range is required"):
        premise_activity_mix_to_database_table(
            mix,
            exchange_map,
            refined_range=[8, 10],
            freq="h",
        )


def test_map_consumption_mix_technologies_xr_uses_country_specific_weights():
    consumption_mix_xr = xr.DataArray(
        np.array([[[[0.5], [0.5]]]]),
        dims=("time", "consumer_country", "source_country", "technology"),
        coords={
            "time": pd.to_datetime(["2050-01-01 00:00"]),
            "consumer_country": ["LU"],
            "source_country": ["FR", "DE"],
            "technology": ["wind"],
        },
        name="consumption_mix",
    )
    tech_map = pd.DataFrame(
        {
            ("FR", "wind"): [0.25, 0.75],
            ("DE", "wind"): [0.8, 0.2],
        },
        index=pd.Index(["wind small", "wind large"], name="premise_activity"),
    )
    tech_map.columns.names = ["source_country", "technology"]

    mapped = map_consumption_mix_technologies_xr(
        consumption_mix_xr,
        tech_map,
        check=True,
    )

    expected = xr.DataArray(
        np.array([[[[0.125, 0.375], [0.4, 0.1]]]]),
        dims=("time", "consumer_country", "source_country", "premise_activity"),
        coords={
            "time": consumption_mix_xr.time,
            "consumer_country": ["LU"],
            "source_country": ["FR", "DE"],
            "premise_activity": ["wind small", "wind large"],
        },
        name="consumption_mix",
    )
    xr.testing.assert_allclose(mapped, expected)


def test_map_consumption_mix_technologies_xr_multiplies_technology_axis():
    consumption_mix_xr = xr.DataArray(
        np.array(
            [
                [
                    [
                        [0.1, 0.15, 0.25],
                        [0.05, 0.1, 0.35],
                    ]
                ]
            ]
        ),
        dims=("time", "consumer_country", "source_country", "technology"),
        coords={
            "time": pd.to_datetime(["2050-01-01 00:00"]),
            "consumer_country": ["LU"],
            "source_country": ["FR", "DE"],
            "technology": ["solar", "wind", "gas"],
        },
        name="consumption_mix",
    )
    tech_map = pd.DataFrame(
        {
            "solar": [1.0, 0.0],
            "wind": [0.25, 0.75],
            "gas": [0.0, 1.0],
        },
        index=pd.Index(["pv", "thermal"], name="premise_activity"),
    )

    mapped = map_consumption_mix_technologies_xr(
        consumption_mix_xr,
        tech_map,
        check=True,
    )

    expected = xr.DataArray(
        np.array(
            [
                [
                    [
                        [0.1375, 0.3625],
                        [0.075, 0.425],
                    ]
                ]
            ]
        ),
        dims=("time", "consumer_country", "source_country", "premise_activity"),
        coords={
            "time": consumption_mix_xr.time,
            "consumer_country": ["LU"],
            "source_country": ["FR", "DE"],
            "premise_activity": ["pv", "thermal"],
        },
        name="consumption_mix",
    )

    assert mapped.shape == (1, 1, 2, 2)
    xr.testing.assert_allclose(mapped, expected)


def test_map_consumption_mix_regions_xr_multiplies_source_country_axis():
    consumption_mix_xr = xr.DataArray(
        np.array(
            [
                [
                    [
                        [0.1375, 0.3625],
                        [0.075, 0.425],
                    ]
                ]
            ]
        ),
        dims=("time", "consumer_country", "source_country", "premise_activity"),
        coords={
            "time": pd.to_datetime(["2050-01-01 00:00"]),
            "consumer_country": ["LU"],
            "source_country": ["FR", "DE"],
            "premise_activity": ["pv", "thermal"],
        },
        name="consumption_mix",
    )
    premise_region_map = pd.DataFrame(
        {"remind": ["EUR", "EUR"]},
        index=pd.Index(["FR", "DE"], name="country"),
    )

    mapped = map_consumption_mix_regions_xr(
        consumption_mix_xr,
        premise_region_map,
        iam_model="remind",
        check=True,
    )

    expected = xr.DataArray(
        np.array([[[[0.2125], [0.7875]]]]),
        dims=("time", "consumer_country", "premise_activity", "premise_region"),
        coords={
            "time": consumption_mix_xr.time,
            "consumer_country": ["LU"],
            "premise_activity": ["pv", "thermal"],
            "premise_region": ["EUR"],
        },
        name="consumption_mix",
    )
    expected.attrs["iam_model"] = "remind"

    assert mapped.shape == (1, 1, 2, 1)
    xr.testing.assert_allclose(mapped, expected)


def test_premise_consumption_mix_mapper_maps_technologies_and_regions():
    consumption_mix_xr = xr.DataArray(
        np.array([[[[0.4, 0.1], [0.2, 0.3]]]]),
        dims=("time", "consumer_country", "source_country", "technology"),
        coords={
            "time": pd.to_datetime(["2050-01-01 00:00"]),
            "consumer_country": ["LU"],
            "source_country": ["FR", "DE"],
            "technology": ["solar", "gas"],
        },
        name="consumption_mix",
    )
    technology_map = pd.DataFrame(
        {
            "solar": [1.0, 0.0],
            "gas": [0.0, 1.0],
        },
        index=pd.Index(["pv", "thermal"], name="premise_activity"),
    )
    premise_region_map = pd.DataFrame(
        {"remind": ["EUR", "EUR"]},
        index=pd.Index(["FR", "DE"], name="country"),
    )

    premise_consumption_mix_mapper = PremiseConsumptionMixMapper(
        technology_mapping=technology_map,
        iam_model="remind",
    )

    mapped = premise_consumption_mix_mapper.map_regions(
        premise_consumption_mix_mapper.map_technologies(consumption_mix_xr),
        premise_region_map=premise_region_map,
    )

    expected = xr.DataArray(
        np.array([[[[0.6], [0.4]]]]),
        dims=("time", "consumer_country", "premise_activity", "premise_region"),
        coords={
            "time": consumption_mix_xr.time,
            "consumer_country": ["LU"],
            "premise_activity": ["pv", "thermal"],
            "premise_region": ["EUR"],
        },
        name="consumption_mix",
    )
    expected.attrs["iam_model"] = "remind"

    xr.testing.assert_allclose(mapped, expected)
