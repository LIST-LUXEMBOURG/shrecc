import numpy as np
import pandas as pd
import xarray as xr

from shrecc.premise_mapping import (
    PremiseConsumptionMixMapper,
    map_consumption_mix_regions_xr,
    map_consumption_mix_technologies_xr,
)


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
