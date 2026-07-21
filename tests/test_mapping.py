import numpy as np
import pandas as pd
import xarray as xr

from shrecc.mapping import (
    activity_mix_to_database_table,
    map_consumption_mix_to_ecoinvent_activities,
)


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
        == "electricity, high voltage, European attribute mix"
    )


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
                "RER",
                "electricity, high voltage, European attribute mix",
                "electricity, high voltage",
                "kWh",
            ),
            "FR",
        ],
        0.4,
    )
