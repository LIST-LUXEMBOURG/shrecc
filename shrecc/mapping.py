"""Shared consumption-mix filtering and background activity mapping."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_COUNTRY_ALIASES = {"NIE": "GB", "UK": "GB"}
DEFAULT_FALLBACK_ACTIVITY = (
    "RER",
    "electricity, high voltage, European attribute mix",
    "electricity, high voltage",
    "kWh",
)


def load_ecoinvent_mapping(mapping_location):
    """Load the country and technology allocation table used by SHRECC.

    ``mapping_location`` may point directly to a CSV file or to a directory
    containing ``el_map_all_norm.csv``.
    """
    mapping_file = (
        Path(mapping_location)
        if isinstance(mapping_location, (str, Path))
        else mapping_location
    )
    if mapping_file.is_dir():
        mapping_file = mapping_file / "el_map_all_norm.csv"

    return pd.read_csv(
        mapping_file,
        index_col=[0, 1, 2, 3],
        header=[0, 1],
    )


def map_consumption_mix_to_ecoinvent_activities(
    consumption_mix,
    activity_mapping,
    *,
    country_aliases=None,
    fallback_activity=DEFAULT_FALLBACK_ACTIVITY,
    check=True,
):
    """Map source-country technologies to ecoinvent activities.

    Unmapped shares are assigned to ``fallback_activity``, reproducing the
    historical SHRECC fallback without first constructing ``Z_cons``.
    """
    required_dims = {"source_country", "technology", "consumer_country"}
    missing_dims = required_dims.difference(consumption_mix.dims)
    if missing_dims:
        raise ValueError(
            "consumption_mix is missing required dimensions: "
            + ", ".join(sorted(missing_dims))
        )
    if not isinstance(activity_mapping, pd.DataFrame):
        raise TypeError("activity_mapping must be a pandas DataFrame")
    if not isinstance(activity_mapping.index, pd.MultiIndex):
        raise ValueError("activity_mapping index must be a MultiIndex")
    if activity_mapping.index.nlevels != 4:
        raise ValueError("activity_mapping index must have four activity levels")
    if not isinstance(activity_mapping.columns, pd.MultiIndex):
        raise ValueError("activity_mapping columns must be a MultiIndex")
    if activity_mapping.columns.nlevels != 2:
        raise ValueError(
            "activity_mapping columns must contain country and technology levels"
        )

    aliases = dict(DEFAULT_COUNTRY_ALIASES)
    if country_aliases is not None:
        aliases.update(country_aliases)

    source_countries = consumption_mix["source_country"].to_index()
    technologies = consumption_mix["technology"].to_index()
    weights = np.zeros(
        (len(source_countries), len(technologies), len(activity_mapping)),
        dtype=float,
    )
    mapping_countries = activity_mapping.columns.get_level_values(0)

    for country_idx, source_country in enumerate(source_countries):
        mapping_country = aliases.get(source_country, source_country)
        if mapping_country not in mapping_countries:
            continue
        country_mapping = activity_mapping.xs(
            mapping_country,
            axis=1,
            level=0,
        ).reindex(columns=technologies, fill_value=0)
        weights[country_idx] = country_mapping.to_numpy(dtype=float).T

    weights_xr = xr.DataArray(
        weights,
        dims=("source_country", "technology", "activity"),
        coords={
            "source_country": source_countries,
            "technology": technologies,
            "activity": np.arange(len(activity_mapping)),
        },
    )
    activity_mix = xr.dot(
        consumption_mix,
        weights_xr,
        dim=["source_country", "technology"],
    )
    mapped_total = activity_mix.sum("activity")
    input_total = consumption_mix.sum(["source_country", "technology"])
    unmapped = (input_total - mapped_total).clip(min=0)

    fallback = unmapped.expand_dims(activity=[len(activity_mapping)])
    activity_mix = xr.concat([activity_mix, fallback], dim="activity")
    activity_index = list(activity_mapping.index) + [tuple(fallback_activity)]
    activity_mix = activity_mix.assign_coords(
        geography=("activity", [item[0] for item in activity_index]),
        activity_name=("activity", [item[1] for item in activity_index]),
        product=("activity", [item[2] for item in activity_index]),
        unit=("activity", [item[3] for item in activity_index]),
    ).rename("activity_mix")
    activity_mix.attrs["unit"] = "dimensionless"

    if check:
        if (mapped_total > input_total + 1e-8).any():
            raise ValueError("Activity mapping allocates more than the input mix")
        np.testing.assert_allclose(
            activity_mix.sum("activity"),
            input_total,
            atol=1e-8,
        )

    return activity_mix


def activity_mix_to_database_table(
    activity_mix,
    *,
    countries=None,
    times=None,
    general_range=None,
    refined_range=None,
    freq=None,
):
    """Aggregate a mapped activity mix to the existing database table shape."""
    required_dims = {"activity", "consumer_country"}
    missing_dims = required_dims.difference(activity_mix.dims)
    if missing_dims:
        raise ValueError(
            "activity_mix is missing required dimensions: "
            + ", ".join(sorted(missing_dims))
        )
    required_coords = {"geography", "activity_name", "product", "unit"}
    missing_coords = required_coords.difference(activity_mix.coords)
    if missing_coords:
        raise ValueError(
            "activity_mix is missing required coordinates: "
            + ", ".join(sorted(missing_coords))
        )

    selected = activity_mix
    if countries is not None:
        selected = selected.sel(consumer_country=list(countries))
    selected = filter_consumption_mix_time(
        selected,
        times=times,
        general_range=general_range,
        refined_range=refined_range,
        freq=freq,
    )
    if "time" in selected.dims:
        selected = selected.mean("time")

    table = selected.to_dataframe(name="share").reset_index()
    table = (
        table.groupby(
            ["geography", "activity_name", "product", "unit", "consumer_country"],
            dropna=False,
        )["share"]
        .sum()
        .unstack("consumer_country", fill_value=0)
    )
    table.index.names = ["geography", "activityName", "product", "unit"]
    table.columns.name = None
    return table


def filter_consumption_mix_time(
    mix,
    *,
    times=None,
    general_range=None,
    refined_range=None,
    freq=None,
):
    """Apply the time-selection semantics used by ``filt_cutoff``."""
    has_time_filter = any(
        value is not None for value in (times, general_range, refined_range, freq)
    )
    if "time" not in mix.dims:
        if has_time_filter:
            raise ValueError("Cannot filter a consumption mix without a time dimension")
        return mix

    if times is not None and len(times):
        requested_times = pd.DatetimeIndex(pd.to_datetime(times))
        mix = mix.sel(time=mix["time"].isin(requested_times))

    if general_range is not None:
        if len(general_range) != 2:
            raise ValueError("general_range must contain a start and end timestamp")
        start, end = pd.to_datetime(general_range)
        mix = mix.sel(time=slice(start, end))

        if refined_range is not None and len(refined_range):
            if freq is None:
                raise ValueError("freq is required when refined_range is supplied")
            timestamps = pd.date_range(start=start, end=end, freq=freq)
            if len(refined_range) > 1:
                timestamps = timestamps[
                    (timestamps.hour >= refined_range[0])
                    & (timestamps.hour <= refined_range[-1])
                ]
            mix = mix.sel(time=mix["time"].isin(timestamps))
    elif refined_range is not None and len(refined_range):
        raise ValueError("general_range is required when refined_range is supplied")

    if mix.sizes["time"] == 0:
        raise ValueError(
            "The requested time selection contains no available timestamps"
        )
    return mix
