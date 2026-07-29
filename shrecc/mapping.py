"""Shared consumption-mix filtering and background activity mapping."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix, hstack

DEFAULT_COUNTRY_ALIASES = {"NIE": "GB", "UK": "GB"}
DEFAULT_FALLBACK_ACTIVITY = (
    None,
    "electricity, high voltage, production mix",
    "electricity, high voltage",
    "kWh",
)
DEFAULT_MAPPING_TIME_CHUNK_SIZE = 168
VALID_CONSUMPTION_PROFILES = {"flat", "national_demand"}
VALID_INVENTORY_RESOLUTIONS = {"annual", "monthly"}


def _build_ecoinvent_mapping_operator(
    activity_mapping,
    source_countries,
    technologies,
    aliases,
    fallback_geography,
):
    """Build the sparse source-technology to activity allocation operator."""
    source_countries = pd.Index(source_countries)
    technologies = pd.Index(technologies)
    mapping_countries = activity_mapping.columns.get_level_values(0)
    source_count = len(source_countries)
    technology_count = len(technologies)
    activity_count = len(activity_mapping)

    row_indices = []
    column_indices = []
    allocation_values = []

    for country_idx, source_country in enumerate(source_countries):
        mapping_country = aliases.get(source_country, source_country)
        if mapping_country not in mapping_countries:
            continue

        country_mapping = (
            activity_mapping.xs(mapping_country, axis=1, level=0)
            .reindex(columns=technologies, fill_value=0)
            .fillna(0)
            .to_numpy(dtype=float)
        )
        activity_indices, technology_indices = np.nonzero(country_mapping)
        row_indices.extend(
            country_idx * technology_count + technology_indices
        )
        column_indices.extend(activity_indices)
        allocation_values.extend(
            country_mapping[activity_indices, technology_indices]
        )

    feature_count = source_count * technology_count
    direct_mapping = csr_matrix(
        (allocation_values, (row_indices, column_indices)),
        shape=(feature_count, activity_count),
        dtype=float,
    )
    mapping_coverage = np.asarray(direct_mapping.sum(axis=1)).ravel()
    fallback_weights = np.clip(1 - mapping_coverage, a_min=0, a_max=None)

    if fallback_geography is None:
        fallback_columns = np.repeat(
            np.arange(source_count),
            technology_count,
        )
        fallback_count = source_count
    else:
        fallback_columns = np.zeros(feature_count, dtype=int)
        fallback_count = 1

    fallback_rows = np.flatnonzero(fallback_weights)
    fallback_mapping = csr_matrix(
        (
            fallback_weights[fallback_rows],
            (fallback_rows, fallback_columns[fallback_rows]),
        ),
        shape=(feature_count, fallback_count),
        dtype=float,
    )
    return (
        hstack([direct_mapping, fallback_mapping], format="csr"),
        mapping_coverage.reshape(source_count, technology_count),
        activity_count,
    )


def _apply_sparse_mapping_operator(
    consumption_mix,
    mapping_operator,
    *,
    time_chunk_size,
):
    """Apply a sparse mapping operator without copying the full input array."""
    mapping_dims = {"source_country", "technology"}
    outer_dims = [dim for dim in consumption_mix.dims if dim not in mapping_dims]
    transposed = consumption_mix.transpose(
        *outer_dims,
        "source_country",
        "technology",
    )
    outer_shape = tuple(transposed.sizes[dim] for dim in outer_dims)
    feature_count = (
        transposed.sizes["source_country"] * transposed.sizes["technology"]
    )
    output = np.empty(
        outer_shape + (mapping_operator.shape[1],),
        dtype=float,
    )

    if not outer_dims:
        input_values = np.asarray(transposed.data, dtype=float).reshape(
            1,
            feature_count,
        )
        output[...] = mapping_operator.T.dot(input_values.T).T.reshape(
            mapping_operator.shape[1]
        )
        return output, outer_dims

    chunk_dim = outer_dims[0]
    chunk_length = transposed.sizes[chunk_dim]
    if time_chunk_size is None:
        time_chunk_size = chunk_length
    if not isinstance(time_chunk_size, (int, np.integer)) or time_chunk_size <= 0:
        raise ValueError("time_chunk_size must be a positive integer or None")

    for start in range(0, chunk_length, time_chunk_size):
        stop = min(start + time_chunk_size, chunk_length)
        chunk = transposed.isel({chunk_dim: slice(start, stop)})
        input_values = np.asarray(chunk.data, dtype=float).reshape(
            -1,
            feature_count,
        )
        mapped_values = mapping_operator.T.dot(input_values.T).T
        output[start:stop] = mapped_values.reshape(
            (stop - start,)
            + outer_shape[1:]
            + (mapping_operator.shape[1],)
        )
    return output, outer_dims


def load_ecoinvent_mapping(mapping_location):
    """Load the country and technology allocation table used by SHRECC.

    ``mapping_location`` may be an existing DataFrame, point directly to a CSV
    file, or point to a directory containing ``el_map_all_norm.csv``.

    Args:
        mapping_location: Allocation DataFrame, mapping CSV, or its directory.

    Returns:
        A copy of the supplied DataFrame or the mapping loaded from CSV. Rows
        identify ecoinvent activities and columns identify country-technology
        combinations.
    """
    if isinstance(mapping_location, pd.DataFrame):
        return mapping_location.copy()

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


def validate_consumption_profile(consumption_profile):
    """Validate and normalize an inventory consumption profile.

    String profiles select built-in weighting behavior. A pandas Series
    supplies custom, non-negative weights indexed by timestamp. Custom values
    may be relative weights or physical consumption because aggregation
    normalizes by their sum.

    Args:
        consumption_profile: ``"flat"``, ``"national_demand"``, or a pandas
            Series with a unique DatetimeIndex.

    Returns:
        A normalized built-in profile name or a defensive float copy of the
        custom Series.

    Raises:
        TypeError: If the profile is not a supported string or Series.
        ValueError: If the profile name, index, or values are invalid.
    """
    if isinstance(consumption_profile, str):
        normalized = consumption_profile.strip().lower()
        if normalized not in VALID_CONSUMPTION_PROFILES:
            raise ValueError(
                "consumption_profile must be 'flat', 'national_demand', "
                "or a pandas Series"
            )
        return normalized

    if not isinstance(consumption_profile, pd.Series):
        raise TypeError(
            "consumption_profile must be 'flat', 'national_demand', "
            "or a pandas Series"
        )
    if not isinstance(consumption_profile.index, pd.DatetimeIndex):
        raise ValueError(
            "A custom consumption_profile must use a pandas DatetimeIndex"
        )
    if consumption_profile.empty:
        raise ValueError("A custom consumption_profile cannot be empty")
    if consumption_profile.index.has_duplicates:
        raise ValueError(
            "A custom consumption_profile cannot contain duplicate timestamps"
        )

    try:
        normalized = consumption_profile.astype(float).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "A custom consumption_profile must contain numeric values"
        ) from exc

    values = normalized.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(
            "A custom consumption_profile cannot contain missing or "
            "non-finite values"
        )
    if (values < 0).any():
        raise ValueError(
            "A custom consumption_profile cannot contain negative values"
        )
    return normalized


def validate_inventory_resolution(inventory_resolution):
    """Validate and normalize the foreground inventory resolution.

    ``"yearly"`` is accepted as an alias for the canonical ``"annual"``
    spelling.

    Args:
        inventory_resolution: ``"annual"``, ``"monthly"``, or the
            ``"yearly"`` alias.

    Returns:
        The normalized resolution name.

    Raises:
        TypeError: If the value is not a string.
        ValueError: If the resolution is unsupported.
    """
    if not isinstance(inventory_resolution, str):
        raise TypeError("inventory_resolution must be a string")
    normalized = inventory_resolution.strip().lower()
    if normalized == "yearly":
        normalized = "annual"
    if normalized == "hourly":
        raise ValueError(
            "Hourly inventories are not written to Brightway. Use "
            "NewDatabase.lcia() to calculate hourly impacts."
        )
    if normalized not in VALID_INVENTORY_RESOLUTIONS:
        raise ValueError(
            "inventory_resolution must be 'annual' or 'monthly'"
        )
    return normalized


def aggregate_consumption_mix(
    consumption_mix,
    *,
    consumption_profile="flat",
    consumption_volume=None,
    inventory_resolution="annual",
    countries=None,
    times=None,
    general_range=None,
    refined_range=None,
    freq=None,
):
    """Resolve hourly canonical mixes to the requested inventory resolution.

    ``"flat"`` gives every selected hour equal weight.
    ``"national_demand"`` weights each consumer country by its corresponding
    hourly ``consumption_volume``. A custom Series supplies one common set of
    timestamp weights for all consumer countries. Profiles are applied within
    each annual or monthly inventory.

    Time and country selection happen before weighting. Annual output has no
    time dimension. Monthly output uses month-start timestamps.

    Args:
        consumption_mix: Dimensionless canonical DataArray resolved by time,
            consumer country, source country, and technology.
        consumption_profile: Built-in profile name or custom pandas Series.
        consumption_volume: Hourly volume DataArray required by
            ``"national_demand"``.
        inventory_resolution: ``"annual"`` or ``"monthly"``.
        countries: Optional consumer countries to include.
        times: Optional exact timestamps to include.
        general_range: Optional inclusive start and end timestamps.
        refined_range: Optional inclusive clock-hour range.
        freq: Frequency used to construct the refined selection.

    Returns:
        Dimensionless canonical mix at the requested inventory resolution.

    Raises:
        ValueError: If dimensions, timestamp coverage, weights, or national
            demand volumes are invalid.
    """
    required_dims = {
        "time",
        "consumer_country",
        "source_country",
        "technology",
    }
    missing_dims = required_dims.difference(consumption_mix.dims)
    if missing_dims:
        raise ValueError(
            "consumption_mix is missing required dimensions: "
            + ", ".join(sorted(missing_dims))
        )

    resolution = validate_inventory_resolution(inventory_resolution)
    selected = consumption_mix
    if countries is not None:
        selected = selected.sel(consumer_country=list(countries))
    selected = filter_consumption_mix_time(
        selected,
        times=times,
        general_range=general_range,
        refined_range=refined_range,
        freq=freq,
    )

    profile = validate_consumption_profile(consumption_profile)
    selected_times = selected["time"].to_index()
    selected_countries = selected["consumer_country"].to_index()
    if isinstance(profile, str) and profile == "flat":
        weights = xr.DataArray(
            np.ones(len(selected_times), dtype=float),
            dims="time",
            coords={"time": selected_times},
            name="consumption_weight",
        )
        profile_name = "flat"
    elif isinstance(profile, str) and profile == "national_demand":
        if consumption_volume is None:
            raise ValueError(
                "consumption_volume is required when consumption_profile is "
                "'national_demand'"
            )
        required_volume_dims = {"time", "consumer_country"}
        missing_volume_dims = required_volume_dims.difference(
            consumption_volume.dims
        )
        if missing_volume_dims:
            raise ValueError(
                "consumption_volume is missing required dimensions: "
                + ", ".join(sorted(missing_volume_dims))
            )
        try:
            weights = consumption_volume.sel(
                time=selected_times,
                consumer_country=selected_countries,
            ).astype(float)
        except KeyError as exc:
            raise ValueError(
                "consumption_volume does not cover all selected timestamps "
                "and consumer countries"
            ) from exc
        profile_name = "national_demand"
    else:
        if selected_times.tz != profile.index.tz:
            raise ValueError(
                "Custom consumption_profile timestamps and SHRECC timestamps "
                "must use the same timezone"
            )
        missing_times = selected_times.difference(profile.index)
        if not missing_times.empty:
            examples = ", ".join(map(str, missing_times[:5]))
            raise ValueError(
                "Custom consumption_profile is missing selected timestamps: "
                + examples
            )
        weights = xr.DataArray(
            profile.reindex(selected_times).to_numpy(dtype=float),
            dims="time",
            coords={"time": selected_times},
            name="consumption_weight",
        )
        profile_name = "custom"

    weight_values = weights.to_numpy()
    if not np.isfinite(weight_values).all():
        raise ValueError("Selected consumption weights must be finite")
    if (weight_values < 0).any():
        raise ValueError("Selected consumption weights cannot be negative")

    if resolution == "monthly":
        monthly_mixes = []
        month_periods = selected_times.to_period("M")
        for month in month_periods.unique():
            positions = np.flatnonzero(month_periods == month)
            period_mix = selected.isel(time=positions)
            period_weights = weights.isel(time=positions)
            weight_totals = period_weights.sum("time")
            _raise_for_zero_weight_totals(
                weight_totals,
                period=str(month),
            )
            if profile_name == "flat":
                aggregated_period = period_mix.mean("time")
            else:
                aggregated_period = xr.dot(
                    period_mix,
                    period_weights / weight_totals,
                    dim="time",
                )
            monthly_mixes.append(
                aggregated_period.expand_dims(
                    time=[month.start_time],
                )
            )
        aggregated = xr.concat(monthly_mixes, dim="time")
        aggregated = aggregated.transpose(
            "time",
            "consumer_country",
            "source_country",
            "technology",
        )
    else:
        weight_totals = weights.sum("time")
        _raise_for_zero_weight_totals(weight_totals)
        if profile_name == "flat":
            aggregated = selected.mean("time")
        else:
            aggregated = xr.dot(
                selected,
                weights / weight_totals,
                dim="time",
            )
        aggregated = aggregated.transpose(
            "consumer_country",
            "source_country",
            "technology",
        )
    aggregated = aggregated.rename("consumption_mix")
    aggregated.attrs.update(consumption_mix.attrs)
    aggregated.attrs["unit"] = "dimensionless"
    aggregated.attrs["consumption_profile"] = profile_name
    aggregated.attrs["inventory_resolution"] = resolution
    if profile_name == "custom" and profile.name is not None:
        aggregated.attrs["consumption_profile_name"] = str(profile.name)
    return aggregated


def _raise_for_zero_weight_totals(weight_totals, period=None):
    """Raise with coordinate details when a weighted period has no demand."""
    zero_totals = weight_totals <= 0
    if not bool(zero_totals.any()):
        return

    zero_positions = np.argwhere(zero_totals.to_numpy())
    examples = []
    for position in zero_positions[:5]:
        labels = [
            str(zero_totals[dimension].to_numpy()[index])
            for dimension, index in zip(zero_totals.dims, position)
        ]
        examples.append(" / ".join(labels))
    details = ": " + ", ".join(examples) if examples else ""
    if period is not None:
        details = f" in {period}" + details
    raise ValueError(
        "Selected consumption weights must sum to more than zero" + details
    )


def map_consumption_mix_to_ecoinvent_activities(
    consumption_mix,
    activity_mapping,
    *,
    country_aliases=None,
    fallback_activity=DEFAULT_FALLBACK_ACTIVITY,
    check=True,
    return_mapping_gaps=False,
    time_chunk_size=DEFAULT_MAPPING_TIME_CHUNK_SIZE,
):
    """Map source-country technologies to ecoinvent activities.

    By default, unmapped shares are assigned to the high-voltage production
    mix of their source country. This preserves national generation without
    embedding network infrastructure. A fallback activity with an explicit
    geography can still be supplied to aggregate all unmapped shares there.

    Set ``return_mapping_gaps`` to return the detailed, pre-fallback shares by
    source country and technology alongside the mapped activity mix.

    Args:
        consumption_mix: Dimensionless xarray DataArray resolved by time,
            consumer country, source country, and source technology.
        activity_mapping: Ecoinvent allocation DataFrame returned by
            :func:`load_ecoinvent_mapping`.
        country_aliases: Optional source-to-ecoinvent country-code overrides.
        fallback_activity: Four-item tuple containing fallback geography,
            activity name, product, and unit. A ``None`` geography preserves
            each source country; an explicit geography aggregates all gaps.
        check: Verify that mapping does not over-allocate and that fallback
            assignment conserves the complete input mix.
        return_mapping_gaps: Also return detailed shares that had no direct
            source-country and technology mapping.
        time_chunk_size: Number of positions along the first non-mapping
            dimension to process per sparse multiplication. The default maps
            one week at a time for canonical hourly results. Set to ``None``
            to process the complete array at once.

    Returns:
        Mapped activity-mix DataArray. If ``return_mapping_gaps`` is true,
        returns ``(activity_mix, mapping_gap)`` instead.

    Raises:
        TypeError: If ``activity_mapping`` is not a DataFrame.
        ValueError: If required dimensions or mapping index levels are absent.
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
    fallback_geography, fallback_name, fallback_product, fallback_unit = (
        fallback_activity
    )
    mapping_operator, mapping_coverage, direct_activity_count = (
        _build_ecoinvent_mapping_operator(
            activity_mapping,
            source_countries,
            technologies,
            aliases,
            fallback_geography,
        )
    )
    mapped_values, outer_dims = _apply_sparse_mapping_operator(
        consumption_mix,
        mapping_operator,
        time_chunk_size=time_chunk_size,
    )

    fallback_countries = [
        aliases.get(country, country) for country in source_countries
    ]
    if fallback_geography is None:
        fallback_index = [
            (country, fallback_name, fallback_product, fallback_unit)
            for country in fallback_countries
        ]
    else:
        fallback_index = [tuple(fallback_activity)]

    activity_index = list(activity_mapping.index) + fallback_index
    preserved_coords = {
        name: coord
        for name, coord in consumption_mix.coords.items()
        if set(coord.dims).issubset(outer_dims)
    }
    preserved_coords["activity"] = np.arange(len(activity_index))
    activity_mix = xr.DataArray(
        mapped_values,
        dims=outer_dims + ["activity"],
        coords=preserved_coords,
        name="activity_mix",
    )
    activity_mix = activity_mix.assign_coords(
        geography=("activity", [item[0] for item in activity_index]),
        activity_name=("activity", [item[1] for item in activity_index]),
        product=("activity", [item[2] for item in activity_index]),
        unit=("activity", [item[3] for item in activity_index]),
    )
    activity_mix.attrs["unit"] = "dimensionless"

    mapped_total = activity_mix.isel(
        activity=slice(0, direct_activity_count)
    ).sum("activity")
    input_total = consumption_mix.sum(["source_country", "technology"])
    mapping_gaps = None
    if return_mapping_gaps:
        mapping_coverage_xr = xr.DataArray(
            mapping_coverage,
            dims=("source_country", "technology"),
            coords={
                "source_country": source_countries,
                "technology": technologies,
            },
        )
        mapping_gaps = (
            consumption_mix * (1 - mapping_coverage_xr).clip(min=0)
        ).rename("mapping_gap")
        mapping_gaps.attrs["unit"] = "dimensionless"

    if check:
        if (mapped_total > input_total + 1e-8).any():
            raise ValueError("Activity mapping allocates more than the input mix")
        np.testing.assert_allclose(
            activity_mix.sum("activity"),
            input_total,
            atol=1e-8,
        )

    if return_mapping_gaps:
        return activity_mix, mapping_gaps
    return activity_mix


def mapping_gap_to_report(
    mapping_gap,
    *,
    countries=None,
    times=None,
    general_range=None,
    refined_range=None,
    freq=None,
    atol=1e-12,
):
    """Summarize detailed activity-mapping gaps for a selected period.

    Args:
        mapping_gap: Dimensionless xarray DataArray containing unmapped shares
            with source-country, technology, and consumer-country dimensions.
        countries: Optional consumer countries to include.
        times: Optional exact timestamps to include.
        general_range: Optional inclusive start and end timestamps.
        refined_range: Optional inclusive hour range inside ``general_range``.
        freq: Pandas frequency used to construct the refined selection.
        atol: Ignore rows whose absolute values do not exceed this tolerance.

    Returns:
        DataFrame indexed by source country and technology, with consumer
        countries as columns. Values are mean shares over the selected times.
    """
    required_dims = {"source_country", "technology", "consumer_country"}
    missing_dims = required_dims.difference(mapping_gap.dims)
    if missing_dims:
        raise ValueError(
            "mapping_gap is missing required dimensions: "
            + ", ".join(sorted(missing_dims))
        )

    selected = mapping_gap
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

    report = selected.to_dataframe(name="share").reset_index()
    report = (
        report.groupby(
            ["source_country", "technology", "consumer_country"],
            dropna=False,
        )["share"]
        .sum()
        .unstack("consumer_country", fill_value=0)
    )
    report.columns.name = None
    report = report.loc[report.abs().gt(atol).any(axis=1)]
    return report.sort_index()


def activity_mix_to_database_table(
    activity_mix,
    *,
    countries=None,
    times=None,
    general_range=None,
    refined_range=None,
    freq=None,
    inventory_resolution="annual",
    preserve_time=False,
):
    """Aggregate an activity mix into the table consumed by ``create_database``.

    Args:
        activity_mix: Dimensionless mapped xarray DataArray with activity and
            consumer-country dimensions and activity metadata coordinates.
        countries: Optional consumer countries to include.
        times: Optional exact timestamps to include.
        general_range: Optional inclusive start and end timestamps.
        refined_range: Optional inclusive hour range inside ``general_range``.
        freq: Pandas frequency used to construct the refined selection.
        inventory_resolution: Whether columns represent annual or monthly
            inventories.
        preserve_time: Keep hourly timestamps instead of applying annual
            aggregation. Intended for in-memory assessment inputs.

    Returns:
        DataFrame indexed by activity geography, name, product, and unit, with
        one normalized electricity-mix column per consumer country.
    """
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

    resolution = validate_inventory_resolution(inventory_resolution)
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
    if "time" in selected.dims and resolution == "annual" and not preserve_time:
        selected = selected.mean("time")

    table = selected.to_dataframe(name="share").reset_index()
    row_levels = ["geography", "activity_name", "product", "unit"]
    if "time" in selected.dims:
        table = (
            table.groupby(
                [*row_levels, "time", "consumer_country"],
                dropna=False,
            )["share"]
            .sum()
            .unstack(["time", "consumer_country"], fill_value=0)
        )
        table.columns.names = ["time", "country"]
    else:
        table = (
            table.groupby(
                [*row_levels, "consumer_country"],
                dropna=False,
            )["share"]
            .sum()
            .unstack("consumer_country", fill_value=0)
        )
        table.columns.name = None
    table.index.names = ["geography", "activityName", "product", "unit"]
    return table


def filter_consumption_mix_time(
    mix,
    *,
    times=None,
    general_range=None,
    refined_range=None,
    freq=None,
):
    """Apply the time-selection semantics used by ``filt_cutoff``.

    Exact ``times`` can be combined with ``general_range``. A
    ``refined_range`` selects inclusive clock hours within ``general_range``
    using timestamps generated at ``freq``.

    Args:
        mix: xarray DataArray or Dataset to select.
        times: Optional exact timestamps to retain.
        general_range: Optional inclusive start and end timestamps.
        refined_range: Optional inclusive start and end hours.
        freq: Pandas frequency used to construct the refined selection.

    Returns:
        The selected xarray object, retaining all non-time dimensions.

    Raises:
        ValueError: If the selection is inconsistent or contains no timestamps.
    """
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
