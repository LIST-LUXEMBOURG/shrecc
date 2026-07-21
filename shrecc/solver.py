"""Shared electricity-network solving and consumption-mix data model."""

import numpy as np
import pandas as pd
import xarray as xr

PRODUCTION_DIMS = ("time", "producer_country", "technology")
TRADE_DIMS = ("time", "exporter_country", "importer_country")
CONSUMPTION_DIMS = ("time", "consumer_country")


def solve_consumption_system(
    production_volume,
    trade_volume,
    *,
    consumption_volume=None,
    check=True,
    zero_consumption="raise",
    unresolved_technology=None,
    volume_unit="MWh",
    include_consumption_mix_volume=True,
):
    """Resolve country-to-country exchanges and retain volumes and shares.

    ``production_volume`` contains domestic generation before electricity is
    exchanged. ``trade_volume`` contains positive directional flows from
    exporter to importer. Only the country trade block is solved; production
    technologies are projected through the resolved network afterwards.

    If ``consumption_volume`` is omitted, it is calculated from the country
    balance ``production + imports - exports``. The returned Dataset contains
    raw production and trade, total country consumption, consumption resolved
    by origin and technology, and its normalized mix. Set
    ``include_consumption_mix_volume`` to false when only normalized shares are
    needed, avoiding a second large four-dimensional result array.

    ``unresolved_technology`` can name a fallback origin for country-hours
    which export or consume electricity despite having no recorded production
    or imports. This preserves incomplete source data explicitly instead of
    silently losing its contribution during the reduced solve.
    """
    _validate_input_array(production_volume, "production_volume", PRODUCTION_DIMS)
    _validate_input_array(trade_volume, "trade_volume", TRADE_DIMS)

    valid_zero_consumption = {"keep_zero", "month_hour_average", "raise"}
    if zero_consumption not in valid_zero_consumption:
        raise ValueError(
            "zero_consumption must be one of: "
            + ", ".join(sorted(valid_zero_consumption))
        )

    production_volume, trade_volume, countries = _align_country_axes(
        production_volume,
        trade_volume,
    )
    times = production_volume["time"]
    country_values = countries.to_numpy()

    production = production_volume.transpose(*PRODUCTION_DIMS).to_numpy()
    trade = trade_volume.transpose(*TRADE_DIMS).to_numpy()
    if check:
        if not np.isfinite(production).all() or not np.isfinite(trade).all():
            raise ValueError("Production and trade volumes must be finite")
        if (production < 0).any():
            raise ValueError("production_volume must contain nonnegative generation")
        if (trade < 0).any():
            raise ValueError("trade_volume must contain positive directional flows")

    # Diagonal trade is domestic supply and is already represented by
    # production. Ignore it if a source dataset happens to include it.
    diagonal = np.arange(len(countries))
    trade[:, diagonal, diagonal] = 0

    domestic_production = production.sum(axis=2)
    imports = trade.sum(axis=1)
    exports = trade.sum(axis=2)
    node_supply = domestic_production + imports

    if consumption_volume is None:
        consumption = node_supply - exports
    else:
        _validate_input_array(
            consumption_volume,
            "consumption_volume",
            CONSUMPTION_DIMS,
        )
        consumption_volume = consumption_volume.reindex(
            time=times,
            consumer_country=country_values,
        )
        if consumption_volume.isnull().any():
            raise ValueError(
                "consumption_volume does not cover all solver times and countries"
            )
        consumption = consumption_volume.transpose(*CONSUMPTION_DIMS).to_numpy()

    zero_supply_mask = node_supply == 0
    if check and (consumption < -1e-8).any():
        positions = np.argwhere(consumption < -1e-8)
        examples = [
            f"{pd.Timestamp(times.values[t])} / {countries[c]}"
            for t, c in positions[:10]
        ]
        raise ValueError(
            "Country balance gives negative consumption volumes. First examples: "
            + "; ".join(examples)
        )
    consumption = np.where(np.abs(consumption) < 1e-12, 0, consumption)

    unresolved_origin_mask = zero_supply_mask & ((exports > 0) | (consumption > 0))
    if unresolved_technology is not None and unresolved_origin_mask.any():
        production_volume, production = _append_zero_technology(
            production_volume,
            production,
            unresolved_technology,
        )
    technologies = production_volume["technology"]

    direct_production = np.divide(
        production,
        node_supply[:, :, None],
        out=np.zeros_like(production, dtype=float),
        where=node_supply[:, :, None] != 0,
    )

    # Solver orientation is consumer country by supplying country.
    direct_trade = np.divide(
        trade.transpose(0, 2, 1),
        node_supply[:, :, None],
        out=np.zeros_like(trade, dtype=float),
        where=node_supply[:, :, None] != 0,
    )

    direct_total = direct_production.sum(axis=2) + direct_trade.sum(axis=2)
    if check:
        if zero_supply_mask.any() and zero_consumption == "raise":
            positions = np.argwhere(zero_supply_mask)
            examples = [
                f"{pd.Timestamp(times.values[t])} / {countries[c]}"
                for t, c in positions[:10]
            ]
            raise ValueError(
                "Cannot normalize direct supply shares for country-hours with "
                "zero total consumption. First examples: " + "; ".join(examples)
            )
        np.testing.assert_allclose(
            direct_total[~zero_supply_mask],
            1,
            atol=1e-8,
        )

    # Right-hand side is (time, consumer_country, producer_country-technology).
    direct_origin = np.zeros(
        (
            len(times),
            len(countries),
            len(countries) * len(technologies),
        ),
        dtype=float,
    )
    rows = np.arange(len(countries))
    origin_slices = (
        rows[:, None] * len(technologies) + np.arange(len(technologies))[None, :]
    )
    direct_origin[:, rows[:, None], origin_slices] = direct_production
    if unresolved_technology is not None and unresolved_origin_mask.any():
        unresolved_idx = len(technologies) - 1
        unresolved_rows = np.argwhere(unresolved_origin_mask)
        for time_idx, country_idx in unresolved_rows:
            origin_idx = country_idx * len(technologies) + unresolved_idx
            direct_origin[time_idx, country_idx, origin_idx] = 1

    system = np.eye(len(countries))[None, :, :] - direct_trade
    consumption_mix_flat = np.linalg.solve(system, direct_origin)
    consumption_mix = consumption_mix_flat.reshape(
        len(times),
        len(countries),
        len(countries),
        len(technologies),
    )

    imputation = None
    if zero_supply_mask.any() and zero_consumption == "month_hour_average":
        consumption_mix, imputation = _impute_zero_consumption(
            consumption_mix,
            node_supply,
            zero_supply_mask,
            times.to_index(),
            countries,
        )

    if check:
        expected_mix_total = np.ones_like(node_supply)
        if zero_consumption == "keep_zero":
            has_unresolved_origin = np.zeros_like(zero_supply_mask)
            if unresolved_technology is not None:
                has_unresolved_origin = unresolved_origin_mask
            expected_mix_total[zero_supply_mask & ~has_unresolved_origin] = 0
        np.testing.assert_allclose(
            consumption_mix.sum(axis=(2, 3)),
            expected_mix_total,
            atol=1e-8,
        )

    mix_dims = (
        "time",
        "consumer_country",
        "source_country",
        "technology",
    )
    mix_coords = {
        "time": times,
        "consumer_country": country_values,
        "source_country": country_values,
        "technology": technologies,
    }
    consumption_mix_xr = xr.DataArray(
        consumption_mix,
        dims=mix_dims,
        coords=mix_coords,
        name="consumption_mix",
        attrs={"unit": "dimensionless"},
    )
    consumption_volume_xr = xr.DataArray(
        consumption,
        dims=CONSUMPTION_DIMS,
        coords={"time": times, "consumer_country": country_values},
        name="consumption_volume",
        attrs={"unit": volume_unit},
    )

    data_vars = {
        "production_volume": production_volume.assign_attrs(unit=volume_unit),
        "trade_volume": trade_volume.assign_attrs(unit=volume_unit),
        "consumption_volume": consumption_volume_xr,
        "consumption_mix": consumption_mix_xr,
    }
    if include_consumption_mix_volume:
        data_vars["consumption_mix_volume"] = (
            consumption_mix_xr * consumption_volume_xr
        ).assign_attrs(unit=volume_unit)

    result = xr.Dataset(
        data_vars,
        attrs={
            "volume_unit": volume_unit,
            "solver": "country_trade_block",
        },
    )

    debug = {
        "direct_production": direct_production,
        "direct_trade": direct_trade,
        "direct_total": direct_total,
        "node_supply_volume": node_supply,
        "zero_consumption_mask": zero_supply_mask,
        "unresolved_origin_mask": unresolved_origin_mask,
        "unresolved_technology": unresolved_technology,
        "zero_consumption": zero_consumption,
        "zero_consumption_imputed": imputation,
    }
    return result, debug


def iter_consumption_system_results(
    production_volume,
    trade_volume,
    *,
    consumption_volume=None,
    time_chunk_size=168,
    **solver_options,
):
    """Solve a long time series in bounded chunks.

    Each yielded Dataset has the same canonical schema as
    :func:`solve_consumption_system`. Chunking limits peak memory while leaving
    callers free to persist, map, or visualize each result independently.

    ``month_hour_average`` requires information from times outside an
    individual chunk and is therefore only accepted when one chunk covers the
    complete input. A future two-pass imputer can lift this restriction without
    changing the result schema.
    """
    _validate_input_array(production_volume, "production_volume", PRODUCTION_DIMS)
    _validate_input_array(trade_volume, "trade_volume", TRADE_DIMS)

    if not isinstance(time_chunk_size, int) or time_chunk_size <= 0:
        raise ValueError("time_chunk_size must be a positive integer")

    time_count = production_volume.sizes["time"]
    if trade_volume.sizes["time"] != time_count:
        raise ValueError("production_volume and trade_volume must use the same times")
    if consumption_volume is not None:
        _validate_input_array(
            consumption_volume,
            "consumption_volume",
            CONSUMPTION_DIMS,
        )
        if consumption_volume.sizes["time"] != time_count:
            raise ValueError(
                "production_volume and consumption_volume must use the same times"
            )

    zero_consumption = solver_options.get("zero_consumption", "raise")
    if zero_consumption == "month_hour_average" and time_chunk_size < time_count:
        raise ValueError(
            "zero_consumption='month_hour_average' cannot yet be used across "
            "multiple time chunks"
        )

    for start in range(0, time_count, time_chunk_size):
        time_slice = slice(start, min(start + time_chunk_size, time_count))
        chunk_consumption = None
        if consumption_volume is not None:
            chunk_consumption = consumption_volume.isel(time=time_slice)

        results, _ = solve_consumption_system(
            production_volume.isel(time=time_slice),
            trade_volume.isel(time=time_slice),
            consumption_volume=chunk_consumption,
            **solver_options,
        )
        yield results


def _append_zero_technology(production_volume, production, technology):
    if not isinstance(technology, str) or not technology:
        raise ValueError("unresolved_technology must be a non-empty string")
    if technology in production_volume["technology"].to_index():
        raise ValueError(
            f"unresolved_technology {technology!r} already exists in production data"
        )

    technologies = production_volume["technology"].to_index().append(
        pd.Index([technology], name="technology")
    )
    production_volume = production_volume.reindex(
        technology=technologies,
        fill_value=0,
    )
    production = np.pad(production, ((0, 0), (0, 0), (0, 1)))
    return production_volume, production


def _validate_input_array(data, name, expected_dims):
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray")
    if set(data.dims) != set(expected_dims):
        raise ValueError(
            f"{name} must have dimensions {expected_dims}; got {data.dims}"
        )


def _align_country_axes(production_volume, trade_volume):
    production_times = production_volume["time"].to_index()
    trade_times = trade_volume["time"].to_index()
    if not production_times.equals(trade_times):
        raise ValueError("production_volume and trade_volume must use the same times")

    countries = (
        production_volume["producer_country"]
        .to_index()
        .union(trade_volume["exporter_country"].to_index())
        .union(trade_volume["importer_country"].to_index())
        .sort_values()
        .rename(None)
    )
    production_volume = production_volume.reindex(
        producer_country=countries,
        fill_value=0,
    )
    trade_volume = trade_volume.reindex(
        exporter_country=countries,
        importer_country=countries,
        fill_value=0,
    )
    return production_volume, trade_volume, countries


def _impute_zero_consumption(
    consumption_mix,
    weights,
    zero_consumption_mask,
    times,
    countries,
):
    if not isinstance(times, pd.DatetimeIndex):
        raise ValueError(
            "zero_consumption='month_hour_average' requires a DatetimeIndex"
        )

    imputed = []
    for country_idx, country in enumerate(countries):
        zero_times = np.flatnonzero(zero_consumption_mask[:, country_idx])
        if len(zero_times) == 0:
            continue

        country_weights = weights[:, country_idx]
        valid = country_weights > 0
        if not valid.any():
            raise ValueError(
                "Cannot impute zero-consumption mix for country "
                f"{country}: no nonzero-consumption hours are available."
            )

        for time_idx in zero_times:
            candidate_masks = [
                valid
                & (times.month == times[time_idx].month)
                & (times.hour == times[time_idx].hour),
                valid & (times.hour == times[time_idx].hour),
                valid,
            ]
            for fallback_level, candidate_mask in zip(
                ("month_hour", "hour", "annual"),
                candidate_masks,
            ):
                if candidate_mask.any():
                    consumption_mix[time_idx, country_idx] = np.average(
                        consumption_mix[candidate_mask, country_idx],
                        axis=0,
                        weights=country_weights[candidate_mask],
                    )
                    imputed.append(
                        {
                            "time": times[time_idx],
                            "country": country,
                            "fallback": fallback_level,
                            "sample_count": int(candidate_mask.sum()),
                            "weight_sum": float(country_weights[candidate_mask].sum()),
                        }
                    )
                    break

    return consumption_mix, pd.DataFrame(
        imputed,
        columns=["time", "country", "fallback", "sample_count", "weight_sum"],
    )
