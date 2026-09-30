"""Historical full-matrix TYNDP solver retained for equivalence validation."""

from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr


def _log(message, verbose):
    if verbose:
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}")


def _legacy_consumption_mix_from_z_gross(
    Z_gross,
    check=True,
    return_debug=False,
    zero_consumption="raise",
    verbose=False,
):
    """Calculate consumption mixes with the former full-matrix algorithm."""
    valid_zero_consumption = {"raise", "month_hour_average"}
    if zero_consumption not in valid_zero_consumption:
        raise ValueError(
            "zero_consumption must be one of: "
            + ", ".join(sorted(valid_zero_consumption))
        )

    _log("Building domestic production and trade denominator matrices", verbose)
    Z_domestic = (
        Z_gross["production mix"]
        .T.groupby(level=["country from", "country to"])
        .sum()
        .T
    )
    Z_trade = (
        pd.concat(
            [Z_gross["trade"].droplevel("source", axis=1), Z_domestic],
            axis=1,
        )
        .swaplevel(axis=1)
        .sort_index(axis=1)
    )
    country_total = Z_trade.T.groupby(level="country to").sum().T

    _log("Calculating direct production and trade shares", verbose)
    A_prod = (
        Z_gross["production mix"]
        .div(country_total, level="country to", axis=1)
        .droplevel("country to", axis=1)
        .fillna(0)
    )
    A_trade = Z_trade.div(country_total, level="country to", axis=1).fillna(0)

    countries = (
        A_trade.columns.get_level_values("country to")
        .unique()
        .union(A_trade.columns.get_level_values("country from"))
        .unique()
        .sort_values()
    )
    activities = A_prod.columns.get_level_values("source").unique().sort_values()
    country_total_ordered = country_total.reindex(columns=countries, fill_value=0)
    _log(
        f"Using {len(countries)} countries and {len(activities)} technologies",
        verbose,
    )

    _log("Reshaping production shares", verbose)
    prod_cols = pd.MultiIndex.from_product(
        [activities, countries],
        names=["source", "country from"],
    )
    A_prod_ordered = A_prod.copy()
    if list(A_prod_ordered.columns.names) == ["country from", "source"]:
        A_prod_ordered = A_prod_ordered.reorder_levels(
            ["source", "country from"], axis=1
        )
    A_prod_ordered = A_prod_ordered.reindex(columns=prod_cols, fill_value=0)

    A_prod_3d = A_prod_ordered.to_numpy().reshape(
        len(A_prod_ordered),
        len(activities),
        len(countries),
    )
    time_count, technology_count, country_count = A_prod_3d.shape
    A_prod_expanded = np.zeros(
        (time_count, technology_count * country_count, country_count),
        dtype=A_prod_3d.dtype,
    )
    rows = (
        np.arange(country_count)[:, None] * technology_count
        + np.arange(technology_count)[None, :]
    ).ravel()
    cols = np.repeat(np.arange(country_count), technology_count)
    values = np.transpose(A_prod_3d, (0, 2, 1)).reshape(
        time_count, country_count * technology_count
    )
    A_prod_expanded[:, rows, cols] = values

    _log("Reshaping trade shares", verbose)
    trade_cols = pd.MultiIndex.from_product(
        [countries, countries],
        names=["country to", "country from"],
    )
    A_trade_wide = A_trade.reindex(columns=trade_cols, fill_value=0)
    self_index = pd.MultiIndex.from_arrays(
        [countries, countries],
        names=["country to", "country from"],
    )
    A_trade_wide.loc[:, self_index] = 0
    A_trade_3d = A_trade_wide.to_numpy().reshape(
        time_count, country_count, country_count
    )

    direct_total = A_prod_expanded.sum(axis=1) + A_trade_3d.sum(axis=2)
    zero_consumption_mask = country_total_ordered.to_numpy() == 0
    if check:
        _log("Checking direct shares sum to one", verbose)
        if zero_consumption_mask.any() and zero_consumption == "raise":
            zero_positions = np.argwhere(zero_consumption_mask)
            examples = [
                f"{country_total_ordered.index[time_idx]} / {countries[country_idx]}"
                for time_idx, country_idx in zero_positions[:10]
            ]
            raise ValueError(
                "Cannot normalize direct supply shares for country-hours with "
                "zero total consumption. First examples: "
                + "; ".join(examples)
            )
        np.testing.assert_allclose(
            direct_total[~zero_consumption_mask],
            1,
            atol=1e-8,
        )

    _log("Solving hourly trade system", verbose)
    matrix = np.eye(country_count)[None, :, :] - A_trade_3d
    consumption_mix_3d = np.linalg.solve(
        matrix,
        A_prod_expanded.transpose(0, 2, 1),
    ).transpose(0, 2, 1)

    zero_consumption_imputed = None
    if zero_consumption_mask.any() and zero_consumption == "month_hour_average":
        _log(
            "Imputing zero-consumption country-hours with month-hour averages",
            verbose,
        )
        zero_consumption_imputed = _impute_zero_consumption_month_hour_average(
            consumption_mix_3d,
            country_total_ordered,
            zero_consumption_mask,
            countries,
        )

    if check:
        _log("Checking final consumption mix sums to one", verbose)
        np.testing.assert_allclose(consumption_mix_3d.sum(axis=1), 1, atol=1e-8)

    _log("Building xarray DataArray", verbose)
    tech_country_index = pd.MultiIndex.from_product(
        [countries, activities],
        names=["source_country", "technology"],
    )
    consumption_mix_xr = xr.DataArray(
        consumption_mix_3d,
        dims=("time", "source", "consumer_country"),
        coords={
            "time": Z_gross.index,
            "source": tech_country_index,
            "consumer_country": countries,
        },
        name="consumption_mix",
    ).unstack("source")
    _log(f"Built consumption_mix_xr with shape {consumption_mix_xr.shape}", verbose)

    if return_debug:
        return consumption_mix_xr, {
            "A_prod": A_prod,
            "A_trade": A_trade,
            "A_prod_3d": A_prod_3d,
            "A_trade_3d": A_trade_3d,
            "A_prod_expanded": A_prod_expanded,
            "country_total": country_total,
            "country_total_ordered": country_total_ordered,
            "countries": countries,
            "direct_total": direct_total,
            "zero_consumption": zero_consumption,
            "zero_consumption_mask": zero_consumption_mask,
            "zero_consumption_imputed": zero_consumption_imputed,
        }
    return consumption_mix_xr


def _impute_zero_consumption_month_hour_average(
    consumption_mix_3d,
    country_total_ordered,
    zero_consumption_mask,
    countries,
):
    """Fill undefined mixes from weighted averages for the same country."""
    if not isinstance(country_total_ordered.index, pd.DatetimeIndex):
        raise ValueError(
            "zero_consumption='month_hour_average' requires a DatetimeIndex"
        )

    times = country_total_ordered.index
    weights = country_total_ordered.to_numpy()
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
                ["month_hour", "hour", "annual"],
                candidate_masks,
            ):
                if candidate_mask.any():
                    mix = np.average(
                        consumption_mix_3d[candidate_mask, :, country_idx],
                        axis=0,
                        weights=country_weights[candidate_mask],
                    )
                    consumption_mix_3d[time_idx, :, country_idx] = mix
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

    return pd.DataFrame(
        imputed,
        columns=["time", "country", "fallback", "sample_count", "weight_sum"],
    )
