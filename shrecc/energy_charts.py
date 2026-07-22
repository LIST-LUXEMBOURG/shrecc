# Copyright © 2024 Luxembourg Institute of Science and Technology
# Licensed under the MIT License (see LICENSE file for details).
# Authors: [Sabina Bednářová, Thomas Gibon]

import json
import os
import time
from datetime import datetime
from importlib.resources import files
from pathlib import Path
from zoneinfo import ZoneInfo  # Only available in Python 3.9+

import appdirs
import pandas as pd
import requests
import xarray as xr
from tqdm import tqdm

from shrecc.result_store import (
    consumption_result_cache_path,
    load_pickle as load_from_pickle,
    save_pickle as save_to_pickle,
    write_solved_consumption_result_cache,
)
from shrecc.solver import solve_consumption_system

API_REQUEST_TIMEOUT = 20
UNRESOLVED_IMPORT_TECHNOLOGY = "Import balance"
ENERGY_CHARTS_COUNTRIES = (
    "AL",
    "AM",
    "AT",
    "AZ",
    "BA",
    "BE",
    "BG",
    "BY",
    "CH",
    "CY",
    "CZ",
    "DE",
    "DK",
    "EE",
    "ES",
    "FI",
    "FR",
    "GE",
    "GR",
    "HR",
    "HU",
    "IE",
    "IT",
    "LT",
    "LU",
    "LV",
    "MD",
    "ME",
    "MK",
    "MT",
    "NIE",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "RS",
    "RU",
    "SE",
    "SK",
    "SI",
    "TR",
    "UA",
    "UK",
    "XK",
)
TRANSIENT_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class EnergyChartsDownloadError(RuntimeError):
    """Raised when a yearly Energy Charts download remains incomplete."""


class _CountryUnavailable(Exception):
    """Internal signal for API responses that definitively lack country data."""


__all__ = (
    "API_REQUEST_TIMEOUT",
    "UNRESOLVED_IMPORT_TECHNOLOGY",
    "build_energy_charts_solver_inputs",
    "cleaning_data",
    "consumption_results_from_energy_charts",
    "data_processing",
    "energy_charts_cached_countries",
    "get_data",
    "get_package_user_data_dir",
    "get_prod",
    "get_trade",
    "write_energy_charts_consumption_cache",
    "year_to_unix",
)


def data_processing(
    data_df,
    year,
    path_to_data=None,
    *,
    time_chunk_size=168,
    include_consumption_mix_volume=False,
    legacy=False,
):
    """Process cleaned Energy Charts data with the shared reduced solver.

    Canonical results are persisted in bounded time chunks and are consumed
    directly by :func:`shrecc.database.filt_cutoff`. Set ``legacy=True`` only
    while validating workflows that require the historical matrix files.
    """
    if legacy:
        from shrecc._legacy_treatment import _legacy_data_processing

        return _legacy_data_processing(data_df, year, path_to_data=path_to_data)

    if path_to_data is None:
        data_root = Path(files("shrecc.data"))
    elif isinstance(path_to_data, (str, Path)):
        data_root = Path(path_to_data)
    else:
        raise TypeError("path_to_data must be None, str, or pathlib.Path")

    cache_dir = consumption_result_cache_path(data_root, year)
    if cache_dir.exists():
        manifest = cache_dir / "manifest.json"
        if manifest.is_file():
            print(f"Canonical consumption results loaded from {cache_dir}")
            return cache_dir
        raise FileExistsError(
            f"Incomplete canonical consumption-result cache exists at {cache_dir}"
        )

    print("Processing data with the reduced country-network solver...")
    write_energy_charts_consumption_cache(
        data_df,
        cache_dir,
        time_chunk_size=time_chunk_size,
        include_consumption_mix_volume=include_consumption_mix_volume,
    )
    print(f"Canonical consumption results saved to {cache_dir}")
    return cache_dir


def build_energy_charts_solver_inputs(data):
    """Convert cleaned Energy Charts data to canonical solver inputs.

    Positive bilateral trade values are imports into the reporting country;
    negative counterparts are discarded so each physical flow is represented
    once. Negative production values and the market import-balance pseudo-series
    are excluded because they are electricity uses or accounting corrections.
    """
    _validate_energy_charts_data(data)

    production = data.xs("production mix", axis=1, level="type").drop(
        columns="Import balance (market)",
        level="source",
        errors="ignore",
    )
    production = _aggregate_columns(
        production.clip(lower=0),
        ["country", "source"],
    )
    trade = _aggregate_columns(
        data.xs("trade", axis=1, level="type").clip(lower=0),
        ["country", "source"],
    )
    load = _aggregate_columns(
        data.xs("load", axis=1, level="type"),
        ["country"],
    )

    countries = (
        production.columns.get_level_values("country")
        .unique()
        .union(trade.columns.get_level_values("country").unique())
        .union(trade.columns.get_level_values("source").unique())
        .sort_values()
        .rename(None)
    )
    technologies = (
        production.columns.get_level_values("source")
        .unique()
        .sort_values()
        .rename(None)
    )

    production_columns = pd.MultiIndex.from_product(
        [countries, technologies],
        names=["country", "source"],
    )
    production = production.reindex(columns=production_columns, fill_value=0)

    # API columns are (reporting/importing country, partner/exporting country).
    trade = trade.reorder_levels(["source", "country"], axis=1)
    trade.columns.names = ["exporter_country", "importer_country"]
    trade_columns = pd.MultiIndex.from_product(
        [countries, countries],
        names=["exporter_country", "importer_country"],
    )
    trade = trade.reindex(columns=trade_columns, fill_value=0)
    load = load.reindex(columns=countries, fill_value=0)

    production_volume = xr.DataArray(
        production.to_numpy().reshape(
            len(data),
            len(countries),
            len(technologies),
        ),
        dims=("time", "producer_country", "technology"),
        coords={
            "time": data.index,
            "producer_country": countries.to_numpy(),
            "technology": technologies.to_numpy(),
        },
        name="production_volume",
    )
    trade_volume = xr.DataArray(
        trade.to_numpy().reshape(
            len(data),
            len(countries),
            len(countries),
        ),
        dims=("time", "exporter_country", "importer_country"),
        coords={
            "time": data.index,
            "exporter_country": countries.to_numpy(),
            "importer_country": countries.to_numpy(),
        },
        name="trade_volume",
    )
    consumption_volume = xr.DataArray(
        load.to_numpy(),
        dims=("time", "consumer_country"),
        coords={
            "time": data.index,
            "consumer_country": countries.to_numpy(),
        },
        name="consumption_volume",
    )
    return production_volume, trade_volume, consumption_volume


def consumption_results_from_energy_charts(
    data,
    *,
    check=True,
    zero_consumption="keep_zero",
    volume_unit="MWh",
    include_consumption_mix_volume=True,
    return_debug=False,
):
    """Calculate canonical consumption volumes and mixes from Energy Charts."""
    production_volume, trade_volume, consumption_volume = (
        build_energy_charts_solver_inputs(data)
    )
    results, debug = solve_consumption_system(
        production_volume,
        trade_volume,
        consumption_volume=consumption_volume,
        check=check,
        zero_consumption=zero_consumption,
        unresolved_technology=UNRESOLVED_IMPORT_TECHNOLOGY,
        volume_unit=volume_unit,
        include_consumption_mix_volume=include_consumption_mix_volume,
    )
    if return_debug:
        return results, debug
    return results


def write_energy_charts_consumption_cache(
    data,
    cache_dir,
    *,
    time_chunk_size=168,
    check=True,
    zero_consumption="keep_zero",
    volume_unit="MWh",
    include_consumption_mix_volume=True,
):
    """Solve Energy Charts data in chunks and persist canonical results."""
    production_volume, trade_volume, consumption_volume = (
        build_energy_charts_solver_inputs(data)
    )
    return write_solved_consumption_result_cache(
        production_volume,
        trade_volume,
        cache_dir,
        consumption_volume=consumption_volume,
        time_chunk_size=time_chunk_size,
        check=check,
        zero_consumption=zero_consumption,
        unresolved_technology=UNRESOLVED_IMPORT_TECHNOLOGY,
        volume_unit=volume_unit,
        include_consumption_mix_volume=include_consumption_mix_volume,
    )


def _aggregate_columns(data, levels):
    return data.T.groupby(level=levels, sort=True).sum().T


def _validate_energy_charts_data(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("data columns must be a MultiIndex")
    required_levels = {"country", "type", "source"}
    missing_levels = required_levels.difference(data.columns.names)
    if missing_levels:
        raise ValueError(
            "Energy Charts columns are missing required levels: "
            + ", ".join(sorted(missing_levels))
        )
    required_types = {"production mix", "trade", "load"}
    missing_types = required_types.difference(data.columns.get_level_values("type"))
    if missing_types:
        raise ValueError(
            "Energy Charts data is missing required types: "
            + ", ".join(sorted(missing_types))
        )


def get_prod(
    start, end, country, cumul=False, rolling=False, timeout=API_REQUEST_TIMEOUT
):
    """
    Downloads production data from the Energy Charts API. Gets called from `get_data()`.

    Args:
        start (int): Start of the download period (output of `year_to_unix()`) in unix seconds.
        end (int): End of the download period (output of `year_to_unix()`) in unix seconds.
        country (list of str): The country for which data needs to be downloaded.
        cumul (bool): If True, calculate the cumulative sum of production.
        rolling (bool): If True, calculate the rolling average of production.
        timeout (int): The request timeout in seconds.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: A tuple containing:
            - A dataframe of production.
            - A dataframe of load.
            - An array of all available technologies.
    """
    s = requests.Session()
    url = f"https://api.energy-charts.info/public_power?country={country}&start={start}&end={end}"
    try:
        r = s.get(url, timeout=timeout)
    finally:
        s.close()
    r.raise_for_status()
    response = json.loads(r.text)
    techs = []
    prod = []
    for r in response["production_types"]:
        try:
            techs.append(r["name"])
            prod.append(r["data"])
        except TypeError:
            print("Somethings wrong")
    ticks = [
        pd.to_datetime(d, unit="s", origin="unix") for d in response["unix_seconds"]
    ]
    load_dict = {"en": "Load"}
    print(url)
    prod_df = pd.DataFrame(data=prod, index=techs, columns=ticks).T
    col_exclude = ["Residual load", "Renewable Share"]
    for col in col_exclude:
        if col in prod_df.columns:
            prod_df.drop(col, axis=1, inplace=True)
    if rolling:
        prod_df = prod_df.rolling(rolling).sum()
    if cumul:
        prod_df = prod_df.cumsum()
    try:
        load_df = prod_df[load_dict["en"]]
        prod_df.drop(load_dict["en"], axis=1, inplace=True)
    except Exception as e:
        print(e)
        load_df = None
    print(f"...production for {country} OK.")
    return prod_df, load_df, techs


def get_trade(start, end, country, timeout=API_REQUEST_TIMEOUT):
    """
    Downloads trade data from the Energy Charts API. Gets called from `get_data()`.
    Uses the "cbpf" endpoint of the API.

    Args:
        start (int): Start of the download period (output of `year_to_unix()`) in unix seconds.
        end (int): End of the download period (output of `year_to_unix()`) in unix seconds.
        country (list of str): The country for which data needs to be downloaded.
        timeout (int): The request timeout in seconds.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: A tuple containing:
            - A dataframe of trades between countries.
            - An array of all available regions.
    """
    s = requests.Session()
    url = (
        f"https://api.energy-charts.info/cbpf?country={country}&start={start}&end={end}"
    )
    try:
        r = s.get(url, timeout=timeout)
    finally:
        s.close()
    r.raise_for_status()
    response = json.loads(r.text)
    ticks = [
        pd.to_datetime(d, unit="s", origin="unix") for d in response["unix_seconds"]
    ]
    trade = []
    regions = []
    for r in response["countries"]:
        trade.append(r["data"])
        regions.append(r["name"])
    print(url)
    trade_df = pd.DataFrame(data=trade, index=regions, columns=ticks).T
    print("...trade for " + country + " OK.")
    return trade_df, regions


def get_data(
    year,
    path_to_data=None,
    max_retries=4,
    retry_delay=15,
    request_timeout=API_REQUEST_TIMEOUT,
    request_interval=2,
    required_countries=None,
):
    """
    Main function for downloading data.

    Args:
        year (int): The selected year for which data is to be downloaded, e.g., 2023.
        path_to_data (str or Path): location of the data.
        max_retries (int): The maximum number of retries for each country download in case of problems.
        retry_delay (int): The delay in seconds between retries.
        request_timeout (int): The request timeout in seconds.
        request_interval (float): Delay between successful endpoint requests.
        required_countries: Optional country codes which must be available in
            the completed download.

    Returns:
        pd.DataFrame: A dataframe containing both production and trade data for all countries in the selected year.
    """
    if path_to_data is None:
        data_dir = files("shrecc.data")
    else:
        data_dir = Path(path_to_data)
    filename = data_dir / f"{year}" / f"prod_and_trade_data_{year}.pkl"
    status_filename = filename.with_suffix(".download.json")
    filename.parent.mkdir(parents=True, exist_ok=True)
    start, end = year_to_unix(year)
    required_countries = {
        str(country).upper() for country in (required_countries or [])
    }

    cache_exists = filename.exists()
    if cache_exists:
        data = load_from_pickle(filename)
        if not isinstance(data, dict):
            raise TypeError(f"Energy Charts cache at {filename} is not a dictionary")
    else:
        data = {}

    if cache_exists and not status_filename.is_file() and not required_countries:
        print("Legacy API data loaded successfully.")
        return cleaning_data(data, files("shrecc.data"))

    status = _load_energy_charts_download_status(status_filename)
    for country in data:
        status[str(country).upper()] = "success"

    unresolved_countries = [
        country
        for country in ENERGY_CHARTS_COUNTRIES
        if status.get(country) not in {"success", "unavailable"}
    ]
    failures = {}
    if unresolved_countries:
        if data:
            print(
                "Resuming Energy Charts download for "
                f"{len(unresolved_countries)} unresolved countries."
            )
        for country in (pbar := tqdm(unresolved_countries)):
            pbar.set_description(f"Fetching data for country {country}")
            country_api = country.lower()
            try:
                prod_df, load_df, _ = _call_energy_charts_with_retries(
                    get_prod,
                    country=country_api,
                    endpoint="production",
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    start=start,
                    end=end,
                    cumul=False,
                    rolling=1,
                    timeout=request_timeout,
                )
                time.sleep(request_interval)
                trade_df, _ = _call_energy_charts_with_retries(
                    get_trade,
                    country=country_api,
                    endpoint="trade",
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    start=start,
                    end=end,
                    timeout=request_timeout,
                )
            except _CountryUnavailable as exc:
                status[country] = "unavailable"
                print(f"\tCountry {country} unavailable: {exc}")
                continue
            except EnergyChartsDownloadError as exc:
                failures[country] = str(exc)
                print(f"\t{exc}")
                continue

            data[country_api] = {
                "production mix": prod_df,
                "load": load_df,
                "trade": trade_df,
            }
            status[country] = "success"
            time.sleep(request_interval)

        save_to_pickle(data, filename)
        _save_energy_charts_download_status(status, status_filename, year)

    if failures:
        failed = ", ".join(sorted(failures))
        raise EnergyChartsDownloadError(
            f"Energy Charts download for {year} is incomplete after transient "
            f"API failures for: {failed}. Successful countries were saved; "
            "rerun the same request to resume only the unresolved countries."
        )

    missing_required = required_countries.difference(
        str(country).upper() for country in data
    )
    if missing_required:
        raise ValueError(
            f"Energy Charts has no complete {year} data for required countries: "
            + ", ".join(sorted(missing_required))
        )

    print("API data loaded successfully.")
    data_df = cleaning_data(data, files("shrecc.data"))
    return data_df


def energy_charts_cached_countries(year, path_to_data=None):
    """Return countries with complete raw API records, or ``None`` without a cache."""
    if path_to_data is None:
        data_dir = files("shrecc.data")
    else:
        data_dir = Path(path_to_data)
    filename = data_dir / f"{year}" / f"prod_and_trade_data_{year}.pkl"
    if not filename.is_file():
        return None

    data = load_from_pickle(filename)
    if not isinstance(data, dict):
        raise TypeError(f"Energy Charts cache at {filename} is not a dictionary")
    return frozenset(str(country).upper() for country in data)


def _call_energy_charts_with_retries(
    function,
    *,
    country,
    endpoint,
    max_retries,
    retry_delay,
    **kwargs,
):
    """Call one API endpoint with rate-limit-aware exponential backoff."""
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    for attempt in range(max_retries):
        try:
            return function(country=country, **kwargs)
        except requests.HTTPError as exc:
            response = exc.response
            status_code = response.status_code if response is not None else None
            if status_code in {400, 404}:
                raise _CountryUnavailable(
                    f"{endpoint} endpoint returned HTTP {status_code}"
                ) from exc
            if status_code not in TRANSIENT_HTTP_STATUS_CODES:
                raise EnergyChartsDownloadError(
                    f"HTTP {status_code} from the {endpoint} endpoint for "
                    f"{country.upper()}"
                ) from exc
            error = exc
            response_for_delay = response
        except requests.Timeout as exc:
            raise EnergyChartsDownloadError(
                f"Timeout from the {endpoint} endpoint for {country.upper()}"
            ) from exc
        except (requests.ConnectionError, requests.RequestException) as exc:
            error = exc
            response_for_delay = getattr(exc, "response", None)
        except Exception as exc:
            error = exc
            response_for_delay = None

        if attempt == max_retries - 1:
            break
        delay = _energy_charts_retry_delay(
            response_for_delay,
            retry_delay,
            attempt,
        )
        print(
            f"\t{endpoint.capitalize()} request for {country.upper()} failed "
            f"({error}); retrying in {delay:g}s "
            f"[{attempt + 2}/{max_retries}]"
        )
        time.sleep(delay)

    raise EnergyChartsDownloadError(
        f"{endpoint.capitalize()} request failed for {country.upper()} after "
        f"{max_retries} attempts: {error}"
    ) from error


def _energy_charts_retry_delay(response, retry_delay, attempt):
    """Return Retry-After seconds or exponential fallback delay."""
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0)
            except ValueError:
                pass
    return max(float(retry_delay), 0) * (2**attempt)


def _load_energy_charts_download_status(filename):
    if not filename.is_file():
        return {}
    with filename.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    return dict(payload.get("countries", {}))


def _save_energy_charts_download_status(status, filename, year):
    payload = {
        "format": "shrecc-energy-charts-download",
        "version": 1,
        "year": int(year),
        "countries": dict(sorted(status.items())),
    }
    with filename.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)


def year_to_unix(year):
    """
    Converts a year to Unix timestamps representing the start and end of the year in UTC.

    Args:
        year (int): The selected year, passed from `get_data()`.

    Returns:
        Tuple[int, int]: A tuple containing:
            - The start of the year in Unix seconds (UTC).
            - The end of the year in Unix seconds (UTC).
    """
    start_of_year = datetime(year, 1, 1, 0, 0, tzinfo=ZoneInfo("UTC"))
    end_of_year = datetime(
        year, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("UTC")
    )  # include full last second
    start_unix = int(start_of_year.timestamp())
    end_unix = int(end_of_year.timestamp())
    return start_unix, end_unix


def cleaning_data(data, data_dir):
    """
    Cleans the data and adds missing countries. Note that missing countries need to be manually added to `country_codes`.
    Gets called from `get_data()`.

    Args:
        data (pd.DataFrame): The dataframe containing production and trade data.
        root (Path): location of the data.

    Returns:
        pd.DataFrame: A dataframe with missing countries added.
    """
    techs = []
    partners = []
    for country, datasets in data.items():
        try:
            techs.extend(datasets["production mix"].columns)
            partners.extend(datasets["trade"].columns)
        except:  # noqa E722
            pass
    filename = data_dir / "generation_units_by_country.csv"
    if filename.exists():
        gen_units_per_country = pd.read_csv(filename, index_col=1)["short"]
    country_codes = {
        p: gen_units_per_country.loc[p]
        for p in set(partners)
        if p in gen_units_per_country.index
    }
    country_codes = {
        **country_codes,
        **{
            "Armenia": "AM",
            "Azerbaijan": "AZ",
            "Cyprus": "CY",  # Does not appear?
            "Ireland": "IE",
            "Malta": "MT",
            "North Macedonia": "MK",
            "Serbia": "RS",
            "Slovakia": "SK",
        },
    }
    data_clean = {}
    filename = data_dir / "techs_agg.json"
    if filename.exists():
        with open(filename, "r") as f:
            techs_agg = json.load(f)
    agg_dict = {
        "production mix": techs_agg,
        "trade": country_codes,
        "load": {"Load": "load"},
    }
    scale_dict = {"production mix": 1, "trade": 1000, "load": 1}

    for country in data.keys():
        print(f"Processing country: {country}")
        data_clean[country.upper()] = {}
        for k, v in data[country].items():
            if type(v) is pd.DataFrame:  # axis = 1 will soon be depreciated
                grouped = v.T.groupby(agg_dict[k]).sum().T
                grouped.index = pd.to_datetime(grouped.index)
                data_clean[country.upper()][k] = (
                    grouped.resample("h").mean() * scale_dict[k]
                )

            elif type(v) is pd.Series:
                v.index = pd.to_datetime(v.index)
                data_clean[country.upper()][k] = v.resample("h").mean() * scale_dict[k]

    data_clean = {k: v for k, v in data_clean.items() if v != {}}
    P = pd.concat(
        [
            pd.concat(
                {country: pd.concat(data_clean[country], axis=1)},
                axis=1,
                names=["country", "type", "source"],
            )
            for country in data_clean.keys()
        ],
        axis=1,
    )
    return P


def get_package_user_data_dir(package_name="shrecc"):
    """
    Get the user data dir through appdirs.
    If it doesn't exist, it will create it.

    Args:
        package_name (str): the name of the package
    Returns
        Path : the existing or newly created directory.
    """
    destination_directory = appdirs.user_data_dir(package_name)
    os.makedirs(destination_directory, exist_ok=True)
    return destination_directory
