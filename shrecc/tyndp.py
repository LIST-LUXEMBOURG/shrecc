"""TYNDP electricity data ingestion and hourly consumption-mix calculation.

This module contains an xarray-oriented variant of the treatment workflow.  It
starts from the raw hourly TYNDP production and cross-border exchange data,
builds a gross supply table, and returns, for each consuming country and hour,
the technology and country of origin of the electricity consumed. The TYNDP
data can be provided either as the original Excel/XLSB workbook or as pickled
DataFrames extracted from that workbook.

The core idea is to avoid building the full square technology-country network.
Only the country-to-country trade block is inverted.  Domestic production shares
are then projected through the resolved trade network to attribute each unit of
country-level consumption back to producing technologies and countries.
"""

import zipfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import requests
import xarray as xr

from shrecc.solver import solve_consumption_system
from shrecc.result_store import (
    load_pickle as _load_pickle,
    save_pickle as _save_pickle,
    write_solved_consumption_result_cache,
)
from shrecc._legacy_tyndp import _legacy_consumption_mix_from_z_gross

TYNDP_SCENARIO_URL_ROOT = (
    "https://2024-data.entsos-tyndp-scenarios.eu/files/scenarios-outputs"
)
TYNDP_SCENARIOS = {
    "DE": "Distributed Energy",
    "GA": "Global Ambition",
    "NT": "National Trends",
}
TYNDP_SCENARIO_YEARS = {
    "DE": (2035, 2040, 2050),
    "GA": (2035, 2040, 2050),
    "NT": (2030, 2040),
}
TYNDP_CLIMATE_YEARS = (1995, 2008, 2009)
TYNDP_DEMAND_CATEGORY = "Demand [MW] (losses included)"


def build_z_gross_from_tyndp_pickles(
    production_pickle,
    trade_pickle,
    technology_mapping,
    country_mapping,
    model_year=2050,
    technology_sheet="concordance",
    countries_sheet="countries",
    connections_sheet="connections",
    verbose=False,
):
    """Build the gross hourly supply table from original TYNDP pickle files.

    Args:
        production_pickle: Pickle file containing the hourly TYNDP production
            table, read from the ``"Hourly Market Data emarket"`` sheet in the
            original workbook.
        trade_pickle: Pickle file containing the hourly TYNDP cross-border
            exchange table, read from the ``"Crossborder exchanges"`` sheet in
            the original workbook.
        technology_mapping: SHRECC/FIONA Excel workbook containing the
            technology concordance. The function uses this file to keep only
            production categories that are mapped in the concordance. It does
            not aggregate to premise activities here; the resulting production
            technologies remain the original TYNDP/ENTSO-E categories.
        country_mapping: SHRECC/FIONA Excel workbook containing country and
            connection mappings. Expected sheets are ``countries_sheet`` and
            ``connections_sheet``.
        model_year: Year appended to the TYNDP day/month/hour labels when
            constructing the hourly datetime index.
        technology_sheet: Sheet name in the SHRECC/FIONA
            ``technology_mapping`` workbook with technology/category mappings.
        countries_sheet: Sheet name in the SHRECC/FIONA ``country_mapping``
            workbook mapping TYNDP node labels to country codes. The sheet is
            expected to have a ``"Country code"`` column.
        connections_sheet: Sheet name in the SHRECC/FIONA ``country_mapping``
            workbook mapping cross-border line labels to ``"Country from"`` and
            ``"Country to"``.
        verbose: If True, print progress messages for the main parsing and
            aggregation steps.

    Returns:
        Gross hourly system table with rows indexed by time and columns indexed
        by ``("type", "country from", "country to", "source")``. The ``type``
        level contains ``"production mix"`` for domestic production by
        technology, ``"trade"`` for positive directional exchanges, and
        ``"consumption"`` for measured country demand.

    Notes:
        This function is the reusable version of the parsing steps in
        ``notebooks/parsing_tyndp.ipynb``. Production nodes such as ``XX00`` or
        ``XX00RETE`` are mapped to country codes and aggregated; exchange
        columns are mapped to country pairs; negative exchanges are reversed so
        all trade flows are positive and directional; and production categories
        absent from the technology concordance are dropped.
    """
    _log("Loading TYNDP pickle files", verbose)
    production = _load_pickle(production_pickle).copy()
    trade = _load_pickle(trade_pickle).copy()

    return _build_z_gross_from_tyndp_tables(
        production=production,
        trade=trade,
        technology_mapping=technology_mapping,
        country_mapping=country_mapping,
        model_year=model_year,
        technology_sheet=technology_sheet,
        countries_sheet=countries_sheet,
        connections_sheet=connections_sheet,
        verbose=verbose,
    )


def build_z_gross_from_tyndp_excel(
    excel_file,
    technology_mapping,
    country_mapping,
    model_year=2050,
    production_sheet="Hourly Market Data emarket",
    trade_sheet="Crossborder exchanges",
    technology_sheet="concordance",
    countries_sheet="countries",
    connections_sheet="connections",
    engine="calamine",
    production_pickle=None,
    trade_pickle=None,
    verbose=False,
):
    """Build the gross hourly supply table directly from the TYNDP workbook.

    Args:
        excel_file: Original TYNDP Excel/XLSB workbook containing the hourly
            production and cross-border exchange sheets.
        technology_mapping: SHRECC/FIONA Excel workbook containing the
            technology concordance.
        country_mapping: SHRECC/FIONA Excel workbook containing country and
            connection mappings.
        model_year: Year appended to the TYNDP day/month/hour labels when
            constructing the hourly datetime index.
        production_sheet: Sheet in the original TYNDP workbook containing
            hourly production data.
        trade_sheet: Sheet in the original TYNDP workbook containing hourly
            cross-border exchanges.
        technology_sheet: Sheet name in the SHRECC/FIONA technology mapping
            workbook.
        countries_sheet: Sheet name in the SHRECC/FIONA country mapping
            workbook for TYNDP node-to-country mappings.
        connections_sheet: Sheet name in the SHRECC/FIONA country mapping
            workbook for cross-border connection mappings.
        engine: Excel engine passed to :func:`pandas.read_excel`. The original
            TYNDP workbook used in the notebook is ``.xlsb``, so ``"calamine"``
            is the default. Use ``None`` to let pandas infer the engine for
            ordinary ``.xlsx`` files.
        production_pickle: Optional path where the raw extracted production
            sheet should be cached.
        trade_pickle: Optional path where the raw extracted trade sheet should
            be cached.
        verbose: If True, print progress messages for reading, parsing, and
            aggregation.

    Returns:
        Gross hourly supply table with the same structure as
        :func:`build_z_gross_from_tyndp_pickles`.
    """
    production, trade = read_tyndp_excel_tables(
        excel_file,
        production_sheet=production_sheet,
        trade_sheet=trade_sheet,
        engine=engine,
        production_pickle=production_pickle,
        trade_pickle=trade_pickle,
        verbose=verbose,
    )

    return _build_z_gross_from_tyndp_tables(
        production=production,
        trade=trade,
        technology_mapping=technology_mapping,
        country_mapping=country_mapping,
        model_year=model_year,
        technology_sheet=technology_sheet,
        countries_sheet=countries_sheet,
        connections_sheet=connections_sheet,
        verbose=verbose,
    )


def read_tyndp_excel_tables(
    excel_file,
    production_sheet="Hourly Market Data emarket",
    trade_sheet="Crossborder exchanges",
    engine="calamine",
    production_pickle=None,
    trade_pickle=None,
    verbose=False,
):
    """Read raw TYNDP production and trade sheets from the workbook.

    Args:
        excel_file: Original TYNDP Excel/XLSB workbook.
        production_sheet: Sheet in the original TYNDP workbook containing
            hourly production data.
        trade_sheet: Sheet in the original TYNDP workbook containing hourly
            cross-border exchanges.
        engine: Excel engine passed to :func:`pandas.read_excel`. Use ``None``
            to let pandas infer the engine.
        production_pickle: Optional path where the raw extracted production
            sheet should be cached.
        trade_pickle: Optional path where the raw extracted trade sheet should
            be cached.
        verbose: If True, print progress messages.

    Returns:
        A tuple containing the raw production and trade DataFrames as read from
        the workbook.

    Raises:
        ValueError: If only one of ``production_pickle`` and ``trade_pickle`` is
            provided.
    """
    _log("Reading hourly production sheet from TYNDP workbook", verbose)
    production = _read_excel_dataframe(
        excel_file,
        sheet_name=production_sheet,
        header=[10, 11, 12],
        index_col=[0, 1],
        engine=engine,
    )

    _log("Reading cross-border exchange sheet from TYNDP workbook", verbose)
    trade = _read_excel_dataframe(
        excel_file,
        sheet_name=trade_sheet,
        header=[10],
        index_col=[0, 1],
        engine=engine,
    )

    if production_pickle is not None or trade_pickle is not None:
        if production_pickle is None or trade_pickle is None:
            raise ValueError(
                "production_pickle and trade_pickle must be provided together"
            )

        _log("Saving extracted TYNDP sheets as pickle files", verbose)
        _save_pickle(production, production_pickle)
        _save_pickle(trade, trade_pickle)

    return production, trade


def _build_z_gross_from_tyndp_tables(
    production,
    trade,
    technology_mapping,
    country_mapping,
    model_year,
    technology_sheet,
    countries_sheet,
    connections_sheet,
    verbose,
):
    """Build ``Z_gross`` from already-loaded TYNDP production and trade tables."""
    _log("Parsing hourly datetime index", verbose)
    production.index = _parse_tyndp_datetime_index(production.index, model_year)
    trade.index = _parse_tyndp_datetime_index(trade.index, model_year)

    production = production.dropna(axis=1, how="all")
    trade = trade.dropna(axis=1, how="all")

    _log("Loading technology and country mappings", verbose)
    concordance = load_technology_concordance(technology_mapping, technology_sheet)
    country_mapping_df, connections = load_tyndp_country_mapping(
        country_mapping,
        countries_sheet=countries_sheet,
        connections_sheet=connections_sheet,
    )
    country_codes = country_mapping_df["Country code"]

    _log("Aggregating hourly production by country and technology", verbose)
    production_agg = _aggregate_tyndp_production(
        production,
        concordance.columns,
        country_codes,
    )
    consumption_agg = _aggregate_tyndp_consumption(
        production,
        country_codes,
    )

    connections = connections[~connections.index.duplicated(keep="first")]

    _log("Aggregating hourly trade to positive directional country flows", verbose)
    trade_directional = _aggregate_tyndp_trade(trade, connections)

    production_agg.columns.names = trade_directional.columns.names = [
        "Country",
        "Source",
    ]

    production_columns = pd.MultiIndex.from_arrays(
        [
            production_agg.columns.get_level_values("Country"),
            production_agg.columns.get_level_values("Country"),
            production_agg.columns.get_level_values("Source"),
        ],
        names=["country from", "country to", "source"],
    )
    production_agg.columns = production_columns

    consumption_columns = pd.MultiIndex.from_arrays(
        [
            consumption_agg.columns,
            consumption_agg.columns,
            [TYNDP_DEMAND_CATEGORY] * len(consumption_agg.columns),
        ],
        names=["country from", "country to", "source"],
    )
    consumption_agg.columns = consumption_columns

    trade_gross = pd.concat(
        [trade_directional],
        keys=["trade"],
        names=["source"],
        axis=1,
    )
    trade_gross.columns.names = ["source", "country to", "country from"]
    trade_gross.columns = trade_gross.columns.swaplevel(0, 2)

    _log("Combining production and trade into Z_gross", verbose)
    Z_gross = pd.concat(
        [production_agg, trade_gross, consumption_agg],
        axis=1,
        keys=["production mix", "trade", "consumption"],
        names=["type"],
    )
    _log(f"Built Z_gross with shape {Z_gross.shape}", verbose)

    return Z_gross


def build_z_gross_from_tyndp_scenario(
    scenario,
    year,
    climate_year,
    data_dir,
    technology_mapping,
    country_mapping,
    *,
    download=True,
    url_root=TYNDP_SCENARIO_URL_ROOT,
    keep_zip=True,
    session=None,
    model_year=None,
    production_sheet="Hourly Market Data emarket",
    trade_sheet="Crossborder exchanges",
    technology_sheet="concordance",
    countries_sheet="countries",
    connections_sheet="connections",
    engine="calamine",
    verbose=False,
):
    """Build ``Z_gross`` for a TYNDP scenario, using cached files when possible.

    If both pickle files already exist, they are loaded directly. Otherwise, an
    existing workbook is parsed and the pickle files are created. If neither the
    pickle files nor the workbook exist and ``download`` is true, the matching
    zip file is downloaded and extracted first.

    ``production_sheet`` and ``trade_sheet`` refer to sheets in the downloaded
    TYNDP workbook. ``technology_sheet``, ``countries_sheet``, and
    ``connections_sheet`` refer to sheets in the SHRECC/FIONA mapping workbooks.

    Args:
        scenario: TYNDP scenario code, such as ``"DE"``, ``"GA"``, or ``"NT"``.
        year: TYNDP scenario year.
        climate_year: TYNDP climate year.
        data_dir: Local cache directory for zip files, extracted workbooks, and
            generated pickle files.
        technology_mapping: SHRECC/FIONA Excel workbook containing the
            technology concordance.
        country_mapping: SHRECC/FIONA Excel workbook containing country and
            connection mappings.
        download: If True, download the official scenario zip when neither the
            pickles nor workbook exist locally.
        url_root: Base URL for official TYNDP scenario zip files.
        keep_zip: If True, keep the downloaded zip after extracting the
            workbook.
        session: Optional requests-compatible session used for downloads.
        model_year: Year used to construct hourly timestamps. Defaults to
            ``year``.
        production_sheet: Sheet in the TYNDP workbook containing hourly
            production data.
        trade_sheet: Sheet in the TYNDP workbook containing cross-border
            exchanges.
        technology_sheet: Sheet in the SHRECC/FIONA technology mapping
            workbook.
        countries_sheet: Sheet in the SHRECC/FIONA country mapping workbook
            for TYNDP node-to-country mappings.
        connections_sheet: Sheet in the SHRECC/FIONA country mapping workbook
            for cross-border connection mappings.
        engine: Excel engine passed to :func:`pandas.read_excel`.
        verbose: If True, print progress messages.

    Returns:
        Gross hourly supply table with the same structure as
        :func:`build_z_gross_from_tyndp_pickles`.
    """
    paths = tyndp_scenario_paths(
        data_dir=data_dir,
        scenario=scenario,
        year=year,
        climate_year=climate_year,
        url_root=url_root,
    )
    model_year = year if model_year is None else model_year

    if paths["production_pickle"].exists() and paths["trade_pickle"].exists():
        return build_z_gross_from_tyndp_pickles(
            production_pickle=paths["production_pickle"],
            trade_pickle=paths["trade_pickle"],
            technology_mapping=technology_mapping,
            country_mapping=country_mapping,
            model_year=model_year,
            technology_sheet=technology_sheet,
            countries_sheet=countries_sheet,
            connections_sheet=connections_sheet,
            verbose=verbose,
        )

    if not paths["workbook"].exists():
        ensure_tyndp_workbook(
            scenario=scenario,
            year=year,
            climate_year=climate_year,
            data_dir=data_dir,
            download=download,
            url_root=url_root,
            keep_zip=keep_zip,
            session=session,
            verbose=verbose,
        )

    return build_z_gross_from_tyndp_excel(
        excel_file=paths["workbook"],
        technology_mapping=technology_mapping,
        country_mapping=country_mapping,
        model_year=model_year,
        production_sheet=production_sheet,
        trade_sheet=trade_sheet,
        technology_sheet=technology_sheet,
        countries_sheet=countries_sheet,
        connections_sheet=connections_sheet,
        engine=engine,
        production_pickle=paths["production_pickle"],
        trade_pickle=paths["trade_pickle"],
        verbose=verbose,
    )


def consumption_mix_from_tyndp_pickles(
    production_pickle,
    trade_pickle,
    technology_mapping,
    country_mapping,
    model_year=2050,
    check=True,
    return_debug=False,
    zero_consumption="raise",
    verbose=False,
):
    """Run the full TYNDP pickle-to-consumption-mix pipeline.

    This is a convenience wrapper around
    :func:`build_z_gross_from_tyndp_pickles` and
    :func:`consumption_mix_from_z_gross`.

    Args:
        production_pickle: Pickle file containing the hourly TYNDP production
            table.
        trade_pickle: Pickle file containing the hourly TYNDP cross-border
            exchange table.
        technology_mapping: SHRECC/FIONA Excel workbook containing the
            technology concordance.
        country_mapping: SHRECC/FIONA Excel workbook containing country and
            connection mappings.
        model_year: Year used to construct the hourly datetime index from TYNDP
            labels.
        check: Forwarded to :func:`consumption_mix_from_z_gross`.
        return_debug: If True, return ``(consumption_mix_xr, debug)``. The
            debug dictionary includes the constructed ``Z_gross`` table in
            addition to the coefficient matrices returned by
            :func:`consumption_mix_from_z_gross`.
        zero_consumption: Forwarded to :func:`consumption_mix_from_z_gross`.
        verbose: If True, print progress messages for the gross-table
            construction and consumption-mix solve.

    Returns:
        Hourly consumption mix by consuming country, producing country, and
        technology. If ``return_debug`` is True, also returns a debug
        dictionary.
    """
    Z_gross = build_z_gross_from_tyndp_pickles(
        production_pickle=production_pickle,
        trade_pickle=trade_pickle,
        technology_mapping=technology_mapping,
        country_mapping=country_mapping,
        model_year=model_year,
        verbose=verbose,
    )

    if not return_debug:
        return consumption_mix_from_z_gross(
            Z_gross,
            check=check,
            return_debug=False,
            zero_consumption=zero_consumption,
            verbose=verbose,
        )

    consumption_mix_xr, debug = cast(
        tuple[xr.DataArray, dict[str, object]],
        consumption_mix_from_z_gross(
            Z_gross,
            check=check,
            return_debug=True,
            zero_consumption=zero_consumption,
            verbose=verbose,
        ),
    )
    debug["Z_gross"] = Z_gross

    return consumption_mix_xr, debug


def consumption_mix_from_tyndp_scenario(
    scenario,
    year,
    climate_year,
    data_dir,
    technology_mapping,
    country_mapping,
    *,
    download=True,
    url_root=TYNDP_SCENARIO_URL_ROOT,
    keep_zip=True,
    session=None,
    model_year=None,
    production_sheet="Hourly Market Data emarket",
    trade_sheet="Crossborder exchanges",
    technology_sheet="concordance",
    countries_sheet="countries",
    connections_sheet="connections",
    check=True,
    return_debug=False,
    zero_consumption="raise",
    engine="calamine",
    verbose=False,
):
    """Run the full TYNDP scenario-to-consumption-mix pipeline.

    ``production_sheet`` and ``trade_sheet`` refer to sheets in the downloaded
    TYNDP workbook. ``technology_sheet``, ``countries_sheet``, and
    ``connections_sheet`` refer to sheets in the SHRECC/FIONA mapping workbooks.

    Args:
        scenario: TYNDP scenario code, such as ``"DE"``, ``"GA"``, or ``"NT"``.
        year: TYNDP scenario year.
        climate_year: TYNDP climate year.
        data_dir: Local cache directory for zip files, extracted workbooks, and
            generated pickle files.
        technology_mapping: SHRECC/FIONA Excel workbook containing the
            technology concordance.
        country_mapping: SHRECC/FIONA Excel workbook containing country and
            connection mappings.
        download: If True, download the official scenario zip when neither the
            pickles nor workbook exist locally.
        url_root: Base URL for official TYNDP scenario zip files.
        keep_zip: If True, keep the downloaded zip after extracting the
            workbook.
        session: Optional requests-compatible session used for downloads.
        model_year: Year used to construct hourly timestamps. Defaults to
            ``year``.
        production_sheet: Sheet in the TYNDP workbook containing hourly
            production data.
        trade_sheet: Sheet in the TYNDP workbook containing cross-border
            exchanges.
        technology_sheet: Sheet in the SHRECC/FIONA technology mapping
            workbook.
        countries_sheet: Sheet in the SHRECC/FIONA country mapping workbook
            for TYNDP node-to-country mappings.
        connections_sheet: Sheet in the SHRECC/FIONA country mapping workbook
            for cross-border connection mappings.
        check: Forwarded to :func:`consumption_mix_from_z_gross`.
        return_debug: If True, return ``(consumption_mix_xr, debug)``.
        zero_consumption: Forwarded to :func:`consumption_mix_from_z_gross`.
        engine: Excel engine passed to :func:`pandas.read_excel`.
        verbose: If True, print progress messages.

    Returns:
        Hourly consumption mix by consuming country, producing country, and
        technology. If ``return_debug`` is True, also returns a debug
        dictionary.
    """
    Z_gross = build_z_gross_from_tyndp_scenario(
        scenario=scenario,
        year=year,
        climate_year=climate_year,
        data_dir=data_dir,
        technology_mapping=technology_mapping,
        country_mapping=country_mapping,
        download=download,
        url_root=url_root,
        keep_zip=keep_zip,
        session=session,
        model_year=model_year,
        production_sheet=production_sheet,
        trade_sheet=trade_sheet,
        technology_sheet=technology_sheet,
        countries_sheet=countries_sheet,
        connections_sheet=connections_sheet,
        engine=engine,
        verbose=verbose,
    )

    if not return_debug:
        return consumption_mix_from_z_gross(
            Z_gross,
            check=check,
            return_debug=False,
            zero_consumption=zero_consumption,
            verbose=verbose,
        )

    consumption_mix_xr, debug = cast(
        tuple[xr.DataArray, dict[str, object]],
        consumption_mix_from_z_gross(
            Z_gross,
            check=check,
            return_debug=True,
            zero_consumption=zero_consumption,
            verbose=verbose,
        ),
    )
    debug["Z_gross"] = Z_gross

    return consumption_mix_xr, debug


def consumption_mix_from_tyndp_excel(
    excel_file,
    technology_mapping,
    country_mapping,
    model_year=2050,
    production_sheet="Hourly Market Data emarket",
    trade_sheet="Crossborder exchanges",
    technology_sheet="concordance",
    countries_sheet="countries",
    connections_sheet="connections",
    check=True,
    return_debug=False,
    zero_consumption="raise",
    engine="calamine",
    verbose=False,
):
    """Run the full TYNDP workbook-to-consumption-mix pipeline.

    This is the Excel/XLSB equivalent of
    :func:`consumption_mix_from_tyndp_pickles`. It reads the production and
    cross-border exchange sheets directly from the original TYNDP workbook,
    builds ``Z_gross``, and calculates the hourly consumption mix.

    Args:
        excel_file: Original TYNDP Excel/XLSB workbook.
        technology_mapping: SHRECC/FIONA Excel workbook containing the
            technology concordance.
        country_mapping: SHRECC/FIONA Excel workbook containing country and
            connection mappings.
        model_year: Year used to construct the hourly datetime index from TYNDP
            labels.
        production_sheet: Sheet in the original TYNDP workbook containing
            hourly production data.
        trade_sheet: Sheet in the original TYNDP workbook containing hourly
            cross-border exchanges.
        technology_sheet: Sheet in the SHRECC/FIONA technology mapping
            workbook.
        countries_sheet: Sheet in the SHRECC/FIONA country mapping workbook
            for TYNDP node-to-country mappings.
        connections_sheet: Sheet in the SHRECC/FIONA country mapping workbook
            for cross-border connection mappings.
        check: Forwarded to :func:`consumption_mix_from_z_gross`.
        return_debug: If True, return ``(consumption_mix_xr, debug)``. The
            debug dictionary includes the constructed ``Z_gross`` table.
        zero_consumption: Forwarded to :func:`consumption_mix_from_z_gross`.
        engine: Excel engine passed to :func:`pandas.read_excel`. Use ``None``
            to let pandas infer the engine.
        verbose: If True, print progress messages for reading, parsing, and
            solving.

    Returns:
        Hourly consumption mix by consuming country, producing country, and
        technology. If ``return_debug`` is True, also returns a debug
        dictionary.
    """
    Z_gross = build_z_gross_from_tyndp_excel(
        excel_file=excel_file,
        technology_mapping=technology_mapping,
        country_mapping=country_mapping,
        model_year=model_year,
        production_sheet=production_sheet,
        trade_sheet=trade_sheet,
        technology_sheet=technology_sheet,
        countries_sheet=countries_sheet,
        connections_sheet=connections_sheet,
        engine=engine,
        verbose=verbose,
    )

    if not return_debug:
        return consumption_mix_from_z_gross(
            Z_gross,
            check=check,
            return_debug=False,
            zero_consumption=zero_consumption,
            verbose=verbose,
        )

    consumption_mix_xr, debug = cast(
        tuple[xr.DataArray, dict[str, object]],
        consumption_mix_from_z_gross(
            Z_gross,
            check=check,
            return_debug=True,
            zero_consumption=zero_consumption,
            verbose=verbose,
        ),
    )
    debug["Z_gross"] = Z_gross

    return consumption_mix_xr, debug


def tyndp_scenario_paths(
    data_dir,
    scenario,
    year,
    climate_year,
    *,
    url_root=TYNDP_SCENARIO_URL_ROOT,
):
    """Return cache paths and download URL for a TYNDP scenario combination.

    Args:
        data_dir: Local cache directory for zip files, extracted workbooks, and
            generated pickle files.
        scenario: TYNDP scenario code.
        year: TYNDP scenario year.
        climate_year: TYNDP climate year.
        url_root: Base URL for official TYNDP scenario zip files.

    Returns:
        Dictionary with ``production_pickle``, ``trade_pickle``, ``workbook``,
        ``zip``, and ``url`` entries.

    Raises:
        ValueError: If the scenario tuple is not supported.
    """
    scenario, year, climate_year = validate_tyndp_scenario(
        scenario,
        year,
        climate_year,
    )
    data_dir = Path(data_dir)
    stem = f"{scenario}{year}_CY{climate_year}"

    if scenario == "NT":
        zip_name = f"NT{year}CY{climate_year}.zip"
        url = (
            f"{url_root.rstrip('/')}/MMStandardOutputFile_NT{year}_Plexos"
            f"_CY{climate_year}_2.5_v40.xlsx.zip"
        )
        workbook = (
            data_dir
            / f"MMStandardOutputFile_NT{year}_Plexos_CY{climate_year}_2.5_v40.xlsx"
        )
    else:
        zip_name = f"{scenario}{year}CY{climate_year}.zip"
        url = f"{url_root.rstrip('/')}/{zip_name}"
        workbook = (
            data_dir
            / f"MMStandardOutputFile_{scenario}{year}_Plexos_CY{climate_year}_v11_SoS.xlsb"
        )

    return {
        "production_pickle": data_dir / f"{stem}_prod.pkl",
        "trade_pickle": data_dir / f"{stem}_trade.pkl",
        "workbook": workbook,
        "zip": data_dir / zip_name,
        "url": url,
    }


def validate_tyndp_scenario(scenario, year, climate_year):
    """Validate and normalize a TYNDP scenario tuple.

    Args:
        scenario: TYNDP scenario code.
        year: TYNDP scenario year.
        climate_year: TYNDP climate year.

    Returns:
        A normalized ``(scenario, year, climate_year)`` tuple.

    Raises:
        ValueError: If the scenario code, year, or climate year is unsupported.
    """
    scenario = str(scenario).upper()
    year = int(year)
    climate_year = int(climate_year)

    if scenario not in TYNDP_SCENARIOS:
        raise ValueError(
            "Unknown TYNDP scenario "
            f"{scenario!r}. Expected one of: {', '.join(TYNDP_SCENARIOS)}."
        )
    valid_years = TYNDP_SCENARIO_YEARS[scenario]
    if year not in valid_years:
        raise ValueError(
            f"Unknown TYNDP scenario year {year!r} for scenario {scenario!r}. "
            "Expected one of: "
            + ", ".join(map(str, valid_years))
            + "."
        )
    if climate_year not in TYNDP_CLIMATE_YEARS:
        raise ValueError(
            f"Unknown TYNDP climate year {climate_year!r}. Expected one of: "
            + ", ".join(map(str, TYNDP_CLIMATE_YEARS))
            + "."
        )

    return scenario, year, climate_year


def ensure_tyndp_workbook(
    scenario,
    year,
    climate_year,
    data_dir,
    *,
    download=True,
    url_root=TYNDP_SCENARIO_URL_ROOT,
    keep_zip=True,
    session=None,
    verbose=False,
):
    """Ensure the XLSB workbook for a TYNDP scenario exists locally.

    Args:
        scenario: TYNDP scenario code.
        year: TYNDP scenario year.
        climate_year: TYNDP climate year.
        data_dir: Local cache directory for zip files and extracted workbooks.
        download: If True, download the official scenario zip when the workbook
            and zip are both missing.
        url_root: Base URL for official TYNDP scenario zip files.
        keep_zip: If True, keep the downloaded zip after extracting the
            workbook.
        session: Optional requests-compatible session used for downloads.
        verbose: If True, print progress messages.

    Returns:
        Path to the local XLSB workbook.

    Raises:
        FileNotFoundError: If the workbook is missing and downloading is
            disabled, or if the expected workbook cannot be found in the zip.
        ValueError: If the scenario tuple is not supported.
    """
    paths = tyndp_scenario_paths(
        data_dir=data_dir,
        scenario=scenario,
        year=year,
        climate_year=climate_year,
        url_root=url_root,
    )

    if paths["workbook"].exists():
        return paths["workbook"]

    if not paths["zip"].exists():
        if not download:
            raise FileNotFoundError(
                "TYNDP workbook is not cached and downloading is disabled: "
                f"{paths['workbook']}"
            )
        download_tyndp_scenario_zip(
            scenario=scenario,
            year=year,
            climate_year=climate_year,
            data_dir=data_dir,
            url_root=url_root,
            session=session,
            verbose=verbose,
        )

    _extract_tyndp_workbook_from_zip(paths["zip"], paths["workbook"], verbose=verbose)

    if not keep_zip:
        paths["zip"].unlink(missing_ok=True)

    return paths["workbook"]



def download_tyndp_scenario_zip(
    scenario,
    year,
    climate_year,
    data_dir,
    *,
    url_root=TYNDP_SCENARIO_URL_ROOT,
    session=None,
    verbose=False,
):
    """Download the official TYNDP scenario zip file and return its path.

    Args:
        scenario: TYNDP scenario code.
        year: TYNDP scenario year.
        climate_year: TYNDP climate year.
        data_dir: Local cache directory for the downloaded zip file.
        url_root: Base URL for official TYNDP scenario zip files.
        session: Optional requests-compatible session used for downloads.
        verbose: If True, print progress messages.

    Returns:
        Path to the downloaded zip file.

    Raises:
        requests.HTTPError: If the download response has an HTTP error status.
        ValueError: If the scenario tuple is not supported.
    """

    paths = tyndp_scenario_paths(
        data_dir=data_dir,
        scenario=scenario,
        year=year,
        climate_year=climate_year,
        url_root=url_root,
    )
    paths["zip"].parent.mkdir(parents=True, exist_ok=True)

    _log(f"Downloading {paths['url']}", verbose)
    http = session if session is not None else requests.Session()
    response = http.get(paths["url"], stream=True, timeout=120)
    response.raise_for_status()

    with paths["zip"].open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)

    return paths["zip"]


def _extract_tyndp_workbook_from_zip(zip_file, workbook, verbose=False):
    """Extract the expected TYNDP workbook from a downloaded zip archive."""
    zip_file = Path(zip_file)
    workbook = Path(workbook)
    workbook.parent.mkdir(parents=True, exist_ok=True)

    _log(f"Extracting {workbook.name} from {zip_file.name}", verbose)
    with zipfile.ZipFile(zip_file) as archive:
        names_by_basename = {Path(name).name: name for name in archive.namelist()}
        archive_name = names_by_basename.get(workbook.name)

        if archive_name is None:
            excel_files = [
                name
                for name in archive.namelist()
                if Path(name).suffix.lower() in (".xlsb", ".xlsx")
                and not Path(name).name.startswith(".")
            ]
            if len(excel_files) != 1:
                raise FileNotFoundError(
                    f"Could not find {workbook.name!r} in {zip_file}. "
                    f"Found Excel files: {excel_files}"
                )
            archive_name = excel_files[0]

        with archive.open(archive_name) as source, workbook.open("wb") as target:
            target.write(source.read())


def _read_excel_dataframe(filename, **kwargs):
    """Read one Excel sheet and return it as a DataFrame.

    ``pandas.read_excel`` has several overloads because it can return either a
    DataFrame or a dictionary of DataFrames. In this module every call reads a
    single sheet, so the runtime result is a DataFrame. The cast keeps static
    checkers such as Pylance from treating the result as a union.

    Args:
        filename: Excel file path.
        **kwargs: Keyword arguments forwarded to :func:`pandas.read_excel`.

    Returns:
        The requested Excel sheet as a DataFrame.
    """
    read_kwargs: dict[str, Any] = dict(kwargs)

    if "engine" in read_kwargs and read_kwargs["engine"] is None:
        read_kwargs.pop("engine")

    try:
        return cast(pd.DataFrame, pd.read_excel(filename, **read_kwargs))
    except ImportError as exc:
        # Some engines (e.g. "calamine") are optional. NT workbooks are .xlsx
        # files that do not require calamine, so retry without a specific
        # engine and let pandas choose the default openpyxl-based reader.
        if "engine" not in read_kwargs:
            raise
        read_kwargs.pop("engine")
        return cast(pd.DataFrame, pd.read_excel(filename, **read_kwargs))
    except ValueError as exc:
        sheet_name = read_kwargs.get("sheet_name")
        if not isinstance(sheet_name, str) or "not found" not in str(exc).lower():
            raise
        # TYNDP workbooks aren't consistently named across scenarios (e.g. the
        # "Global Ambition" workbook uses "Hourly Market Data emarket" while
        # "National Trends" just uses "Hourly Market Data"). Fall back to a
        # sheet whose name is a prefix/suffix match of the requested one
        # before giving up.
        available = pd.ExcelFile(filename, engine=read_kwargs.get("engine")).sheet_names
        candidates = [
            name
            for name in available
            if sheet_name.startswith(name) or name.startswith(sheet_name)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Worksheet named {sheet_name!r} not found in {filename}, and no "
                f"unambiguous fallback match among {available}"
                + (f" (candidates: {candidates})" if candidates else "")
            ) from exc
        read_kwargs["sheet_name"] = candidates[0]
        return cast(pd.DataFrame, pd.read_excel(filename, **read_kwargs))


def _log(message, verbose):
    """Print a timestamped progress message when verbose output is enabled."""
    if verbose:
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}")


def _parse_tyndp_datetime_index(index, model_year):
    """Convert the TYNDP hourly index to a DatetimeIndex.

    The original files used in the notebook have a two-level index where the
    second level stores labels such as ``"01Jan00:00"`` without a year. If the
    pickle already contains datetimes, they are returned unchanged.

    NT workbooks use all-caps month abbreviations (``"01SEP00:00"``); these are
    normalised to title-case before parsing so that ``%b`` matches correctly.

    Args:
        index: Original TYNDP index.
        model_year: Year appended to labels that do not already contain one.

    Returns:
        Parsed datetime index.
    """
    if isinstance(index, pd.DatetimeIndex):
        return index

    raw_index = index.get_level_values(1) if isinstance(index, pd.MultiIndex) else index
    raw_index = pd.Index(raw_index)

    if pd.api.types.is_datetime64_any_dtype(raw_index):
        return pd.DatetimeIndex(raw_index)

    labels = raw_index.astype(str).str.strip()

    if labels.str.contains(r"\d{4}", regex=True).all():
        return pd.to_datetime(labels)

    # Map month abbreviations to their numeric value instead of relying on
    # strptime's locale-dependent %b directive (English locale variants don't
    # agree on abbreviations -- e.g. "English (Ireland)" uses "Sept" for
    # September instead of the standard "Sep", which silently breaks %b).
    _MONTH_NUMBERS = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }

    def _normalise(label):
        upper = label.upper()
        for month, number in _MONTH_NUMBERS.items():
            if month in upper:
                day_str, rest = upper.split(month, 1)
                return f"{day_str}{number:02d}{rest}"
        return label

    labels = labels.map(_normalise)

    # Parse without the year (pandas' Cython strptime is strict about %M width
    # when %Y is appended), then replace the placeholder year with model_year.
    parsed = pd.to_datetime(labels, format="%d%m%H:%M")
    return parsed.map(lambda t: t.replace(year=model_year))


def load_technology_concordance(filename, sheet_name):
    """Read and normalize the technology concordance used to filter production.

    Args:
        filename: SHRECC/FIONA technology mapping workbook or CSV file.
        sheet_name: Sheet containing the technology concordance. Ignored for
            CSV files.

    Returns:
        Normalized technology concordance DataFrame.
    """
    filename = Path(filename)

    if filename.suffix.lower() == ".csv":
        concordance = pd.read_csv(filename, index_col=0)
    else:
        concordance = _read_excel_dataframe(
            filename,
            sheet_name=sheet_name,
            index_col=0,
            skipfooter=1,
        ).iloc[:, :-1]

    concordance = concordance.loc[
        concordance.sum(axis=1) > 0,
        concordance.sum(axis=0) > 0,
    ]
    concordance = concordance.div(concordance.sum(axis=0), axis=1)
    concordance.columns.name = "Category"

    return concordance


_load_technology_concordance = load_technology_concordance


def load_tyndp_country_mapping(
    mapping_location,
    countries_sheet="countries",
    connections_sheet="connections",
):
    """Read TYNDP node and connection mappings.

    Args:
        mapping_location: Either an Excel workbook with ``countries_sheet`` and
            ``connections_sheet`` sheets, or a directory containing
            ``tyndp_countries.csv`` and ``tyndp_connections.csv``.
        countries_sheet: Sheet name used for Excel workbooks.
        connections_sheet: Sheet name used for Excel workbooks.

    Returns:
        Tuple of ``(countries, connections)`` DataFrames.
    """
    mapping_location = Path(mapping_location)

    if mapping_location.is_dir():
        countries = pd.read_csv(
            mapping_location / "tyndp_countries.csv",
            index_col=0,
        )
        connections = pd.read_csv(
            mapping_location / "tyndp_connections.csv",
            index_col=0,
        )
    else:
        countries = _read_excel_dataframe(
            mapping_location,
            sheet_name=countries_sheet,
            index_col=0,
        )
        connections = _read_excel_dataframe(
            mapping_location,
            sheet_name=connections_sheet,
            index_col=0,
        )

    return countries, connections


def _aggregate_tyndp_production(production, accepted_categories, country_codes):
    """Aggregate production nodes to countries and keep mapped categories only."""
    if not isinstance(production.columns, pd.MultiIndex):
        raise TypeError("production columns must be a pandas MultiIndex")

    category_level = (
        "Category" if "Category" in production.columns.names else 0
    )
    country_level = "Country" if "Country" in production.columns.names else 1

    categories = production.columns.get_level_values(category_level)
    country_nodes = production.columns.get_level_values(country_level)

    mapped_countries = _map_tyndp_nodes_to_countries(country_nodes, country_codes)
    category_values = categories.to_numpy()
    mapped_country_values = mapped_countries.to_numpy()
    keep_columns = (
        np.asarray(categories.isin(accepted_categories))
        & mapped_countries.notna().to_numpy()
    )

    production = production.loc[:, keep_columns]
    production.columns = pd.MultiIndex.from_arrays(
        [
            mapped_country_values[keep_columns],
            category_values[keep_columns],
        ],
        names=["Country", "Source"],
    )

    production = production.groupby(
        level=["Country", "Source"],
        axis=1,
    ).sum()

    return production


def _aggregate_tyndp_consumption(production, country_codes):
    """Aggregate measured TYNDP electricity demand to country level."""
    if not isinstance(production.columns, pd.MultiIndex):
        raise TypeError("production columns must be a pandas MultiIndex")

    category_level = (
        "Category" if "Category" in production.columns.names else 0
    )
    country_level = "Country" if "Country" in production.columns.names else 1

    categories = production.columns.get_level_values(category_level)
    normalized_categories = (
        pd.Index(categories)
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    country_nodes = production.columns.get_level_values(country_level)
    mapped_countries = _map_tyndp_nodes_to_countries(country_nodes, country_codes)

    keep_columns = (
        (normalized_categories == TYNDP_DEMAND_CATEGORY)
        & mapped_countries.notna().to_numpy()
    )
    if not keep_columns.any():
        raise ValueError(
            f"TYNDP production table has no {TYNDP_DEMAND_CATEGORY!r} columns"
        )

    consumption = production.loc[:, keep_columns].copy()
    consumption.columns = pd.Index(
        mapped_countries.to_numpy()[keep_columns],
        name="Country",
    )
    return consumption.T.groupby(level="Country", sort=True).sum().T


def _map_tyndp_nodes_to_countries(nodes, country_codes):
    """Map TYNDP node labels to country codes with a conservative fallback."""
    nodes = pd.Index(nodes).astype(str)
    index = np.arange(len(nodes))
    valid_codes = set(country_codes.dropna().astype(str))

    mapped = pd.Series(nodes.map(country_codes.to_dict()), index=index)
    fallback = pd.Series(nodes, index=index).str.extract(
        r"^([A-Z]{2})",
        expand=False,
    )

    mapped = mapped.fillna(fallback.where(fallback.isin(valid_codes)))

    return mapped


def _aggregate_tyndp_trade(trade, connections):
    """Aggregate TYNDP exchange lines to positive directional country flows."""
    trade = trade.copy()
    raw_labels = pd.Index(trade.columns).astype(str)
    cleaned_labels = _clean_tyndp_connection_labels(raw_labels)
    detail_columns = raw_labels.str.contains(
        r"\s*/?(?:Concept|Real)\s+\d+\s*$",
        regex=True,
    )
    has_base_column = (
        pd.Series(~detail_columns, index=cleaned_labels)
        .groupby(level=0)
        .transform("any")
        .to_numpy()
    )

    # The unsuffixed series is the aggregate interconnector flow. Real/Concept
    # series are components or alternatives and must not be added to it. Where
    # no aggregate exists, retain and combine the available detail series.
    keep_columns = (~detail_columns) | (~has_base_column)
    trade = trade.loc[:, keep_columns]
    trade.columns = cleaned_labels[keep_columns]

    mapping = connections.reindex(trade.columns)
    required_columns = ["Country from", "Country to"]
    missing_columns = [col for col in required_columns if col not in mapping.columns]

    if missing_columns:
        raise KeyError(
            "connections mapping is missing required columns: "
            + ", ".join(missing_columns)
        )

    missing = mapping[required_columns].isna().any(axis=1)
    if missing.any():
        missing_lines = mapping.index[missing].tolist()
        warnings.warn(
            "Dropping trade columns without country-pair mappings: "
            + ", ".join(map(str, missing_lines[:10])),
            stacklevel=2,
        )
        trade = trade.loc[:, ~missing.to_numpy()]
        mapping = mapping.loc[~missing]

    if trade.empty:
        raise ValueError("No mapped trade columns remain after applying connections")

    trade.columns = pd.MultiIndex.from_arrays(
        [
            mapping["Country from"],
            mapping["Country to"],
        ],
        names=["Country from", "Country to"],
    )

    trade = trade.T.groupby(level=["Country from", "Country to"]).sum().T

    # Positive values are already origin -> destination. Negative values are
    # reversed and made positive so every column is a physical positive flow.
    positive = trade.clip(lower=0)
    negative_reversed = -trade.clip(upper=0)
    negative_reversed.columns = pd.MultiIndex.from_arrays(
        [
            negative_reversed.columns.get_level_values("Country to"),
            negative_reversed.columns.get_level_values("Country from"),
        ],
        names=["Country from", "Country to"],
    )

    return (
        pd.concat([positive, negative_reversed], axis=1)
        .T.groupby(level=["Country from", "Country to"])
        .sum()
        .T.swaplevel(axis=1)
    )


def _clean_tyndp_connection_labels(labels):
    """Remove TYNDP concept suffixes from cross-border connection labels.

    Also normalises the from/to separator: some TYNDP scenario workbooks
    (e.g. "National Trends") export connection labels as ``"AL00->GR00"``,
    while others (e.g. "Global Ambition") and the ``connections`` mapping
    both use ``"AL00-GR00"``.
    """
    return (
        pd.Index(labels)
        .astype(str)
        .str.replace(r"\s*/?(Concept|Real)\s+\d+\s*$", "", regex=True)
        .str.replace("->", "-", regex=False)
        .str.strip()
    )


def consumption_results_from_z_gross(
    Z_gross,
    check=True,
    return_debug=False,
    zero_consumption="raise",
    verbose=False,
    volume_unit="MWh",
    include_consumption_mix_volume=True,
):
    """Return consumption volumes and normalized mixes from ``Z_gross``.

    This is the canonical result-oriented interface. The returned Dataset
    retains raw production and trade volumes, measured TYNDP country demand,
    consumption resolved by producing country and technology, and the
    normalized consumption mix. For backward-compatible ``Z_gross`` inputs
    without measured demand, country consumption is inferred from the physical
    balance. ``consumption_mix_volume`` can be omitted to reduce memory use
    when only normalized shares are required.
    """
    _log("Preparing canonical production and trade volumes", verbose)
    production_volume, trade_volume = _solver_inputs_from_z_gross(Z_gross)
    consumption_volume = _consumption_volume_from_z_gross(Z_gross)

    _log("Solving hourly country trade systems", verbose)
    results, debug = solve_consumption_system(
        production_volume,
        trade_volume,
        consumption_volume=consumption_volume,
        check=check,
        zero_consumption=zero_consumption,
        volume_unit=volume_unit,
        include_consumption_mix_volume=include_consumption_mix_volume,
    )
    if consumption_volume is not None:
        results.attrs["consumption_volume_source"] = TYNDP_DEMAND_CATEGORY
        results["consumption_volume"].attrs["source_category"] = (
            TYNDP_DEMAND_CATEGORY
        )
    _log(f"Built consumption results with sizes {dict(results.sizes)}", verbose)

    if return_debug:
        countries = results["consumption_mix"]["consumer_country"].to_index()
        technologies = results["consumption_mix"]["technology"].to_index()
        direct_production = debug["direct_production"]
        debug.update(
            {
                "A_prod_3d": direct_production.transpose(0, 2, 1),
                "A_trade_3d": debug["direct_trade"],
                "country_total_ordered": pd.DataFrame(
                    debug["node_supply_volume"],
                    index=Z_gross.index,
                    columns=countries,
                ),
                "countries": countries,
                "activities": technologies,
            }
        )
        return results, debug

    return results


def write_z_gross_consumption_cache(
    Z_gross,
    cache_dir,
    *,
    time_chunk_size=168,
    check=True,
    zero_consumption="raise",
    volume_unit="MWh",
    include_consumption_mix_volume=True,
):
    """Solve a TYNDP gross table in chunks and persist canonical results."""
    production_volume, trade_volume = _solver_inputs_from_z_gross(Z_gross)
    consumption_volume = _consumption_volume_from_z_gross(Z_gross)
    return write_solved_consumption_result_cache(
        production_volume,
        trade_volume,
        cache_dir,
        consumption_volume=consumption_volume,
        time_chunk_size=time_chunk_size,
        check=check,
        zero_consumption=zero_consumption,
        volume_unit=volume_unit,
        include_consumption_mix_volume=include_consumption_mix_volume,
    )


def consumption_mix_from_z_gross(
    Z_gross,
    check=True,
    return_debug=False,
    zero_consumption="raise",
    verbose=False,
):
    """Calculate the normalized hourly consumption mix from ``Z_gross``.

    This backward-compatible interface returns only ``consumption_mix``. Use
    :func:`consumption_results_from_z_gross` to retain physical volumes too.
    """
    result = consumption_results_from_z_gross(
        Z_gross,
        check=check,
        return_debug=return_debug,
        zero_consumption=zero_consumption,
        verbose=verbose,
        include_consumption_mix_volume=False,
    )
    if return_debug:
        results, debug = result
        return results["consumption_mix"], debug
    return result["consumption_mix"]


def _solver_inputs_from_z_gross(Z_gross):
    """Convert a TYNDP gross table to shared solver input arrays."""
    if not isinstance(Z_gross, pd.DataFrame):
        raise TypeError("Z_gross must be a pandas DataFrame")
    if not isinstance(Z_gross.columns, pd.MultiIndex):
        raise ValueError("Z_gross columns must be a MultiIndex")

    required_levels = {"type", "country from", "country to", "source"}
    missing_levels = required_levels.difference(Z_gross.columns.names)
    if missing_levels:
        raise ValueError(
            "Z_gross columns are missing required levels: "
            + ", ".join(sorted(missing_levels))
        )
    required_types = {"production mix", "trade"}
    missing_types = required_types.difference(
        Z_gross.columns.get_level_values("type")
    )
    if missing_types:
        raise ValueError(
            "Z_gross is missing required column types: "
            + ", ".join(sorted(missing_types))
        )

    production = Z_gross["production mix"]
    production_from = production.columns.get_level_values("country from")
    production_to = production.columns.get_level_values("country to")
    if not production_from.equals(production_to):
        raise ValueError("Production volumes must be domestic")
    production = production.T.groupby(
        level=["country from", "source"],
        sort=True,
    ).sum().T

    trade = Z_gross["trade"].droplevel("source", axis=1)
    trade = trade.T.groupby(
        level=["country from", "country to"],
        sort=True,
    ).sum().T

    countries = (
        production.columns.get_level_values("country from")
        .unique()
        .union(trade.columns.get_level_values("country from").unique())
        .union(trade.columns.get_level_values("country to").unique())
        .sort_values()
    )
    technologies = (
        production.columns.get_level_values("source").unique().sort_values()
    )

    production_columns = pd.MultiIndex.from_product(
        [countries, technologies],
        names=["country from", "source"],
    )
    trade_columns = pd.MultiIndex.from_product(
        [countries, countries],
        names=["country from", "country to"],
    )
    production = production.reindex(columns=production_columns, fill_value=0)
    trade = trade.reindex(columns=trade_columns, fill_value=0)

    production_volume = xr.DataArray(
        production.to_numpy().reshape(
            len(Z_gross),
            len(countries),
            len(technologies),
        ),
        dims=("time", "producer_country", "technology"),
        coords={
            "time": Z_gross.index,
            "producer_country": countries.to_numpy(),
            "technology": technologies.to_numpy(),
        },
        name="production_volume",
    )
    trade_volume = xr.DataArray(
        trade.to_numpy().reshape(
            len(Z_gross),
            len(countries),
            len(countries),
        ),
        dims=("time", "exporter_country", "importer_country"),
        coords={
            "time": Z_gross.index,
            "exporter_country": countries.to_numpy(),
            "importer_country": countries.to_numpy(),
        },
        name="trade_volume",
    )
    return production_volume, trade_volume


def _consumption_volume_from_z_gross(Z_gross):
    """Return measured country demand when it is available in ``Z_gross``."""
    if "consumption" not in Z_gross.columns.get_level_values("type"):
        return None

    consumption = Z_gross["consumption"]
    consumption_from = consumption.columns.get_level_values("country from")
    consumption_to = consumption.columns.get_level_values("country to")
    if not consumption_from.equals(consumption_to):
        raise ValueError("Consumption volumes must be domestic")

    consumption = consumption.T.groupby(
        level="country to",
        sort=True,
    ).sum().T
    return xr.DataArray(
        consumption.to_numpy(),
        dims=("time", "consumer_country"),
        coords={
            "time": Z_gross.index,
            "consumer_country": consumption.columns.to_numpy(),
        },
        name="consumption_volume",
    )
