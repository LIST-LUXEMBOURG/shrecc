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

import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr


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

    Parameters
    ----------
    production_pickle : str or pathlib.Path
        Pickle file containing the hourly TYNDP production table, read from the
        ``"Hourly Market Data emarket"`` sheet in the original workbook.
    trade_pickle : str or pathlib.Path
        Pickle file containing the hourly TYNDP cross-border exchange table,
        read from the ``"Crossborder exchanges"`` sheet in the original
        workbook.
    technology_mapping : str or pathlib.Path
        Excel workbook containing the technology concordance. The function uses
        this file to keep only production categories that are mapped in the
        concordance. It does not aggregate to premise activities here; the
        resulting production technologies remain the original TYNDP/ENTSO-E
        categories.
    country_mapping : str or pathlib.Path
        Excel workbook containing country and connection mappings. Expected
        sheets are ``countries_sheet`` and ``connections_sheet``.
    model_year : int, default 2050
        Year appended to the TYNDP day/month/hour labels when constructing the
        hourly datetime index.
    technology_sheet : str, default "concordance"
        Sheet name in ``technology_mapping`` with technology/category mappings.
    countries_sheet : str, default "countries"
        Sheet name in ``country_mapping`` mapping TYNDP node labels to country
        codes. The sheet is expected to have a ``"Country code"`` column.
    connections_sheet : str, default "connections"
        Sheet name in ``country_mapping`` mapping cross-border line labels to
        ``"Country from"`` and ``"Country to"``.
    verbose : bool, default False
        If True, print progress messages for the main parsing and aggregation
        steps.

    Returns
    -------
    pandas.DataFrame
        Gross hourly supply table with rows indexed by time and columns indexed
        by ``("type", "country from", "country to", "source")``. The ``type``
        level contains:

        - ``"production mix"``: production by country and technology, encoded
          as domestic flows where ``country from == country to``;
        - ``"trade"``: positive directional imports from ``country from`` to
          ``country to`` with ``source == "trade"``.

    Notes
    -----
    This function is the reusable version of the parsing steps in
    ``notebooks/parsing_tyndp.ipynb``. In particular:

    - production nodes such as ``XX00`` or ``XX00RETE`` are mapped to country
      codes and aggregated;
    - exchange columns are mapped to country pairs;
    - negative exchanges are reversed so all trade flows are positive and
      directional;
    - production categories absent from the technology concordance are dropped.
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
    engine="pyxlsb",
    production_pickle=None,
    trade_pickle=None,
    verbose=False,
):
    """Build the gross hourly supply table directly from the TYNDP workbook.

    Parameters
    ----------
    excel_file : str or pathlib.Path
        Original TYNDP Excel/XLSB workbook containing the hourly production and
        cross-border exchange sheets.
    technology_mapping, country_mapping
        See :func:`build_z_gross_from_tyndp_pickles`.
    model_year : int, default 2050
        Year appended to the TYNDP day/month/hour labels when constructing the
        hourly datetime index.
    production_sheet : str, default "Hourly Market Data emarket"
        Sheet containing hourly production data.
    trade_sheet : str, default "Crossborder exchanges"
        Sheet containing hourly cross-border exchanges.
    technology_sheet, countries_sheet, connections_sheet
        See :func:`build_z_gross_from_tyndp_pickles`.
    engine : str or None, default "pyxlsb"
        Excel engine passed to :func:`pandas.read_excel`. The original TYNDP
        workbook used in the notebook is ``.xlsb``, so ``"pyxlsb"`` is the
        default. Use ``None`` to let pandas infer the engine for ordinary
        ``.xlsx`` files.
    production_pickle, trade_pickle : str or pathlib.Path, optional
        If both are provided, save the raw extracted production and trade sheets
        to these pickle files after reading the workbook. The saved files can be
        reused with :func:`build_z_gross_from_tyndp_pickles`.
    verbose : bool, default False
        If True, print progress messages for reading, parsing, and aggregation.

    Returns
    -------
    pandas.DataFrame
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
    engine="pyxlsb",
    production_pickle=None,
    trade_pickle=None,
    verbose=False,
):
    """Read raw TYNDP production and trade sheets from the workbook.

    Parameters
    ----------
    excel_file : str or pathlib.Path
        Original TYNDP Excel/XLSB workbook.
    production_sheet : str, default "Hourly Market Data emarket"
        Sheet containing hourly production data.
    trade_sheet : str, default "Crossborder exchanges"
        Sheet containing hourly cross-border exchanges.
    engine : str or None, default "pyxlsb"
        Excel engine passed to :func:`pandas.read_excel`.
    production_pickle, trade_pickle : str or pathlib.Path, optional
        If both are provided, save the raw extracted sheets to these pickle
        files.
    verbose : bool, default False
        If True, print progress messages.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Raw production and trade DataFrames as read from the workbook.
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
    country_mapping_df = _read_excel_dataframe(
        country_mapping,
        sheet_name=countries_sheet,
        index_col=0,
    )
    country_codes = country_mapping_df["Country code"]

    _log("Aggregating hourly production by country and technology", verbose)
    production_agg = _aggregate_tyndp_production(
        production,
        concordance.columns,
        country_codes,
    )

    connections = _read_excel_dataframe(
        country_mapping,
        sheet_name=connections_sheet,
        index_col=0,
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
        [production_agg, trade_gross],
        axis=1,
        keys=["production mix", "trade"],
        names=["type"],
    )
    _log(f"Built Z_gross with shape {Z_gross.shape}", verbose)

    return Z_gross


def consumption_mix_from_tyndp_pickles(
    production_pickle,
    trade_pickle,
    technology_mapping,
    country_mapping,
    model_year=2050,
    check=True,
    return_debug=False,
    verbose=False,
):
    """Run the full TYNDP pickle-to-consumption-mix pipeline.

    This is a convenience wrapper around
    :func:`build_z_gross_from_tyndp_pickles` and
    :func:`consumption_mix_from_z_gross`.

    Parameters
    ----------
    production_pickle, trade_pickle, technology_mapping, country_mapping
        See :func:`build_z_gross_from_tyndp_pickles`.
    model_year : int, default 2050
        Year used to construct the hourly datetime index from TYNDP labels.
    check : bool, default True
        Forwarded to :func:`consumption_mix_from_z_gross`.
    return_debug : bool, default False
        If True, return ``(consumption_mix_xr, debug)``. The debug dictionary
        includes the constructed ``Z_gross`` table in addition to the coefficient
        matrices returned by :func:`consumption_mix_from_z_gross`.
    verbose : bool, default False
        If True, print progress messages for the gross-table construction and
        consumption-mix solve.

    Returns
    -------
    xarray.DataArray or tuple[xarray.DataArray, dict]
        Hourly consumption mix by consuming country, producing country, and
        technology.
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
            verbose=verbose,
        )

    consumption_mix_xr, debug = cast(
        tuple[xr.DataArray, dict[str, object]],
        consumption_mix_from_z_gross(
            Z_gross,
            check=check,
            return_debug=True,
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
    engine="pyxlsb",
    verbose=False,
):
    """Run the full TYNDP workbook-to-consumption-mix pipeline.

    This is the Excel/XLSB equivalent of
    :func:`consumption_mix_from_tyndp_pickles`. It reads the production and
    cross-border exchange sheets directly from the original TYNDP workbook,
    builds ``Z_gross``, and calculates the hourly consumption mix.

    Parameters
    ----------
    excel_file, technology_mapping, country_mapping
        See :func:`build_z_gross_from_tyndp_excel`.
    model_year : int, default 2050
        Year used to construct the hourly datetime index from TYNDP labels.
    production_sheet : str, default "Hourly Market Data emarket"
        Sheet containing hourly production data.
    trade_sheet : str, default "Crossborder exchanges"
        Sheet containing hourly cross-border exchanges.
    technology_sheet, countries_sheet, connections_sheet
        See :func:`build_z_gross_from_tyndp_excel`.
    check : bool, default True
        Forwarded to :func:`consumption_mix_from_z_gross`.
    return_debug : bool, default False
        If True, return ``(consumption_mix_xr, debug)``. The debug dictionary
        includes the constructed ``Z_gross`` table in addition to the coefficient
        matrices returned by :func:`consumption_mix_from_z_gross`.
    engine : str or None, default "pyxlsb"
        Excel engine passed to :func:`pandas.read_excel`.
    verbose : bool, default False
        If True, print progress messages for reading, parsing, and solving.

    Returns
    -------
    xarray.DataArray or tuple[xarray.DataArray, dict]
        Hourly consumption mix by consuming country, producing country, and
        technology.
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
            verbose=verbose,
        )

    consumption_mix_xr, debug = cast(
        tuple[xr.DataArray, dict[str, object]],
        consumption_mix_from_z_gross(
            Z_gross,
            check=check,
            return_debug=True,
            verbose=verbose,
        ),
    )
    debug["Z_gross"] = Z_gross

    return consumption_mix_xr, debug


def _load_pickle(filename):
    """Load a pickled object from disk."""
    with Path(filename).open("rb") as handle:
        return pickle.load(handle)


def _save_pickle(obj, filename):
    """Save an object to a pickle file, creating parent directories if needed."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with filename.open("wb") as handle:
        pickle.dump(obj, handle)


def _read_excel_dataframe(filename, **kwargs):
    """Read one Excel sheet and return it as a DataFrame.

    ``pandas.read_excel`` has several overloads because it can return either a
    DataFrame or a dictionary of DataFrames. In this module every call reads a
    single sheet, so the runtime result is a DataFrame. The cast keeps static
    checkers such as Pylance from treating the result as a union.
    """
    read_kwargs: dict[str, Any] = dict(kwargs)

    if "engine" in read_kwargs and read_kwargs["engine"] is None:
        read_kwargs.pop("engine")

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
    """
    if isinstance(index, pd.DatetimeIndex):
        return index

    raw_index = index.get_level_values(1) if isinstance(index, pd.MultiIndex) else index
    raw_index = pd.Index(raw_index)

    if pd.api.types.is_datetime64_any_dtype(raw_index):
        return pd.DatetimeIndex(raw_index)

    labels = raw_index.astype(str)

    if labels.str.contains(r"\d{4}", regex=True).all():
        return pd.to_datetime(labels)

    return pd.to_datetime(
        labels + str(model_year),
        format="%d%b%H:%M%Y",
    )


def load_technology_concordance(filename, sheet_name):
    """Read and normalize the technology concordance used to filter production."""
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
    trade.columns = _clean_tyndp_connection_labels(trade.columns)

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

    trade = trade.groupby(level=["Country from", "Country to"], axis=1).sum()

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
    """Remove TYNDP concept suffixes from cross-border connection labels."""
    return (
        pd.Index(labels)
        .astype(str)
        .str.replace(r"\s*/?(Concept|Real)\s+\d+\s*$", "", regex=True)
        .str.strip()
    )


def consumption_mix_from_z_gross(
    Z_gross,
    check=True,
    return_debug=False,
    verbose=False,
):
    """Calculate hourly consumption mixes from a gross supply matrix.

    Parameters
    ----------
    Z_gross : pandas.DataFrame
        Hourly gross electricity supply table. Rows must be hourly timestamps or
        another time-like index. Columns must be a ``MultiIndex`` with the first
        level named ``"type"`` and containing at least:

        - ``"production mix"``: domestic production by source technology.
        - ``"trade"``: cross-border electricity flows.

        The remaining column levels are expected to describe the flow as
        ``"country from"``, ``"country to"``, and ``"source"``. For production
        columns, ``"source"`` is the production technology. For trade columns,
        ``"source"`` is dropped and trade is aggregated to country-to-country
        flows.
    check : bool, default True
        If True, run two conservation checks:

        - direct domestic production shares plus direct import shares must sum
          to one for every hour and consuming country;
        - the final consumption mix must also sum to one for every hour and
          consuming country.

        These checks are intentionally strict because most errors in this
        workflow are axis-orientation or denominator mismatches.
    return_debug : bool, default False
        If True, return a tuple ``(consumption_mix_xr, debug)`` where ``debug``
        contains intermediate coefficient matrices and the direct-total check.
    verbose : bool, default False
        If True, print progress messages for coefficient construction,
        conservation checks, the matrix solve, and xarray conversion.

    Returns
    -------
    xarray.DataArray or tuple[xarray.DataArray, dict]
        DataArray named ``"consumption_mix"`` with dimensions:

        - ``time``: copied from ``Z_gross.index``;
        - ``consumer_country``: country where electricity is consumed;
        - ``source_country``: country where electricity was produced;
        - ``technology``: production technology.

        Values are shares of each consuming country's hourly electricity
        consumption. For each ``time`` and ``consumer_country``, summing over
        ``source_country`` and ``technology`` should give one.

    Notes
    -----
    Axis convention is the main thing to keep straight:

    - ``A_prod_3d`` is ``(time, technology, source_country)``.
    - ``A_prod_expanded`` is ``(time, technology-source_country, source_country)``.
    - ``A_trade_3d`` is ``(time, consumer_country, source_country)``.

    With this convention, direct shares satisfy::

        A_prod_expanded.sum(axis=1) + A_trade_3d.sum(axis=2) == 1

    The Leontief-style solve is therefore performed with ``I - A_trade_3d``
    directly, not with its transpose.
    """
    # Build a domestic country-to-country block from production. It gives the
    # total domestic production available for each consumer country and is used
    # only to define the denominator for direct shares.
    _log("Building domestic production and trade denominator matrices", verbose)
    Z_domestic = (
        Z_gross["production mix"]
        .groupby(["country from", "country to"], axis=1)
        .sum()
    )

    Z_trade = (
        pd.concat(
            [
                Z_gross["trade"].droplevel("source", axis=1),
                Z_domestic,
            ],
            axis=1,
        )
        .swaplevel(axis=1)
        .sort_index(axis=1)
    )

    # Common denominator for all direct supply shares: total electricity
    # consumption in each country and hour, i.e. domestic production plus
    # imports. Using the same denominator for production and trade is essential
    # for the direct shares to sum to one.
    country_total = Z_trade.groupby("country to", axis=1).sum()

    # Direct production shares by technology and producing country. The
    # consuming-country level is no longer needed after normalization because
    # production is domestic: country from == country to.
    _log("Calculating direct production and trade shares", verbose)
    A_prod = (
        Z_gross["production mix"]
        .div(country_total, level="country to", axis=1)
        .droplevel("country to", axis=1)
        .fillna(0)
    )

    # Direct trade shares by consuming country and source country. This still
    # includes the domestic diagonal temporarily; it is removed after reshaping
    # because domestic supply is represented by A_prod.
    A_trade = (
        Z_trade
        .div(country_total, level="country to", axis=1)
        .fillna(0)
    )

    countries = (
        A_trade.columns.get_level_values("country to")
        .unique()
        .union(A_trade.columns.get_level_values("country from"))
        .unique()
        .sort_values()
    )
    activities = A_prod.columns.get_level_values("source").unique().sort_values()
    _log(
        f"Using {len(countries)} countries and {len(activities)} technologies",
        verbose,
    )

    # Arrange production columns into a dense, deterministic
    # (technology, source_country) order before reshaping.
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

    # Expand production into technology-country rows. Each producing country
    # occupies one block of K technology rows, and only the matching country
    # column is populated.
    T, K, C = A_prod_3d.shape
    A_prod_expanded = np.zeros((T, K * C, C), dtype=A_prod_3d.dtype)

    rows = (np.arange(C)[:, None] * K + np.arange(K)[None, :]).ravel()
    cols = np.repeat(np.arange(C), K)
    values = np.transpose(A_prod_3d, (0, 2, 1)).reshape(T, C * K)

    A_prod_expanded[:, rows, cols] = values

    # Arrange trade columns into (consumer_country, source_country) order. This
    # orientation is important: import shares sum over axis 2.
    _log("Reshaping trade shares", verbose)
    trade_cols = pd.MultiIndex.from_product(
        [countries, countries],
        names=["country to", "country from"],
    )

    A_trade_wide = A_trade.reindex(columns=trade_cols, fill_value=0)

    # Domestic supply is represented by A_prod, not by country-to-country trade.
    self_index = pd.MultiIndex.from_arrays(
        [countries, countries],
        names=["country to", "country from"],
    )
    A_trade_wide.loc[:, self_index] = 0

    A_trade_3d = A_trade_wide.to_numpy().reshape(T, C, C)

    direct_total = A_prod_expanded.sum(axis=1) + A_trade_3d.sum(axis=2)

    if check:
        _log("Checking direct shares sum to one", verbose)
        np.testing.assert_allclose(direct_total, 1, atol=1e-8)

    # Resolve indirect trade dependencies. Since A_trade_3d is oriented as
    # (consumer_country, source_country), solve with M directly.
    _log("Solving hourly trade system", verbose)
    M = np.eye(C)[None, :, :] - A_trade_3d

    consumption_mix_3d = np.linalg.solve(
        M,
        A_prod_expanded.transpose(0, 2, 1),
    ).transpose(0, 2, 1)

    if check:
        _log("Checking final consumption mix sums to one", verbose)
        np.testing.assert_allclose(consumption_mix_3d.sum(axis=1), 1, atol=1e-8)

    # Use a temporary MultiIndex dimension for technology-country sources, then
    # unstack it so users can select by source_country and technology directly.
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
            "direct_total": direct_total,
        }

    return consumption_mix_xr
