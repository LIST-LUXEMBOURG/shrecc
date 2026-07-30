# Copyright © 2024,2025,2026 Luxembourg Institute of Science and Technology
# Licensed under the MIT License (see LICENSE file for details).
# Authors: [Sabina Bednářová, Thomas Gibon]

from datetime import datetime
from importlib.resources import files
from packaging.version import parse as vparse
from pathlib import Path
import re

import bw2data as bd
import pandas as pd
from bw2data.query import Filter, Query

from shrecc._legacy_database import (
    apply_mapping,
    filter_by_countries,
    filter_by_range,
    filter_by_times,
    load_time_series_data,
    prepare_consumption_data,
    tech_mapping,
)
from shrecc.mapping import (
    activity_mix_to_database_table,
    load_ecoinvent_mapping as load_mapping_data,
    map_consumption_mix_to_ecoinvent_activities,
    validate_inventory_resolution,
)
from shrecc.result_store import (
    MANIFEST_FILENAME,
    consumption_result_cache_path,
    load_consumption_result_cache,
)

UNUSED_SOURCE = "Import balance (physical)"
DEFAULT_ACTIVITY_FALLBACK_LOCATIONS = ("RER", "RoW", "GLO")


def filt_cutoff(
    countries,
    times=[],
    general_range=0,
    refined_range=0,
    freq=0,
    cutoff=1e-3,
    include_cutoff=True,
    path_to_data=None,
):
    """
    Filters data based on selected countries and times (either one-off, a range, or periodical range).

    Args:
        year (int): Selected year of the downloaded data.
        countries (list of str): Countries selected by the user for their database.
            E.g. countries=['FR', 'DE'].
        times (list of str): Selecting one specific time, e.g. times = ['2023-06-16 8:00:00', '2023-06-16 22:00:00'].
            Can be applied alone.
        general_range (list of str): Selecting a general range, e.g. for the month of June
            general_range = ['2023-06-01 01:00:00', '2023-06-30 23:00:00']. Can be applied alone.
        refined_range (list of int): Refining range of general range, e.g. mornings of June (previously selected in general_range):
            refined_range = [8, 12], in hours (24-hour format). Can only be
            applied with general_range.
        freq (str): Days to be included, e.g. freq='D' selects calendar days,
            see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
        cutoff (float): Cutoff value for technology values.
        include_cutoff (bool): If True, cutoff is applied and summed at the end to create a new technology "The rest".
            If False, cutoff is applied but new technology not created.
        path_to_data (str or str or Path): location of the data. If none, the data is taken from within the package.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    if path_to_data:
        print(f"Using mapping root: {path_to_data}")
        path_to_data = Path(path_to_data)
    else:
        path_to_data = files("shrecc.data")

    if general_range:
        year = datetime.strptime(general_range[0], "%Y-%m-%d %H:%M:%S").year
    elif len(times):
        times = pd.to_datetime(times)
        year = times[0].year
    else:
        raise ValueError("Either `times` or `general_range` must be provided")

    now = datetime.now()
    print(f"{now} Filtering dataframe...")
    cache_dir = consumption_result_cache_path(path_to_data, year)
    if (cache_dir / MANIFEST_FILENAME).is_file():
        dataframe = _filt_canonical_consumption_results(
            cache_dir,
            countries=countries,
            times=times,
            general_range=general_range,
            refined_range=refined_range,
            freq=freq,
        )
    else:
        dataframe = tech_mapping(year, path_to_data)
        dataframe = dataframe.droplevel("source", axis=1)
        dataframe = filter_by_countries(dataframe, countries)

        if len(times):
            # Ensure datetime is used in the backwards-compatible filtering.
            times = pd.to_datetime(times)
            dataframe = filter_by_times(dataframe, times)
        if general_range:
            dataframe = filter_by_range(dataframe, general_range, refined_range, freq)
    dataframe = apply_cutoff(dataframe, cutoff, include_cutoff)
    now = datetime.now()
    print(f"{now} Dataframe filtered.")
    return dataframe


def _filt_canonical_consumption_results(
    cache_dir,
    *,
    countries,
    times,
    general_range,
    refined_range,
    freq,
):
    selected_times = times if len(times) else None
    selected_range = general_range if general_range else None
    results = load_consumption_result_cache(
        cache_dir,
        times=selected_times,
        general_range=selected_range,
        variables=["consumption_mix"],
    )
    activity_mapping = load_mapping_data(files("shrecc.data"))
    activity_mix = map_consumption_mix_to_ecoinvent_activities(
        results["consumption_mix"],
        activity_mapping,
    )
    return activity_mix_to_database_table(
        activity_mix,
        countries=countries,
        times=selected_times,
        general_range=selected_range,
        refined_range=refined_range if refined_range else None,
        freq=freq if freq else None,
    )


def apply_cutoff(df_filt, cutoff, include_cutoff):
    """
    Apply a cutoff value to filter out smaller values in the dataframe and optionally include a "rest" category.

    Args:
        df_filt (pd.DataFrame): The filtered dataframe.
        cutoff (float): The cutoff value for technology values.
        include_cutoff (bool): If True, sums values below cutoff and includes them as a new technology "The rest".

    Returns:
        pd.DataFrame: A dataframe with values below the cutoff set to zero, optionally including a "rest" category.
    """
    cutoff_totals = df_filt.where(df_filt.le(cutoff), 0).sum(axis=0)
    result = df_filt.mask(df_filt.lt(cutoff), 0)
    if not include_cutoff:
        return result

    rest_row = (
        "RER",
        "market group for electricity, high voltage",
        "electricity, high voltage",
        "kWh",
    )
    rest = cutoff_totals.to_frame().T
    rest.index = pd.MultiIndex.from_tuples([rest_row], names=result.index.names)
    return pd.concat([result.drop(index=rest_row, errors="ignore"), rest])


def setup_database(project_name, db_name):
    """
    Sets up the BW2 database for the given project.

    Args:
        project_name (str): The name of the BW project.
        db_name (str): The name of the BW database to set up.

    Returns:
        bd.Database: The newly registered BW2 database.
    """
    bd.projects.set_current(project_name)
    if db_name in bd.databases:
        del bd.databases[db_name]
        bd.projects.purge_deleted_directories()
    elec_db = bd.Database(db_name)
    elec_db.register()
    return elec_db


def map_known_inputs(
    eidb_name,
    dataframe_filt,
    strict=False,
    include_network=True,
    fallback_locations=DEFAULT_ACTIVITY_FALLBACK_LOCATIONS,
):
    """
    Maps known inputs from the ecoinvent database to the filtered dataframe.

    Args:
        eidb_name (str): The name of the ecoinvent database in the BW project.
        dataframe_filt (pd.DataFrame): The filtered dataframe containing technology data.
        strict (bool): Raise an error if a required activity cannot be matched uniquely.
        include_network (bool): Also resolve the fixed electricity-network inputs.
        fallback_locations: Ordered fallback geographies tried when the requested
            activity name and unit do not exist at the requested geography.

    Returns:
        dict: A dictionary mapping known inputs to their corresponding entries in the ecoinvent database.
    """
    ei_db = bd.Database(eidb_name)
    ei_db_data = ei_db.load()
    known_inputs = {}
    missing_inputs = []
    country_to_code = {
        "Germany": "DE",
        "France": "FR",
    }
    active_rows = dataframe_filt.ne(0).any(axis=1)
    for idx in dataframe_filt.index[active_rows]:
        original_loc, original_name, prod, unit = idx
        loc = original_loc
        name = original_name
        # Region names in ENTSOE and ecoinvent don't exactly match
        # Only UK seems concerned but consider using a dictionary
        if loc == "UK":
            loc = "GB"

        # For ecoinvent > 3.10, the activity names have changed
        # They now use a country code, instead of a full name
        # We deal with them here:
        def repl(match):
            return f"from {country_to_code[match.group(1)]}"

        if any(v in eidb_name for v in ("3.11", "3.12")):
            pattern = re.compile(r"from (Germany|France)")
            name = pattern.sub(repl, name)
        candidate_locations = [loc]
        candidate_locations.extend(
            fallback_loc
            for fallback_loc in fallback_locations or ()
            if fallback_loc != loc
        )
        results = []
        resolved_loc = loc
        for candidate_loc in candidate_locations:
            q = Query()
            q.add(Filter("name", "is", name))
            q.add(Filter("location", "is", candidate_loc))
            q.add(Filter("unit", "is", "kilowatt hour"))
            results = q(ei_db_data)
            resolved_loc = candidate_loc
            if results:
                break

        if len(results) == 1:
            exchange = list(results).pop()
            known_inputs[(original_loc, original_name, unit)] = exchange
            known_inputs[(loc, name, unit)] = exchange
            known_inputs[(resolved_loc, name, unit)] = exchange
            if resolved_loc != loc:
                print(
                    f"Using fallback activity:{name}, {resolved_loc} "
                    f"for requested geography {loc}"
                )
        else:
            print("Couldnt find activity:" + name + ", " + loc)
            missing_inputs.append((loc, name, len(results)))

    network = get_network_activities(eidb_name) if include_network else []
    known_inputs_network = {}
    for act in network:
        loc = act["loc"]
        name = act["name"]
        q = Query()
        filter_name = Filter("name", "is", name)
        filter_loc = Filter("location", "is", loc)
        q.add(filter_name)
        q.add(filter_loc)
        results = q(ei_db_data)
        if len(results) == 1:
            known_inputs_network[(loc, name)] = list(results).pop()
        else:
            print("Couldnt find activity:" + name)
            missing_inputs.append((loc, name, len(results)))

    if strict and missing_inputs:
        details = "; ".join(
            f"{name}, {loc} ({matches} matches)"
            for loc, name, matches in missing_inputs
        )
        raise ValueError(
            f"Could not uniquely map required activities in {eidb_name!r}: "
            + details
        )

    return known_inputs, known_inputs_network


def get_network_activities(eidb_name):
    """Return fixed electricity-network exchanges for a background version.

    Args:
        eidb_name: Background database name. Version markers in the name select
            the locations used by newer ecoinvent releases.

    Returns:
        List of dictionaries containing activity name, location, and exchange
        amount per kilowatt hour of supplied electricity.
    """
    activities = [
        "market for distribution network, electricity, low voltage",
        "market for transmission network, electricity, medium voltage",
        "market for sulfur hexafluoride, liquid",
        "market for transmission network, electricity, high voltage direct current aerial line",
        "market for transmission network, electricity, high voltage direct current land cable",
        "market for transmission network, electricity, high voltage direct current subsea cable",
        "transmission network construction, electricity, high voltage",
    ]
    if any(v in eidb_name for v in ("3.10", "3.11", "3.12")):
        locations = [
            "GLO",
            "GLO",
            "RER",
            "RER",
            "RER",
            "RER",
            "CH",
        ]
    else:
        locations = [
            "GLO",
            "GLO",
            "RER",
            "GLO",
            "GLO",
            "GLO",
            "CH",
        ]
    values = [
        8.679076855e-8,
        1.86646177072e-8,
        1.27657893204915e-7,
        8.38e-9,
        3.47e-10,
        5.66e-10,
        6.58e-9,
    ]
    network_act = [
        {"name": activity, "loc": location, "val": value}
        for activity, location, value in zip(activities, locations, values)
    ]
    return network_act


def get_country_network_activities(eidb_name, country):
    """Return the fixed network exchanges included for one consumer country."""
    specific_network = {
        "land": (
            "market for transmission network, electricity, high voltage "
            "direct current land cable"
        ),
        "subsea": (
            "market for transmission network, electricity, high voltage "
            "direct current subsea cable"
        ),
        "construction": (
            "transmission network construction, electricity, high voltage"
        ),
    }
    land_cable_countries = {
        "AT",
        "BE",
        "DK",
        "EE",
        "ES",
        "FI",
        "FR",
        "GR",
        "HR",
        "IT",
        "LT",
        "LV",
        "ME",
        "MT",
        "NO",
        "SE",
        "TR",
        "UK",
    }
    subsea_cable_countries = {
        "DK",
        "EE",
        "ES",
        "FI",
        "FR",
        "GR",
        "HR",
        "IT",
        "LT",
        "ME",
        "MT",
        "NO",
        "PL",
        "UK",
    }
    selected = []
    for exchange in get_network_activities(eidb_name):
        name = exchange["name"]
        if name not in specific_network.values():
            selected.append(exchange)
        elif name == specific_network["land"] and country in land_cable_countries:
            selected.append(exchange)
        elif name == specific_network["subsea"] and country in subsea_cable_countries:
            selected.append(exchange)
        elif name == specific_network["construction"] and country == "CH":
            selected.append(exchange)
    return selected


def create_activity_dict(
    dataframe_filt,
    known_inputs,
    known_inputs_network,
    db_name,
    eidb_name=None,
    year=None,
    consumption_profile=None,
    inventory_resolution=None,
):
    """
    Creates a dictionary of activities for the BW database based on the filtered dataframe and known inputs.

    Args:
        dataframe_filt (pd.DataFrame): The filtered dataframe containing technology data.
        known_inputs (dict): A dictionary mapping known inputs to ecoinvent database entries.
        known_inputs_network (dict): A dictionary mapping known network inputs to ecoinvent database entries.
        db_name (str): The name of the BW database.
        eidb_name (str): Background database name used to select version-specific
            network activities. Defaults to ``db_name`` for backwards compatibility.
        year (int, optional): Model year stored on each activity and included in
            its name. If a time-indexed column already contains the year, the
            name is left unchanged to avoid repeating it.
        consumption_profile (str, optional): Temporal weighting represented by
            the foreground inventory, stored as activity metadata.
        inventory_resolution (str, optional): ``"annual"`` or ``"monthly"``.
            When omitted, time-indexed columns are treated as monthly and
            country-only columns as annual for compatibility.

    Returns:
        dict: A dictionary containing activities to be written to the BW2 database.
    """
    activities = {}
    activity_year = int(year) if year is not None else None
    resolution = (
        validate_inventory_resolution(inventory_resolution)
        if inventory_resolution is not None
        else ("monthly" if dataframe_filt.columns.nlevels > 1 else "annual")
    )
    for i, col in enumerate(dataframe_filt.columns):
        if dataframe_filt.columns.nlevels > 1:
            time, country = col
            period = _format_inventory_period(time, resolution)
        else:
            country = col
            period = str(activity_year) if activity_year is not None else None
        name = f"Electricity mix in {country}"
        if period is not None:
            name = f"{name}, {period}"
        code = f"electricity {i}"
        bd_version = bd.__version__
        if not isinstance(bd_version, str):
            bd_version = ".".join(map(str, bd_version))
        BW2 = vparse(bd_version) < vparse("4")
        if BW2:
            act_type = "process"
            prod_exchange_type = "production"
        else:
            act_type = bd.labels.process_node_default
            prod_exchange_type = bd.labels.production_edge_default
        act = {
            "name": name,
            "unit": "kWh",
            "code": code,
            "location": str(country),
            "reference product": "Electricity mix",
            "type": act_type,
            "inventory_resolution": resolution,
            "comment": _inventory_comment(
                resolution,
                period,
                consumption_profile,
            ),
            "exchanges": [],
        }
        if activity_year is not None:
            act["year"] = activity_year
        if consumption_profile is not None:
            act["consumption_profile"] = str(consumption_profile)
        # Add the production exchange
        act["exchanges"].append(
            {
                "input": (db_name, code),
                "name": name,
                "location": str(country),
                "unit": "kWh",
                "amount": 1,
                "type": prod_exchange_type,
            }
        )
        for idx in dataframe_filt.index:
            source, exch_name, prod, unit = idx
            if float(dataframe_filt.loc[idx, col]) != 0:
                exchange = known_inputs.get((source, exch_name, unit))
                if exchange:
                    new_exchange = {
                        "input": exchange,
                        "amount": float(dataframe_filt.loc[idx, col]),
                        "type": "technosphere",
                    }
                    act["exchanges"].append(new_exchange)
        for exch_net in get_country_network_activities(eidb_name or db_name, country):
            exchange = known_inputs_network.get((exch_net["loc"], exch_net["name"]))
            if exchange:
                act["exchanges"].append(
                    {
                        "input": exchange,
                        "amount": float(exch_net["val"]),
                        "type": "technosphere",
                    }
                )
        activities[(db_name, code)] = act
    return activities


def create_database(
    dataframe_filt,
    project_name,
    db_name,
    eidb_name,
    network=True,
    strict=False,
    year=None,
    consumption_profile=None,
    inventory_resolution=None,
):
    """
    Creates an "ecoinvent-like" BW database based on a previously filtered dataframe.

    Args:
        dataframe_filt (pd.DataFrame): Scaled and filtered dataframe.
        project_name (str): BW project name to which the database will be saved.
        db_name (str): Name of the BW database to be created.
        eidb_name (str): Name of the ecoinvent or premise background database.
            Must be the same as in the BW project.
        network (bool): If True, network activities will be considered. The legacy
            strings ``"True"`` and ``"False"`` are also accepted.
        strict (bool): Raise before writing if any required background activity
            cannot be matched uniquely.
        year (int, optional): Model year included in foreground activity names
            and stored as activity metadata.
        consumption_profile (str, optional): Temporal weighting stored as
            foreground activity metadata.
        inventory_resolution (str, optional): Temporal resolution stored in
            activity names, metadata, and comments.

    Returns:
        None
    """
    include_network = network is True or (
        isinstance(network, str) and network.lower() == "true"
    )
    bd.projects.set_current(project_name)
    known_inputs, known_inputs_network = map_known_inputs(
        eidb_name,
        dataframe_filt,
        strict=strict,
        include_network=include_network,
    )
    activities = create_activity_dict(
        dataframe_filt,
        known_inputs,
        known_inputs_network,
        db_name,
        eidb_name=eidb_name,
        year=year,
        consumption_profile=consumption_profile,
        inventory_resolution=inventory_resolution,
    )
    elec_db = setup_database(project_name, db_name)
    elec_db.write(activities)


def _format_inventory_period(time, inventory_resolution):
    """Format a table-column timestamp for an activity name."""
    timestamp = pd.Timestamp(time)
    if inventory_resolution == "monthly":
        return timestamp.strftime("%Y-%m")
    return str(timestamp.year)


def _inventory_comment(
    inventory_resolution,
    period,
    consumption_profile,
):
    """Build ActivityBrowser documentation for a foreground inventory."""
    parts = [
        "SHRECC electricity consumption mix.",
        f"Inventory resolution: {inventory_resolution}.",
    ]
    if period is not None:
        parts.append(f"Inventory period: {period}.")
    if consumption_profile is not None:
        parts.append(f"Consumption profile: {consumption_profile}.")
    return " ".join(parts)
