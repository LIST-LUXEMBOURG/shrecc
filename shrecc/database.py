# Copyright © 2024 Luxembourg Institute of Science and Technology
# Licensed under the MIT License (see LICENSE file for details).
# Authors: [Sabina Bednářová, Thomas Gibon]

from datetime import datetime
from importlib.resources import files
from pathlib import Path

import bw2data as bd
import numpy as np
import pandas as pd
from bw2data.query import Filter, Query

from shrecc.treatment import load_from_pickle, save_to_pickle


def filt_cutoff(
    countries,
    times=0,
    general_range=0,
    refined_range=0,
    freq=0,
    cutoff=1e-3,
    include_cutoff=True,
    path_to_data=None,
):
    """
    Filters data based on selected countries and times (either one-off, a range, or periodical range). Gets called from `create_database()`.

    Args:
        year (int): Selected year of the downloaded data.
        countries (list of str): Countries selected by the user for their database.
            E.g. countries=['FR', 'DE'].
        times (list of str): Selecting one specific time, e.g. times = ['2023-06-16 8:00:00', '2023-06-16 22:00:00'].
            Can be applied alone.
        general_range (list of str): Selecting a general range, e.g. for the month of June
            general_range = ['2023-06-01 01:00:00', '2023-06-30 23:00:00']. Can be applied alone.
        refined_range (list of int): Refining range of general range, e.g. mornings of June (previously selected in general_range):
            refined_range = [8, 9, 10, 11]. Can only be applied with general_range.
        freq (str): Days to be included, e.g. freq='D' selects calendar days,
            see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
        cutoff (float): Cutoff value for technology values.
        include_cutoff (bool): If True, cutoff is applied and summed at the end to create a new technology "The rest".
            If False, cutoff is applied but new technology not created.
        path_to_data (Path): location of the data. If none, the data is taken from within the package.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    if path_to_data:
        print(f"Using mapping root: {path_to_data}")
        path_to_data = Path(path_to_data)
    else:
        path_to_data = files("shrecc.data")
    year = datetime.strptime(general_range[0], "%Y-%m-%d %H:%M:%S").year
    dataframe = tech_mapping(year, path_to_data)
    print("Filtering dataframe...")
    dataframe = dataframe.droplevel("source", axis=1)
    dataframe = filter_by_countries(dataframe, countries)
    if times:
        dataframe = filter_by_times(dataframe, times)
    if general_range:
        dataframe = filter_by_range(dataframe, general_range, refined_range, freq)
    dataframe = apply_cutoff(dataframe, cutoff, include_cutoff)
    print("Dataframe filtered.")
    return dataframe


def load_mapping_data(data_dir):
    """
    Load the mapping data from an Excel file.

    Args:
        data_dir (str or Path): The path to the directory containing the mapping data.

    Returns:
        pd.DataFrame: A DataFrame containing the technology mapping data from the Excel file.
    """
    return pd.read_csv(
        data_dir / "el_map_all_norm.csv",
        index_col=[0, 1, 2, 3],
        header=[0, 1],
    )


def load_time_series_data(path_to_data, year):
    """
    Load the time series data from a pickle file and format it as a DataFrame.

    Args:
        path_to_data (str or Path): The path to the directory containing the time series data.
        year (int): The year corresponding to the time series data.

    Returns:
        pd.DataFrame: A DataFrame containing the time series data, with levels reordered and sorted.
    """
    filename = path_to_data / f"{year}" / f"indices_{year}.pkl"
    indices = load_from_pickle(filename)
    Z_cons_sp = load_from_pickle(path_to_data / f"{year}" / f"Z_cons_{year}.pkl")
    Z_cons = pd.DataFrame(
        np.float32(Z_cons_sp.todense()),
        index=indices["index"],
        columns=indices["columns"],
    )
    return Z_cons.reorder_levels(["time", "source", "country"], axis=1).sort_index(
        axis=1
    )


def prepare_consumption_data(Z_cons):
    """
    Prepare the consumption data by removing trade data and adjusting indices.

    Args:
        Z_cons (pd.DataFrame): The original consumption data DataFrame.

    Returns:
        pd.DataFrame: The prepared consumption data, with the trade data removed and indices swapped.
    """
    Z_cons = Z_cons.sort_index()
    Z_cons_to_multiply = Z_cons.drop("trade", axis=0).copy()
    Z_cons_to_multiply.index.names = ["source", "geography_mix"]
    return Z_cons_to_multiply.swaplevel()


def apply_mapping(Z_cons_to_multiply, el_map_all_norm):
    """
    Apply the technology mapping to the consumption data.

    Args:
        Z_cons_to_multiply (pd.DataFrame): The consumption data to be mapped.
        el_map_all_norm (pd.DataFrame): The normalized mapping data.

    Returns:
        pd.DataFrame: The resulting DataFrame after applying the technology mapping.
    """

    # Changing data types to float32 significantly reduces the product time
    el_map_to_multiply = el_map_all_norm.reindex(
        Z_cons_to_multiply.index, axis=1
    ).astype("float32")
    el_map_to_multiply["NIE"] = el_map_all_norm["GB"][
        el_map_to_multiply["NIE"].columns
    ].astype("float32")
    el_map_to_multiply["UK"] = el_map_all_norm["GB"][
        el_map_to_multiply["UK"].columns
    ].astype("float32")

    LCI_cons = el_map_to_multiply.dot(Z_cons_to_multiply)

    if LCI_cons.isna().sum().sum():
        return LCI_cons.fillna(0)

    return LCI_cons


def tech_mapping(year, path_to_data):
    """
    Main function to map the technologies and scale them to 1 kWh.

    Args:
        year (int): The year corresponding to the data.
        path_to_data (path): Root directory of the mapping data.

    Returns:
        pd.DataFrame: A DataFrame with the scaled technology mappings.
    """
    print("Mapping technologies...")
    data_dir = files("shrecc.data")
    el_map_all_norm = load_mapping_data(data_dir)
    Z_cons = load_time_series_data(path_to_data, year)
    Z_cons_to_multiply = prepare_consumption_data(Z_cons)
    filename = Path(path_to_data / f"{year}" / f"LCI_cons_scaled_{year}.pkl")
    if filename.exists():
        LCI_cons_scaled = load_from_pickle(filename)
    else:
        LCI_cons = apply_mapping(Z_cons_to_multiply, el_map_all_norm)
        sum = LCI_cons.sum()
        filename = Path(path_to_data / f"{year}" / f"Z_load_{year}.pkl")
        load = load_from_pickle(filename)
        load_difference_row = (
            "RER",
            "electricity, high voltage, European attribute mix",
            "electricity, high voltage",
            "kWh",
        )

        load_stacked = load.stack()
        load_stacked.index.names = ["country", "time"]
        load_stacked = load_stacked[load_stacked != 0]
        merged = load_stacked.reset_index(name="load_value").merge(
            sum.reset_index(), how="inner", on=["time", "country"]
        )
        merged.rename(columns={0: "sum"}, inplace=True)
        merged["difference"] = merged["load_value"] - merged["sum"]
        merged = merged[merged["difference"] > 0]
        merged.set_index(["time", "source", "country"], inplace=True)

        LCI_cons.sort_index(inplace=True)
        LCI_cons.sort_index(axis=1, inplace=True)
        LCI_cons.loc[load_difference_row] = merged["difference"]
        LCI_cons.loc[load_difference_row].fillna(0, inplace=True)
        LCI_cons_scaled = LCI_cons / LCI_cons.sum()
        save_to_pickle(
            LCI_cons_scaled,
            Path(path_to_data / f"{year}" / f"LCI_cons_scaled_{year}.pkl"),
        )
    print("Technologies mapped.")
    return LCI_cons_scaled


def filter_by_countries(dataframe, countries):
    """
    Filter the dataframe by selected countries.

    Args:
        dataframe (pd.DataFrame): The original dataframe containing data for multiple countries.
        countries (list of str): A list of country codes to filter by.

    Returns:
        pd.DataFrame: A dataframe filtered by the specified countries.
    """
    if "country" not in dataframe.columns.names:
        print("Couldnt find country to filter")
        return

    return dataframe.loc[
        :, dataframe.columns.get_level_values("country").isin(countries)
    ]


def filter_by_times(dataframe, times):
    """
    Filter the dataframe by specific times.

    Args:
        dataframe (pd.DataFrame): The original dataframe containing data for multiple times.
        times (list of str): A list of specific times to filter by.

    Returns:
        pd.DataFrame: A dataframe filtered by the specified times.
    """
    return dataframe.loc[:, dataframe.columns.get_level_values("time").isin(times)]


def filter_by_range(dataframe, general_range, refined_range, freq):
    """
    Filter the dataframe by a general time range and optionally by a refined time range.

    Args:
        dataframe (pd.DataFrame): The original dataframe containing data.
        general_range (list of str): The start and end of the general range to filter by (e.g., ['2023-06-01', '2023-06-30']).
        refined_range (list of int): A list specifying the refined range of hours to filter within the general range.
        freq (str): The frequency for generating timestamps (e.g., 'D' for daily).

    Returns:
        pd.DataFrame: A dataframe filtered by the specified time range and refined range.
    """
    df_filt = dataframe.loc[:, general_range[0] : general_range[1]]
    if refined_range and len(refined_range) > 1:
        timestamp = pd.date_range(
            start=general_range[0], end=general_range[1], freq=freq, normalize=True
        ) + pd.Timedelta(hours=refined_range[0])
        timestamps_range = timestamp[
            (timestamp.hour >= refined_range[0]) & (timestamp.hour <= refined_range[1])
        ]
        df_filt = df_filt.loc[
            :, df_filt.columns.get_level_values("time").isin(timestamps_range)
        ]
    elif refined_range and len(refined_range) == 1:
        timestamp = pd.date_range(
            start=general_range[0], end=general_range[1], freq=freq, normalize=True
        ) + pd.Timedelta(hours=refined_range[0])
        df_filt = df_filt.loc[
            :, df_filt.columns.get_level_values("time").isin(timestamp)
        ]
    return df_filt.T.groupby(level="country").mean().T


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
    sums = []
    for col in df_filt.columns:
        select = df_filt.loc[:, col]
        sum_smaller_than_cutoff = select[select.le(cutoff)].sum()
        sums.append(sum_smaller_than_cutoff)
    df_filt[df_filt < cutoff] = 0
    if include_cutoff:
        df_filt.loc[
            (
                "RER",
                "market group for electricity, high voltage",
                "electricity, high voltage",
                "kWh",
            ),
            :,
        ] = sums
    return df_filt


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


def map_known_inputs(eidb_name, dataframe_filt):
    """
    Maps known inputs from the ecoinvent database to the filtered dataframe.

    Args:
        eidb_name (str): The name of the ecoinvent database in the BW project.
        dataframe_filt (pd.DataFrame): The filtered dataframe containing technology data.

    Returns:
        dict: A dictionary mapping known inputs to their corresponding entries in the ecoinvent database.
    """
    ei_db = bd.Database(eidb_name)
    ei_db_data = ei_db.load()
    known_inputs = {}
    for idx in dataframe_filt.index:
        loc, name, prod, unit = idx
        # Region names in ENTSOE and ecoinvent don't exactly match
        # Only UK seems concerned but consider using a dictionary
        if loc == "UK":
            loc = "GB"
        q = Query()
        filter_name = Filter("name", "is", name)
        filter_loc = Filter("location", "is", loc)
        filter_unit = Filter("unit", "is", "kilowatt hour")
        q.add(filter_name)
        q.add(filter_loc)
        q.add(filter_unit)
        results = q(ei_db_data)
        if len(results) == 1:
            known_inputs[(loc, name, unit)] = list(results).pop()
        else:
            print("Couldnt find activity:" + name + ", " + loc)
    network = get_network_activities(eidb_name)
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
    return known_inputs, known_inputs_network


def get_network_activities(eidb_name):
    activities = [
        "market for distribution network, electricity, low voltage",
        "market for transmission network, electricity, medium voltage",
        "market for sulfur hexafluoride, liquid",
        "market for transmission network, electricity, high voltage direct current aerial line",
        "market for transmission network, electricity, high voltage direct current land cable",
        "market for transmission network, electricity, high voltage direct current subsea cable",
        "transmission network construction, electricity, high voltage",
    ]
    if "3.11" in eidb_name:
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


def create_activity_dict(dataframe_filt, known_inputs, known_inputs_network, db_name):
    """
    Creates a dictionary of activities for the BW database based on the filtered dataframe and known inputs.

    Args:
        dataframe_filt (pd.DataFrame): The filtered dataframe containing technology data.
        known_inputs (dict): A dictionary mapping known inputs to ecoinvent database entries.
        db_name (str): The name of the BW database.

    Returns:
        dict: A dictionary containing activities to be written to the BW2 database.
    """
    activities = {}
    for i, col in enumerate(dataframe_filt.columns):
        if dataframe_filt.columns.nlevels > 1:
            time, country = col
            activity = f"{time} Electricity mix"
            name = f"{activity} in {country}"
        else:
            country = col
            name = f"Electricity mix in {country}"
        code = f"electricity {i}"
        act = {
            "name": name,
            "unit": "kWh",
            "code": code,
            "location": str(country),
            "reference product": "Electricity mix",
            "type": "process",
            "exchanges": [],
        }
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
        network = get_network_activities(db_name)
        specific_network = [
            "market for transmission network, electricity, high voltage direct current land cable",
            "market for transmission network, electricity, high voltage direct current subsea cable",
            "transmission network construction, electricity, high voltage",
        ]
        land_cable = [
            "FR",
            "IT",
            "GR",
            "DK",
            "AT",
            "BE",
            "EE",
            "ES",
            "FI",
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
        ]
        subsea_cable = [
            "FR",
            "IT",
            "GR",
            "DK",
            "EE",
            "ES",
            "FI",
            "HR",
            "IT",
            "LT",
            "ME",
            "MT",
            "NO",
            "PL",
            "UK",
        ]
        for exch_net in network:
            if exch_net["name"] not in specific_network:
                exchange = known_inputs_network.get((exch_net["loc"], exch_net["name"]))
                if exchange:
                    new_exchange = {
                        "input": exchange,
                        "amount": float(exch_net["val"]),
                        "type": "technosphere",
                    }
                    act["exchanges"].append(new_exchange)
            if (
                country in land_cable and exch_net["name"] == specific_network[0]
            ):  # writing the land cable
                exchange = known_inputs_network.get((exch_net["loc"], exch_net["name"]))
                if exchange:
                    new_exchange = {
                        "input": exchange,
                        "amount": float(exch_net["val"]),
                        "type": "technosphere",
                    }
                    act["exchanges"].append(new_exchange)
            if (
                country in subsea_cable and exch_net["name"] == specific_network[1]
            ):  # writing the subsea cable
                exchange = known_inputs_network.get((exch_net["loc"], exch_net["name"]))
                if exchange:
                    new_exchange = {
                        "input": exchange,
                        "amount": float(exch_net["val"]),
                        "type": "technosphere",
                    }
                    act["exchanges"].append(new_exchange)
            if country == "CH" and exch_net["name"] == specific_network[2]:
                exchange = known_inputs_network.get((exch_net["loc"], exch_net["name"]))
                if exchange:
                    new_exchange = {
                        "input": exchange,
                        "amount": float(exch_net["val"]),
                        "type": "technosphere",
                    }
                    act["exchanges"].append(new_exchange)
        activities[(db_name, code)] = act
    return activities


def create_database(dataframe_filt, project_name, db_name, eidb_name, network="True"):
    """
    Creates an "ecoinvent-like" BW database based on a previously filtered dataframe.

    Args:
        dataframe_filt (pd.DataFrame): Scaled and filtered dataframe.
        project_name (str): BW project name to which the database will be saved.
        db_name (str): Name of the BW database to be created.
        eidb_name (str): Name of the ecoinvent database. Must be the same as in the BW project.

    Returns:
        None
    """
    elec_db = setup_database(project_name, db_name)
    if network == "True":
        known_inputs, known_inputs_network = map_known_inputs(eidb_name, dataframe_filt)
        activities = create_activity_dict(
            dataframe_filt, known_inputs, known_inputs_network, db_name
        )
    else:
        known_inputs, _ = map_known_inputs(eidb_name, dataframe_filt)
        activities = create_activity_dict(dataframe_filt, known_inputs, _, db_name)
    elec_db.write(activities)
