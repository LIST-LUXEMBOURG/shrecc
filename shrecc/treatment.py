# Copyright © 2024 Luxembourg Institute of Science and Technology
# Licensed under the MIT License (see LICENSE file for details).
# Authors: [Sabina Bednářová, Thomas Gibon]

import pickle
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pypardiso import spsolve
from scipy.linalg import block_diag
from scipy.sparse import coo_array, csc_matrix, csr_matrix, identity


def data_processing(data_df, year, path_to_data=None):
    """
    Processes data, adds missing countries, and correctly divides them between consumption and demand.

    Args:
        data_df (pd.DataFrame): All of the downloaded data for the selected year (from `get_data`).
        year (int): The selected year, e.g., 2023.
        path_to_data (Path): location of the data.

    Returns:
        None: Doesn't return anything, but saves files to the directory 'data'.
    """
    if path_to_data is None:
        data_dir = files("shrecc.data")
    elif isinstance(path_to_data, str):
        data_dir = Path(path_to_data)
    print("Processing data...")

    # If pandas is recent enough, use the future_stack argument in stack
    params = {"level": 0}
    if pd.__version__ >= "2.1.0":
        params.update({"future_stack": True})

    Z = (
        data_df.T.unstack("country")
        .drop(
            [
                col
                for col in [("load", "Load"), ("Load", "Load")]
                if col in data_df.T.index
            ],
            errors="ignore",
        )
        .fillna(0)
        .stack(**params)
    )
    Z_trade = pd.concat(
        [Z.loc["trade"]], keys=["trade"], names=["source", "country", "time"]
    )

    missing_countries = set(Z.loc["trade"].columns.values) ^ set(
        Z.loc["trade"].index.get_level_values(0).unique()
    )
    missing_countries.add("CY")
    missing_index = missing_countries - set(Z_trade.index.get_level_values(1).unique())
    missing_columns = missing_countries - set(Z_trade.columns)

    Z_trade[list(missing_columns)] = 0
    t_index = Z.index.levels[2]
    n_t = len(t_index)
    n_c = len(Z_trade.columns)

    to_append_index = pd.DataFrame(
        np.zeros([len(missing_index) * n_t, n_c]),
        index=pd.MultiIndex.from_product([["trade"], missing_index, t_index]),
        columns=Z_trade.columns,
    )

    Z_trade = (
        pd.concat([Z_trade, to_append_index], axis=0).sort_index().sort_index(axis=1)
    )
    Z_trade_sq = Z_trade.unstack()
    Z_trade_sq.index.names = ["source", "country"]
    Z_trade_sq.columns.names = ["country", "time"]
    cols_trade = Z_trade_sq.columns

    Z[list(missing_columns)] = 0
    c_index = Z_trade.columns
    p_index = (
        Z.loc["production mix"]
        .drop(["Import balance (market)"])
        .index.get_level_values(0)
        .unique()
    )  # Sometimes,
    # 'Day Ahead Auction' needs to be dropped as well.
    n_p = len(p_index)
    Zu = Z.unstack()
    Z_prod_diag = pd.DataFrame(
        block_diag(
            *[
                Zu.loc["production mix", c].drop(["Import balance (market)"])
                for c in c_index
            ]
        ),
        index=pd.MultiIndex.from_product([c_index, p_index]),
        columns=cols_trade,
    ).swaplevel(0, 1)
    Z_trade_sq[Z_trade_sq < 0] = 0
    Zx = pd.concat(
        [pd.concat([Z_prod_diag, Z_trade_sq], axis=0)],
        keys=["trade"],
        names=["source"],
        axis=1,
    )
    Z_net = Zx.reorder_levels(["time", "source", "country"], axis=1).sort_index(axis=1)

    Z_indices = {key: getattr(Z_net, key) for key in ["index", "columns"]}
    hydro_pumped = (
        Z_net[(Z_net < 0).any(axis=1)]
        .droplevel("source", axis=0)
        .pipe(lambda df: df.droplevel(["source", "country"], axis=1))
        .loc[:, lambda df: ~(df == 0).all(axis=0)]
        .T.groupby(level=0)
        .sum()
    )
    Z_load = (
        Z_net.sum()
        .unstack(level="time")
        .sub(Z_net.loc["trade"].sum().unstack(level="time"))
        .droplevel("source")
        .sub(hydro_pumped.T, fill_value=0)
    )
    save_to_pickle(Z_load, data_dir / f"{year}" / f"Z_load_{year}.pkl")
    save_to_pickle(Z_indices, data_dir / f"{year}" / f"indices_{year}.pkl")
    save_to_pickle(Z_net, data_dir / f"{year}" / f"Z_net_{year}.pkl")
    print("Treating data:")
    treating_data(year, n_c, n_p, t_index, Z_net, data_dir)
    print("..all done!")


def save_to_pickle(obj, filename):
    """
    Saves an object to a pickle file.

    Args:
        obj: The object to be saved.
        filename (Path): Path to the filename where the object will be saved.

    Returns:
        None
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(filename):
    """
    Loads an object from a pickle file.

    Args:
        filename (Path): Path to the filename where the object will be saved.

    Returns:
        object: The object loaded from the pickle file.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def add_missing_elements(df, existing_elements, all_elements, axis=0, fill_value=0):
    """
    Adds missing elements (rows or columns) to a dataframe.

    Args:
        df (pd.DataFrame): The dataframe to which elements will be added.
        existing_elements (set): The set of existing elements in the dataframe.
        all_elements (set): The set of all possible elements.
        axis (int): The axis along which to add missing elements (0 for index, 1 for columns).
        fill_value: The value to fill for the missing elements.

    Returns:
        pd.DataFrame: The dataframe with missing elements added.
    """
    missing_elements = all_elements - existing_elements
    for element in missing_elements:
        if axis == 0:
            df.loc[element] = fill_value
        else:
            df[element] = fill_value
    if axis == 0:
        df.index = pd.Index(df.index, dtype=object)
    else:
        df.columns = pd.Index(df.columns, dtype=object)

    return df.sort_index(
        axis=axis, key=lambda x: x.astype(str) if x.dtype == object else x
    )


def create_block_diagonal_matrix(data_list):
    """
    Creates a block diagonal matrix from a list of data arrays.

    Args:
        data_list (list): A list of data arrays to be combined into a block diagonal matrix.

    Returns:
        pd.DataFrame: A block diagonal matrix containing the input data.
    """
    return pd.DataFrame(block_diag(*data_list))


def process_matrix(df, operation, axis=1, **kwargs):
    """
    Applies a specified operation to a dataframe or matrix.

    Args:
        df (pd.DataFrame): The dataframe or matrix to process.
        operation (str): The operation to perform (e.g., 'normalize', 'reorder_levels').
        axis (int): The axis along which the operation should be applied, if applicable.
        kwargs: Additional arguments for specific operations.

    Returns:
        pd.DataFrame: The processed dataframe or matrix.
    """
    if operation == "normalize":
        return df.div(df.sum(axis=axis), axis=axis).fillna(0)
    elif operation == "reorder_levels":
        return df.reorder_levels(kwargs["order"], axis=axis).sort_index(axis=axis)
    else:
        raise ValueError(f"Unknown operation: {operation}")


def treating_data(year, n_c, n_p, t_index, Z_net, data_dir):
    """
    Function called from `data_processing`. This includes heavy operations, so it's advised to run this part on a server.

    Args:
        year (int): The selected year, passed from `data_processing`.
        n_c (int): The number of all countries (including missing ones) in the dataframe, passed from `data_processing`.
        t_index ():
        data_dir (Path): location of the data.

    Returns:
        None: Doesn't return anything; everything gets saved to the directory 'data'.
    """
    Z_indices = load_from_pickle(data_dir / f"{year}" / f"indices_{year}.pkl")
    Z_load = load_from_pickle(data_dir / f"{year}" / f"Z_load_{year}.pkl")
    results, results_load = calculate_results(
        year, n_c, n_p, t_index, Z_net, Z_load, data_dir
    )
    filename = data_dir / f"{year}" / f"results_light_{year}.pkl"
    results_light = process_results_light(results, filename, n_c)

    filename = data_dir / f"{year}" / f"results_light_load_{year}.pkl"

    L = concatenate_results(results_light, Z_net)
    output = Z_net.sum().reorder_levels(["source", "country", "time"])
    L_series = process_matrix(
        L, "reorder_levels", order=["source", "country", "time"], axis=1
    )
    output = output.reindex(L_series.columns)
    filename = data_dir / f"{year}" / f"Z_cons_{year}.pkl"
    Z_cons = calculate_Z_cons(filename, L_series, output, Z_indices)  # noqa: F841
    save_to_pickle(results_load, data_dir / f"{year}" / f"Z_load_lv_{year}.pkl")


def calculate_results(year, n_c, n_p, t_index, Z_net, Z_load, data_dir):
    """
    Calculates results by inverting matrices over time.

    Args:
        year (int): The selected year.
        n_c (int): The number of countries.
        n_p (int): The number of production mix elements.
        t_index (pd.Index): The time index.
        Z_net (pd.DataFrame): The net consumption dataframe.
        Z_load (pd.DataFrame): The load dataframe.
        data_dir (Path): location of the data.

    Returns:
        dict: The results dictionary.
    """

    def process_time_series(data, I, t_index, filename, processor):
        if filename.exists():
            results = load_from_pickle(filename)
            print(f"Results loaded from {filename.name}")
        else:
            results = dict()
            month = None
            for t in t_index:
                if t.month != month:
                    month = t.month
                    print(f"month {month}/{t.year}")
                processor(data, t, I, results)
            save_to_pickle(results, filename)
        return results

    def process_case1(data, t, I, results):
        Z_t = data[t].reindex(data.index, axis=1, fill_value=0).values
        x_t = Z_t.sum(axis=0)
        x_t[x_t == 0] = 1  # prevent division by zero
        A_t_sparse = csr_matrix(Z_t / x_t)
        M = csr_matrix(I) - A_t_sparse
        X = spsolve(M, identity(I.shape[0], format="csc", dtype="float32"))
        results[t] = csr_matrix(X)

    def apply_load_losses(Z_load, loss_factor=0.965):
        """Applies a fixed loss factor to the entire Z_load matrix at once."""
        return Z_load * loss_factor

    # case 1: process Z_net
    filename = data_dir / f"{year}" / f"cons_results_{year}.pkl"
    I_net = np.eye((n_p + 1) * n_c)
    print("Solving exchange network graph")
    results = process_time_series(Z_net, I_net, t_index, filename, process_case1)
    print("Echange network graph solved")

    # case 2: process Z_load
    filename_load = data_dir / f"{year}" / f"load_results_{year}.pkl"
    print("Applying load losses")
    results_load = apply_load_losses(Z_load)
    save_to_pickle(results_load, filename_load)
    print("Load losses applied and saved")

    return results, results_load


def process_results_light(results, filename, n_c):
    """
    Processes the results to a lighter format.

    Args:
        results (dict): The results dictionary.
        n_c (int): The number of countries.

    Returns:
        dict: The light results dictionary.
    """
    if filename.exists():
        results_light = load_from_pickle(filename)
        print("Light results loaded")
    else:
        print("Results light computation started")
        results_light = {}

        for k, v in results.items():
            if isinstance(v, np.ndarray):
                results_light[k] = v.astype("float32")[:, -n_c:]
            else:
                results_light[k] = csc_matrix(v[:, -n_c:], dtype="float32")

        print("Results light computation finished")
    save_to_pickle(results_light, filename)
    return results_light


def concatenate_results(results, Z):
    """
    Concatenates results into a single DataFrame.

    Args:
        results_light (dict): The light results dictionary.
        Z (pd.DataFrame): The network dataframe.

    Returns:
        pd.DataFrame: The concatenated DataFrame.
    """
    sparse_matrices = [r for r in results.values()]
    L_sparse = sp.hstack(sparse_matrices)
    num_time_steps = len(results)
    num_sub_columns = L_sparse.shape[1] // num_time_steps
    time_labels = list(results.keys())
    multi_index = pd.MultiIndex.from_tuples(
        [(time, sub) for time in time_labels for sub in range(num_sub_columns)],
        names=["time", "index"],
    )
    L_df = pd.DataFrame.sparse.from_spmatrix(L_sparse, columns=multi_index)
    L_df.columns = Z.columns
    L_df.index = Z.index
    return L_df


def calculate_Z_cons(filename, L_series, output, Z_indices):
    """
    Calculates the consumption matrix Z_cons.

    Args:
        year (int): The selected year.
        L_series (pd.DataFrame): The L series DataFrame.
        output (pd.Series): The output series.
        Z_indices (dict): The Z indices dictionary.

    Returns:
        pd.DataFrame: The consumption matrix Z_cons.
    """
    L_array = L_series.to_numpy()
    output_array = output.to_numpy().reshape(1, -1)
    Z_cons_array = L_array * output_array
    Z_cons = pd.DataFrame(Z_cons_array, index=L_series.index, columns=L_series.columns)
    if filename.exists():
        Z_cons_sp = load_from_pickle(filename)
        Z_cons = pd.DataFrame(
            Z_cons_sp.todense(), index=Z_indices["index"], columns=Z_indices["columns"]
        )
    else:
        Z_cons_sp = coo_array(
            process_matrix(
                Z_cons, "reorder_levels", order=["time", "source", "country"], axis=1
            )
        )
        save_to_pickle(Z_cons_sp, filename)
    return Z_cons
