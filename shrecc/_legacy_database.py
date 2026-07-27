"""Historical file-based database-table preparation retained for compatibility."""

from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

from shrecc.mapping import load_ecoinvent_mapping as load_mapping_data
from shrecc.result_store import (
    load_pickle as load_from_pickle,
    save_pickle as save_to_pickle,
)


def load_time_series_data(path_to_data, year):
    """Load a legacy sparse consumption matrix and restore its labels."""
    path_to_data = Path(path_to_data)
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
    """Remove legacy trade rows and arrange rows for activity mapping."""
    Z_cons = Z_cons.sort_index()

    if "trade" in Z_cons.index.get_level_values("source"):
        Z_cons_to_multiply = Z_cons.drop("trade", axis=0).copy()
    else:
        Z_cons_to_multiply = Z_cons.copy()
    Z_cons_to_multiply.index.names = ["source", "geography_mix"]

    return Z_cons_to_multiply.swaplevel()


def apply_mapping(Z_cons_to_multiply, el_map_all_norm):
    """Apply the historical matrix-form technology mapping."""
    el_map_to_multiply = el_map_all_norm.reindex(
        Z_cons_to_multiply.index, axis=1
    ).astype("float32")
    el_map_to_multiply["NIE"] = el_map_all_norm["GB"][
        el_map_to_multiply["NIE"].columns
    ].astype("float32")
    el_map_to_multiply["UK"] = el_map_all_norm["GB"][
        el_map_to_multiply["UK"].columns
    ].astype("float32")

    el_map_to_multiply = el_map_to_multiply.fillna(0)
    LCI_cons = el_map_to_multiply.dot(Z_cons_to_multiply)
    return LCI_cons.fillna(0) if LCI_cons.isna().sum().sum() else LCI_cons


def tech_mapping(year, path_to_data, path_to_mapping=None):
    """Map a legacy consumption matrix and normalize each mix to one kWh."""
    now = datetime.now()
    print(f"{now} Mapping technologies...")
    if path_to_mapping:
        el_map_all_norm = load_mapping_data(path_to_mapping)
    else:
        el_map_all_norm = load_mapping_data(files("shrecc.data"))
    Z_cons = load_time_series_data(path_to_data, year)
    Z_cons_to_multiply = prepare_consumption_data(Z_cons)
    filename = Path(path_to_data / f"{year}" / f"LCI_cons_scaled_{year}.pkl")
    if filename.exists():
        LCI_cons_scaled = load_from_pickle(filename)
    else:
        LCI_cons = apply_mapping(Z_cons_to_multiply, el_map_all_norm)
        mapped_total = LCI_cons.sum()
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
            mapped_total.reset_index(), how="inner", on=["time", "country"]
        )
        merged.rename(columns={0: "sum"}, inplace=True)
        merged["difference"] = merged["load_value"] - merged["sum"]
        merged = merged[merged["difference"] > 0]
        merged.set_index(["time", "source", "country"], inplace=True)

        LCI_cons.sort_index(inplace=True)
        LCI_cons.sort_index(axis=1, inplace=True)
        LCI_cons.loc[load_difference_row] = merged["difference"]
        LCI_cons.loc[load_difference_row] = LCI_cons.loc[
            load_difference_row
        ].fillna(0)
        LCI_cons_scaled = LCI_cons / LCI_cons.sum()
        save_to_pickle(
            LCI_cons_scaled,
            Path(path_to_data / f"{year}" / f"LCI_cons_scaled_{year}.pkl"),
        )
    now = datetime.now()
    print(f"{now} Technologies mapped.")
    return LCI_cons_scaled


def filter_by_countries(dataframe, countries):
    """Filter a legacy activity table to selected consumer countries."""
    if "country" not in dataframe.columns.names:
        print("Couldnt find country to filter")
        return None

    return dataframe.loc[
        :, dataframe.columns.get_level_values("country").isin(countries)
    ]


def filter_by_times(dataframe, times):
    """Filter a legacy activity table to specific timestamps."""
    available_times = pd.to_datetime(dataframe.columns.get_level_values("time"))
    requested_times = pd.to_datetime(times)
    return dataframe.loc[:, available_times.isin(requested_times)]


def filter_by_range(dataframe, general_range, refined_range, freq):
    """Filter and aggregate a legacy activity table over a time range."""
    df_filt = dataframe.loc[:, general_range[0] : general_range[1]]
    if refined_range and len(refined_range) > 1:
        timestamp = pd.date_range(
            start=general_range[0], end=general_range[1], freq=freq
        )
        timestamps_range = timestamp[
            (timestamp.hour >= refined_range[0]) & (timestamp.hour <= refined_range[1])
        ]
        df_filt = df_filt.loc[
            :,
            pd.to_datetime(df_filt.columns.get_level_values("time")).isin(
                timestamps_range
            ),
        ]
    elif refined_range and len(refined_range) == 1:
        timestamp = pd.date_range(
            start=general_range[0], end=general_range[1], freq=freq
        )
        df_filt = df_filt.loc[
            :, pd.to_datetime(df_filt.columns.get_level_values("time")).isin(timestamp)
        ]
    return df_filt.T.groupby(level="country").mean().T
