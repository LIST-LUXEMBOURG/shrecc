# Copyright © 2024 Luxembourg Institute of Science and Technology
# Licensed under the MIT License (see LICENSE file for details).
# Authors: [Sabina Bednářová, Thomas Gibon]

import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path
import requests
import json
import os
import datetime
import time
import appdirs
import sys
import tempfile, subprocess
from shrecc.treatment import save_to_pickle, load_from_pickle


def get_prod(start, end, country, cumul=False, rolling=False):
    """
    Downloads production data from the Energy Charts API. Gets called from `get_data()`.

    Args:
        start (int): Start of the download period (output of `year_to_unix()`) in unix seconds.
        end (int): End of the download period (output of `year_to_unix()`) in unix seconds.
        country (list of str): The country for which data needs to be downloaded.
        cumul (bool): If True, calculate the cumulative sum of production.
        rolling (bool): If True, calculate the rolling average of production.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: A tuple containing:
            - A dataframe of production.
            - A dataframe of load.
            - An array of all available technologies.
    """
    s = requests.Session()
    url = f"https://api.energy-charts.info/public_power?country={country}&start={start}&end={end}"
    r = s.get(url)
    s.close()
    if r.status_code == 404:
        print("File not found!")
        return None, None, None
    if r.status_code == 400:
        print("Bad request! Country not found.")
        return None, None, None
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
    print("...production for " + country + " OK.")
    return prod_df, load_df, techs


def get_trade(start, end, country):
    """
    Downloads trade data from the Energy Charts API. Gets called from `get_data()`.

    Args:
        start (int): Start of the download period (output of `year_to_unix()`) in unix seconds.
        end (int): End of the download period (output of `year_to_unix()`) in unix seconds.
        country (list of str): The country for which data needs to be downloaded.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: A tuple containing:
            - A dataframe of trades between countries.
            - An array of all available regions.
    """
    s = requests.Session()
    url = (
        f"https://api.energy-charts.info/cbpf?country={country}&start={start}&end={end}"
    )
    r = s.get(url)
    s.close()
    if r.status_code == 404:
        print("File not found!")
        return None, None
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


def get_data(year, root=None, max_retries=3, retry_delay=5):
    """
    Main function for downloading data.

    Args:
        year (int): The selected year for which data is to be downloaded, e.g., 2023.
        root (Path): location of the data.
        max_retries (int): The maximum number of retries for each country download in case of problems.
        retry_delay (int): The delay in seconds between retries.

    Returns:
        pd.DataFrame: A dataframe containing both production and trade data for all countries in the selected year.
    """
    if root is None:
        package_name = sys.modules["__main__"].__package__
        if package_name is None:
            package_name = "shrecc"
        root = Path(appdirs.user_data_dir(package_name))
    elif isinstance(root, str):
        root = Path(root)
    countries = [
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
    ]
    filename = Path(root) / "data" / f"{year}" / f"prod_and_trade_data_{year}.pkl"
    filename.parent.mkdir(parents=True, exist_ok=True)
    start, end = year_to_unix(year)
    if filename.exists():
        data = load_from_pickle(filename)
        print("API data loaded successfully.")
    else:
        data = dict()
        for country in countries:
            country = country.lower()
            print(country)
            for attempt in range(max_retries):
                try:
                    prod_df, load_df, _ = get_prod(
                        start=start, end=end, country=country, cumul=False, rolling=1
                    )

                    trade_df, _ = get_trade(
                        start=start,
                        end=end,
                        country=country,
                    )

                    data[country] = {
                        "production mix": prod_df,
                        "load": load_df,
                        "trade": trade_df,
                    }
                    break
                except requests.ConnectionError as e:
                    print(
                        f"Network error: {e}, retrying {attempt + 1}/{max_retries}..."
                    )
                    time.sleep(retry_delay)
                except Exception as e:
                    print(f"Error: {e}, retrying {attempt + 1}/{max_retries}...")
                    time.sleep(retry_delay)
            else:
                print(f"Failed to fetch data for {country}.")
        save_to_pickle(data, filename)
    data_df = cleaning_data(data, root)
    return data_df


def year_to_unix(year):
    """
    Converts a year to Unix timestamps representing the start and end of the year.

    Args:
        year (int): The selected year, passed from `get_data()`.

    Returns:
        Tuple[int, int]: A tuple containing:
            - The start of the year in Unix seconds.
            - The end of the year in Unix seconds.
    """
    start_of_year = datetime.datetime(year, 1, 1, 0, 0)
    end_of_year = datetime.datetime(year, 12, 31, 23, 59)
    start_unix = int(time.mktime(start_of_year.timetuple()))
    end_unix = int(time.mktime(end_of_year.timetuple()))
    return start_unix, end_unix


def cleaning_data(data, root):
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
        except:
            pass
    filename = Path(root) / "data" / f"generation_units_by_country.csv"
    if filename.exists():
        gen_units_per_country = pd.read_csv(filename, index_col=1)["short"]
    country_codes = {
        p: gen_units_per_country.loc[p]
        for p in set(partners)
        if p in gen_units_per_country.index
    }
    missing_countries = set(partners) ^ country_codes.keys()
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
    data_clean = dict()
    filename = Path(root) / "data" / f"techs_agg.json"
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
        data_clean[country.upper()] = dict()
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


def download_shrecc_data(
    repo_url="https://git.list.lu/shrecc_project/shrecc_data/",
    destination_directory=None,
    branch="main",
) -> None:
    """
    Utility function to download the whole pre-calculate dataset.

    Args:
        repo_url (str): the default location of the repository with the data
        destination_directory: destination of data download
        branch (str): branch of the repository

    """
    clone_last_commit(repo_url, destination_directory, branch)


def clone_last_commit(repo_url, destination_directory=None, branch="main"):
    """
    Clone the latest commit from a specified branch of a Git repository.

    This function performs a shallow clone of the latest commit from a specified
    branch of a Git repository and copies the repository contents to the destination
    directory, excluding the `.git` directory, `.gitattributes`, and `README.md`.

    Args:
        repo_url (str): The URL of the Git repository.
        destination_directory (str): The directory where the repository contents will be copied.
                                     If not specified, it will be stored in the SHRECC module appdir.
        branch (str): The branch to clone. Defaults to 'main'.

    Raises:
        subprocess.CalledProcessError: If the git clone command fails.
    """
    if destination_directory is None:
        package_name = sys.modules["__main__"].__package__
        if package_name is None:
            package_name = "shrecc"
        destination_directory = appdirs.user_data_dir(package_name)

    os.makedirs(destination_directory, exist_ok=True)
    ignore_from_download = [".git", ".gitattributes", "README.md"]

    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    branch,
                    repo_url,
                    tmpdirname,
                ],
                check=True,
            )
            print(
                f"Repository cloned successfully into temporary directory '{tmpdirname}'."
            )
            print(f"Moving downloaded contents to '{destination_directory}'.")
            for item in os.listdir(tmpdirname):
                s = os.path.join(tmpdirname, item)
                d = os.path.join(destination_directory, item)
                if os.path.exists(s) and item in ignore_from_download:
                    continue
                if os.path.isdir(s):
                    if not os.path.exists(d):
                        shutil.copytree(s, d)
                    else:
                        print(f"Directory '{d}' already exists, skipping.")
                else:
                    if not os.path.exists(d):
                        shutil.copy2(s, d)
                    else:
                        print(f"File '{d}' already exists, skipping.")
            print(
                f"Repository contents copied successfully to '{destination_directory}'."
            )

        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository. Error: {e}")
            print("Please make sure you have git installed.")
