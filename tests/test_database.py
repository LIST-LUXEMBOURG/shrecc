import pytest
import pandas as pd
import numpy as np
from pytest import approx
from unittest.mock import patch, MagicMock
import types
from shrecc.database import (
    filt_cutoff,
    load_mapping_data,
    load_time_series_data,
    prepare_consumption_data,
    apply_mapping,
    tech_mapping,
    filter_by_countries,
    filter_by_times,
    filter_by_range,
    apply_cutoff,
    setup_database,
    map_known_inputs,
    get_network_activities,
    create_activity_dict,
    create_database,
)


# ─────────────────────────────────────────────────────────────
# Tests for: load_mapping_data() — requires multiple sub-tests
# ─────────────────────────────────────────────────────────────
def test_load_mapping_data_basic(tmp_path):
    # Create a dummy CSV file with multiindex and multiheader
    csv_content = (
        "0,,,,AL,AL,AM,AM\n"
        "1,,,,Biomass,Brown coal,Biomass,Brown coal\n"
        "geography,activityName,product,unit,,,,\n"
        'AT,"electricity production, deep geothermal","electricity, high voltage",kWh,0.1,0.2,0.3,0.4\n'
        'AT,"electricity production, hard coal","electricity, high voltage",kWh,0.5,0.6,0.7,0.8\n'
    )
    # Write CSV file
    csv_file = tmp_path / "el_map_all_norm.csv"
    with open(csv_file, "w") as f:
        f.write(csv_content)
    # Load the mapping data
    df = load_mapping_data(tmp_path)
    # Check index and columns
    assert isinstance(df, pd.DataFrame)
    assert df.index.names == ["geography", "activityName", "product", "unit"]
    assert isinstance(df.columns, pd.MultiIndex)
    # Check values
    assert df.shape[0] == 2
    assert df.shape[1] == 4
    assert float(df.iloc[0, 0]) == 0.1
    assert float(df.iloc[1, 2]) == 0.7
    # Check if the function works with path as string
    df_string = load_mapping_data(str(tmp_path))
    # Check if both methods return the same DataFrame
    assert df.equals(df_string)


def test_load_mapping_data_file_not_found(tmp_path):
    # No file present
    with pytest.raises(FileNotFoundError):
        load_mapping_data(tmp_path)


# ────────────────────────────────────────────────────────────────
# Tests for: load_time_series_data() — requires multiple sub-tests
# ────────────────────────────────────────────────────────────────
def test_load_time_series_data_basic(monkeypatch, tmp_path):
    # Prepare fake indices and Z_cons_sp
    year = 2023
    data_dir = tmp_path / str(year)
    data_dir.mkdir(parents=True)
    indices = {
        "index": pd.MultiIndex.from_tuples(
            [
                ("Hydro pumped", "CY"),
                ("Solar", "FR"),
                ("Wind", "DE"),
            ],
            names=["source", "country"],
        ),
        "columns": pd.MultiIndex.from_tuples(
            [
                ("2023-01-01 00:00:00", "trade", "DE"),
                ("2023-01-01 00:00:00", "trade", "FR"),
                ("2023-01-01 01:00:00", "trade", "DE"),
                ("2023-01-01 01:00:00", "trade", "FR"),
            ],
            names=["time", "source", "country"],
        ),
    }

    # Fake sparse matrix with .todense()
    class FakeSparse:
        def todense(self):
            return np.array(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                ],
                dtype=np.float64,
            )

    Z_cons_sp = FakeSparse()

    # Patch load_from_pickle to return indices and Z_cons_sp in order
    calls = []

    def mock_load_from_pickle(path):
        calls.append(str(path))
        if str(path).endswith("indices_2023.pkl"):
            return indices
        elif str(path).endswith("Z_cons_2023.pkl"):
            return Z_cons_sp
        else:
            raise ValueError("Unexpected file: " + str(path))

    monkeypatch.setattr("shrecc.database.load_from_pickle", mock_load_from_pickle)

    df = load_time_series_data(tmp_path, year)
    # Check DataFrame shape and content
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 4)
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.equals(indices["index"])
    assert isinstance(df.columns, pd.MultiIndex)
    assert df.columns.equals(indices["columns"])
    # Should be sorted by columns
    assert df.columns.get_level_values("time").is_monotonic_increasing
    # Values should match the fake matrix
    np.testing.assert_array_equal(
        df.values,
        np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
            dtype=np.float32,
        ),
    )

    # Check if the function works with path as string
    df_string = load_time_series_data(str(tmp_path), year)
    assert isinstance(df_string, pd.DataFrame)
    assert df_string.shape == (3, 4)
    assert df_string.iloc[0, 0] == 1.0


def test_load_time_series_data_file_not_found(monkeypatch, tmp_path):
    # Patch load_from_pickle to raise FileNotFoundError
    def fake_load_from_pickle(path):
        raise FileNotFoundError(str(path))

    monkeypatch.setattr("shrecc.database.load_from_pickle", fake_load_from_pickle)
    with pytest.raises(FileNotFoundError):
        load_time_series_data(tmp_path, 2023)


# ────────────────────────────────────────────────────────────────────
# Tests for: prepare_consumption_data() — requires multiple sub-tests
# ────────────────────────────────────────────────────────────────────
def test_prepare_consumption_data_basic():
    # Create a DataFrame with 'trade' in the index and other rows
    index = pd.MultiIndex.from_tuples(
        [
            ("trade", "FR"),
            ("Solar", "FR"),
            ("Wind", "FR"),
            ("trade", "DE"),
            ("Solar", "DE"),
            ("Wind", "DE"),
        ],
        names=["source", "country"],
    )
    columns = pd.MultiIndex.from_tuples(
        [
            ("2023-01-01 00:00:00", "trade", "DE"),
            ("2023-01-01 00:00:00", "trade", "FR"),
            ("2023-01-01 01:00:00", "trade", "DE"),
            ("2023-01-01 01:00:00", "trade", "FR"),
        ],
        names=["time", "source", "country"],
    )
    data = [
        [1, 2, 3, 4],  # trade row (should be dropped)
        [5, 6, 7, 8],  # Solar row
        [9, 10, 11, 12],  # Wind row
        [13, 14, 15, 16],  # trade row for DE (should be dropped)
        [17, 18, 19, 20],  # Solar row for DE
        [21, 22, 23, 24],  # Wind row for DE
    ]
    df = pd.DataFrame(data, index=index, columns=columns)

    # Call the function to prepare consumption data
    result = prepare_consumption_data(df)

    assert isinstance(result, pd.DataFrame)
    assert set(result.index.get_level_values(0)) == {"DE", "FR", "DE", "FR"}
    assert set(result.index.get_level_values(1)) == {"Solar", "Solar", "Wind", "Wind"}
    # 'trade' row should be gone
    assert "trade" not in result.index.get_level_values("source")

    expected_result = pd.DataFrame(
        [
            [17, 18, 19, 20],
            [5, 6, 7, 8],
            [21, 22, 23, 24],
            [9, 10, 11, 12],
        ],
        index=pd.MultiIndex.from_tuples(
            [
                ("DE", "Solar"),
                ("FR", "Solar"),
                ("DE", "Wind"),
                ("FR", "Wind"),
            ],
            names=["geography_mix", "source"],
        ),
        columns=columns,
    )
    pd.testing.assert_frame_equal(result, expected_result)


def test_prepare_consumption_data_no_trade_row():
    # DataFrame without 'trade' in index
    index_no_trade = pd.MultiIndex.from_tuples(
        [
            ("Solar", "FR"),
            ("Wind", "FR"),
        ],
        names=["source", "country"],
    )
    columns = pd.MultiIndex.from_tuples(
        [
            ("2023-01-01 00:00:00", "trade", "DE"),
            ("2023-01-01 00:00:00", "trade", "FR"),
            ("2023-01-01 01:00:00", "trade", "DE"),
            ("2023-01-01 01:00:00", "trade", "FR"),
        ],
        names=["time", "source", "country"],
    )
    data = [
        [5, 6, 7, 8],  # Solar row
        [9, 10, 11, 12],  # Wind row
    ]
    df = pd.DataFrame(data, index=index_no_trade, columns=columns)

    # Call the function to prepare consumption data
    result = prepare_consumption_data(df)

    expected_result = pd.DataFrame(
        [
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
        index=pd.MultiIndex.from_tuples(
            [
                ("FR", "Solar"),
                ("FR", "Wind"),
            ],
            names=["geography_mix", "source"],
        ),
        columns=columns,
    )
    pd.testing.assert_frame_equal(result, expected_result)


# ──────────────────────────────────────────────────────────
# Tests for: apply_mapping() — requires multiple sub-tests
# ──────────────────────────────────────────────────────────
@pytest.fixture
def create_two_dataframes():
    # Create a simple Z_cons_to_multiply DataFrame
    idx = pd.MultiIndex.from_tuples(
        [
            ("DE", "geothermal"),
            ("DE", "hard coal"),
            ("DE", "hydro"),
            ("NIE", "hydro"),
            ("NIE", "wind"),
            ("UK", "geothermal"),
            ("UK", "hydro"),
            ("GB", "hydro"),
        ],
        names=["geography_mix", "source"],
    )
    cols = pd.MultiIndex.from_tuples(
        [
            ("2023-01-01 00:00:00", "trade", "DE"),
            ("2023-01-01 00:00:00", "trade", "NIE"),
            ("2023-01-01 00:00:00", "trade", "UK"),
            ("2023-01-01 00:00:00", "trade", "GB"),
        ],
        names=["time", "source", "country"],
    )
    Z_cons_to_multiply = pd.DataFrame(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
            [25, 26, 27, 28],
            [29, 30, 31, 32],
        ],
        index=idx,
        columns=cols,
    )

    # Create el_map_all_norm with matching columns and MultiIndex columns for "GB"
    index = pd.MultiIndex.from_tuples(
        [
            (
                "DE",
                "electricity production, deep geothermal",
                "electricity, high voltage",
                "kWh",
            ),
            (
                "DE",
                "electricity production, hard coal",
                "electricity, high voltage",
                "kWh",
            ),
            ("DE", "electricity production, hydro", "electricity, high voltage", "kWh"),
            (
                "NIE",
                "electricity production, hydro",
                "electricity, high voltage",
                "kWh",
            ),
            ("NIE", "electricity production, wind", "electricity, high voltage", "kWh"),
            (
                "UK",
                "electricity production, deep geothermal",
                "electricity, high voltage",
                "kWh",
            ),
            ("UK", "electricity production, hydro", "electricity, high voltage", "kWh"),
            ("GB", "electricity production, hydro", "electricity, high voltage", "kWh"),
        ],
        names=["geography", "activityName", "product", "unit"],
    )
    columns = pd.MultiIndex.from_tuples(
        [
            ("DE", "geothermal"),
            ("DE", "hard coal"),
            ("DE", "hydro"),
            ("DE", "wind"),
            ("NIE", "geothermal"),
            ("NIE", "hard coal"),
            ("NIE", "hydro"),
            ("NIE", "wind"),
            ("UK", "geothermal"),
            ("UK", "hard coal"),
            ("UK", "hydro"),
            ("UK", "wind"),
            ("GB", "geothermal"),
            ("GB", "hard coal"),
            ("GB", "hydro"),
            ("GB", "wind"),
        ],
        names=[None, None],
    )
    el_map_all_norm = pd.DataFrame(
        [
            [
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                2.3,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                3.7,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.8,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                5.9,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                6.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                8.5,
                0.0,
            ],
        ],
        index=index,
        columns=columns,
    )
    return Z_cons_to_multiply, el_map_all_norm


def test_apply_mapping_basic(create_two_dataframes):
    """Test if apply_mapping works correctly with basic case,
    i.e., columns of el_map_all_norm should contain all indices of Z_cons_to_multiply
    """
    # Unpack the fixture
    Z_cons_to_multiply, el_map_all_norm = create_two_dataframes

    result = apply_mapping(Z_cons_to_multiply, el_map_all_norm)

    assert isinstance(result, pd.DataFrame)
    # Should have same index as el_map_to_multiply and columns as Z_cons_to_multiply
    assert set(result.index) == set(el_map_all_norm.index)
    assert set(result.columns) == set(Z_cons_to_multiply.columns)
    # Should not contain NaN (should be filled with 0 if any)
    assert not result.isnull().any().any()
    # Check if the values are correct
    assert result.loc[
        ("DE", "electricity production, hydro", "electricity, high voltage", "kWh"),
        ("2023-01-01 00:00:00", "trade", "NIE"),
    ] == approx(23.0)
    assert result.loc[
        ("UK", "electricity production, hydro", "electricity, high voltage", "kWh"),
        ("2023-01-01 00:00:00", "trade", "UK"),
    ] == approx(18.3)


def test_apply_mapping_nan_handling(create_two_dataframes):
    """Test if NaN values are handled correctly
    Special case: Z_cons_to_multiply contains an index not found in el_map_all_norm
    """
    Z_cons_to_multiply, el_map_all_norm = create_two_dataframes
    # Add an index to Z_cons_to_multiply that is not present in el_map_all_norm
    new_row = pd.Series(
        [1, 2, 3, 4], index=Z_cons_to_multiply.columns, name=("DE", "solar")
    )
    Z_cons_to_multiply = pd.concat(
        [Z_cons_to_multiply, new_row.to_frame().T]
    ).sort_index()

    result = apply_mapping(Z_cons_to_multiply, el_map_all_norm)

    assert isinstance(result, pd.DataFrame)
    assert (result.fillna(0) == result).all().all()
    # Check if the values are correct
    assert result.values.sum() == approx(2620.80)
    assert result.loc[
        ("DE", "electricity production, hydro", "electricity, high voltage", "kWh"),
        ("2023-01-01 00:00:00", "trade", "NIE"),
    ] == approx(23.0)


# ──────────────────────────────────────────────────────────
# Tests for: tech_mapping() — requires multiple sub-tests
# ──────────────────────────────────────────────────────────
def test_tech_mapping_computes_and_saves(monkeypatch, tmp_path):
    """Test if tech_mapping computes the expected DataFrame and saves it correctly"""
    # Setup
    year = 2023
    data_dir = tmp_path / str(year)
    data_dir.mkdir(parents=True)
    # Dummy el_map_all_norm, Z_cons, Z_cons_to_multiply, LCI_cons, load
    el_map_all_norm = MagicMock(name="el_map_all_norm")
    Z_cons = MagicMock(name="Z_cons")
    Z_cons_to_multiply = MagicMock(name="Z_cons_to_multiply")
    # Patch load_mapping_data, load_time_series_data, prepare_consumption_data
    monkeypatch.setattr("shrecc.database.load_mapping_data", lambda _: el_map_all_norm)
    monkeypatch.setattr(
        "shrecc.database.load_time_series_data", lambda path, year: Z_cons
    )
    monkeypatch.setattr(
        "shrecc.database.prepare_consumption_data", lambda _: Z_cons_to_multiply
    )
    # if the file exists, we should remove it to ensure we test the saving functionality
    scaled_path = data_dir / "LCI_cons_scaled_2023.pkl"
    if scaled_path.exists():
        scaled_path.unlink()
    # Patch apply_mapping to return a DataFrame with the expected structure
    # LCI_cons DataFrame with columns: MultiIndex with time, source, country
    columns = pd.MultiIndex.from_tuples(
        [
            ("2023-01-01 00:00:00", "trade", "FR"),
            ("2023-01-01 01:00:00", "trade", "FR"),
        ],
        names=["time", "source", "country"],
    )
    index = pd.MultiIndex.from_tuples(
        [
            ("FR", "tech1", "electricity, high voltage", "kWh"),
            ("DE", "tech2", "electricity, high voltage", "kWh"),
        ],
        names=["source", "exch_name", "prod", "unit"],
    )
    lci_cons_data = [[1.0, 2.0], [3.0, 4.0]]
    LCI_cons = pd.DataFrame(lci_cons_data, index=index, columns=columns)
    # Patch apply_mapping to return LCI_cons
    monkeypatch.setattr("shrecc.database.apply_mapping", lambda z, m: LCI_cons)
    # Patch load_from_pickle for Z_load_2023.pkl
    load = pd.DataFrame(
        [[10.0, 20.0], [30.0, 40.0]],
        index=["FR", "DE"],
        columns=["2023-01-01 00:00:00", "2023-01-01 01:00:00"],
    )

    def fake_load_from_pickle(path):
        if str(path).endswith("Z_load_2023.pkl"):
            return load
        raise FileNotFoundError(str(path))

    monkeypatch.setattr("shrecc.database.load_from_pickle", fake_load_from_pickle)
    # Patch save_to_pickle to record call
    saved = {}

    def fake_save_to_pickle(obj, path):
        saved["obj"] = obj
        saved["path"] = path

    monkeypatch.setattr("shrecc.database.save_to_pickle", fake_save_to_pickle)
    # Patch files to avoid importlib.resources.files
    monkeypatch.setattr("shrecc.database.files", lambda pkg: tmp_path)
    # Call tech_mapping
    result = tech_mapping(year, tmp_path)
    # Should return a DataFrame
    assert isinstance(result, pd.DataFrame)
    # Should have same columns as LCI_cons
    assert set(result.columns) == set(LCI_cons.columns)
    # Should have a row for load_difference_row
    load_difference_row = (
        "RER",
        "electricity, high voltage, European attribute mix",
        "electricity, high voltage",
        "kWh",
    )
    assert load_difference_row in result.index
    # Should have saved the result
    assert "obj" in saved
    assert saved["obj"].equals(result)
    assert str(saved["path"]).endswith("LCI_cons_scaled_2023.pkl")
    # Should have added the load_difference_row with correct values
    assert result.loc[load_difference_row].sum() == approx(1.3)


@patch("shrecc.database.save_to_pickle")
@patch("shrecc.database.load_mapping_data")
@patch("shrecc.database.load_time_series_data")
@patch("shrecc.database.prepare_consumption_data")
@patch("shrecc.database.apply_mapping")
def test_tech_mapping_if_file_exists(
    mock_apply_mapping,
    mock_consumption_data,
    mock_time_series,
    mock_load_mapping,
    mock_save_to_pickle,
    monkeypatch,
    tmp_path,
):
    """Test if tech_mapping loads existing file if it exists and does not call corresponding functions"""
    # Setup
    year = 2023
    data_dir = tmp_path / str(year)
    data_dir.mkdir(parents=True)
    scaled_path = data_dir / "LCI_cons_scaled_2023.pkl"
    # Dummy DataFrame to be loaded
    dummy_df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
    # Patch Path.exists to True only for scaled_path
    monkeypatch.setattr(
        "pathlib.Path.exists", lambda self: str(self) == str(scaled_path)
    )

    # Patch load_from_pickle to return dummy_df
    def fake_load_from_pickle(path):
        if str(path).endswith("LCI_cons_scaled_2023.pkl"):
            return dummy_df
        raise FileNotFoundError(str(path))

    monkeypatch.setattr("shrecc.database.load_from_pickle", fake_load_from_pickle)
    # Patch files to avoid importlib.resources.files
    monkeypatch.setattr("shrecc.database.files", lambda pkg: tmp_path)
    # Call tech_mapping
    result = tech_mapping(year, tmp_path)
    # Should return the dummy DataFrame and correctly called the functions
    assert result.equals(dummy_df)
    mock_load_mapping.assert_called_once()
    mock_time_series.assert_called_once()
    mock_consumption_data.assert_called_once()
    mock_apply_mapping.assert_not_called()
    mock_save_to_pickle.assert_not_called()


def test_tech_mapping_handles_empty_difference(monkeypatch, tmp_path):
    """Test if tech_mapping works well when the difference is not positive"""
    year = 2023
    data_dir = tmp_path / str(year)
    data_dir.mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
    # Dummy LCI_cons and load with no positive difference
    columns = pd.MultiIndex.from_tuples(
        [("2023-01-01 00:00:00", "trade", "FR")], names=["time", "source", "country"]
    )
    index = pd.MultiIndex.from_tuples(
        [("FR", "tech1", "electricity, high voltage", "kWh")],
        names=["source", "exch_name", "prod", "unit"],
    )
    LCI_cons = pd.DataFrame([[5.0]], index=index, columns=columns)
    monkeypatch.setattr("shrecc.database.apply_mapping", lambda z, m: LCI_cons)
    monkeypatch.setattr("shrecc.database.load_mapping_data", lambda path: None)
    monkeypatch.setattr(
        "shrecc.database.load_time_series_data", lambda path, year: None
    )
    monkeypatch.setattr("shrecc.database.prepare_consumption_data", lambda z: None)
    # load DataFrame with same value as LCI_cons.sum()
    load = pd.DataFrame([[5.0]], index=["FR"], columns=["2023-01-01 00:00:00"])

    def fake_load_from_pickle(path):
        if str(path).endswith("Z_load_2023.pkl"):
            return load
        raise FileNotFoundError(str(path))

    monkeypatch.setattr("shrecc.database.load_from_pickle", fake_load_from_pickle)
    monkeypatch.setattr("shrecc.database.save_to_pickle", lambda obj, path: None)
    monkeypatch.setattr("shrecc.database.files", lambda pkg: tmp_path)
    result = tech_mapping(year, tmp_path)
    # Should not add any new row for load_difference_row (difference <= 0)
    load_difference_row = (
        "RER",
        "electricity, high voltage, European attribute mix",
        "electricity, high voltage",
        "kWh",
    )
    # If difference is not positive, merged will be empty, so row will be filled with NaN and then fillna(0)
    assert load_difference_row in result.index
    assert (result.loc[load_difference_row] == 0).all()


# ────────────────────────────────────────────────────────────────
# Tests for: filter_by_countries() — requires multiple sub-tests
# ────────────────────────────────────────────────────────────────
def test_filter_by_countries_basic():
    """Test if filter_by_countries works correctly with basic case"""
    # Create a DataFrame with MultiIndex columns and index
    columns = pd.MultiIndex.from_tuples(
        [
            ("2023-06-01 08:00:00", "FR"),
            ("2023-06-01 09:00:00", "DE"),
            ("2023-06-01 10:00:00", "IT"),
            ("2023-06-01 11:00:00", "FR"),
        ],
        names=["time", "country"],
    )
    index = pd.MultiIndex.from_tuples(
        [("FR", "tech1", "electricity, high voltage", "kWh")],
        names=["geography", "activityName", "prod", "unit"],
    )
    data = [[1, 2, 3, 4]]
    df = pd.DataFrame(data, index=index, columns=columns)
    # Filter for FR and DE
    result = filter_by_countries(df, ["FR", "DE"])
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns.get_level_values("country")) == {"FR", "DE"}
    assert result.shape[1] == 3


def test_filter_by_countries_no_country_level(capsys):
    # DataFrame with columns missing 'country' level
    columns = pd.MultiIndex.from_tuples(
        [
            ("2023-06-01 08:00:00",),
            ("2023-06-01 09:00:00",),
        ],
        names=["time"],
    )
    index = pd.MultiIndex.from_tuples(
        [("FR", "tech1", "electricity, high voltage", "kWh")],
        names=["geography", "activityName", "prod", "unit"],
    )
    data = [[1, 2]]
    df = pd.DataFrame(data, index=index, columns=columns)
    result = filter_by_countries(df, ["FR"])
    # Should print warning and return None
    captured = capsys.readouterr()
    assert "Couldnt find country to filter" in captured.out
    assert result is None


def test_filter_by_countries_no_country_data():
    """Test if filter_by_countries returns empty DataFrame and prints message when no countries match"""
    columns = pd.MultiIndex.from_tuples(
        [
            ("2023-06-01 08:00:00", "FR"),
            ("2023-06-01 09:00:00", "DE"),
        ],
        names=["time", "country"],
    )
    index = pd.MultiIndex.from_tuples(
        [("FR", "tech1", "electricity, high voltage", "kWh")],
        names=["geography", "activityName", "prod", "unit"],
    )
    data = [[1, 2]]
    df = pd.DataFrame(data, index=index, columns=columns)
    result = filter_by_countries(df, ["IT"])
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 0


def test_filter_by_countries_empty_countries_list():
    # DataFrame with several countries, filter with empty list
    columns = pd.MultiIndex.from_tuples(
        [
            ("2023-06-01 08:00:00", "FR"),
            ("2023-06-01 09:00:00", "DE"),
        ],
        names=["time", "country"],
    )
    index = pd.MultiIndex.from_tuples(
        [("FR", "tech1", "electricity, high voltage", "kWh")],
        names=["source", "exch_name", "prod", "unit"],
    )
    data = [[1, 2]]
    df = pd.DataFrame(data, index=index, columns=columns)
    result = filter_by_countries(df, [])
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 0


# ───────────────────────────────────────────────────────────
# Tests for: filter_by_times() — requires multiple sub-tests
# ───────────────────────────────────────────────────────────
def test_filter_by_times_basic():
    """Test if filter_by_times works correctly with basic case"""
    # DataFrame with MultiIndex columns containing 'time'
    columns = pd.MultiIndex.from_tuples(
        [
            ("2023-06-01 08:00:00", "FR"),
            ("2023-06-01 09:00:00", "DE"),
            ("2023-06-01 10:00:00", "IT"),
            ("2023-06-01 11:00:00", "FR"),
        ],
        names=["time", "country"],
    )
    index = pd.MultiIndex.from_tuples(
        [("FR", "tech1", "electricity, high voltage", "kWh")],
        names=["geography", "activityName", "prod", "unit"],
    )
    data = [[1, 2, 3, 4]]
    df = pd.DataFrame(data, index=index, columns=columns)
    # Filter for two times
    times = ["2023-06-01 08:00:00", "2023-06-01 10:00:00"]
    result = filter_by_times(df, times)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns.get_level_values("time")) == set(times)
    assert result.shape[1] == 2


def test_filter_by_times_empty_times():
    """Test if filter_by_times returns empty DataFrame when times list is empty"""
    columns = pd.MultiIndex.from_tuples(
        [
            ("2023-06-01 08:00:00", "FR"),
            ("2023-06-01 09:00:00", "DE"),
        ],
        names=["time", "country"],
    )
    index = pd.MultiIndex.from_tuples(
        [("FR", "tech1", "electricity, high voltage", "kWh")],
        names=["geography", "activityName", "prod", "unit"],
    )
    data = [[1, 2]]
    df = pd.DataFrame(data, index=index, columns=columns)
    # Filter with empty times list
    result = filter_by_times(df, [])
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 0


def test_filter_by_times_no_matching_times():
    """Test if filter_by_times returns empty DataFrame when no times match"""
    columns = pd.MultiIndex.from_tuples(
        [
            ("2023-06-01 08:00:00", "FR"),
            ("2023-06-01 09:00:00", "DE"),
        ],
        names=["time", "country"],
    )
    index = pd.MultiIndex.from_tuples(
        [("FR", "tech1", "electricity, high voltage", "kWh")],
        names=["geography", "activityName", "prod", "unit"],
    )
    data = [[1, 2]]
    df = pd.DataFrame(data, index=index, columns=columns)
    # Filter for a time not present
    times = ["2023-06-01 10:00:00"]
    result = filter_by_times(df, times)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == 0


# ────────────────────────────────────────────────────────────────
# Tests for: filter_by_range() — requires multiple sub-tests
# ────────────────────────────────────────────────────────────────
def test_filter_by_range_basic():

    time_range = pd.date_range(
        start="2023-06-01 00:00:00", end="2023-08-31 23:00:00", freq="h"
    )
    sources = ["trade"]
    countries = ["FR", "DE"]
    columns = pd.MultiIndex.from_product(
        [time_range, sources, countries], names=["time", "source", "country"]
    )
    index = pd.MultiIndex.from_tuples(
        [("FR", "tech1", "electricity, high voltage", "kWh")],
        names=["geography", "activityName", "prod", "unit"],
    )
    data = np.arange(4416).reshape(1, 4416)
    dataframe = pd.DataFrame(data, index=index, columns=columns)

    general_range = ["2023-06-01 00:00:00", "2023-06-30 23:00:00"]
    refined_range = [20, 22]
    freq = "h"

    result = filter_by_range(dataframe, general_range, refined_range, freq)

    data = [[739.0, 738.0]]
    cols = pd.Index(["DE", "FR"], name="country")
    expected_result = pd.DataFrame(data, index=index, columns=cols)

    assert result.shape == (1, 2)
    # Should group by country and average
    assert (
        result.iloc[0, result.columns.get_level_values("country") == "FR"].values[0]
        == 738.0
    )
    assert (
        result.iloc[0, result.columns.get_level_values("country") == "DE"].values[0]
        == 739.0
    )
    pd.testing.assert_frame_equal(result, expected_result)


def test_filter_by_range_empty_result():
    # DataFrame with times outside the general_range
    columns = pd.MultiIndex.from_tuples(
        [
            ("2023-06-01 06:00:00", "FR"),
            ("2023-06-01 07:00:00", "FR"),
        ],
        names=["time", "country"],
    )
    index = pd.MultiIndex.from_tuples(
        [("FR", "tech1", "electricity, high voltage", "kWh")],
        names=["geography", "activityName", "prod", "unit"],
    )
    data = [[1, 2]]
    df = pd.DataFrame(data, index=index, columns=columns)
    general_range = ["2023-06-01 08:00:00", "2023-06-01 09:00:00"]
    refined_range = []
    freq = "h"
    result = filter_by_range(df, general_range, refined_range, freq)
    # Should be empty
    assert result.shape[1] == 0


# ─────────────────────────────────────────────────────────
# Tests for: apply_cutoff() — requires multiple sub-tests
# ─────────────────────────────────────────────────────────
def test_apply_cutoff_include_rest_row():
    # DataFrame with values below and above cutoff
    columns = pd.MultiIndex.from_tuples(
        [("FR",), ("DE",)],
        names=["country"],
    )
    index = pd.MultiIndex.from_tuples(
        [
            ("FR", "tech1", "electricity, high voltage", "kWh"),
            ("DE", "tech2", "electricity, high voltage", "kWh"),
        ],
        names=["geography", "activityName", "prod", "unit"],
    )
    data = [[0.5, 0.2], [0.1, 0.05]]
    df = pd.DataFrame(data, index=index, columns=columns)
    cutoff = 0.15
    result = apply_cutoff(df.copy(), cutoff, include_cutoff=True)

    assert (result.values >= 0).all()
    # Vaules above cutoff should be the same as before
    assert (
        result.loc[("FR", "tech1", "electricity, high voltage", "kWh"), "FR"].item()
        == 0.5
    )
    assert (
        result.loc[("FR", "tech1", "electricity, high voltage", "kWh"), "DE"].item()
        == 0.20
    )
    # Values below cutoff should be zero
    assert (
        result.loc[("DE", "tech2", "electricity, high voltage", "kWh"), "FR"].item()
        == 0
    )
    assert (
        result.loc[("DE", "tech2", "electricity, high voltage", "kWh"), "DE"].item()
        == 0
    )
    # Rest row should exist and be sum of values below cutoff per column
    rest_row = (
        "RER",
        "market group for electricity, high voltage",
        "electricity, high voltage",
        "kWh",
    )
    assert rest_row in result.index
    # For first column: only 0.1 is below cutoff, for second column: 0.05 and 0.2
    assert result.loc[rest_row, "FR"].item() == 0.1
    assert result.loc[rest_row, "DE"].item() == 0.05


def test_apply_cutoff_exclude_rest_row():
    columns = pd.MultiIndex.from_tuples(
        [("FR",), ("DE",)],
        names=["country"],
    )
    index = pd.MultiIndex.from_tuples(
        [
            ("FR", "tech1", "electricity, high voltage", "kWh"),
            ("DE", "tech2", "electricity, high voltage", "kWh"),
        ],
        names=["geography", "activityName", "prod", "unit"],
    )
    data = [[0.5, 0.2], [0.1, 0.05]]
    df = pd.DataFrame(data, index=index, columns=columns)
    cutoff = 0.15
    result = apply_cutoff(df.copy(), cutoff, include_cutoff=False)
    # Rest row should not exist
    rest_row = (
        "RER",
        "market group for electricity, high voltage",
        "electricity, high voltage",
        "kWh",
    )
    assert rest_row not in result.index
    # Values below cutoff should be zero
    assert (
        result.loc[("DE", "tech2", "electricity, high voltage", "kWh"), "FR"].item()
        == 0
    )
    assert (
        result.loc[("DE", "tech2", "electricity, high voltage", "kWh"), "DE"].item()
        == 0
    )


def test_apply_cutoff_all_above_cutoff():
    columns = pd.MultiIndex.from_tuples(
        [("FR",)],
        names=["country"],
    )
    index = pd.MultiIndex.from_tuples(
        [
            ("FR", "tech1", "electricity, high voltage", "kWh"),
            ("DE", "tech2", "electricity, high voltage", "kWh"),
        ],
        names=["geography", "activityName", "prod", "unit"],
    )
    data = [[0.5], [0.4]]
    df = pd.DataFrame(data, index=index, columns=columns)
    cutoff = 0.15
    result = apply_cutoff(df.copy(), cutoff, include_cutoff=True)
    # No values below cutoff, so rest row should be zero
    rest_row = (
        "RER",
        "market group for electricity, high voltage",
        "electricity, high voltage",
        "kWh",
    )
    assert rest_row in result.index
    assert result.loc[rest_row, "FR"].item() == 0


def test_apply_cutoff_all_below_cutoff():
    columns = pd.MultiIndex.from_tuples(
        [("FR",)],
        names=["country"],
    )
    index = pd.MultiIndex.from_tuples(
        [
            ("FR", "tech1", "electricity, high voltage", "kWh"),
            ("DE", "tech2", "electricity, high voltage", "kWh"),
        ],
        names=["geography", "activityName", "prod", "unit"],
    )
    data = [[0.05], [0.1]]
    df = pd.DataFrame(data, index=index, columns=columns)
    cutoff = 0.15
    result = apply_cutoff(df.copy(), cutoff, include_cutoff=True)
    # All values below cutoff, so rest row should be sum of all
    rest_row = (
        "RER",
        "market group for electricity, high voltage",
        "electricity, high voltage",
        "kWh",
    )
    assert rest_row in result.index
    assert result.loc[rest_row, "FR"].item() == approx(0.15)
    # All other values should be zero
    assert (result.drop(rest_row).values == 0).all()


def test_apply_cutoff_empty_dataframe():
    columns = pd.MultiIndex.from_tuples(
        [("FR",)],
        names=["country"],
    )
    index = pd.MultiIndex.from_tuples(
        [], names=["geography", "activityName", "prod", "unit"]
    )
    df = pd.DataFrame([], index=index, columns=columns)
    cutoff = 0.15
    result = apply_cutoff(df.copy(), cutoff, include_cutoff=True)
    # Should still add rest row with zero
    rest_row = (
        "RER",
        "market group for electricity, high voltage",
        "electricity, high voltage",
        "kWh",
    )
    assert rest_row in result.index
    assert result.loc[rest_row, "FR"].item() == 0


# ────────────────────────────────────────────────────────────────
# Tests for: filt_cutoff() — additional direct tests for coverage
# ────────────────────────────────────────────────────────────────
@pytest.fixture
def dummy_dataframe_for_filt_cutoff():
    # MultiIndex for columns: (time, country)
    columns = pd.MultiIndex.from_tuples(
        [
            ("2023-06-01 08:00:00", "trade", "DE"),
            ("2023-06-01 08:00:00", "trade", "FR"),
            ("2023-06-01 09:00:00", "trade", "DE"),
            ("2023-06-01 09:00:00", "trade", "FR"),
            ("2023-06-01 10:00:00", "trade", "DE"),
            ("2023-06-01 10:00:00", "trade", "FR"),
            ("2023-06-01 11:00:00", "trade", "DE"),
            ("2023-06-01 11:00:00", "trade", "FR"),
            ("2023-06-01 12:00:00", "trade", "DE"),
            ("2023-06-01 12:00:00", "trade", "FR"),
        ],
        names=["time", "source", "country"],
    )
    index = pd.MultiIndex.from_tuples(
        [
            ("FR", "tech1", "electricity, high voltage", "kWh"),
            ("DE", "tech2", "electricity, high voltage", "kWh"),
        ],
        names=["geography", "activityName", "prod", "unit"],
    )
    data = [
        [0.5, 0.2, 0.1, 0.05, 0.01, 0.1, 0.03, 0.2, 0.5, 0.03],
        [0.3, 0.4, 0.2, 0.01, 0.06, 0.3, 0.07, 0.1, 0.3, 0.04],
    ]
    df = pd.DataFrame(data, index=index, columns=columns)
    return df


@patch("shrecc.database.tech_mapping")
def test_filt_cutoff_filtering_by_times(
    mock_df_tech_mapping, dummy_dataframe_for_filt_cutoff
):

    df = dummy_dataframe_for_filt_cutoff
    mock_df_tech_mapping.return_value = df

    result = filt_cutoff(
        countries=["FR"],
        times=["2023-06-01 08:00:00"],
        # general_range = ["2023-06-01 08:00:00", "2023-06-01 11:00:00"], #for testing
        freq="h",
        cutoff=0.15,
        include_cutoff=True,
        path_to_data="dummy_root",
    )
    rest_row = (
        "RER",
        "market group for electricity, high voltage",
        "electricity, high voltage",
        "kWh",
    )
    assert rest_row in result.index
    assert result.shape == (3, 1)
    assert all(result.columns.get_level_values("country") == "FR")
    assert (result.values >= 0).all()
    assert result.loc[rest_row, ("FR")] == 0
    assert (
        result.loc[("FR", "tech1", "electricity, high voltage", "kWh"), ("FR")] == 0.2
    )
    assert (
        result.loc[("DE", "tech2", "electricity, high voltage", "kWh"), ("FR")] == 0.4
    )


@patch("shrecc.database.tech_mapping")
def test_filt_cutoff_filtering_by_range(
    mock_df_tech_mapping, dummy_dataframe_for_filt_cutoff
):

    df = dummy_dataframe_for_filt_cutoff
    mock_df_tech_mapping.return_value = df

    general_range = ["2023-06-01 08:00:00", "2023-06-01 11:00:00"]
    refined_range = [9, 10]

    result = filt_cutoff(
        countries=["FR"],
        general_range=general_range,
        refined_range=refined_range,
        freq="h",
        cutoff=0.15,
        include_cutoff=True,
        path_to_data="dummy_root",
    )
    rest_row = (
        "RER",
        "market group for electricity, high voltage",
        "electricity, high voltage",
        "kWh",
    )
    assert rest_row in result.index
    assert result.shape == (3, 1)
    assert all(result.columns.get_level_values("country") == "FR")
    assert (result.values >= 0).all()
    assert result.loc[rest_row, "FR"].item() == approx(0.075)
    assert (
        result.loc[("FR", "tech1", "electricity, high voltage", "kWh"), "FR"].item()
        == 0
    )
    assert result.loc[
        ("DE", "tech2", "electricity, high voltage", "kWh"), "FR"
    ].item() == approx(0.155)


# ────────────────────────────────────────────────────────────────
# Tests for: setup_database() — requires multiple sub-tests
# ────────────────────────────────────────────────────────────────
def test_setup_database_registers_and_returns(monkeypatch):
    # Prepare mocks
    mock_projects = MagicMock()
    mock_databases = {}
    mock_elec_db = MagicMock()
    mock_elec_db.register = MagicMock()
    # Patch bd
    monkeypatch.setattr("shrecc.database.bd", MagicMock())
    bd = __import__("shrecc.database").database.bd
    bd.projects = mock_projects
    bd.databases = mock_databases
    bd.Database = MagicMock(return_value=mock_elec_db)

    # Call setup_database
    result = setup_database("my_project", "my_db")

    # Should set current project
    mock_projects.set_current.assert_called_once_with("my_project")
    # Should create and register the database
    bd.Database.assert_called_once_with("my_db")
    mock_elec_db.register.assert_called_once()
    # Should return the database object
    assert result == mock_elec_db


def test_setup_database_deletes_existing_db(monkeypatch):
    # Prepare mocks
    mock_projects = MagicMock()
    mock_databases = {"my_db": "something"}
    mock_elec_db = MagicMock()
    mock_elec_db.register = MagicMock()
    # Patch bd
    monkeypatch.setattr("shrecc.database.bd", MagicMock())
    bd = __import__("shrecc.database").database.bd
    bd.projects = mock_projects
    bd.databases = mock_databases
    bd.Database = MagicMock(return_value=mock_elec_db)

    # Call setup_database
    result = setup_database("my_project", "my_db")

    # Should delete the db and purge directories
    assert "my_db" not in bd.databases
    mock_projects.purge_deleted_directories.assert_called_once()
    # Should return the database object
    assert result == mock_elec_db


def test_setup_database_handles_multiple_calls(monkeypatch):
    # Prepare mocks
    mock_projects = MagicMock()
    mock_databases = {"my_db": "something"}
    mock_elec_db = MagicMock()
    mock_elec_db.register = MagicMock()
    # Patch bd
    monkeypatch.setattr("shrecc.database.bd", MagicMock())
    bd = __import__("shrecc.database").database.bd
    bd.projects = mock_projects
    bd.databases = mock_databases
    bd.Database = MagicMock(return_value=mock_elec_db)

    # Call setup_database twice to check idempotency
    result1 = setup_database("my_project", "my_db")
    # After first call, db should be deleted
    assert "my_db" not in bd.databases
    # Add it back and call again
    bd.databases["my_db"] = "something"
    result2 = setup_database("my_project", "my_db")
    assert "my_db" not in bd.databases
    assert result1 == mock_elec_db
    assert result2 == mock_elec_db


# ──────────────────────────────────────────────────────────
# Tests for: tech_mapping() — requires multiple sub-tests
# ──────────────────────────────────────────────────────────
def test_get_network_activities_basic():

    eidb_name = "ecoinvent-3.8-cutoff"
    result = get_network_activities(eidb_name)
    assert isinstance(result, list)
    assert len(result) == 7
    # Check structure of each dict
    for entry in result:
        assert set(entry.keys()) == {"name", "loc", "val"}
        assert isinstance(entry["name"], str)
        assert isinstance(entry["loc"], str)
        assert isinstance(entry["val"], float)
    # Check names are unchanged
    expected_names = [
        "market for distribution network, electricity, low voltage",
        "market for transmission network, electricity, medium voltage",
        "market for sulfur hexafluoride, liquid",
        "market for transmission network, electricity, high voltage direct current aerial line",
        "market for transmission network, electricity, high voltage direct current land cable",
        "market for transmission network, electricity, high voltage direct current subsea cable",
        "transmission network construction, electricity, high voltage",
    ]
    assert [entry["name"] for entry in result] == expected_names
    # Check locations for default (not 3.11)
    expected_locations = ["GLO", "GLO", "RER", "GLO", "GLO", "GLO", "CH"]
    assert [entry["loc"] for entry in result] == expected_locations
    # Check values
    expected_values = [
        8.679076855e-8,
        1.86646177072e-8,
        1.27657893204915e-7,
        8.38e-9,
        3.47e-10,
        5.66e-10,
        6.58e-9,
    ]
    assert [entry["val"] for entry in result] == expected_values


def test_get_network_activities_311():

    eidb_name = "ecoinvent-3.11-cutoff"
    result = get_network_activities(eidb_name)
    assert isinstance(result, list)
    assert len(result) == 7
    # Check locations for 3.11
    expected_locations = ["GLO", "GLO", "RER", "RER", "RER", "RER", "CH"]
    assert [entry["loc"] for entry in result] == expected_locations


def test_get_network_activities_other_versions():

    # Should not match "3.11" if not present
    for eidb_name in ["ecoinvent 3.7", "ei3.8.1", "eco3.9", ""]:
        result = get_network_activities(eidb_name)
        expected_locations = ["GLO", "GLO", "RER", "GLO", "GLO", "GLO", "CH"]
        assert [entry["loc"] for entry in result] == expected_locations


# ────────────────────────────────────────────────────────────────
# Tests for: map_known_inputs() — requires multiple sub-tests
# ────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_dataframe():
    idx = pd.MultiIndex.from_tuples(
        [
            ("FR", "Hydro", "electricity, high voltage", "kWh"),
            ("UK", "Wind", "electricity, high voltage", "kWh"),
            ("DE", "Wind", "electricity, high voltage", "kWh"),
        ],
        names=["loc", "name", "prod", "unit"],
    )
    return pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=idx, columns=["A", "B"])


@patch("shrecc.database.get_network_activities")
@patch("shrecc.database.Filter")
@patch("shrecc.database.Query")
@patch("shrecc.database.bd.Database")
def test_map_known_inputs_basic(
    mock_database_cls,
    mock_query_cls,
    mock_filter_cls,
    mock_get_network,
    sample_dataframe,
    capsys,
):
    # Mock bd.Database().load()
    mock_db_instance = MagicMock()
    mock_db_instance.load.return_value = object()  # mock ei_db_data
    mock_database_cls.return_value = mock_db_instance

    # Prepare results map
    results_map = {
        ("FR", "Hydro"): ["fr_hydro_result"],
        ("GB", "Wind"): ["gb_wind_result"],
    }

    # Setup the fake Query behavior
    def fake_query_factory():
        class Q:
            def __init__(self):
                self.filters = []

            def add(self, f):
                self.filters.append(f)

            def __call__(self, data):
                name, loc = None, None
                for f in self.filters:
                    if hasattr(f, "field") and f.field == "name":
                        name = f.value
                    if hasattr(f, "field") and f.field == "location":
                        loc = f.value
                if name and loc:
                    if name.startswith("market for") or name.startswith(
                        "transmission network"
                    ):
                        return ["network_result"]
                    return results_map.get((loc, name), [])
                return []

        return Q()

    mock_query_cls.side_effect = fake_query_factory

    # Fake Filter class
    class DummyFilter:
        def __init__(self, field, op, value):
            self.field = field
            self.value = value

    mock_filter_cls.side_effect = DummyFilter

    # Mock get_network_activities
    mock_get_network.return_value = [
        {
            "name": "market for distribution network, electricity, low voltage",
            "loc": "GLO",
            "val": "8.6",
        },
        {
            "name": "transmission network construction, electricity, high voltage",
            "loc": "CH",
            "val": "1.8",
        },
    ]

    # Run the function
    known_inputs, known_inputs_network = map_known_inputs("eidb", sample_dataframe)

    # Assert known inputs
    assert ("FR", "Hydro", "kWh") in known_inputs
    assert known_inputs[("FR", "Hydro", "kWh")] == "fr_hydro_result"
    assert ("GB", "Wind", "kWh") in known_inputs
    assert known_inputs[("GB", "Wind", "kWh")] == "gb_wind_result"
    # Assert network inputs
    assert (
        "GLO",
        "market for distribution network, electricity, low voltage",
    ) in known_inputs_network
    assert (
        "CH",
        "transmission network construction, electricity, high voltage",
    ) in known_inputs_network
    assert (
        known_inputs_network[
            ("GLO", "market for distribution network, electricity, low voltage")
        ]
        == "network_result"
    )
    assert (
        known_inputs_network[
            ("CH", "transmission network construction, electricity, high voltage")
        ]
        == "network_result"
    )
    # Should print warning for the missing activity
    captured = capsys.readouterr()
    assert "Couldnt find activity:Wind, DE" in captured.out


@patch("shrecc.database.get_network_activities", return_value=[])
@patch("shrecc.database.Filter")
@patch("shrecc.database.Query")
@patch("shrecc.database.bd.Database")
def test_map_known_inputs_empty_dataframe(
    mock_database_cls, mock_query_cls, mock_filter_cls, mock_get_network_activities
):
    # Create empty DataFrame with MultiIndex
    idx = pd.MultiIndex.from_tuples([], names=["loc", "name", "prod", "unit"])
    df = pd.DataFrame([], index=idx, columns=["A"])

    # Set up mock Database and .load()
    mock_db_instance = mock_database_cls.return_value
    mock_db_instance.load.return_value = object()

    # Set up mock Query instance
    mock_query_instance = mock_query_cls.return_value
    mock_query_instance.filters = []
    mock_query_instance.add.side_effect = lambda f: mock_query_instance.filters.append(
        f
    )
    mock_query_instance.side_effect = lambda data: []

    # Set up mock Filter (not actually used in this test since DataFrame is empty)
    mock_filter_cls.side_effect = lambda field, op, value: types.SimpleNamespace(
        field=field, value=value
    )

    mock_get_network_activities.return_value = []

    # Call the function under test
    known_inputs, known_inputs_network = map_known_inputs("eidb", df)

    # Assertions
    assert known_inputs == {}
    assert known_inputs_network == {}


# ────────────────────────────────────────────────────────────────
# Tests for: create_activity_dict() — requires multiple sub-tests
# ────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_dataframe_filt():
    # MultiIndex columns: (time, country)
    columns = pd.MultiIndex.from_tuples(
        [
            ("2023-06-01 08:00:00", "FR"),
            ("2023-06-01 09:00:00", "DE"),
        ],
        names=["time", "country"],
    )
    index = pd.MultiIndex.from_tuples(
        [
            ("FR", "Hydro", "electricity, high voltage", "kWh"),
            ("DE", "Wind", "electricity, high voltage", "kWh"),
        ],
        names=["source", "exch_name", "prod", "unit"],
    )
    data = [
        [0.5, 0.0],  # Only FR/08:00:00 has value
        [0.0, 0.7],  # Only DE/09:00:00 has value
    ]
    return pd.DataFrame(data, index=index, columns=columns)


@pytest.fixture
def known_inputs_fixture():
    # Only provide mapping for the nonzero entries
    return {
        ("FR", "Hydro", "kWh"): "fr_hydro_ei",
        ("DE", "Wind", "kWh"): "de_wind_ei",
    }


@pytest.fixture
def known_inputs_network_fixture():
    # Provide mapping for two network activities
    return {
        (
            "GLO",
            "market for distribution network, electricity, low voltage",
        ): "net_dist_lv",
        (
            "CH",
            "transmission network construction, electricity, high voltage",
        ): "net_trans_ch",
        (
            "GLO",
            "market for transmission network, electricity, medium voltage",
        ): "net_trans_mv",
        (
            "GLO",
            "market for transmission network, electricity, high voltage direct current subsea cable",
        ): "net_trans_subsea",
    }


@pytest.fixture
def fake_network_activities():
    return [
        {
            "name": "market for distribution network, electricity, low voltage",
            "loc": "GLO",
            "val": 1.1,
        },
        {
            "name": "market for transmission network, electricity, medium voltage",
            "loc": "GLO",
            "val": 2.2,
        },
        {
            "name": "market for transmission network, electricity, high voltage direct current land cable",
            "loc": "GLO",
            "val": 4.4,
        },
        {
            "name": "market for transmission network, electricity, high voltage direct current subsea cable",
            "loc": "GLO",
            "val": 5.5,
        },
        {
            "name": "transmission network construction, electricity, high voltage",
            "loc": "CH",
            "val": 3.3,
        },
    ]


@patch("shrecc.database.get_network_activities")
def test_create_activity_dict_basic(
    mock_get_network_activities,
    sample_dataframe_filt,
    known_inputs_fixture,
    known_inputs_network_fixture,
    fake_network_activities,
):
    mock_get_network_activities.return_value = fake_network_activities
    db_name = "test_db"
    activities = create_activity_dict(
        sample_dataframe_filt,
        known_inputs_fixture,
        known_inputs_network_fixture,
        db_name,
    )
    # There should be one activity per column
    assert len(activities) == 2
    # Check keys
    assert set(activities.keys()) == {
        (db_name, "electricity 0"),
        (db_name, "electricity 1"),
    }
    # Check structure of activity dict
    for act in activities.values():
        assert act["unit"] == "kWh"
        assert act["type"] == "process"
        assert isinstance(act["exchanges"], list)
        # Should have at least one exchange (the technology)
        assert any(e["type"] == "technosphere" for e in act["exchanges"])
        # Should include network exchanges if known_inputs_network_fixture matches
        assert any(
            e["input"] in known_inputs_network_fixture.values()
            for e in act["exchanges"]
        )

    # Check that the correct technology exchange is present for each activity
    # act0 is for ("2023-06-01 08:00:00", "FR")
    act0 = activities[(db_name, "electricity 0")]
    inputs0 = [e["input"] for e in act0["exchanges"]]
    assert set(inputs0) == {
        "fr_hydro_ei",
        "net_trans_subsea",
        "net_dist_lv",
        "net_trans_mv",
    }
    assert "de_wind_ei" not in inputs0
    assert "net_trans_ch" not in inputs0
    tech_ex0 = [e for e in act0["exchanges"] if e["input"] == "fr_hydro_ei"]
    assert len(tech_ex0) == 1
    assert tech_ex0[0]["amount"] == 0.5
    tech_ex1 = [e for e in act0["exchanges"] if e["input"] == "net_trans_subsea"]
    assert len(tech_ex1) == 1
    assert tech_ex1[0]["amount"] == 5.5  # specific network
    # act1 is for ("2023-06-01 09:00:00", "DE")
    act1 = activities[(db_name, "electricity 1")]
    inputs1 = [e["input"] for e in act1["exchanges"]]
    assert set(inputs1) == {"de_wind_ei", "net_dist_lv", "net_trans_mv"}
    assert "net_trans_subsea" not in inputs1
    assert "net_trans_ch" not in inputs1
    tech_ex2 = [e for e in act1["exchanges"] if e["input"] == "de_wind_ei"]
    assert len(tech_ex2) == 1
    assert tech_ex2[0]["amount"] == 0.7


@patch("shrecc.database.get_network_activities")
def test_create_activity_dict_no_network_exchanges(
    mock_get_network_activities, sample_dataframe_filt, known_inputs_fixture
):
    # No known_inputs_network, so only technology exchanges
    mock_get_network_activities.return_value = []
    db_name = "test_db"
    activities = create_activity_dict(
        sample_dataframe_filt,
        known_inputs_fixture,
        {},
        db_name,
    )
    for act in activities.values():
        # Only technology exchanges should be present
        assert all(
            e["input"] in known_inputs_fixture.values() for e in act["exchanges"]
        )


def test_create_activity_dict_empty_dataframe():
    # Should return empty dict if dataframe_filt has no columns
    df = pd.DataFrame()
    activities = create_activity_dict(df, {}, {}, "db")
    assert activities == {}


# ────────────────────────────────────────────────────────────
# Tests for: create_database() — requires multiple sub-tests
# ────────────────────────────────────────────────────────────
@pytest.fixture
def dataframe_filt_for_create_database():
    columns = pd.MultiIndex.from_tuples(
        [("2023-06-01 08:00:00", "FR"), ("2023-06-01 09:00:00", "DE")],
        names=["time", "country"],
    )
    index = pd.MultiIndex.from_tuples(
        [
            ("FR", "Hydro", "electricity, high voltage", "kWh"),
            ("DE", "Wind", "electricity, high voltage", "kWh"),
        ],
        names=["source", "exch_name", "prod", "unit"],
    )
    data = [
        [0.5, 0.0],
        [0.0, 0.7],
    ]
    return pd.DataFrame(data, index=index, columns=columns)


@patch("shrecc.database.setup_database")
@patch("shrecc.database.map_known_inputs")
@patch("shrecc.database.create_activity_dict")
def test_create_database_with_network_true(
    mock_create_activity_dict,
    mock_map_known_inputs,
    mock_setup_database,
    dataframe_filt_for_create_database,
):
    # Prepare mocks
    mock_db = MagicMock()
    mock_setup_database.return_value = mock_db
    known_inputs = {"a": 1}
    known_inputs_network = {"b": 2}
    mock_map_known_inputs.return_value = (known_inputs, known_inputs_network)
    activities = {"activity": {"name": "test"}}
    mock_create_activity_dict.return_value = activities

    # Call function
    create_database(
        dataframe_filt_for_create_database,
        "proj",
        "db",
        "eidb",
        network="True",
    )

    # Check calls
    mock_setup_database.assert_called_once_with("proj", "db")
    mock_map_known_inputs.assert_called_once_with(
        "eidb", dataframe_filt_for_create_database
    )
    mock_create_activity_dict.assert_called_once_with(
        dataframe_filt_for_create_database, known_inputs, known_inputs_network, "db"
    )
    mock_db.write.assert_called_once_with(activities)


@patch("shrecc.database.setup_database")
@patch("shrecc.database.map_known_inputs")
@patch("shrecc.database.create_activity_dict")
def test_create_database_with_network_false(
    mock_create_activity_dict,
    mock_map_known_inputs,
    mock_setup_database,
    dataframe_filt_for_create_database,
):
    mock_db = MagicMock()
    mock_setup_database.return_value = mock_db
    known_inputs = {"a": 1}
    mock_map_known_inputs.return_value = (known_inputs, None)
    activities = {"activity": {"name": "test"}}
    mock_create_activity_dict.return_value = activities

    create_database(
        dataframe_filt_for_create_database,
        "proj",
        "db",
        "eidb",
        network="False",
    )

    mock_setup_database.assert_called_once_with("proj", "db")
    mock_map_known_inputs.assert_called_once_with(
        "eidb", dataframe_filt_for_create_database
    )
    # The third argument to create_activity_dict should be None (the _ from map_known_inputs)
    mock_create_activity_dict.assert_called_once_with(
        dataframe_filt_for_create_database, known_inputs, None, "db"
    )
    mock_db.write.assert_called_once_with(activities)


@patch("shrecc.database.setup_database")
@patch("shrecc.database.map_known_inputs")
@patch("shrecc.database.create_activity_dict")
def test_create_database_empty_activities(
    mock_create_activity_dict,
    mock_map_known_inputs,
    mock_setup_database,
    dataframe_filt_for_create_database,
):
    mock_db = MagicMock()
    mock_setup_database.return_value = mock_db
    known_inputs = {}
    known_inputs_network = {}
    mock_map_known_inputs.return_value = (known_inputs, known_inputs_network)
    mock_create_activity_dict.return_value = {}

    create_database(
        dataframe_filt_for_create_database,
        "proj",
        "db",
        "eidb",
        network="True",
    )

    mock_db.write.assert_called_once_with({})
