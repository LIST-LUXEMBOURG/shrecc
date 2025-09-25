import pytest
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix, csc_matrix, coo_array
from unittest.mock import patch
from shrecc.treatment import (
    data_processing,
    save_to_pickle,
    load_from_pickle,
    add_missing_elements,
    create_block_diagonal_matrix,
    process_matrix,
    treating_data,
    calculate_results,
    process_results_light,
    concatenate_results,
    calculate_Z_cons,
)


# ───────────────────────────────────────────────────────────
# Tests for: data_processing() — requires multiple sub-tests
# ───────────────────────────────────────────────────────────
@pytest.fixture
def mock_data_df():
    # Create a minimal DataFrame with the required structure for data_processing
    # MultiIndex: (country, type, source)
    col_tuples = [
        ("DE", "production mix", "solar"),
        ("DE", "production mix", "wind"),
        ("DE", "trade", "FR"),
        ("DE", "trade", "LU"),
        ("DE", "trade", "IE"),
        ("DE", "load", "load"),
        ("FR", "production mix", "Hydro pumped storage consumption"),
        ("FR", "production mix", "wind"),
        ("FR", "trade", "LU"),
        ("FR", "load", "load"),
    ]
    columns = pd.MultiIndex.from_tuples(col_tuples, names=["country", "type", "source"])
    idx = pd.to_datetime(
        [
            "2023-01-01 00:00:00",
            "2023-01-01 01:00:00",
            "2023-01-01 02:00:00",
        ]
    )
    data = [
        [1, 11, 1100, 2100, 3100, 10, -4, 12, 4100, 20],
        [2, 13, 1200, 2200, 3200, 11, -5, 14, 4200, 21],
        [3, 15, 1300, 2300, 3300, 12, -6, 16, 4300, 22],
    ]
    df = pd.DataFrame(data, index=idx, columns=columns)
    return df


@patch("shrecc.treatment.treating_data")
def test_data_processing(mock_treating_data, tmp_path, mock_data_df):
    # Make treating_data do nothing
    mock_treating_data.return_value = None

    data_df = mock_data_df
    year = 2023

    expected_output_dir = tmp_path / str(year)
    expected_output_dir.mkdir(parents=True, exist_ok=True)

    # Run data_processing
    data_processing(data_df, year, path_to_data=tmp_path)
    # Check that the expected files are created
    expected_Z_load = pd.DataFrame(
        [
            [0.0, 0.0, 0.0],
            [12.0, 15.0, 18.0],
            [12.0, 14.0, 16.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        index=["CY", "DE", "FR", "IE", "LU"],
        columns=pd.to_datetime(
            ["2023-01-01 00:00:00", "2023-01-01 01:00:00", "2023-01-01 02:00:00"]
        ),
    )
    expected_Z_indices = {
        "index": pd.MultiIndex.from_tuples(
            [
                ("Hydro pumped storage consumption", "CY"),
                ("solar", "CY"),
                ("wind", "CY"),
                ("Hydro pumped storage consumption", "DE"),
                ("solar", "DE"),
                ("wind", "DE"),
                ("Hydro pumped storage consumption", "FR"),
                ("solar", "FR"),
                ("wind", "FR"),
                ("Hydro pumped storage consumption", "IE"),
                ("solar", "IE"),
                ("wind", "IE"),
                ("Hydro pumped storage consumption", "LU"),
                ("solar", "LU"),
                ("wind", "LU"),
                ("trade", "CY"),
                ("trade", "DE"),
                ("trade", "FR"),
                ("trade", "IE"),
                ("trade", "LU"),
            ],
            names=["source", "country"],
        ),
        "columns": pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2023-01-01 00:00:00"), "trade", "CY"),
                (pd.Timestamp("2023-01-01 00:00:00"), "trade", "DE"),
                (pd.Timestamp("2023-01-01 00:00:00"), "trade", "FR"),
                (pd.Timestamp("2023-01-01 00:00:00"), "trade", "IE"),
                (pd.Timestamp("2023-01-01 00:00:00"), "trade", "LU"),
                (pd.Timestamp("2023-01-01 01:00:00"), "trade", "CY"),
                (pd.Timestamp("2023-01-01 01:00:00"), "trade", "DE"),
                (pd.Timestamp("2023-01-01 01:00:00"), "trade", "FR"),
                (pd.Timestamp("2023-01-01 01:00:00"), "trade", "IE"),
                (pd.Timestamp("2023-01-01 01:00:00"), "trade", "LU"),
                (pd.Timestamp("2023-01-01 02:00:00"), "trade", "CY"),
                (pd.Timestamp("2023-01-01 02:00:00"), "trade", "DE"),
                (pd.Timestamp("2023-01-01 02:00:00"), "trade", "FR"),
                (pd.Timestamp("2023-01-01 02:00:00"), "trade", "IE"),
                (pd.Timestamp("2023-01-01 02:00:00"), "trade", "LU"),
            ],
            names=["time", "source", "country"],
        ),
    }

    # Check that pickle files are created
    data_dir = tmp_path / f"{year}"
    assert (data_dir / f"Z_load_{year}.pkl").exists()
    assert (data_dir / f"indices_{year}.pkl").exists()
    assert (data_dir / f"Z_net_{year}.pkl").exists()

    # Check that the pickles can be loaded
    z_load = load_from_pickle(data_dir / f"Z_load_{year}.pkl")
    z_indices = load_from_pickle(data_dir / f"indices_{year}.pkl")
    z_net = load_from_pickle(data_dir / f"Z_net_{year}.pkl")

    assert isinstance(z_load, pd.DataFrame)
    assert isinstance(z_indices, dict)
    assert isinstance(z_net, pd.DataFrame)

    # Check the contents of the loaded DataFrames
    assert z_load.shape == expected_Z_load.shape, "Z_load shape does not match expected"
    assert z_load.columns.equals(
        expected_Z_load.columns
    ), "Z_load columns do not match expected"
    assert z_load.index.equals(
        expected_Z_load.index
    ), "Z_load index does not match expected"
    assert z_load.equals(expected_Z_load), "Z_load does not match expected DataFrame"
    assert z_indices["index"].equals(
        expected_Z_indices["index"]
    ), "Z_indices index does not match expected"
    assert z_indices["columns"].equals(
        expected_Z_indices["columns"]
    ), "Z_indices columns do not match expected"


# ───────────────────────────────────────────────────────────
# Tests for: save_to_pickle() — requires multiple sub-tests
# ───────────────────────────────────────────────────────────
def test_save_to_pickle_writes_file(tmp_path):
    """Test that save_to_pickle actually writes a file to disk."""
    data = {"foo": "bar"}
    file_path = tmp_path / "mydata.pkl"
    save_to_pickle(data, file_path)
    assert file_path.exists(), "Pickle file should be created on disk"


def test_save_to_pickle_and_load(tmp_path):
    """Test that data saved with save_to_pickle can be loaded and matches original."""
    data = [1, 2, 3, {"a": 10}]
    file_path = tmp_path / "array.pkl"
    save_to_pickle(data, file_path)
    loaded = load_from_pickle(file_path)
    assert loaded == data, "Loaded data should match the original data"


def test_save_to_pickle_overwrites_file(tmp_path):
    """Test that save_to_pickle overwrites an existing file."""
    file_path = tmp_path / "overwrite.pkl"
    save_to_pickle({"x": 1}, file_path)
    save_to_pickle({"x": 2}, file_path)
    loaded = load_from_pickle(file_path)
    assert loaded == {"x": 2}, "File should be overwritten with new data"


def test_save_to_pickle_with_various_types(tmp_path):
    """Test save_to_pickle with different data types."""
    file_path = tmp_path / "types.pkl"
    test_cases = [
        123,
        3.14,
        "hello",
        [1, 2, 3],
        {"a": 1, "b": [2, 3]},
        (1, 2, 3),
        None,
    ]
    for obj in test_cases:
        save_to_pickle(obj, file_path)
        loaded = load_from_pickle(file_path)
        assert loaded == obj, f"Loaded object should match original: {obj}"


# ───────────────────────────────────────────────────────────
# Tests for: load_from_pickle() — requires multiple sub-tests
# ───────────────────────────────────────────────────────────
def test_load_from_pickle_reads_correct_data(tmp_path):
    """Test that load_from_pickle reads back the correct data."""
    data = {"alpha": [1, 2, 3], "beta": {"x": 10}}
    file_path = tmp_path / "test_pickle.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    loaded = load_from_pickle(file_path)
    assert loaded == data, "Loaded data should match the pickled data"


def test_load_from_pickle_with_nonexistent_file(tmp_path):
    """Test that load_from_pickle raises FileNotFoundError for missing file."""
    file_path = tmp_path / "does_not_exist.pkl"
    with pytest.raises(FileNotFoundError):
        load_from_pickle(file_path)


def test_load_from_pickle_with_invalid_pickle(tmp_path):
    """Test that load_from_pickle raises an exception for invalid pickle file."""
    file_path = tmp_path / "invalid.pkl"
    file_path.write_text("not a pickle")
    with pytest.raises(Exception):
        load_from_pickle(file_path)


# ───────────────────────────────────────────────────────────────
# Tests for: add_missing_elements() — requires multiple sub-tests
# ───────────────────────────────────────────────────────────────
def test_add_missing_elements_adds_missing_rows():
    """Test add_missing_elements adds missing rows to a DataFrame."""
    df = pd.DataFrame({"A": [1, 2]}, index=["x", "y"])
    existing_elements = set(df.index)
    all_elements = {"x", "y", "z"}
    result = add_missing_elements(
        df.copy(), existing_elements, all_elements, axis=0, fill_value=99
    )
    assert "z" in result.index
    assert (result.loc["z"] == 99).all()
    assert set(result.index) == all_elements
    assert result["A"].dtype == df["A"].dtype


def test_add_missing_elements_adds_missing_columns():
    """Test add_missing_elements adds missing columns to a DataFrame."""
    df = pd.DataFrame({"A": [1, 2]}, index=["x", "y"])
    existing_elements = set(df.columns)
    all_elements = {"A", "B"}
    result = add_missing_elements(
        df.copy(), existing_elements, all_elements, axis=1, fill_value=-1
    )
    assert "B" in result.columns
    assert (result["B"] == -1).all()
    assert set(result.columns) == all_elements
    assert result["A"].dtype == df["A"].dtype


def test_add_missing_elements_no_missing_elements():
    """Test add_missing_elements when there are no missing elements."""
    df = pd.DataFrame({"A": [1, 2]}, index=["x", "y"])
    existing_elements = set(df.index)
    all_elements = set(df.index)
    result = add_missing_elements(
        df.copy(), existing_elements, all_elements, axis=0, fill_value=0
    )
    pd.testing.assert_frame_equal(result, df)


def test_add_missing_elements_with_non_default_fill_value():
    """Test add_missing_elements with a custom fill_value."""
    df = pd.DataFrame({"A": [1]}, index=["x"])
    existing_elements = set(df.index)
    all_elements = {"x", "y"}
    result = add_missing_elements(
        df.copy(), existing_elements, all_elements, axis=0, fill_value="foo"
    )
    assert result.loc["y", "A"] == "foo"


def test_add_missing_elements_sorting():
    """Test add_missing_elements sorts the index/columns as expected."""
    df = pd.DataFrame({"B": [1], "A": [2]}, index=["y"])
    existing_elements = set(df.columns)
    all_elements = {"A", "B", "C"}
    result = add_missing_elements(
        df.copy(), existing_elements, all_elements, axis=1, fill_value=0
    )
    assert list(result.columns) == ["A", "B", "C"]


# ────────────────────────────────────────────────────────────────────────
# Tests for: create_block_diagonal_matrix() — requires multiple sub-tests
# ────────────────────────────────────────────────────────────────────────
def test_create_block_diagonal_matrix_basic():
    """Test block diagonal creation with matrices of different shapes."""
    arr1 = np.ones((2, 3))
    arr2 = np.full((1, 2), 5)
    arr3 = np.eye(3)
    result = create_block_diagonal_matrix([arr1, arr2, arr3])
    # Shape should be (2+1+3, 3+2+3) = (6, 8)
    assert result.shape == (6, 8)
    # Check blocks
    np.testing.assert_array_equal(result.iloc[0:2, 0:3].values, arr1)
    np.testing.assert_array_equal(result.iloc[2:3, 3:5].values, arr2)
    np.testing.assert_array_equal(result.iloc[3:6, 5:8].values, arr3)
    # Off-diagonal blocks should be zeros
    assert (result.iloc[0:2, 3:8] == 0).all().all()
    assert (result.iloc[2:3, list(range(0, 3)) + list(range(5, 8))] == 0).all().all()
    assert (result.iloc[3:6, 0:5] == 0).all().all()


def test_create_block_diagonal_matrix_empty_list():
    """Test with an empty list."""
    result = create_block_diagonal_matrix([])
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (1, 0)


# ──────────────────────────────────────────────────────────
# Tests for: process_matrix() — requires multiple sub-tests
# ──────────────────────────────────────────────────────────
def test_process_matrix_normalize_rows():
    """Test normalization along rows (axis=0)."""
    df = pd.DataFrame([[1, 1], [2, 0]], columns=["A", "B"])
    result = process_matrix(df, "normalize", axis=0)
    expected = pd.DataFrame([[0.5, 0.5], [1.0, 0.0]], columns=["A", "B"])
    pd.testing.assert_frame_equal(result, expected)


def test_process_matrix_normalize_columns():
    """Test normalization along columns (axis=1)."""
    df = pd.DataFrame([[2, 3], [2, 9]], columns=["A", "B"])
    result = process_matrix(df, "normalize", axis=1)
    expected = pd.DataFrame([[0.5, 0.25], [0.5, 0.75]], columns=["A", "B"])
    pd.testing.assert_frame_equal(result, expected)


def test_process_matrix_normalize_with_zeros():
    """Test normalization where some sums are zero."""
    df = pd.DataFrame([[0, 0], [1, 1]], columns=["A", "B"])
    result = process_matrix(df, "normalize", axis=0)
    # First row sum is zero, so should be filled with zeros
    expected = pd.DataFrame([[0.0, 0.0], [0.5, 0.5]], columns=["A", "B"])
    pd.testing.assert_frame_equal(result, expected)


def test_process_matrix_reorder_levels_columns():
    """Test reorder_levels on columns MultiIndex."""
    arrays = [["one", "one", "two", "two"], ["A", "B", "A", "B"]]
    columns = pd.MultiIndex.from_arrays(arrays, names=["num", "letter"])
    df = pd.DataFrame(np.arange(8).reshape(2, 4), columns=columns)

    result = process_matrix(df, "reorder_levels", axis=1, order=["letter", "num"])
    expected_columns = pd.MultiIndex.from_arrays(
        [["A", "A", "B", "B"], ["one", "two", "one", "two"]], names=["letter", "num"]
    )
    expected_result = pd.DataFrame(
        [[0, 2, 1, 3], [4, 6, 5, 7]], index=df.index, columns=expected_columns
    )
    assert all(result == expected_result)


def test_process_matrix_reorder_levels_index():
    """Test reorder_levels on index MultiIndex."""
    arrays = [["one", "one", "two", "two"], ["A", "B", "A", "B"]]
    index = pd.MultiIndex.from_arrays(arrays, names=["num", "letter"])
    df = pd.DataFrame(np.arange(4), index=index, columns=["value"])

    result = process_matrix(df, "reorder_levels", axis=0, order=["letter", "num"])
    expected_index = pd.MultiIndex.from_arrays(
        [["A", "A", "B", "B"], ["one", "two", "one", "two"]], names=["letter", "num"]
    )
    expected_result = pd.DataFrame(
        [0, 2, 1, 3], index=expected_index, columns=df.columns
    )
    assert all(result == expected_result)


def test_process_matrix_invalid_operation():
    """Test that an invalid operation raises ValueError."""
    df = pd.DataFrame([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Unknown operation"):
        process_matrix(df, "not_a_real_operation")


# ──────────────────────────────────────────────────────────
# Tests for: treating_data() — requires multiple sub-tests
# ──────────────────────────────────────────────────────────
@pytest.fixture
def mock_treating_data_setup(tmp_path):
    """Fixture to create minimal files and data for treating_data."""
    year = 2023
    n_c = 2
    n_p = 1
    t_index = pd.date_range("2023-01-01 00:00:00", periods=2, freq="h")
    # Minimal Z_net and Z_load DataFrames
    columns = pd.MultiIndex.from_product(
        [t_index, ["solar"], ["DE", "CY"]], names=["time", "source", "country"]
    )
    index = pd.Index([0, 1])
    Z_net = pd.DataFrame(np.ones((2, len(columns))), index=index, columns=columns)
    Z_load = pd.DataFrame(np.ones((2, n_c)), columns=["A", "B"])
    Z_indices = {"index": Z_net.index, "columns": Z_net.columns}

    data_dir = tmp_path / "data"
    data_year_dir = tmp_path / "data" / f"{year}"
    data_year_dir.mkdir(parents=True, exist_ok=True)
    # Save required pickles
    save_to_pickle(Z_indices, data_year_dir / f"indices_{year}.pkl")
    save_to_pickle(Z_load, data_year_dir / f"Z_load_{year}.pkl")

    return {
        "year": year,
        "n_c": n_c,
        "n_p": n_p,
        "t_index": t_index,
        "Z_net": Z_net,
        "data_dir": data_dir,
        "Z_indices": Z_indices,
        "Z_load": Z_load,
    }


@patch("shrecc.treatment.calculate_Z_cons")
@patch("shrecc.treatment.process_matrix")
@patch("shrecc.treatment.concatenate_results")
@patch("shrecc.treatment.process_results_light")
@patch("shrecc.treatment.calculate_results")
def test_treating_data_creates_expected_files(
    mock_calc_results,
    mock_proc_light,
    mock_concat,
    mock_proc_matrix,
    mock_calc_Z_cons,
    mock_treating_data_setup,
):
    setup = mock_treating_data_setup
    # Create dummy matrices to return from mocks
    dummy_results = {"t": np.eye((setup["n_p"] + 1) * setup["n_c"])}
    dummy_results_light = pd.DataFrame(
        np.full((2, 4), 1), index=setup["Z_net"].index, columns=setup["Z_net"].columns
    )
    dummy_L = pd.DataFrame(
        np.full((2, 4), 2), index=setup["Z_net"].index, columns=setup["Z_net"].columns
    )
    dummy_L_series = pd.DataFrame(
        np.full((2, 4), 3), index=setup["Z_net"].index, columns=setup["Z_net"].columns
    )
    dummy_Z_cons = pd.DataFrame(
        np.full((2, 4), 4), index=setup["Z_net"].index, columns=setup["Z_net"].columns
    )

    mock_calc_results.return_value = (dummy_results, setup["Z_load"])
    mock_proc_light.return_value = dummy_results_light
    mock_concat.return_value = dummy_L
    mock_proc_matrix.return_value = dummy_L_series
    mock_calc_Z_cons.return_value = dummy_Z_cons

    year = setup["year"]
    treating_data(
        year,
        setup["n_c"],
        setup["n_p"],
        setup["t_index"],
        setup["Z_net"],
        setup["data_dir"],
    )

    # Check for created files
    assert (setup["data_dir"] / f"{year}" / f"Z_load_lv_{year}.pkl").exists()
    # Check called functions
    args, _ = mock_calc_results.call_args
    assert args[0] == year
    assert args[1] == setup["n_c"]
    assert args[2] == setup["n_p"]
    pd.testing.assert_index_equal(args[3], setup["t_index"])
    pd.testing.assert_frame_equal(args[4], setup["Z_net"])
    pd.testing.assert_frame_equal(args[5], setup["Z_load"])
    assert args[6] == setup["data_dir"]

    args, _ = mock_proc_light.call_args
    assert args[0] == dummy_results
    assert args[2] == setup["n_c"]

    args, _ = mock_concat.call_args
    pd.testing.assert_frame_equal(args[0], dummy_results_light)
    pd.testing.assert_frame_equal(args[1], setup["Z_net"])

    args, kwargs = mock_proc_matrix.call_args
    pd.testing.assert_frame_equal(args[0], dummy_L)
    assert args[1] == "reorder_levels"
    assert kwargs["order"] == ["source", "country", "time"]
    assert kwargs["axis"] == 1

    args, _ = mock_calc_Z_cons.call_args
    pd.testing.assert_frame_equal(args[1], dummy_L_series)
    assert set(args[3].keys()) == set(setup["Z_indices"].keys())
    pd.testing.assert_index_equal(args[3]["columns"], setup["Z_indices"]["columns"])
    pd.testing.assert_index_equal(args[3]["index"], setup["Z_indices"]["index"])


# ─────────────────────────────────────────────────────────────
# Tests for: calculate_results() — requires multiple sub-tests
# ─────────────────────────────────────────────────────────────
@pytest.fixture
def calculate_results_setup(tmp_path):
    """Fixture to create data for calculate_results."""
    year = 2023
    sources = ["Hydro pumped storage consumption", "solar", "wind", "trade"]
    countries = ["CY", "DE", "FR", "IE", "LU"]
    n_c = len(countries)
    n_p = len(sources) - 1  # Exclude 'trade' from production mix

    t_index = pd.date_range("2023-01-01 00:00:00", periods=3, freq="h")
    row_index = pd.MultiIndex.from_product(
        [sources, countries], names=["source", "country"]
    )
    col_index = pd.MultiIndex.from_product(
        [t_index, ["trade"], countries], names=["time", "source", "country"]
    )

    Z_net = pd.DataFrame(
        np.ones((len(row_index), len(col_index))), index=row_index, columns=col_index
    )
    Z_load = pd.DataFrame(
        np.ones((n_c, len(t_index))), index=countries, columns=t_index
    )
    Z_load.index.name = "country"
    Z_load.columns.name = "time"

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return {
        "year": year,
        "n_c": n_c,
        "n_p": n_p,
        "t_index": t_index,
        "Z_net": Z_net,
        "Z_load": Z_load,
        "root": tmp_path,
        "data_dir": data_dir,
    }


def test_calculate_results_basic(calculate_results_setup):
    """Test that calculate_results creates expected pickle files and returns correct types."""
    setup = calculate_results_setup
    year = setup["year"]
    cons_results = setup["data_dir"] / f"{year}" / f"cons_results_{year}.pkl"
    load_results = setup["data_dir"] / f"{year}" / f"load_results_{year}.pkl"
    cons_results.parent.mkdir(parents=True, exist_ok=True)
    results, results_load = calculate_results(
        setup["year"],
        setup["n_c"],
        setup["n_p"],
        setup["t_index"],
        setup["Z_net"],
        setup["Z_load"],
        setup["data_dir"],
    )

    # Check that pickle files exist
    assert cons_results.exists()
    assert load_results.exists()
    # Check returned types
    assert isinstance(results, dict)
    assert isinstance(results_load, pd.DataFrame)
    # Check that calculate_results returns csr_matrix in results dict.
    import scipy.sparse

    for v in results.values():
        assert isinstance(v, scipy.sparse.csr_matrix)
    # Check that results dict has keys for each t_index
    for t in setup["t_index"]:
        assert t in results
    # Check that the load loss factor has been applied correctly
    expected_load = pd.DataFrame(
        0.965, index=setup["Z_load"].index, columns=setup["Z_load"].columns
    )
    pd.testing.assert_frame_equal(results_load, expected_load)


@patch("shrecc.treatment.spsolve")
def test_calculate_results_loads_existing_pickles(
    mock_spsolve, calculate_results_setup
):
    """Test that calculate_results loads from existing pickle if present."""
    setup = calculate_results_setup
    year = setup["year"]

    # Pre-create cons_results pickle
    dummy_results = {
        t: np.eye((setup["n_p"] + 1) * setup["n_c"]) for t in setup["t_index"]
    }
    cons_results = setup["data_dir"] / f"{year}" / f"cons_results_{year}.pkl"
    cons_results.parent.mkdir(parents=True, exist_ok=True)
    save_to_pickle(dummy_results, cons_results)

    # Call function
    results, _ = calculate_results(
        setup["year"],
        setup["n_c"],
        setup["n_p"],
        setup["t_index"],
        setup["Z_net"],
        setup["Z_load"],
        setup["data_dir"],
    )

    # Assertions
    for t in dummy_results:
        assert t in results, f"Missing timestamp {t} in results"
        assert np.array_equal(results[t], dummy_results[t]), f"Mismatch at time {t}"

    mock_spsolve.assert_not_called()


def test_calculate_results_handles_zero_columns(calculate_results_setup):
    """Test that calculate_results does not divide by zero when Z_t.sum(axis=0) == 0."""
    setup = calculate_results_setup
    year = setup["year"]
    cons_results = setup["data_dir"] / f"{year}" / f"cons_results_{year}.pkl"
    cons_results.parent.mkdir(parents=True, exist_ok=True)
    # Set Z_net to zeros for one time step
    setup["Z_net"].iloc[:, :] = 0
    # Should not raise
    calculate_results(
        setup["year"],
        setup["n_c"],
        setup["n_p"],
        setup["t_index"],
        setup["Z_net"],
        setup["Z_load"],
        setup["data_dir"],
    )


# ─────────────────────────────────────────────────────────────────
# Tests for: process_results_light() — requires multiple sub-tests
# ─────────────────────────────────────────────────────────────────
def test_process_results_light_basic(tmp_path):
    """Test process_results_light creates a light results pickle and loads it if exists."""
    # Prepare dummy results dict with both ndarray and sparse matrix
    arr = np.arange(16).reshape(4, 4)
    sparse = csr_matrix(np.arange(16).reshape(4, 4))
    results = {
        "a": arr,
        "b": sparse,
    }
    n_c = 2
    filename = tmp_path / "results_light.pkl"

    # Should create new file and return correct shapes/types
    results_light = process_results_light(results, filename, n_c)
    assert filename.exists()
    assert set(results_light.keys()) == {"a", "b"}
    # ndarray case: should be float32 and shape (4, 2)
    assert results_light["a"].dtype == np.float32
    assert results_light["a"].shape == (4, 2)
    # sparse case: should be csc_matrix, float32, shape (4, 2)
    assert isinstance(results_light["b"], csc_matrix)
    assert results_light["b"].shape == (4, 2)
    assert results_light["b"].dtype == np.float32
    # Check that ndarray and sparse matrix with same data yield similar output.
    np.testing.assert_array_equal(results_light["a"], arr.astype("float32")[:, -n_c:])
    np.testing.assert_array_equal(
        results_light["b"].toarray(), arr.astype("float32")[:, -n_c:]
    )

    # Should load from pickle if file exists (simulate by overwriting)
    dummy = {"a": "b"}
    with open(filename, "wb") as f:
        pickle.dump(dummy, f)
    expected_results = process_results_light(results, filename, n_c)
    assert expected_results == dummy


def test_process_results_light_handles_empty_results(tmp_path):
    """Test process_results_light with empty results dict."""
    filename = tmp_path / "empty.pkl"
    results = {}
    n_c = 1
    results_light = process_results_light(results, filename, n_c)
    assert results_light == {}
    assert filename.exists()


# ──────────────────────────────────────────────────────────────
# Tests for: concatenate_results() — requires multiple sub-tests
# ──────────────────────────────────────────────────────────────
def test_concatenate_results_basic():
    """Test concatenate_results with two time steps and simple sparse matrices."""
    # Prepare input data
    results = {
        "2023-01-01 00:00:00": np.array([[1, 0], [0, 2]]),
        "2023-01-02 01:00:00": csr_matrix([[3, 4], [5, 6]]),
    }
    columns = pd.MultiIndex.from_product([["A", "B"], ["x", "y"]])
    index = pd.Index(["I", "II"])
    Z = pd.DataFrame(np.ones((2, 4)), index=index, columns=columns)
    # Call concatenate_results
    L_df = concatenate_results(results, Z)
    expected_L_df = pd.DataFrame(
        [[1, 0, 3, 4], [0, 2, 5, 6]], index=index, columns=columns
    )
    # Should be DataFrame with same index/columns as Z
    assert isinstance(L_df, pd.DataFrame)
    assert list(L_df.index) == list(Z.index)
    assert all(L_df.columns == Z.columns)
    # Should have correct shape
    assert L_df.shape == Z.shape
    assert (
        L_df.values == expected_L_df.values
    ).all(), "Concatenated DataFrame does not match expected values"


def test_concatenate_results_handles_empty_results():
    """Test concatenate_results with empty results dict."""
    results = {}
    Z = pd.DataFrame()
    L_df = concatenate_results(results, Z)
    assert isinstance(L_df, pd.DataFrame)
    assert L_df.empty


# ──────────────────────────────────────────────────────────────
# Tests for: calculate_Z_cons() — requires multiple sub-tests
# ──────────────────────────────────────────────────────────────
@pytest.fixture
def mock_z_cons_setup(tmp_path):
    # Create minimal L_series DataFrame and output Series
    index = pd.Index(["i1", "i2"])
    columns1 = pd.MultiIndex.from_product(
        [["source1"], ["country1", "country2"], ["time1"]],
        names=["source", "country", "time"],
    )
    columns2 = pd.MultiIndex.from_product(
        [["time1"], ["source1"], ["country1", "country2"]],
        names=["time", "source", "country"],
    )
    L_series = pd.DataFrame([[1, 2], [3, 4]], index=index, columns=columns1)
    output = pd.Series([10, 20], index=columns1)
    Z_indices = {"index": index, "columns": columns2}
    filename = tmp_path / "z_cons.pkl"
    return L_series, output, Z_indices, filename


@patch("shrecc.treatment.process_matrix")
def test_calculate_z_cons_basic(mock_process_matrix, mock_z_cons_setup):
    """Test that calculate_Z_cons computes, saves, and returns correct DataFrame when file does not exist."""
    L_series, output, Z_indices, filename = mock_z_cons_setup

    # Expected Z_cons DataFrame
    expected_Z_cons = pd.DataFrame(
        [[10, 40], [30, 80]], index=L_series.index, columns=L_series.columns
    )
    # Expected Z_cons_sp DataFrame
    expected_sp = pd.DataFrame(
        [[10, 40], [30, 80]], index=Z_indices["index"], columns=Z_indices["columns"]
    )
    mock_process_matrix.return_value = expected_sp
    # Call function (file does not exist yet)
    result = calculate_Z_cons(filename, L_series, output, Z_indices)

    # Should return DataFrame with correct index/columns and values
    pd.testing.assert_frame_equal(result, expected_Z_cons)

    # Check that process_matrix was called with expected arguments
    called_args, called_kwargs = mock_process_matrix.call_args
    # Check positional arguments
    pd.testing.assert_frame_equal(called_args[0], expected_Z_cons)
    assert called_args[1] == "reorder_levels"
    # Check keyword arguments
    assert called_kwargs == {"order": ["time", "source", "country"], "axis": 1}

    # Should have saved a coo_array to file
    assert filename.exists()
    loaded = load_from_pickle(filename)
    # Check that loaded is a coo_array and matches expected values
    assert isinstance(loaded, coo_array)
    np.testing.assert_array_equal(loaded.todense(), expected_sp.values)


def test_calculate_z_cons_loads_existing_sparse(mock_z_cons_setup):
    """Test that calculate_Z_cons loads and returns DataFrame from existing pickle file."""
    L_series, output, Z_indices, filename = mock_z_cons_setup

    # Save a coo_array to file first
    arr = np.array([[100, 200], [300, 400]])
    coo = coo_array(arr)
    save_to_pickle(coo, filename)

    # Call function (file exists)
    result = calculate_Z_cons(filename, L_series, output, Z_indices)

    # Should return DataFrame with correct index/columns and loaded values
    expected_results = pd.DataFrame(
        arr, index=Z_indices["index"], columns=Z_indices["columns"]
    )
    pd.testing.assert_frame_equal(result, expected_results)


@patch("shrecc.treatment.process_matrix", new=lambda df, op, **kwargs: df)
def test_calculate_z_cons_with_zero_output(mock_z_cons_setup):
    """Test calculate_Z_cons with output series of zeros."""
    L_series, output, Z_indices, filename = mock_z_cons_setup
    output[:] = 0
    result = calculate_Z_cons(filename, L_series, output, Z_indices)
    assert (result.values == 0).all()
