import pytest
import pandas as pd
import json
import numpy as np
import pandas.testing as pdt
from unittest.mock import patch, MagicMock, ANY
from shrecc.download import (
    get_prod,
    get_trade,
    get_data,
    year_to_unix,
    cleaning_data,
)


# ────────────────────────────────────────────────
# Tests for: year_to_unix() — single, direct test
# ────────────────────────────────────────────────
@pytest.mark.parametrize(
    "year, expected_start, expected_end",
    [
        (2023, 1672531200, 1704067199),
        (2022, 1640995200, 1672531199),
    ],
)
def test_year_to_unix(year, expected_start, expected_end):
    """Test the year_to_unix function to ensure it correctly converts a year to the start and end of that year in Unix time."""
    start, end = year_to_unix(year)
    assert start == expected_start
    # 1672531200 → 2023-01-01 00:00:00 UTC
    # 1640995200 → 2022-01-01 00:00:00 UTC
    assert end == expected_end
    # 1704067199 → 2023-12-31 23:59:59 UTC
    # 1672531199 → 2022-12-31 23:59:59 UTC


# ─────────────────────────────────────────────────────
# Tests for: get_prod() — requires multiple sub-tests
# ─────────────────────────────────────────────────────
@pytest.fixture
def mock_prod_api_response():
    """Fixture to mock the API response for get_prod."""
    # This is a mock response that simulates the expected structure from the API
    return {
        "production_types": [
            {"name": "Solar", "data": [1.0, 2.0, 3.0]},
            {"name": "Wind", "data": [4.0, 5.0, 6.0]},
            {"name": "Load", "data": [10.0, 11.0, 12.0]},
            {"name": "Residual load", "data": [13.0, 14.0, 15.0]},
        ],
        "unix_seconds": [1672531200, 1672534800, 1672538400],
        # Corresponds to 2023-01-01 00:00:00, 2023-01-01 01:00:00, 2023-01-01 02:00:00
    }


@patch("shrecc.download.requests.Session")
def test_get_prod_success(mock_session, mock_prod_api_response):
    # Mock the requests.Session().get().status_code and .text
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = json.dumps(mock_prod_api_response)
    mock_session.return_value.get.return_value = mock_response

    start = 1672531200
    end = 1672538400
    country = "de"
    prod_df, load_df, techs = get_prod(start, end, country)
    expected_times = [
        pd.Timestamp("2023-01-01 00:00:00"),
        pd.Timestamp("2023-01-01 01:00:00"),
        pd.Timestamp("2023-01-01 02:00:00"),
    ]
    # Check types
    assert isinstance(prod_df, pd.DataFrame)
    assert isinstance(load_df, pd.Series)
    assert isinstance(techs, list)
    # Check columns and index
    assert set(prod_df.columns) == {"Solar", "Wind"}
    assert all(isinstance(idx, pd.Timestamp) for idx in prod_df.index)
    assert list(prod_df.index) == expected_times
    assert set(load_df.index) == set(prod_df.index)
    # Check load_df values
    assert all(load_df.values == [10.0, 11.0, 12.0])
    # Check techs
    assert set(techs) == {"Solar", "Wind", "Load", "Residual load"}


@patch("shrecc.download.requests.Session")
def test_get_prod_404(mock_session):
    """Test the get_prod function for a 404 response."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_session.return_value.get.return_value = mock_response

    prod_df, load_df, techs = get_prod(0, 1, "xx")
    assert prod_df is None
    assert load_df is None
    assert techs is None


@patch("shrecc.download.requests.Session")
def test_get_prod_400(mock_session):
    """Test the get_prod function for a 400 response."""
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_session.return_value.get.return_value = mock_response

    prod_df, load_df, techs = get_prod(0, 1, "invalid")
    assert prod_df is None
    assert load_df is None
    assert techs is None


@patch("shrecc.download.requests.Session")
def test_get_prod_missing_load_column(mock_session, mock_prod_api_response):
    """Test the get_prod function when the 'Load' column is missing."""
    mock_prod_api_response = {
        "production_types": [
            {"name": "Solar", "data": [1.0, 2.0, 3.0]},
            {"name": "Wind", "data": [4.0, 5.0, 6.0]},
        ],
        "unix_seconds": [1672531200, 1672534800, 1672538400],
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = json.dumps(mock_prod_api_response)
    mock_session.return_value.get.return_value = mock_response

    prod_df, load_df, techs = get_prod(1672531200, 1672538400, "de")
    assert isinstance(prod_df, pd.DataFrame)
    assert load_df is None
    assert set(prod_df.columns) == {"Solar", "Wind"}
    assert set(techs) == {"Solar", "Wind"}


@patch("shrecc.download.requests.Session")
def test_get_prod_with_rolling_and_cumul(mock_session, mock_prod_api_response):
    """Test the get_prod function with rolling and cumul parameters."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = json.dumps(mock_prod_api_response)
    mock_session.return_value.get.return_value = mock_response

    prod_df_rolling, _, _ = get_prod(
        1672531200, 1672538400, "de", cumul=False, rolling=2
    )
    prod_df_cumul, _, _ = get_prod(1672531200, 1672538400, "de", cumul=True)
    prod_df_rolling_and_cumul, _, _ = get_prod(
        1672531200, 1672538400, "de", cumul=True, rolling=2
    )
    # Create expected DataFrame
    expected_df_rolling = pd.DataFrame(
        {
            "Solar": [np.nan, 3.0, 5.0],
            "Wind": [np.nan, 9.0, 11.0],
        },
        index=pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 01:00:00",
                "2023-01-01 02:00:00",
            ]
        ),
    )
    expected_df_cumul = pd.DataFrame(
        {
            "Solar": [1.0, 3.0, 6.0],
            "Wind": [4.0, 9.0, 15.0],
        },
        index=pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 01:00:00",
                "2023-01-01 02:00:00",
            ]
        ),
    )
    expected_df_rolling_and_cumul = pd.DataFrame(
        {
            "Solar": [np.nan, 3.0, 8.0],
            "Wind": [np.nan, 9.0, 20.0],
        },
        index=pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 01:00:00",
                "2023-01-01 02:00:00",
            ]
        ),
    )

    # Assert the DataFrame is as expected
    pdt.assert_frame_equal(prod_df_rolling, expected_df_rolling)
    pdt.assert_frame_equal(prod_df_cumul, expected_df_cumul)
    pdt.assert_frame_equal(prod_df_rolling_and_cumul, expected_df_rolling_and_cumul)


# ─────────────────────────────────────────────────────
# Tests for: get_trade() — requires multiple sub-tests
# ─────────────────────────────────────────────────────
@pytest.fixture
def mock_trade_api_response():
    """Fixture to mock the API response for get_trade."""
    return {
        "countries": [
            {"name": "France", "data": [1.1, 2.2, 3.3]},
            {"name": "Luxembourg", "data": [4.4, 5.5, 6.6]},
        ],
        "unix_seconds": [1672531200, 1672534800, 1672538400],
    }


@patch("shrecc.download.requests.Session")
def test_get_trade_success(mock_session, mock_trade_api_response):
    """Test the get_trade function with a successful API response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = json.dumps(mock_trade_api_response)
    mock_session.return_value.get.return_value = mock_response

    start = 1672531200
    end = 1672538400
    country = "de"
    trade_df, regions = get_trade(start, end, country)
    expected_index = pd.to_datetime(
        [
            "2023-01-01 00:00:00",
            "2023-01-01 01:00:00",
            "2023-01-01 02:00:00",
        ]
    )
    expected_columns = ["France", "Luxembourg"]
    assert isinstance(trade_df, pd.DataFrame)
    assert list(trade_df.index) == list(expected_index)
    assert list(trade_df.columns) == expected_columns
    np.testing.assert_array_equal(trade_df["France"].values, [1.1, 2.2, 3.3])
    np.testing.assert_array_equal(trade_df["Luxembourg"].values, [4.4, 5.5, 6.6])
    assert regions == expected_columns


@patch("shrecc.download.requests.Session")
def test_get_trade_404(mock_session):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_session.return_value.get.return_value = mock_response

    trade_df, regions = get_trade(0, 1, "xx")
    assert trade_df is None
    assert regions is None


# ─────────────────────────────────────────────────
# Tests for: cleaning_data() — single, direct test
# ─────────────────────────────────────────────────
def test_cleaning_data(tmp_path):
    """Test the cleaning_data function with basic input."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Prepare generation_units_by_country.csv
    gen_units = pd.DataFrame(
        {"en": ["France", "Luxembourg"], "short": ["FR", "LU"]},
    )
    gen_units.to_csv(data_dir / "generation_units_by_country.csv")
    # Prepare techs_agg.json
    techs_agg = {
        "Solar": "Solar",
        "Wind offshore": "Wind",
        "Wind onshore": "Wind",
        "Load": "Load",
    }
    with open(data_dir / "techs_agg.json", "w") as f:
        json.dump(techs_agg, f)
    # Prepare input data
    idx = pd.date_range("2023-01-01", periods=3, freq="h")
    data = {
        "de": {
            "production mix": pd.DataFrame(
                {
                    "Solar": [1, 2, 3],
                    "Wind offshore": [4, 5, 6],
                    "Wind onshore": [7, 8, 9],
                },
                index=idx,
            ),
            "trade": pd.DataFrame(
                {
                    "France": [1.1, 1.2, 1.3],
                    "Luxembourg": [2.1, 2.2, 2.3],
                    "Ireland": [3.1, 3.2, 3.3],
                },
                index=idx,
            ),
            "load": pd.Series([10, 11, 12], index=idx, name="Load"),
        },
        "fr": {},
    }
    result = cleaning_data(data, data_dir)
    # Check MultiIndex columns
    assert isinstance(result.columns, pd.MultiIndex)
    assert "DE" in result.columns.get_level_values("country")
    assert "FR" not in result.columns.get_level_values("country")
    assert "production mix" in result.columns.get_level_values("type")
    assert "trade" in result.columns.get_level_values("type")
    assert "load" in result.columns.get_level_values("type")
    assert "Solar" in result.columns.get_level_values("source")
    assert "Wind" in result.columns.get_level_values("source")
    # Check values
    assert np.allclose(result["DE", "production mix", "Solar"].values, [1, 2, 3])
    assert np.allclose(result["DE", "production mix", "Wind"].values, [11, 13, 15])
    assert np.allclose(
        result["DE", "trade", "FR"].values, [1100, 1200, 1300]
    )  # trade scaled by 1000
    assert np.allclose(result["DE", "trade", "LU"].values, [2100, 2200, 2300])
    assert np.allclose(result["DE", "trade", "IE"].values, [3100, 3200, 3300])
    assert np.allclose(result["DE", "load", "Load"].values, [10, 11, 12])


# ────────────────────────────────────────────────────
# Tests for: get_data() — requires multiple sub-tests
# ────────────────────────────────────────────────────
@pytest.fixture
def mock_pickle(tmp_path):
    """Fixture to mock a pickle file and return its path and dummy data."""
    data_dir = tmp_path / "data" / "2023"
    data_dir.mkdir(parents=True, exist_ok=True)
    filename = data_dir / "prod_and_trade_data_2023.pkl"
    dummy_data = {
        "de": {
            "production mix": pd.DataFrame(
                {"Solar": [1]}, index=[pd.Timestamp("2023-01-01 00:00:00")]
            ),
            "load": pd.Series([2], index=[pd.Timestamp("2023-01-01 00:00:00")]),
            "trade": pd.DataFrame(
                {"FR": [3]}, index=[pd.Timestamp("2023-01-01 00:00:00")]
            ),
        }
    }

    pd.to_pickle(dummy_data, filename)

    return filename, dummy_data


@patch("shrecc.download.load_from_pickle")
@patch("shrecc.download.cleaning_data")
def test_get_data_loads_existing_pickle(
    mock_cleaning, mock_load, tmp_path, mock_pickle
):
    """Test the get_data function when a pickle file exists."""
    # If pickle exists, should load and call cleaning_data
    filename, dummy_data = mock_pickle
    mock_load.return_value = dummy_data
    mock_cleaning.return_value = "cleaned"

    data_dir = tmp_path / "data"
    result = get_data(2023, path_to_data=data_dir)

    mock_load.assert_called_once_with(filename)
    mock_cleaning.assert_called_once_with(dummy_data, ANY)
    assert result == "cleaned"


@patch("shrecc.download.save_to_pickle")
@patch("shrecc.download.cleaning_data")
@patch("shrecc.download.get_trade")
@patch("shrecc.download.get_prod")
@patch("shrecc.download.load_from_pickle")
@patch("pathlib.Path.exists")
def test_get_data_downloads_and_saves(
    mock_path_exist,
    mock_load,
    mock_get_prod,
    mock_get_trade,
    mock_cleaning,
    mock_save,
    tmp_path,
):
    """Test the get_data function when no pickle file exists."""
    # If pickle does not exist, should call get_prod/get_trade for each country, save, and clean
    mock_path_exist.return_value = False  # Simulate no existing pickle file
    mock_get_prod.return_value = (
        pd.DataFrame({"Solar": [1]}, index=[pd.Timestamp("2023-01-01 00:00:00")]),
        pd.Series([2], index=[pd.Timestamp("2023-01-01 00:00:00")]),
        ["Solar"],
    )
    mock_get_trade.return_value = (
        pd.DataFrame({"FR": [3]}, index=[pd.Timestamp("2023-01-01 00:00:00")]),
        ["FR"],
    )
    mock_cleaning.return_value = "cleaned"

    data_dir = tmp_path / "data"
    result = get_data(2023, path_to_data=data_dir, max_retries=1)

    # Should call get_prod/get_trade for all countries (but we only check calls > 0)
    assert mock_get_prod.call_count > 0
    assert mock_get_trade.call_count > 0
    assert mock_save.called
    assert mock_cleaning.called
    assert mock_load.call_count == 0
    assert result == "cleaned"


@patch("shrecc.download.save_to_pickle")
@patch("shrecc.download.cleaning_data")
@patch("shrecc.download.get_trade")
@patch("shrecc.download.get_prod")
@patch("shrecc.download.load_from_pickle")
@patch("pathlib.Path.exists")
def test_get_data_retries_on_exception(
    mock_path_exist,
    mock_load,
    mock_get_prod,
    mock_get_trade,
    mock_cleaning,
    mock_save,
    tmp_path,
):
    """Test the get_data function with retries on exceptions."""
    # Simulate get_prod raising an exception the first time, then succeeding
    call_count = {"count": 0}

    def prod_side_effect(*args, **kwargs):
        if call_count["count"] == 0:
            call_count["count"] += 1
            raise Exception("fail once")
        return (
            pd.DataFrame({"Solar": [1]}, index=[pd.Timestamp("2023-01-01 00:00:00")]),
            pd.Series([2], index=[pd.Timestamp("2023-01-01 00:00:00")]),
            ["Solar"],
        )

    mock_path_exist.return_value = False  # Simulate no existing pickle file
    mock_get_prod.side_effect = prod_side_effect
    mock_get_trade.return_value = (
        pd.DataFrame({"FR": [3]}, index=[pd.Timestamp("2023-01-01 00:00:00")]),
        ["FR"],
    )
    mock_cleaning.return_value = "cleaned"

    data_dir = tmp_path / "data"
    result = get_data(2023, path_to_data=data_dir, max_retries=2, retry_delay=0)

    assert mock_get_prod.call_count >= 2
    assert mock_get_trade.call_count >= 2
    assert mock_save.called
    mock_load.assert_not_called()
    assert mock_cleaning.called
    assert result == "cleaned"


@patch("shrecc.download.save_to_pickle")
@patch("shrecc.download.cleaning_data")
@patch("shrecc.download.get_trade")
@patch("shrecc.download.get_prod")
@patch("shrecc.download.load_from_pickle")
@patch("pathlib.Path.exists")
def test_get_data_handles_failed_country(
    mock_path_exist,
    mock_load,
    mock_get_prod,
    mock_get_trade,
    mock_cleaning,
    mock_save,
    tmp_path,
    capsys,
):
    """Test the get_data function when get_prod and get_trade fail for a country."""
    # Simulate get_prod and get_trade always failing for a country
    mock_path_exist.return_value = False  # Simulate no existing pickle file
    mock_get_prod.side_effect = Exception("fail always")
    mock_get_trade.side_effect = Exception("fail always")
    mock_cleaning.return_value = "cleaned"

    data_dir = tmp_path / "data"
    get_data(2023, path_to_data=data_dir, max_retries=2, retry_delay=0)

    captured = capsys.readouterr()

    mock_load.assert_not_called()
    assert mock_save.called
    assert "Failed to fetch data" in captured.out
