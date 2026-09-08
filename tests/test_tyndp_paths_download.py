import tempfile
from unittest.mock import MagicMock, patch

import pytest
import requests

from shrecc.tyndp import download_tyndp_scenario_zip, tyndp_scenario_paths, validate_tyndp_scenario

CLIMATE_YEAR = 2009
URL_ROOT = "https://2024-data.entsos-tyndp-scenarios.eu/files/scenarios-outputs/"

# ── path construction ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("scenario, year, expected_zip", [
    ("GA", 2040, "GA2040CY2009.zip"),
    ("DE", 2040, "DE2040CY2009.zip"),
    ("DE", 2035, "DE2035CY1995.zip".replace("1995", str(CLIMATE_YEAR))),
])
def test_ga_de_zip_name(scenario, year, expected_zip):
    with tempfile.TemporaryDirectory() as tmp:
        paths = tyndp_scenario_paths(tmp, scenario, year, CLIMATE_YEAR)
    assert paths["zip"].name == expected_zip


@pytest.mark.parametrize("year", [2030, 2040])
def test_nt_url_matches_known_pattern(year):
    expected = (
        f"{URL_ROOT}MMStandardOutputFile_NT{year}_Plexos"
        f"_CY{CLIMATE_YEAR}_2.5_v40.xlsx.zip"
    )
    with tempfile.TemporaryDirectory() as tmp:
        paths = tyndp_scenario_paths(tmp, "NT", year, CLIMATE_YEAR)
    assert paths["url"] == expected


@pytest.mark.parametrize("year", [2030, 2040])
def test_nt_workbook_is_xlsx_not_xlsb(year):
    with tempfile.TemporaryDirectory() as tmp:
        paths = tyndp_scenario_paths(tmp, "NT", year, CLIMATE_YEAR)
    assert paths["workbook"].suffix == ".xlsx"


@pytest.mark.parametrize("scenario, year", [
    ("GA", 2040),
    ("DE", 2040),
])
def test_ga_de_workbook_is_xlsb(scenario, year):
    with tempfile.TemporaryDirectory() as tmp:
        paths = tyndp_scenario_paths(tmp, scenario, year, CLIMATE_YEAR)
    assert paths["workbook"].suffix == ".xlsb"


# ── validation ────────────────────────────────────────────────────────────────

def test_unsupported_nt_year_raises_value_error():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="scenario year"):
            download_tyndp_scenario_zip("NT", 2050, CLIMATE_YEAR, tmp)


def test_unsupported_scenario_raises_value_error():
    with pytest.raises(ValueError, match="Unknown TYNDP scenario"):
        validate_tyndp_scenario("XX", 2040, CLIMATE_YEAR)


def test_unsupported_climate_year_raises_value_error():
    with pytest.raises(ValueError, match="climate year"):
        validate_tyndp_scenario("GA", 2040, 1999)


# ── download logic (mocked — no network) ─────────────────────────────────────

@pytest.mark.parametrize("scenario, year", [
    ("NT", 2030),
    ("NT", 2040),
    ("GA", 2040),
    ("DE", 2040),
])
def test_download_calls_correct_url(scenario, year):
    """Confirms the function requests the right URL without hitting the network."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = tyndp_scenario_paths(tmp, scenario, year, CLIMATE_YEAR)
        expected_url = paths["url"]

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"data"]
        mock_response.raise_for_status.return_value = None

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        zip_path = download_tyndp_scenario_zip(
            scenario=scenario,
            year=year,
            climate_year=CLIMATE_YEAR,
            data_dir=tmp,
            session=mock_session,
        )

        mock_session.get.assert_called_once_with(
            expected_url,
            stream=True,
            timeout=120,
        )
        assert zip_path.exists()


# ── live URL check (skipped in offline environments) ─────────────────────────

@pytest.mark.parametrize("scenario, year", [
    ("NT", 2030),
    ("NT", 2040),
    ("GA", 2040),
    ("DE", 2040),
])
def test_url_is_reachable(scenario, year):
    """HEAD request to confirm the URL returns 200. Skip if no network."""
    with tempfile.TemporaryDirectory() as tmp:
        url = tyndp_scenario_paths(tmp, scenario, year, CLIMATE_YEAR)["url"]
    try:
        r = requests.head(url, timeout=10, allow_redirects=True)
    except requests.RequestException as e:
        pytest.skip(f"Network unavailable: {e}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code} for {url}"