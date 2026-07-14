import zipfile

import numpy as np
import pandas as pd
import pytest

from shrecc import tyndp


def test_load_technology_concordance_accepts_csv(tmp_path):
    concordance_file = tmp_path / "technology_mapping.csv"
    pd.DataFrame(
        {
            "wind": [1, 3, 0],
            "unused": [0, 0, 0],
        },
        index=pd.Index(["wind small", "wind large", "network"], name="activity"),
    ).to_csv(concordance_file)

    result = tyndp.load_technology_concordance(
        concordance_file,
        sheet_name="ignored for csv",
    )

    expected = pd.DataFrame(
        {"wind": [0.25, 0.75]},
        index=pd.Index(["wind small", "wind large"], name="activity"),
    )
    expected.columns.name = "Category"

    pd.testing.assert_frame_equal(result, expected)


def test_load_tyndp_country_mapping_accepts_csv_directory(tmp_path):
    mapping_dir = tmp_path / "tyndp"
    mapping_dir.mkdir()
    countries_file = mapping_dir / "tyndp_countries.csv"
    connections_file = mapping_dir / "tyndp_connections.csv"

    pd.DataFrame(
        {
            "Country code": ["ES", "PT"],
            "Country name": ["Spain", "Portugal"],
        },
        index=pd.Index(["ES00", "PT00"], name="Bidding Zone"),
    ).to_csv(countries_file)
    pd.DataFrame(
        {
            "BZ from": ["ES00"],
            "BZ to": ["PT00"],
            "Country from": ["ES"],
            "Country to": ["PT"],
            "Country line": ["ES-PT"],
        },
        index=pd.Index(["ES00-PT00"], name="BZ line"),
    ).to_csv(connections_file)

    countries, connections = tyndp.load_tyndp_country_mapping(mapping_dir)

    assert countries.loc["ES00", "Country code"] == "ES"
    assert connections.loc["ES00-PT00", "Country to"] == "PT"


def test_tyndp_scenario_paths_normalizes_inputs(tmp_path):
    paths = tyndp.tyndp_scenario_paths(
        tmp_path,
        scenario="de",
        year="2050",
        climate_year="2009",
    )

    assert paths["production_pickle"] == tmp_path / "DE2050_CY2009_prod.pkl"
    assert paths["trade_pickle"] == tmp_path / "DE2050_CY2009_trade.pkl"
    assert paths["zip"] == tmp_path / "DE2050CY2009.zip"
    assert paths["workbook"] == (
        tmp_path / "MMStandardOutputFile_DE2050_Plexos_CY2009_v11_SoS.xlsb"
    )
    assert paths["url"].endswith("/DE2050CY2009.zip")


def test_tyndp_scenario_paths_supports_de_2035_snapshot(tmp_path):
    paths = tyndp.tyndp_scenario_paths(tmp_path, "DE", 2035, 2009)

    assert paths["production_pickle"] == tmp_path / "DE2035_CY2009_prod.pkl"
    assert paths["trade_pickle"] == tmp_path / "DE2035_CY2009_trade.pkl"
    assert paths["zip"] == tmp_path / "DE2035CY2009.zip"
    assert paths["workbook"] == (
        tmp_path / "MMStandardOutputFile_DE2035_Plexos_CY2009_v11_SoS.xlsb"
    )
    assert paths["url"] == (
        "https://2024-data.entsos-tyndp-scenarios.eu/files/scenarios-outputs/"
        "DE2035CY2009.zip"
    )


@pytest.mark.parametrize(
    "scenario, year, climate_year",
    [
        ("DE", 2035, 2009),
        ("GA", 2035, 2009),
        ("DE", 2040, 1995),
        ("GA", 2050, 2008),
        ("NT", 2030, 2009),
        ("NT", 2040, 2009),
    ],
)
def test_validate_tyndp_scenario_accepts_known_values(
    scenario,
    year,
    climate_year,
):
    assert tyndp.validate_tyndp_scenario(scenario, year, climate_year) == (
        scenario,
        year,
        climate_year,
    )


@pytest.mark.parametrize(
    "scenario, year, climate_year",
    [
        ("bad", 2050, 2009),
        ("DE", 2060, 2009),
        ("DE", 2030, 2009),
        ("NT", 2035, 2009),
        ("NT", 2050, 2009),
        ("DE", 2050, 2010),
    ],
)
def test_validate_tyndp_scenario_rejects_unknown_values(
    scenario,
    year,
    climate_year,
):
    with pytest.raises(ValueError):
        tyndp.validate_tyndp_scenario(scenario, year, climate_year)


def test_ensure_tyndp_workbook_extracts_existing_zip(tmp_path):
    paths = tyndp.tyndp_scenario_paths(tmp_path, "DE", 2050, 2009)
    with zipfile.ZipFile(paths["zip"], "w") as archive:
        archive.writestr(f"nested/{paths['workbook'].name}", b"xlsb-bytes")

    workbook = tyndp.ensure_tyndp_workbook(
        "DE",
        2050,
        2009,
        tmp_path,
        download=False,
    )

    assert workbook == paths["workbook"]
    assert workbook.read_bytes() == b"xlsb-bytes"


def test_ensure_tyndp_workbook_raises_when_download_disabled(tmp_path):
    with pytest.raises(FileNotFoundError):
        tyndp.ensure_tyndp_workbook(
            "DE",
            2050,
            2009,
            tmp_path,
            download=False,
        )


def test_build_z_gross_from_tyndp_scenario_uses_pickles_first(
    tmp_path,
    monkeypatch,
):
    paths = tyndp.tyndp_scenario_paths(tmp_path, "DE", 2050, 2009)
    paths["production_pickle"].write_bytes(b"prod")
    paths["trade_pickle"].write_bytes(b"trade")
    calls = []
    expected = pd.DataFrame({"value": [1]})

    def fake_from_pickles(**kwargs):
        calls.append(("pickles", kwargs))
        return expected

    def fake_from_excel(**kwargs):
        calls.append(("excel", kwargs))
        return pd.DataFrame()

    monkeypatch.setattr(tyndp, "build_z_gross_from_tyndp_pickles", fake_from_pickles)
    monkeypatch.setattr(tyndp, "build_z_gross_from_tyndp_excel", fake_from_excel)

    result = tyndp.build_z_gross_from_tyndp_scenario(
        "DE",
        2050,
        2009,
        tmp_path,
        technology_mapping="tech.xlsx",
        country_mapping="country.xlsx",
    )

    assert result is expected
    assert [name for name, _ in calls] == ["pickles"]
    assert calls[0][1]["production_pickle"] == paths["production_pickle"]
    assert calls[0][1]["trade_pickle"] == paths["trade_pickle"]


def test_build_z_gross_from_tyndp_scenario_uses_excel_and_saves_pickles(
    tmp_path,
    monkeypatch,
):
    paths = tyndp.tyndp_scenario_paths(tmp_path, "DE", 2050, 2009)
    paths["workbook"].write_bytes(b"xlsb")
    calls = []
    expected = pd.DataFrame({"value": [1]})

    def fake_from_excel(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(tyndp, "build_z_gross_from_tyndp_excel", fake_from_excel)

    result = tyndp.build_z_gross_from_tyndp_scenario(
        "DE",
        2050,
        2009,
        tmp_path,
        technology_mapping="tech.xlsx",
        country_mapping="country.xlsx",
        download=False,
    )

    assert result is expected
    assert calls[0]["excel_file"] == paths["workbook"]
    assert calls[0]["production_pickle"] == paths["production_pickle"]
    assert calls[0]["trade_pickle"] == paths["trade_pickle"]


def _tiny_z_gross_with_zero_consumption_hour():
    times = pd.to_datetime(
        [
            "2040-01-01 00:00",
            "2040-01-02 00:00",
            "2040-01-03 00:00",
            "2040-02-01 00:00",
        ]
    )
    columns = pd.MultiIndex.from_tuples(
        [
            ("production mix", "AL", "AL", "Solar"),
            ("production mix", "AL", "AL", "Wind"),
            ("production mix", "ME", "ME", "Solar"),
            ("production mix", "ME", "ME", "Wind"),
            ("trade", "AL", "ME", "electricity"),
            ("trade", "ME", "AL", "electricity"),
        ],
        names=["type", "country from", "country to", "source"],
    )
    return pd.DataFrame(
        [
            [2.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 6.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        index=times,
        columns=columns,
    )


def test_consumption_mix_zero_consumption_month_hour_average_fallback():
    Z_gross = _tiny_z_gross_with_zero_consumption_hour()

    consumption_mix, debug = tyndp.consumption_mix_from_z_gross(
        Z_gross,
        zero_consumption="month_hour_average",
        return_debug=True,
    )

    imputed = consumption_mix.sel(
        time="2040-01-03 00:00",
        consumer_country="AL",
        source_country="AL",
    )
    np.testing.assert_allclose(imputed.sel(technology="Solar"), 0.25)
    np.testing.assert_allclose(imputed.sel(technology="Wind"), 0.75)
    np.testing.assert_allclose(
        consumption_mix.sel(
            time="2040-01-03 00:00",
            consumer_country="AL",
        ).sum(),
        1.0,
    )

    assert debug["zero_consumption_mask"].sum() == 1
    assert debug["zero_consumption_imputed"].iloc[0]["fallback"] == "month_hour"


def test_consumption_mix_zero_consumption_still_raises_by_default():
    Z_gross = _tiny_z_gross_with_zero_consumption_hour()

    with pytest.raises(ValueError, match="zero total consumption"):
        tyndp.consumption_mix_from_z_gross(Z_gross)
