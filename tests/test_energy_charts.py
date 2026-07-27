import numpy as np
import pandas as pd
import xarray as xr

from shrecc.energy_charts import (
    build_energy_charts_solver_inputs,
    consumption_results_from_energy_charts,
    data_processing,
)
from shrecc.database import filt_cutoff


def _energy_charts_data():
    columns = pd.MultiIndex.from_tuples(
        [
            ("A", "production mix", "Solar"),
            ("A", "production mix", "Wind"),
            ("A", "trade", "B"),
            ("A", "load", "load"),
            ("B", "production mix", "Solar"),
            ("B", "production mix", "Wind"),
            ("B", "trade", "A"),
            ("B", "load", "load"),
            ("C", "production mix", "Solar"),
            ("C", "production mix", "Wind"),
            ("C", "production mix", "Hydro pumped storage consumption"),
            ("C", "production mix", "Import balance (market)"),
            ("C", "load", "load"),
        ],
        names=["country", "type", "source"],
    )
    return pd.DataFrame(
        [[
            80.0,
            20.0,
            -40.0,
            60.0,
            0.0,
            60.0,
            40.0,
            100.0,
            0,
            0,
            -15.0,
            -25.0,
            0,
        ]],
        index=pd.to_datetime(["2025-01-01 00:00"]),
        columns=columns,
    )


def test_energy_charts_adapter_builds_directional_trade_and_measured_load():
    production, trade, consumption = build_energy_charts_solver_inputs(
        _energy_charts_data()
    )

    np.testing.assert_allclose(
        trade.sel(exporter_country="A", importer_country="B"),
        40,
    )
    np.testing.assert_allclose(
        trade.sel(exporter_country="B", importer_country="A"),
        0,
    )
    np.testing.assert_allclose(
        production.sel(producer_country="A", technology=["Solar", "Wind"]),
        [[80, 20]],
    )
    np.testing.assert_allclose(
        consumption.sel(consumer_country=["A", "B", "C"]),
        [[60, 100, 0]],
    )
    assert "Import balance (market)" not in production["technology"]
    assert float(production.min()) == 0


def test_energy_charts_results_retain_volume_and_mix():
    results = consumption_results_from_energy_charts(_energy_charts_data())

    np.testing.assert_allclose(
        results["consumption_mix_volume"].sel(
            consumer_country="B",
            source_country="A",
            technology=["Solar", "Wind"],
        ),
        [[32, 8]],
    )
    xr.testing.assert_allclose(
        results["consumption_mix_volume"].sum(
            ["source_country", "technology"]
        ),
        results["consumption_volume"],
    )
    np.testing.assert_allclose(
        results["consumption_mix"].sel(consumer_country="C").sum(),
        0,
    )


def test_energy_charts_preserves_unresolved_export_origin():
    data = _energy_charts_data().copy()
    data[("A", "trade", "C")] = 25.0
    data[("C", "trade", "A")] = -25.0

    results = consumption_results_from_energy_charts(data)

    np.testing.assert_allclose(
        results["consumption_mix"].sel(
            consumer_country="A",
            source_country="C",
            technology="Import balance",
        ),
        0.2,
    )
    xr.testing.assert_allclose(
        results["consumption_mix_volume"].sum(
            ["source_country", "technology"]
        ),
        results["consumption_volume"],
    )


def test_historical_pipeline_uses_canonical_cache_end_to_end(tmp_path):
    data = _energy_charts_data()
    cache_dir = data_processing(
        data,
        2025,
        path_to_data=tmp_path,
        time_chunk_size=1,
    )

    table = filt_cutoff(
        countries=["A", "B"],
        times=["2025-01-01 00:00"],
        cutoff=0,
        include_cutoff=False,
        path_to_data=tmp_path,
    )

    assert (cache_dir / "manifest.json").is_file()
    np.testing.assert_allclose(table[["A", "B"]].sum(), [1, 1])
