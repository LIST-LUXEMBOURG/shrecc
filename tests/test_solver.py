import numpy as np
import pandas as pd
import xarray as xr

from shrecc.solver import iter_consumption_system_results, solve_consumption_system


def _solver_inputs():
    time = pd.to_datetime(["2040-01-01 00:00"])
    production_volume = xr.DataArray(
        [[[100.0], [60.0], [20.0]]],
        dims=("time", "producer_country", "technology"),
        coords={
            "time": time,
            "producer_country": ["A", "B", "C"],
            "technology": ["electricity"],
        },
    )
    trade_volume = xr.DataArray(
        np.zeros((1, 3, 3)),
        dims=("time", "exporter_country", "importer_country"),
        coords={
            "time": time,
            "exporter_country": ["A", "B", "C"],
            "importer_country": ["A", "B", "C"],
        },
    )
    trade_volume.loc[
        {"exporter_country": "A", "importer_country": "B"}
    ] = 40
    return production_volume, trade_volume


def test_solver_accepts_measured_consumption_volume():
    production_volume, trade_volume = _solver_inputs()
    measured_consumption = xr.DataArray(
        [[55.0, 95.0, 18.0]],
        dims=("time", "consumer_country"),
        coords={
            "time": production_volume["time"],
            "consumer_country": ["A", "B", "C"],
        },
    )

    results, _ = solve_consumption_system(
        production_volume,
        trade_volume,
        consumption_volume=measured_consumption,
    )

    xr.testing.assert_allclose(
        results["consumption_volume"],
        measured_consumption,
    )
    xr.testing.assert_allclose(
        results["consumption_mix_volume"].sum(
            ["source_country", "technology"]
        ),
        measured_consumption,
    )
    np.testing.assert_allclose(
        results["consumption_mix"].sel(
            consumer_country="C",
            source_country="C",
        ),
        1,
    )


def test_solver_rejects_negative_directional_trade():
    production_volume, trade_volume = _solver_inputs()
    trade_volume.loc[
        {"exporter_country": "A", "importer_country": "B"}
    ] = -1

    with np.testing.assert_raises_regex(
        ValueError,
        "positive directional flows",
    ):
        solve_consumption_system(production_volume, trade_volume)


def test_solver_rejects_negative_generation():
    production_volume, trade_volume = _solver_inputs()
    production_volume.loc[
        {"producer_country": "A", "technology": "electricity"}
    ] = -1

    with np.testing.assert_raises_regex(ValueError, "nonnegative generation"):
        solve_consumption_system(production_volume, trade_volume)


def test_chunked_solver_matches_single_solve():
    production_volume, trade_volume = _solver_inputs()
    times = pd.date_range("2040-01-01", periods=3, freq="h")
    production_volume = xr.concat(
        [production_volume.isel(time=0)] * len(times),
        dim=pd.Index(times, name="time"),
    )
    trade_volume = xr.concat(
        [trade_volume.isel(time=0)] * len(times),
        dim=pd.Index(times, name="time"),
    )

    expected, _ = solve_consumption_system(production_volume, trade_volume)
    chunks = list(
        iter_consumption_system_results(
            production_volume,
            trade_volume,
            time_chunk_size=2,
        )
    )

    assert [chunk.sizes["time"] for chunk in chunks] == [2, 1]
    xr.testing.assert_allclose(xr.concat(chunks, dim="time"), expected)


def test_chunked_solver_rejects_cross_chunk_month_hour_imputation():
    production_volume, trade_volume = _solver_inputs()
    times = pd.date_range("2040-01-01", periods=2, freq="h")
    production_volume = xr.concat(
        [production_volume.isel(time=0)] * len(times),
        dim=pd.Index(times, name="time"),
    )
    trade_volume = xr.concat(
        [trade_volume.isel(time=0)] * len(times),
        dim=pd.Index(times, name="time"),
    )

    with np.testing.assert_raises_regex(ValueError, "multiple time chunks"):
        list(
            iter_consumption_system_results(
                production_volume,
                trade_volume,
                time_chunk_size=1,
                zero_consumption="month_hour_average",
            )
        )
