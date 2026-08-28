from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import shrecc.lcia as lcia
from shrecc.lcia import (
    LCIAResults,
    ResolvedInventoryBasis,
    build_lcia_dataset,
    build_resolved_inventory_basis,
    calculate_lcia,
    consumption_profile_weights,
    resolve_lcia_methods,
)

METHODS = (
    ("ecoinvent", "EF v3.1", "climate change", "GWP100"),
    ("ecoinvent", "EF v3.1", "water use", "deprivation"),
)


def _columns():
    return pd.MultiIndex.from_product(
        [
            pd.to_datetime(["2040-01-01 00:00", "2040-01-01 01:00"]),
            ["FR"],
        ],
        names=["time", "consumer_country"],
    )


def _basis():
    index = pd.MultiIndex.from_tuples(
        [("background", "wind"), ("background", "gas")],
        names=["database", "code"],
    )
    return ResolvedInventoryBasis(
        year=2040,
        background_database="background",
        coefficients=pd.DataFrame(
            [[0.25, 0.75], [0.75, 0.25]],
            index=index,
            columns=_columns(),
        ),
    )


def _canonical_results():
    times = _columns().levels[0]
    mix = xr.DataArray(
        np.ones((2, 1, 1, 1)),
        dims=("time", "consumer_country", "source_country", "technology"),
        coords={
            "time": times,
            "consumer_country": ["FR"],
            "source_country": ["FR"],
            "technology": ["wind"],
        },
    )
    volume = xr.DataArray(
        [[1.0], [3.0]],
        dims=("time", "consumer_country"),
        coords={"time": times, "consumer_country": ["FR"]},
    )
    return xr.Dataset(
        {
            "consumption_mix": mix,
            "consumption_volume": volume,
        }
    )


def test_default_methods_select_exact_ef_family(monkeypatch):
    installed = [
        METHODS[0],
        METHODS[1],
        ("ecoinvent", "EF v3.1 no LT", "climate change", "GWP100"),
    ]
    monkeypatch.setattr(lcia, "bd", SimpleNamespace(methods=installed))

    assert resolve_lcia_methods() == METHODS


def test_default_methods_raise_when_ef_is_missing(monkeypatch):
    monkeypatch.setattr(lcia, "bd", SimpleNamespace(methods=[]))

    with pytest.raises(ValueError, match="No EF v3.1"):
        resolve_lcia_methods()


def test_resolved_basis_combines_duplicate_inputs_and_network(monkeypatch):
    index = pd.MultiIndex.from_tuples(
        [
            ("FR", "wind one", "electricity", "kWh"),
            ("FR", "wind two", "electricity", "kWh"),
        ],
        names=["geography", "activityName", "product", "unit"],
    )
    table = pd.DataFrame(
        [[0.2, 0.3], [0.8, 0.7]],
        index=index,
        columns=_columns(),
    )
    monkeypatch.setattr(
        lcia,
        "map_known_inputs",
        lambda *args, **kwargs: (
            {
                ("FR", "wind one", "kWh"): ("background", "wind"),
                ("FR", "wind two", "kWh"): ("background", "wind"),
            },
            {("GLO", "network"): ("background", "network")},
        ),
    )
    monkeypatch.setattr(
        lcia,
        "get_country_network_activities",
        lambda *args: [{"loc": "GLO", "name": "network", "val": 0.1}],
    )

    basis = build_resolved_inventory_basis(
        table,
        year=2040,
        background_database="background",
    )

    np.testing.assert_allclose(
        basis.coefficients.loc[("background", "wind")],
        [1.0, 1.0],
    )
    np.testing.assert_allclose(
        basis.coefficients.loc[("background", "network")],
        [0.1, 0.1],
    )


def test_linear_lcia_multiplies_hourly_coefficients_by_source_scores(monkeypatch):
    monkeypatch.setattr(
        lcia,
        "_score_unique_inputs",
        lambda basis, methods, cache=None: np.array([[10.0, 100.0], [20.0, 200.0]]),
    )

    result = calculate_lcia(_basis(), METHODS)

    np.testing.assert_allclose(
        result.sel(consumer_country="FR").to_numpy(),
        [[17.5, 175.0], [12.5, 125.0]],
    )


def test_multilca_engine_matches_linear_composite_scores(monkeypatch):
    monkeypatch.setattr(
        lcia,
        "_basis_demands",
        lambda basis: (
            ["input:0", "input:1"],
            {"input:0": {1: 1.0}, "input:1": {2: 1.0}},
            [1, 2],
        ),
    )

    def score(demands, methods, **kwargs):
        return {
            label: np.array(
                [
                    demand.get(1, 0) * 10 + demand.get(2, 0) * 20,
                    demand.get(1, 0) * 100 + demand.get(2, 0) * 200,
                ]
            )
            for label, demand in demands.items()
        }

    monkeypatch.setattr(lcia, "_run_fast_multilca", score)
    monkeypatch.setattr(lcia, "_multilca_data_objects", lambda *args: object())

    result = calculate_lcia(
        _basis(),
        METHODS,
        engine="multilca",
        batch_size=1,
    )

    np.testing.assert_allclose(
        result.sel(consumer_country="FR").to_numpy(),
        [[17.5, 175.0], [12.5, 125.0]],
    )


def test_consumption_profile_weights_support_demand_and_custom_series():
    results = _canonical_results()
    demand = consumption_profile_weights(results, "national_demand")
    custom = consumption_profile_weights(
        results,
        pd.Series([2.0, 1.0], index=results["time"].to_index()),
    )

    np.testing.assert_allclose(demand.to_numpy().ravel(), [1.0, 3.0])
    np.testing.assert_allclose(custom.to_numpy().ravel(), [2.0, 1.0])


def test_lcia_results_aggregate_profile_weighted_intensity():
    times = _canonical_results()["time"].to_index()
    intensity = xr.DataArray(
        np.array([[[10.0]], [[20.0]]]),
        dims=("time", "consumer_country", "impact_category"),
        coords={
            "time": times,
            "consumer_country": ["FR"],
            "impact_category": ["climate"],
        },
    )
    weights = xr.DataArray(
        [[1.0], [3.0]],
        dims=("time", "consumer_country"),
        coords={"time": times, "consumer_country": ["FR"]},
    )
    dataset = build_lcia_dataset(
        intensity,
        weights,
        year=2040,
        engine="linear",
    )
    result = LCIAResults({2040: dataset}, METHODS[:1], "linear")

    assert result.years == (2040,)
    assert result.impact_categories == ("climate",)
    assert repr(result) == (
        "LCIAResults(years=[2040], impact_categories=1, engine='linear')"
    )
    hourly = result.hourly()
    assert "consuming one kilowatt hour" in hourly["intensity"].attrs["description"]
    assert "Unnormalized" in hourly["consumption_weight"].attrs["description"]
    assert "annual intensity" in hourly["weighted_contribution"].attrs[
        "description"
    ]
    assert result.annual().sel(
        year=2040,
        consumer_country="FR",
        impact_category="climate",
    ).item() == pytest.approx(17.5)
    assert result.monthly().sel(
        year=2040,
        consumer_country="FR",
        impact_category="climate",
    ).item() == pytest.approx(17.5)
