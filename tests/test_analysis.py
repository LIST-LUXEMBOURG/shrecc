from unittest.mock import MagicMock

import pandas as pd
import pytest

from shrecc import analysis


class FakeActivity(dict):
    def __init__(
        self,
        activity_id,
        name,
        *,
        location="FR",
        product="electricity, high voltage",
        unit="kilowatt hour",
    ):
        super().__init__(
            name=name,
            location=location,
            **{"reference product": product, "unit": unit},
        )
        self.id = activity_id


class FakeGraphResolver:
    def __init__(self, nodes, inputs):
        self.nodes = nodes
        self.inputs = inputs

    def node(self, activity_id):
        return self.nodes[activity_id]

    def metadata(self, activity_id):
        activity = self.nodes[activity_id]
        return activity["name"], activity["location"]

    def electricity_inputs(self, activity_id):
        return self.inputs.get(activity_id, ())


def test_resolve_electricity_generation_shares_follows_wrappers():
    market = FakeActivity(1, "market for electricity, low voltage")
    regional_market = FakeActivity(2, "market group for electricity")
    solar = FakeActivity(3, "electricity production, photovoltaic")
    wind = FakeActivity(4, "electricity production, wind")
    resolver = FakeGraphResolver(
        nodes={
            market.id: market,
            regional_market.id: regional_market,
            solar.id: solar,
            wind.id: wind,
        },
        inputs={
            market.id: ((solar.id, 0.4), (regional_market.id, 0.6)),
            regional_market.id: ((wind.id, 1.0),),
        },
    )

    shares, diagnostics = analysis.resolve_electricity_generation_shares(
        market,
        resolver=resolver,
    )

    assert shares.to_dict() == pytest.approx(
        {
            "electricity production, wind": 0.6,
            "electricity production, photovoltaic": 0.4,
        }
    )
    assert diagnostics["iterations"] == 3
    assert diagnostics["nonstandard terminal volume"] == 0


class FakeDatabaseResolver:
    def __init__(self):
        self.activities = {
            "flat-db": FakeActivity(
                1,
                "electricity, consumption mix, 2040",
            ),
            "background-db": FakeActivity(
                2,
                "market for electricity, low voltage",
            ),
        }

    def find_unique(self, database_name, activity_name, location):
        activity = self.activities.get(database_name)
        if activity is None:
            return None
        if activity["name"] == activity_name and activity["location"] == location:
            return activity
        return None


def test_compare_electricity_generation_mixes_builds_long_table(monkeypatch):
    monkeypatch.setattr(
        analysis,
        "_ActivityResolver",
        FakeDatabaseResolver,
    )

    def fake_resolve(activity, **_):
        technology = "wind" if activity.id == 1 else "solar"
        return (
            pd.Series({technology: 1.0}),
            {
                "iterations": 1,
                "terminal generation volume": 1.0,
                "remaining wrapper volume": 0.0,
                "nonstandard terminal volume": 0.0,
            },
        )

    monkeypatch.setattr(
        analysis,
        "resolve_electricity_generation_shares",
        fake_resolve,
    )

    mixes, gaps, diagnostics = analysis.compare_electricity_generation_mixes(
        {"Flat": {2040: "flat-db"}},
        {2040: "background-db"},
        countries=["FR"],
        years=[2040],
    )

    assert gaps.empty
    assert mixes.set_index("model")["technology"].to_dict() == {
        "Flat": "wind",
        "Background": "solar",
    }
    assert diagnostics["model"].tolist() == ["Flat", "Background"]


def test_compare_electricity_lcia_batches_demands(monkeypatch):
    monkeypatch.setattr(
        analysis,
        "_ActivityResolver",
        FakeDatabaseResolver,
    )
    run_multilca = MagicMock(
        side_effect=lambda demands, method: {
            label: float(next(iter(demand))) for label, demand in demands.items()
        }
    )
    monkeypatch.setattr(analysis, "_run_multilca", run_multilca)

    results, gaps = analysis.compare_electricity_lcia(
        {"Flat": {2040: "flat-db"}},
        {2040: "background-db"},
        countries=["FR"],
        years=[2040],
        method=("method",),
    )

    assert gaps.empty
    assert results.set_index("model")["score"].to_dict() == {
        "Flat": 1.0,
        "Background": 2.0,
    }
    run_multilca.assert_called_once()


def test_comparison_requires_every_database_year():
    with pytest.raises(ValueError, match="missing years: 2040"):
        analysis.compare_electricity_generation_mixes(
            {"Flat": {2035: "flat-db"}},
            {2035: "background-db", 2040: "background-db"},
            countries=["FR"],
            years=[2035, 2040],
        )
