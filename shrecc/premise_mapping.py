"""Map SHRECC consumption mixes to premise regions and activities."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    from .tyndp import load_technology_concordance
except ImportError:
    from tyndp import load_technology_concordance


DEFAULT_PREMISE_IAM_MODELS = ("remind", "image", "remind-eu")


def build_country_activity_technology_map(
    activity_concordance,
    country_activity_shares,
    fallback_activity_concordance=False,
):
    """Build country-specific ENTSO-E-to-activity allocation shares.

    The detailed activity concordance identifies activities compatible with
    each ENTSO-E technology. Within each country, compatible activities are
    weighted by their contribution to the country's electricity market and
    then normalized. A technology with exactly one compatible activity is
    allocated fully to that activity even when it is absent from the supplied
    market shares, which supports one-to-one premise-added technologies.

    Args:
        activity_concordance: DataFrame indexed by activity and with ENTSO-E
            technologies as columns. Positive values indicate compatibility.
        country_activity_shares: DataFrame indexed by activity and with source
            countries as columns. Values are resolved electricity-market
            contributions and do not need to be normalized.
        fallback_activity_concordance: If True, use the normalized input
            concordance weights when a country/technology has multiple
            compatible activities but no positive market shares.

    Returns:
        DataFrame indexed by premise activity. Columns are a MultiIndex of
        ``source_country`` and ``technology``.

    Raises:
        ValueError: If inputs contain negative values, have duplicate labels,
            or a country/technology with multiple compatible activities has no
            positive market shares.
    """
    activity_concordance = activity_concordance.copy()
    country_activity_shares = country_activity_shares.copy()

    _validate_activity_share_builder_input(
        activity_concordance,
        country_activity_shares,
    )

    fallback_concordance = activity_concordance.fillna(0).clip(lower=0).astype(float)
    fallback_concordance = fallback_concordance.div(
        fallback_concordance.sum(axis=0).replace(0, np.nan),
        axis=1,
    ).fillna(0)
    activity_concordance = activity_concordance.fillna(0).gt(0).astype(float)
    country_activity_shares = country_activity_shares.fillna(0).reindex(
        activity_concordance.index,
        fill_value=0,
    )

    country_maps = {}
    unresolved = []

    for country in country_activity_shares.columns:
        weighted = activity_concordance.mul(
            country_activity_shares[country],
            axis=0,
        )
        totals = weighted.sum(axis=0)

        for technology in totals.index[totals.eq(0)]:
            matches = activity_concordance.index[
                activity_concordance[technology].gt(0)
            ]
            if len(matches) == 1:
                weighted.loc[matches[0], technology] = 1
                totals.loc[technology] = 1
            elif len(matches) > 1:
                if fallback_activity_concordance:
                    weighted.loc[matches, technology] = fallback_concordance.loc[
                        matches,
                        technology,
                    ]
                    totals.loc[technology] = 1
                else:
                    unresolved.append((country, technology))

        country_maps[country] = weighted.div(totals.replace(0, np.nan), axis=1)

    if unresolved:
        examples = "; ".join(
            f"{country} / {technology}" for country, technology in unresolved[:10]
        )
        raise ValueError(
            "Cannot allocate technologies with multiple compatible activities "
            "and zero country activity shares. First examples: "
            + examples
        )

    result = pd.concat(country_maps, axis=1, names=["source_country", "technology"])
    result.index.name = activity_concordance.index.name or "premise_activity"
    return result.fillna(0)


def build_country_activity_shares_from_ecoinvent_mapping(
    ecoinvent_mapping,
    premise_activities=None,
):
    """Collapse the basic SHRECC ecoinvent mapping to activity-country shares.

    Args:
        ecoinvent_mapping: Either the ``el_map_all_norm.csv`` DataFrame or a
            path to that CSV. Rows must include an ``"activityName"`` level and
            columns must be a MultiIndex whose first level is country.
        premise_activities: Optional activity index used to align the result
            to a premise concordance.

    Returns:
        DataFrame indexed by activity and with source countries as columns.
        Values are not normalized; only their relative magnitude within each
        country and compatible technology is used downstream.
    """
    if isinstance(ecoinvent_mapping, pd.DataFrame):
        mapping = ecoinvent_mapping.copy()
    else:
        mapping = pd.read_csv(
            ecoinvent_mapping,
            index_col=[0, 1, 2, 3],
            header=[0, 1],
        )

    if not isinstance(mapping.index, pd.MultiIndex):
        raise ValueError("ecoinvent_mapping rows must use a MultiIndex")
    if "activityName" not in mapping.index.names:
        raise ValueError("ecoinvent_mapping index must include 'activityName'")
    if not isinstance(mapping.columns, pd.MultiIndex):
        raise ValueError("ecoinvent_mapping columns must use a MultiIndex")

    country_activity_shares = (
        mapping.T.groupby(level=0).sum().T.groupby(level="activityName").sum()
    )
    country_activity_shares.index.name = "premise_activity"

    if premise_activities is not None:
        country_activity_shares = country_activity_shares.reindex(
            premise_activities,
            fill_value=0,
        )
        country_activity_shares.index.name = "premise_activity"

    return country_activity_shares


def build_country_specific_premise_technology_map(
    technology_mapping,
    ecoinvent_mapping,
    technology_sheet="concordance",
    fallback_activity_concordance=True,
):
    """Build a country-specific TYNDP-to-premise technology map.

    The premise/TYNDP concordance identifies compatible premise activities for
    each TYNDP technology. When several activities are compatible, this helper
    uses the country-specific activity shares from the basic SHRECC ecoinvent
    mapping to allocate between them. If a country has no usable ecoinvent
    shares for that technology, the helper falls back to the concordance
    weights, which keeps future-only premise technologies such as CCS usable.

    Args:
        technology_mapping: Premise/TYNDP technology concordance as a DataFrame
            or path to a workbook.
        ecoinvent_mapping: Basic SHRECC ecoinvent mapping as a DataFrame or
            path to ``el_map_all_norm.csv``.
        technology_sheet: Sheet name used when ``technology_mapping`` is an
            Excel workbook.
        fallback_activity_concordance: If True, use concordance weights when
            ecoinvent-derived country shares are unavailable.

    Returns:
        DataFrame indexed by premise activity. Columns are a MultiIndex of
        ``source_country`` and TYNDP ``technology``.
    """
    if isinstance(technology_mapping, pd.DataFrame):
        activity_concordance = technology_mapping.copy()
    else:
        activity_concordance = load_technology_concordance(
            technology_mapping,
            technology_sheet,
        )

    country_activity_shares = build_country_activity_shares_from_ecoinvent_mapping(
        ecoinvent_mapping,
        premise_activities=activity_concordance.index,
    )

    return build_country_activity_technology_map(
        activity_concordance=activity_concordance,
        country_activity_shares=country_activity_shares,
        fallback_activity_concordance=fallback_activity_concordance,
    )


def _validate_activity_share_builder_input(
    activity_concordance,
    country_activity_shares,
):
    """Validate inputs to ``build_country_activity_technology_map``."""
    if not isinstance(activity_concordance, pd.DataFrame):
        raise TypeError("activity_concordance must be a pandas DataFrame")
    if not isinstance(country_activity_shares, pd.DataFrame):
        raise TypeError("country_activity_shares must be a pandas DataFrame")
    if activity_concordance.index.has_duplicates:
        raise ValueError("activity_concordance activity labels must be unique")
    if activity_concordance.columns.has_duplicates:
        raise ValueError("activity_concordance technology labels must be unique")
    if country_activity_shares.index.has_duplicates:
        raise ValueError("country_activity_shares activity labels must be unique")
    if country_activity_shares.columns.has_duplicates:
        raise ValueError("country_activity_shares country labels must be unique")
    if activity_concordance.fillna(0).lt(0).to_numpy().any():
        raise ValueError("activity_concordance cannot contain negative values")
    if country_activity_shares.fillna(0).lt(0).to_numpy().any():
        raise ValueError("country_activity_shares cannot contain negative values")


class PremiseConsumptionMixMapper:
    """Map SHRECC consumption mixes to premise geography and activity axes.

    Args:
        technology_mapping: Technology concordance as a DataFrame or path to a
            SHRECC/FIONA technology mapping workbook.
        iam_model: premise/IAM model used for geography mapping.
        topology_files: Optional mapping from IAM model names to premise-style
            topology JSON files.
        technology_sheet: Sheet name used when ``technology_mapping`` is an
            Excel workbook.
        on_missing_model: Behavior when a premise geography map cannot be
            built. Must be ``"warn"``, ``"raise"``, or ``"ignore"``.
    """

    def __init__(
        self,
        technology_mapping,
        iam_model,
        topology_files=None,
        technology_sheet="concordance",
        on_missing_model="warn",
    ):
        self.technology_mapping = technology_mapping
        self.iam_model = iam_model
        self.topology_files = topology_files or {}
        self.technology_sheet = technology_sheet
        self.on_missing_model = on_missing_model
        self._premise_region_map = None

    def build_region_map(self, countries):
        """Build and cache a source-country to premise-region map.

        Args:
            countries: Iterable of source-country codes.

        Returns:
            DataFrame mapping country codes to premise/IAM regions.
        """
        self._premise_region_map = build_premise_region_map(
            countries,
            iam_models=(self.iam_model,),
            topology_files=self.topology_files,
            on_missing_model=self.on_missing_model,
        )
        return self._premise_region_map

    def map_technologies(self, consumption_mix_xr, check=True):
        """Aggregate the ``technology`` dimension to ``premise_activity``.

        Args:
            consumption_mix_xr: Consumption mix DataArray with a
                ``technology`` dimension.
            check: If True, check that mapped shares sum to one.

        Returns:
            Consumption mix DataArray with ``premise_activity`` replacing
            ``technology``.
        """
        return map_consumption_mix_technologies_xr(
            consumption_mix_xr,
            technology_mapping=self.technology_mapping,
            technology_sheet=self.technology_sheet,
            check=check,
        )

    def map_regions(
        self,
        consumption_mix_xr,
        premise_region_map=None,
        source_dim="source_country",
        region_dim="premise_region",
        check=True,
    ):
        """Aggregate source countries to premise regions for this IAM model.

        Args:
            consumption_mix_xr: Consumption mix DataArray with a source-country
                dimension.
            premise_region_map: Optional precomputed country-to-region map. If
                omitted, a map is built and cached.
            source_dim: Name of the source-country dimension.
            region_dim: Name of the output premise-region dimension.
            check: If True, check that mapped shares sum to one.

        Returns:
            Consumption mix DataArray aggregated to premise regions.
        """
        if premise_region_map is None:
            if self._premise_region_map is None:
                self.build_region_map(consumption_mix_xr[source_dim].to_index())
            premise_region_map = self._premise_region_map

        return map_consumption_mix_regions_xr(
            consumption_mix_xr,
            premise_region_map=premise_region_map,
            iam_model=self.iam_model,
            source_dim=source_dim,
            region_dim=region_dim,
            check=check,
        )

    def map_consumption_mix(self, consumption_mix_xr, check=True):
        """Map technologies and then source countries to premise dimensions.

        Args:
            consumption_mix_xr: Consumption mix DataArray with ``technology``
                and ``source_country`` dimensions.
            check: If True, check that mapped shares sum to one after each
                mapping step.

        Returns:
            Consumption mix DataArray with premise activity and region
            dimensions.
        """
        premise_activity_mix_xr = self.map_technologies(
            consumption_mix_xr,
            check=check,
        )
        return self.map_regions(
            premise_activity_mix_xr,
            check=check,
        )


def build_premise_region_map(
    countries,
    iam_models=DEFAULT_PREMISE_IAM_MODELS,
    on_missing_model="warn",
):
    """Map country codes to premise/IAM regions for one or more IAM models.

    Args:
        countries: Iterable of country codes to map.
        iam_models: IAM model names to include as output columns.
        topology_files: Optional mapping from IAM model names to premise-style
            topology JSON files.
        on_missing_model: Behavior when a premise geography map cannot be
            built. Must be ``"warn"``, ``"raise"``, or ``"ignore"``.

    Returns:
        DataFrame indexed by country code with one column per IAM model.

    Raises:
        ValueError: If ``on_missing_model`` is not supported.
        FileNotFoundError: If a geography map is missing and
            ``on_missing_model`` is ``"raise"``.
    """
    if on_missing_model not in {"warn", "raise", "ignore"}:
        raise ValueError("on_missing_model must be 'warn', 'raise', or 'ignore'")

    from premise.geomap import Geomap

    country_index = pd.Index(list(countries), name="country").unique().sort_values()
    region_map = pd.DataFrame(index=country_index)

    for model in iam_models:

        try:
            geomap = Geomap(model)
        except FileNotFoundError as exc:
            if on_missing_model == "raise":
                raise
            if on_missing_model == "warn":
                warnings.warn(
                    "Could not build geography map for IAM model "
                    f"{model!r}: {exc}",
                    stacklevel=2,
                )
            region_map[model] = pd.NA
            continue

        region_map[model] = [
            geomap.ecoinvent_to_iam_location(country) for country in country_index
        ]

    return region_map


def map_consumption_mix_technologies_xr(
    consumption_mix_xr,
    technology_mapping,
    technology_sheet="concordance",
    check=True,
):
    """Map consumption-mix technologies while keeping the result as xarray.

    Args:
        consumption_mix_xr: Consumption mix DataArray with a ``technology``
            dimension.
        technology_mapping: Technology concordance as a DataFrame or path to a
            SHRECC/FIONA technology mapping workbook.
        technology_sheet: Sheet name used when ``technology_mapping`` is an
            Excel workbook.
        check: If True, check that mapped shares sum to one.

    Returns:
        Consumption mix DataArray with ``premise_activity`` replacing
        ``technology``.

    Raises:
        ValueError: If the input lacks a ``technology`` dimension or if any
            technology is missing from the mapping.
        AssertionError: If conservation checks fail.
    """
    if "technology" not in consumption_mix_xr.dims:
        raise ValueError("consumption_mix_xr must have a 'technology' dimension")

    if isinstance(technology_mapping, pd.DataFrame):
        tech_map = technology_mapping.copy()
    else:
        tech_map = load_technology_concordance(
            technology_mapping,
            technology_sheet,
        )

    technologies = consumption_mix_xr["technology"].to_index()

    if isinstance(tech_map.columns, pd.MultiIndex):
        weights = _country_specific_technology_weights(
            tech_map,
            consumption_mix_xr,
            technologies,
        )
    else:
        missing_technologies = technologies.difference(tech_map.columns)
        if not missing_technologies.empty:
            raise ValueError(
                "No premise technology mapping found for: "
                + ", ".join(map(str, missing_technologies.tolist()))
            )
        tech_map = tech_map.reindex(columns=technologies)
        tech_map.index.name = "premise_activity"
        weights = xr.DataArray(
            tech_map.T.to_numpy(),
            dims=("technology", "premise_activity"),
            coords={
                "technology": technologies,
                "premise_activity": tech_map.index,
            },
            name="technology_weight",
        )

    mapped = xr.dot(consumption_mix_xr, weights, dim="technology")
    mapped = mapped.rename("consumption_mix")
    mapped = mapped.transpose(
        *[dim for dim in consumption_mix_xr.dims if dim != "technology"],
        "premise_activity",
    )

    if check:
        total_dims = [
            dim
            for dim in ("source_country", "premise_activity")
            if dim in mapped.dims
        ]
        totals = mapped.sum(total_dims)
        np.testing.assert_allclose(totals.to_numpy(), 1, atol=1e-8)

    return mapped


def _country_specific_technology_weights(
    tech_map,
    consumption_mix_xr,
    technologies,
):
    """Convert a country-specific technology map to aligned xarray weights."""
    required_levels = {"source_country", "technology"}
    if not required_levels.issubset(tech_map.columns.names):
        raise ValueError(
            "Country-specific technology mapping columns must have "
            "'source_country' and 'technology' levels"
        )
    if "source_country" not in consumption_mix_xr.dims:
        raise ValueError(
            "consumption_mix_xr must have a 'source_country' dimension when "
            "using a country-specific technology mapping"
        )

    source_countries = consumption_mix_xr["source_country"].to_index()
    mapped_countries = tech_map.columns.get_level_values("source_country")
    mapped_technologies = tech_map.columns.get_level_values("technology")
    missing_countries = source_countries.difference(mapped_countries)
    missing_technologies = technologies.difference(mapped_technologies)

    if not missing_countries.empty:
        raise ValueError(
            "No country-specific technology mapping found for source countries: "
            + ", ".join(map(str, missing_countries.tolist()))
        )
    if not missing_technologies.empty:
        raise ValueError(
            "No premise technology mapping found for: "
            + ", ".join(map(str, missing_technologies.tolist()))
        )

    ordered_columns = pd.MultiIndex.from_product(
        [source_countries, technologies],
        names=["source_country", "technology"],
    )
    tech_map = tech_map.reindex(columns=ordered_columns)
    tech_map.index.name = "premise_activity"

    values = tech_map.to_numpy().reshape(
        len(tech_map.index),
        len(source_countries),
        len(technologies),
    )
    return xr.DataArray(
        values.transpose(1, 2, 0),
        dims=("source_country", "technology", "premise_activity"),
        coords={
            "source_country": source_countries,
            "technology": technologies,
            "premise_activity": tech_map.index,
        },
        name="technology_weight",
    )


def map_consumption_mix_regions_xr(
    consumption_mix_xr,
    premise_region_map,
    iam_model,
    source_dim="source_country",
    region_dim="premise_region",
    check=True,
):
    """Map source countries to premise/IAM regions while keeping xarray output.

    Args:
        consumption_mix_xr: Consumption mix DataArray with a source-country
            dimension.
        premise_region_map: DataFrame indexed by source country with IAM model
            columns.
        iam_model: Column in ``premise_region_map`` used for mapping.
        source_dim: Name of the source-country dimension.
        region_dim: Name of the output premise-region dimension.
        check: If True, check that mapped shares sum to one.

    Returns:
        Consumption mix DataArray aggregated to premise/IAM regions.

    Raises:
        ValueError: If the source dimension, IAM model column, source-country
            rows, or region values are missing.
        AssertionError: If conservation checks fail.
    """
    if source_dim not in consumption_mix_xr.dims:
        raise ValueError(f"consumption_mix_xr must have a {source_dim!r} dimension")

    if iam_model not in premise_region_map.columns:
        raise ValueError(f"{iam_model!r} is not a column in premise_region_map")

    source_countries = consumption_mix_xr[source_dim].to_index()
    missing_countries = source_countries.difference(premise_region_map.index)

    if not missing_countries.empty:
        raise ValueError(
            "No premise region mapping found for source countries: "
            + ", ".join(map(str, missing_countries.tolist()))
        )

    regions_by_country = premise_region_map.loc[source_countries, iam_model]
    missing_regions = regions_by_country[regions_by_country.isna()]

    if not missing_regions.empty:
        raise ValueError(
            f"No {iam_model!r} premise region found for source countries: "
            + ", ".join(map(str, missing_regions.index.tolist()))
        )

    premise_regions = pd.Index(regions_by_country.unique(), name=region_dim)
    region_weights = pd.DataFrame(
        0,
        index=source_countries,
        columns=premise_regions,
        dtype=int,
    )

    for country, region in regions_by_country.items():
        region_weights.loc[country, region] = 1

    weights = xr.DataArray(
        region_weights.to_numpy(),
        dims=(source_dim, region_dim),
        coords={
            source_dim: source_countries,
            region_dim: premise_regions,
        },
        name="region_weight",
    )

    mapped = xr.dot(consumption_mix_xr, weights, dim=source_dim)
    mapped = mapped.rename("consumption_mix")
    mapped = mapped.transpose(
        *[dim for dim in consumption_mix_xr.dims if dim != source_dim],
        region_dim,
    )
    mapped.attrs["iam_model"] = iam_model

    if check:
        total_dims = [
            dim
            for dim in (region_dim, "premise_activity", "technology")
            if dim in mapped.dims
        ]
        totals = mapped.sum(total_dims)
        np.testing.assert_allclose(totals.to_numpy(), 1, atol=1e-8)

    return mapped


def _map_countries_with_topology_file(countries, topology_file):
    """Map countries using a premise-style topology JSON file.

    Args:
        countries: Iterable of country codes.
        topology_file: Path to a premise-style topology JSON file.

    Returns:
        List of mapped regions, with missing countries represented as
        ``pandas.NA``.
    """
    with Path(topology_file).open(encoding="utf-8") as handle:
        topology = json.load(handle)

    country_to_region = {}
    for region, geographies in topology.items():
        for geography in geographies:
            country_to_region.setdefault(geography, region)

    return [country_to_region.get(country, pd.NA) for country in countries]
