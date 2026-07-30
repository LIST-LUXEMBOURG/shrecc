"""Map SHRECC consumption mixes to premise regions and activities."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from shrecc.mapping import (
    filter_consumption_mix_time,
    load_ecoinvent_mapping,
    validate_inventory_resolution,
)
from shrecc.tyndp import load_technology_concordance


DEFAULT_PREMISE_IAM_MODELS = ("remind", "image", "remind-eu")
DEFAULT_ECOINVENT_COUNTRY_ALIASES = {"UK": "GB", "NIE": "GB"}
DEFAULT_COUNTRY_SPECIFIC_PREMISE_ACTIVITIES = (
    "electricity production, photovoltaic, commercial",
)


def _load_and_validate_ecoinvent_mapping(ecoinvent_mapping):
    """Load and validate the basic SHRECC ecoinvent mapping."""
    mapping = load_ecoinvent_mapping(ecoinvent_mapping)

    if not isinstance(mapping.index, pd.MultiIndex):
        raise ValueError("ecoinvent_mapping rows must use a MultiIndex")
    if "activityName" not in mapping.index.names:
        raise ValueError("ecoinvent_mapping index must include 'activityName'")
    if "geography_source" not in mapping.index.names:
        raise ValueError("ecoinvent_mapping index must include 'geography_source'")
    if not isinstance(mapping.columns, pd.MultiIndex):
        raise ValueError("ecoinvent_mapping columns must use a MultiIndex")

    return mapping


def _add_country_alias_columns(
    country_frame,
    country_aliases=DEFAULT_ECOINVENT_COUNTRY_ALIASES,
    source_countries=None,
    fill_value=0,
):
    """Add alias country columns using their ecoinvent geography."""
    result = country_frame.copy()
    if country_aliases is None:
        country_aliases = {}

    requested_countries = (
        pd.Index(source_countries)
        if source_countries is not None
        else pd.Index(country_aliases)
    )
    for alias in requested_countries:
        source = country_aliases.get(alias)
        if (
            source is not None
            and alias not in result.columns
            and source in result.columns
        ):
            result[alias] = result[source]

    if source_countries is not None:
        result = result.reindex(
            columns=pd.Index(source_countries), fill_value=fill_value
        )

    return result


def _exchange_country_geography(country, country_aliases):
    """Return the activity exchange geography for a country-specific activity."""
    return (country_aliases or {}).get(country, country)


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
            matches = activity_concordance.index[activity_concordance[technology].gt(0)]
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
            "and zero country activity shares. First examples: " + examples
        )

    result = pd.concat(country_maps, axis=1, names=["source_country", "technology"])
    result.index.name = activity_concordance.index.name or "premise_activity"
    return result.fillna(0)


def build_country_activity_shares_from_ecoinvent_mapping(
    ecoinvent_mapping,
    premise_activities=None,
    source_countries=None,
    country_aliases=DEFAULT_ECOINVENT_COUNTRY_ALIASES,
):
    """Collapse the basic SHRECC ecoinvent mapping to activity-country shares.

    Args:
        ecoinvent_mapping: Either the ``el_map_all_norm.csv`` DataFrame or a
            path to that CSV. Rows must include an ``"activityName"`` level and
            columns must be a MultiIndex whose first level is country.
        premise_activities: Optional activity index used to align the result
            to a premise concordance.
        source_countries: Optional country index used to align the columns.
        country_aliases: Optional mapping from requested country codes to the
            country codes used by ecoinvent.

    Returns:
        DataFrame indexed by activity and with source countries as columns.
        Values are not normalized; only their relative magnitude within each
        country and compatible technology is used downstream.
    """
    mapping = _load_and_validate_ecoinvent_mapping(ecoinvent_mapping)

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

    country_activity_shares = _add_country_alias_columns(
        country_activity_shares,
        country_aliases=country_aliases,
        source_countries=source_countries,
        fill_value=0,
    )

    return country_activity_shares


def build_country_activity_presence_from_ecoinvent_mapping(
    ecoinvent_mapping,
    premise_activities=None,
    source_countries=None,
    country_aliases=DEFAULT_ECOINVENT_COUNTRY_ALIASES,
):
    """Identify country-specific activities present in the ecoinvent mapping.

    Args:
        ecoinvent_mapping: Either the ``el_map_all_norm.csv`` DataFrame or a
            path to that CSV.
        premise_activities: Optional activity index used to align the result.
        source_countries: Optional country index used to align the columns.
        country_aliases: Optional mapping from requested country codes to the
            country codes used by ecoinvent.

    Returns:
        Boolean DataFrame indexed by activity and with source countries as
        columns. ``True`` means the activity exists as an ecoinvent row for
        that source country.
    """
    mapping = _load_and_validate_ecoinvent_mapping(ecoinvent_mapping)

    rows = mapping.index.to_frame(index=False)
    presence = pd.crosstab(
        rows["activityName"],
        rows["geography_source"],
    ).gt(0)
    presence.index.name = "premise_activity"
    presence.columns.name = "source_country"

    if premise_activities is not None:
        presence = presence.reindex(premise_activities, fill_value=False)
        presence.index.name = "premise_activity"

    presence = _add_country_alias_columns(
        presence,
        country_aliases=country_aliases,
        source_countries=source_countries,
        fill_value=False,
    )
    presence.columns.name = "source_country"

    return presence


def build_premise_exchange_geography_map(
    ecoinvent_mapping,
    premise_region_map,
    iam_model,
    premise_activities=None,
    source_countries=None,
    country_aliases=DEFAULT_ECOINVENT_COUNTRY_ALIASES,
    country_specific_premise_activities=DEFAULT_COUNTRY_SPECIFIC_PREMISE_ACTIVITIES,
):
    """Choose source-country or premise-region exchange geography per activity.

    Existing country-specific ecoinvent activities keep their source country as
    exchange geography. Known country-specific premise activities do the same
    even if they are absent from the basic SHRECC ecoinvent mapping. Other
    activities fall back to the corresponding premise/IAM region.

    Args:
        ecoinvent_mapping: Either the ``el_map_all_norm.csv`` DataFrame or a
            path to that CSV.
        premise_region_map: DataFrame indexed by source country with IAM model
            columns.
        iam_model: Column in ``premise_region_map`` used as fallback region.
        premise_activities: Optional activity index used to align the result.
        source_countries: Optional country index used to align the columns.
        country_aliases: Optional mapping from requested country codes to the
            country codes used by ecoinvent.
        country_specific_premise_activities: Optional iterable of premise
            activity names known to exist at country level.

    Returns:
        DataFrame indexed by premise activity and with source countries as
        columns. Values are exchange geographies.
    """
    if iam_model not in premise_region_map.columns:
        raise ValueError(f"{iam_model!r} is not a column in premise_region_map")

    presence = build_country_activity_presence_from_ecoinvent_mapping(
        ecoinvent_mapping,
        premise_activities=premise_activities,
        source_countries=source_countries,
        country_aliases=country_aliases,
    )

    countries = presence.columns
    missing_countries = countries.difference(premise_region_map.index)
    if not missing_countries.empty:
        raise ValueError(
            "No premise region mapping found for source countries: "
            + ", ".join(map(str, missing_countries.tolist()))
        )

    regions = premise_region_map.loc[countries, iam_model]
    missing_regions = regions[regions.isna()]
    if not missing_regions.empty:
        raise ValueError(
            f"No {iam_model!r} premise region found for source countries: "
            + ", ".join(map(str, missing_regions.index.tolist()))
        )

    exchange_geography = pd.DataFrame(
        np.tile(regions.to_numpy(), (len(presence.index), 1)),
        index=presence.index,
        columns=countries,
    )
    country_specific_premise_activities = pd.Index(
        country_specific_premise_activities or []
    )
    country_specific_premise_activities = presence.index.intersection(
        country_specific_premise_activities
    )
    for country in countries:
        country_geography = _exchange_country_geography(country, country_aliases)
        exchange_geography.loc[presence[country], country] = country_geography
        exchange_geography.loc[country_specific_premise_activities, country] = (
            country_geography
        )

    exchange_geography.index.name = "premise_activity"
    exchange_geography.columns.name = "source_country"
    return exchange_geography


def build_country_specific_premise_technology_map(
    technology_mapping,
    ecoinvent_mapping,
    technology_sheet="concordance",
    fallback_activity_concordance=True,
    source_countries=None,
    country_aliases=DEFAULT_ECOINVENT_COUNTRY_ALIASES,
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
        source_countries: Optional country index used to align the mapping to
            the consumption mix source countries.
        country_aliases: Optional mapping from requested country codes to the
            country codes used by ecoinvent.

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
        source_countries=source_countries,
        country_aliases=country_aliases,
    )

    return build_country_activity_technology_map(
        activity_concordance=activity_concordance,
        country_activity_shares=country_activity_shares,
        fallback_activity_concordance=fallback_activity_concordance,
    )


def premise_activity_mix_to_database_table(
    premise_activity_mix_xr,
    exchange_geography_map,
    countries=None,
    times=None,
    general_range=None,
    refined_range=None,
    freq=None,
    inventory_resolution="annual",
    preserve_time=False,
    product="electricity, high voltage",
    unit="kWh",
):
    """Convert a mapped premise activity mix to the SHRECC database table shape.

    Args:
        premise_activity_mix_xr: Consumption mix DataArray with
            ``source_country`` and ``premise_activity`` dimensions.
        exchange_geography_map: DataFrame indexed by premise activity and with
            source countries as columns. Values are exchange geographies.
        countries: Optional consuming countries to select.
        times: Optional explicit timestamps to select.
        general_range: Optional start/end timestamps used as a time slice.
        refined_range: Optional start/end hours used to refine
            ``general_range``, following :func:`shrecc.database.filt_cutoff`.
        freq: Pandas frequency used to generate timestamps within
            ``general_range`` when ``refined_range`` is supplied.
        inventory_resolution: Whether columns represent annual or monthly
            inventories.
        preserve_time: Keep hourly timestamps instead of applying annual
            aggregation. Intended for in-memory assessment inputs.
        product: Product label used in the output MultiIndex.
        unit: Unit label used in the output MultiIndex.

    Returns:
        DataFrame indexed by ``geography``, ``activityName``, ``product``, and
        ``unit``. Columns are consuming countries.
    """
    required_dims = {"source_country", "premise_activity", "consumer_country"}
    missing_dims = required_dims.difference(premise_activity_mix_xr.dims)
    if missing_dims:
        raise ValueError(
            "premise_activity_mix_xr is missing required dimensions: "
            + ", ".join(sorted(missing_dims))
        )

    resolution = validate_inventory_resolution(inventory_resolution)
    mix = premise_activity_mix_xr
    if countries is not None:
        mix = mix.sel(consumer_country=list(countries))

    mix = filter_consumption_mix_time(
        mix,
        times=times,
        general_range=general_range,
        refined_range=refined_range,
        freq=freq,
    )

    if "time" in mix.dims and resolution == "annual" and not preserve_time:
        mix = mix.mean("time")

    source_countries = mix["source_country"].to_index()
    premise_activities = mix["premise_activity"].to_index()
    exchange_geography_map = exchange_geography_map.reindex(
        index=premise_activities,
        columns=source_countries,
    )

    if exchange_geography_map.isna().to_numpy().any():
        raise ValueError("exchange_geography_map is missing required entries")

    exchange_geographies = (
        exchange_geography_map.rename_axis(
            index="premise_activity",
            columns="source_country",
        )
        .stack()
        .rename("geography")
        .reset_index()
    )

    table = mix.to_dataframe(name="share").reset_index()
    table = table.merge(
        exchange_geographies,
        on=["premise_activity", "source_country"],
        how="left",
        validate="many_to_one",
    )
    if table["geography"].isna().any():
        raise ValueError("Could not assign exchange geography to all rows")

    row_levels = ["geography", "premise_activity"]
    if "time" in mix.dims:
        table = (
            table.groupby(
                [*row_levels, "time", "consumer_country"],
                dropna=False,
            )["share"]
            .sum()
            .unstack(["time", "consumer_country"], fill_value=0)
        )
        table.columns.names = ["time", "country"]
    else:
        table = (
            table.groupby(
                [*row_levels, "consumer_country"],
                dropna=False,
            )["share"]
            .sum()
            .unstack("consumer_country", fill_value=0)
        )
        table.columns.name = None

    table.index = pd.MultiIndex.from_arrays(
        [
            table.index.get_level_values("geography"),
            table.index.get_level_values("premise_activity"),
            [product] * len(table),
            [unit] * len(table),
        ],
        names=["geography", "activityName", "product", "unit"],
    )
    return table


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
        ecoinvent_mapping: Optional basic SHRECC ecoinvent mapping. When
            supplied, technology allocation uses country-specific ecoinvent
            shares where available and concordance fallback otherwise.
        topology_files: Optional mapping from IAM model names to premise-style
            topology JSON files.
        technology_sheet: Sheet name used when ``technology_mapping`` is an
            Excel workbook.
        on_missing_model: Behavior when a premise geography map cannot be
            built. Must be ``"warn"``, ``"raise"``, or ``"ignore"``.
        fallback_activity_concordance: If True, use normalized concordance
            weights when country-specific ecoinvent shares are unavailable.
        country_specific_premise_activities: Optional iterable of premise
            activities known to exist at country level.
    """

    def __init__(
        self,
        technology_mapping,
        iam_model,
        ecoinvent_mapping=None,
        topology_files=None,
        technology_sheet="concordance",
        on_missing_model="warn",
        fallback_activity_concordance=True,
        country_aliases=DEFAULT_ECOINVENT_COUNTRY_ALIASES,
        country_specific_premise_activities=DEFAULT_COUNTRY_SPECIFIC_PREMISE_ACTIVITIES,
    ):
        self.technology_mapping = technology_mapping
        self.ecoinvent_mapping = ecoinvent_mapping
        self.iam_model = iam_model
        self.topology_files = topology_files or {}
        self.technology_sheet = technology_sheet
        self.on_missing_model = on_missing_model
        self.fallback_activity_concordance = fallback_activity_concordance
        self.country_aliases = country_aliases
        self.country_specific_premise_activities = country_specific_premise_activities
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
        technology_mapping = self.technology_mapping
        if self.ecoinvent_mapping is not None:
            technology_mapping = build_country_specific_premise_technology_map(
                self.technology_mapping,
                self.ecoinvent_mapping,
                technology_sheet=self.technology_sheet,
                fallback_activity_concordance=self.fallback_activity_concordance,
                source_countries=consumption_mix_xr["source_country"].to_index(),
                country_aliases=self.country_aliases,
            )

        return map_consumption_mix_technologies_xr(
            consumption_mix_xr,
            technology_mapping=technology_mapping,
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

    def build_exchange_geography_map(
        self,
        premise_activity_mix_xr,
        premise_region_map=None,
    ):
        """Build exchange geography map for a mapped premise activity mix."""
        if self.ecoinvent_mapping is None:
            raise ValueError(
                "ecoinvent_mapping is required to build exchange geography map"
            )

        if premise_region_map is None:
            if self._premise_region_map is None:
                self.build_region_map(
                    premise_activity_mix_xr["source_country"].to_index()
                )
            premise_region_map = self._premise_region_map

        return build_premise_exchange_geography_map(
            self.ecoinvent_mapping,
            premise_region_map,
            iam_model=self.iam_model,
            premise_activities=premise_activity_mix_xr["premise_activity"].to_index(),
            source_countries=premise_activity_mix_xr["source_country"].to_index(),
            country_aliases=self.country_aliases,
            country_specific_premise_activities=(
                self.country_specific_premise_activities
            ),
        )


def build_premise_region_map(
    countries,
    iam_models=DEFAULT_PREMISE_IAM_MODELS,
    topology_files=None,
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

    topology_files = topology_files or {}
    country_index = pd.Index(list(countries), name="country").unique().sort_values()
    region_map = pd.DataFrame(index=country_index)
    geomap_cls = None

    for model in iam_models:
        if model in topology_files:
            region_map[model] = _map_countries_with_topology_file(
                country_index,
                topology_files[model],
            )
            continue

        if geomap_cls is None:
            try:
                from premise.geomap import Geomap as geomap_cls
            except ImportError as exc:
                raise ImportError(
                    "build_premise_region_map requires the optional 'premise' "
                    "dependency. Install it with `pip install shrecc[premise]`."
                ) from exc

        try:
            geomap = geomap_cls(model)
        except FileNotFoundError as exc:
            if on_missing_model == "raise":
                raise
            if on_missing_model == "warn":
                warnings.warn(
                    "Could not build geography map for IAM model " f"{model!r}: {exc}",
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
            dim for dim in ("source_country", "premise_activity") if dim in mapped.dims
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
