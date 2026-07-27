"""Compatibility imports for the former FIONA/TYNDP module name.

New code should import TYNDP ingestion functions from :mod:`shrecc.tyndp` and
premise mapping helpers from :mod:`shrecc.premise_mapping`.
"""

from shrecc.premise_mapping import (
    DEFAULT_PREMISE_IAM_MODELS,
    PremiseConsumptionMixMapper,
    build_country_activity_shares_from_ecoinvent_mapping,
    build_country_activity_technology_map,
    build_country_specific_premise_technology_map,
    build_premise_exchange_geography_map,
    build_premise_region_map,
    map_consumption_mix_regions_xr,
    map_consumption_mix_technologies_xr,
    premise_activity_mix_to_database_table,
)
from shrecc.tyndp import (
    TYNDP_CLIMATE_YEARS,
    TYNDP_SCENARIO_URL_ROOT,
    TYNDP_SCENARIO_YEARS,
    TYNDP_SCENARIOS,
    build_z_gross_from_tyndp_excel,
    build_z_gross_from_tyndp_pickles,
    build_z_gross_from_tyndp_scenario,
    consumption_mix_from_tyndp_excel,
    consumption_mix_from_tyndp_pickles,
    consumption_mix_from_tyndp_scenario,
    consumption_mix_from_z_gross,
    download_tyndp_scenario_zip,
    ensure_tyndp_workbook,
    load_technology_concordance,
    load_tyndp_country_mapping,
    read_tyndp_excel_tables,
    tyndp_scenario_paths,
    validate_tyndp_scenario,
)

__all__ = (
    "DEFAULT_PREMISE_IAM_MODELS",
    "PremiseConsumptionMixMapper",
    "TYNDP_CLIMATE_YEARS",
    "TYNDP_SCENARIO_URL_ROOT",
    "TYNDP_SCENARIO_YEARS",
    "TYNDP_SCENARIOS",
    "build_country_activity_shares_from_ecoinvent_mapping",
    "build_country_activity_technology_map",
    "build_country_specific_premise_technology_map",
    "build_premise_exchange_geography_map",
    "build_premise_region_map",
    "build_z_gross_from_tyndp_excel",
    "build_z_gross_from_tyndp_pickles",
    "build_z_gross_from_tyndp_scenario",
    "consumption_mix_from_tyndp_excel",
    "consumption_mix_from_tyndp_pickles",
    "consumption_mix_from_tyndp_scenario",
    "consumption_mix_from_z_gross",
    "download_tyndp_scenario_zip",
    "ensure_tyndp_workbook",
    "load_technology_concordance",
    "load_tyndp_country_mapping",
    "map_consumption_mix_regions_xr",
    "map_consumption_mix_technologies_xr",
    "premise_activity_mix_to_database_table",
    "read_tyndp_excel_tables",
    "tyndp_scenario_paths",
    "validate_tyndp_scenario",
)
