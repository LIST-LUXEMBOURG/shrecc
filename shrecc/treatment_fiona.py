"""Compatibility imports for the FIONA/TYNDP workflow.

New code should import TYNDP ingestion functions from :mod:`shrecc.tyndp` and
premise mapping helpers from :mod:`shrecc.premise_mapping`.
"""

try:
    from shrecc.premise_mapping import (
        DEFAULT_PREMISE_IAM_MODELS,
        PremiseConsumptionMixMapper,
        build_country_activity_technology_map,
        build_premise_region_map,
        map_consumption_mix_regions_xr,
        map_consumption_mix_technologies_xr,
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
        read_tyndp_excel_tables,
        tyndp_scenario_paths,
        validate_tyndp_scenario,
    )
except ImportError:
    import sys
    from pathlib import Path

    module_dir = Path(__file__).resolve().parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

    from premise_mapping import (
        DEFAULT_PREMISE_IAM_MODELS,
        PremiseConsumptionMixMapper,
        build_country_activity_technology_map,
        build_premise_region_map,
        map_consumption_mix_regions_xr,
        map_consumption_mix_technologies_xr,
    )
    from tyndp import (
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
        read_tyndp_excel_tables,
        tyndp_scenario_paths,
        validate_tyndp_scenario,
    )

__all__ = [
    "DEFAULT_PREMISE_IAM_MODELS",
    "PremiseConsumptionMixMapper",
    "TYNDP_CLIMATE_YEARS",
    "TYNDP_SCENARIO_URL_ROOT",
    "TYNDP_SCENARIO_YEARS",
    "TYNDP_SCENARIOS",
    "build_country_activity_technology_map",
    "build_premise_region_map",
    "build_z_gross_from_tyndp_excel",
    "build_z_gross_from_tyndp_pickles",
    "build_z_gross_from_tyndp_scenario",
    "consumption_mix_from_tyndp_scenario",
    "consumption_mix_from_tyndp_excel",
    "consumption_mix_from_tyndp_pickles",
    "consumption_mix_from_z_gross",
    "download_tyndp_scenario_zip",
    "ensure_tyndp_workbook",
    "load_technology_concordance",
    "map_consumption_mix_regions_xr",
    "map_consumption_mix_technologies_xr",
    "read_tyndp_excel_tables",
    "tyndp_scenario_paths",
    "validate_tyndp_scenario",
]
