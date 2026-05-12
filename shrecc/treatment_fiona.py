"""Compatibility imports for the FIONA/TYNDP workflow.

New code should import TYNDP ingestion functions from :mod:`shrecc.tyndp` and
premise mapping helpers from :mod:`shrecc.premise_mapping`.
"""

try:
    from .premise_mapping import (
        DEFAULT_PREMISE_IAM_MODELS,
        PremiseConsumptionMixMapper,
        build_premise_region_map,
        map_consumption_mix_regions_xr,
        map_consumption_mix_technologies_xr,
    )
    from .tyndp import (
        build_z_gross_from_tyndp_excel,
        build_z_gross_from_tyndp_pickles,
        consumption_mix_from_tyndp_excel,
        consumption_mix_from_tyndp_pickles,
        consumption_mix_from_z_gross,
        load_technology_concordance,
        read_tyndp_excel_tables,
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
        build_premise_region_map,
        map_consumption_mix_regions_xr,
        map_consumption_mix_technologies_xr,
    )
    from tyndp import (
        build_z_gross_from_tyndp_excel,
        build_z_gross_from_tyndp_pickles,
        consumption_mix_from_tyndp_excel,
        consumption_mix_from_tyndp_pickles,
        consumption_mix_from_z_gross,
        load_technology_concordance,
        read_tyndp_excel_tables,
    )

__all__ = [
    "DEFAULT_PREMISE_IAM_MODELS",
    "PremiseConsumptionMixMapper",
    "build_premise_region_map",
    "build_z_gross_from_tyndp_excel",
    "build_z_gross_from_tyndp_pickles",
    "consumption_mix_from_tyndp_excel",
    "consumption_mix_from_tyndp_pickles",
    "consumption_mix_from_z_gross",
    "load_technology_concordance",
    "map_consumption_mix_regions_xr",
    "map_consumption_mix_technologies_xr",
    "read_tyndp_excel_tables",
]
