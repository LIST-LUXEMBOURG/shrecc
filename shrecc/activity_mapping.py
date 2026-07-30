"""Compatibility imports for the former activity-mapping module name.

New code should import these functions from :mod:`shrecc.mapping`.
"""

from shrecc.mapping import (
    DEFAULT_COUNTRY_ALIASES,
    DEFAULT_FALLBACK_ACTIVITY,
    activity_mix_to_database_table,
    aggregate_consumption_mix,
    filter_consumption_mix_time,
    load_ecoinvent_mapping,
    map_consumption_mix_to_ecoinvent_activities,
    mapping_gap_to_report,
    validate_consumption_profile,
    validate_inventory_resolution,
)

__all__ = (
    "DEFAULT_COUNTRY_ALIASES",
    "DEFAULT_FALLBACK_ACTIVITY",
    "activity_mix_to_database_table",
    "aggregate_consumption_mix",
    "filter_consumption_mix_time",
    "load_ecoinvent_mapping",
    "map_consumption_mix_to_ecoinvent_activities",
    "mapping_gap_to_report",
    "validate_consumption_profile",
    "validate_inventory_resolution",
)
