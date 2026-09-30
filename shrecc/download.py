"""Compatibility imports for the historical Energy Charts module name.

New code should import these functions from :mod:`shrecc.energy_charts`.
"""

from shrecc.energy_charts import (
    API_REQUEST_TIMEOUT,
    UNRESOLVED_IMPORT_TECHNOLOGY,
    build_energy_charts_solver_inputs,
    cleaning_data,
    consumption_results_from_energy_charts,
    data_processing,
    get_data,
    get_package_user_data_dir,
    get_prod,
    get_trade,
    write_energy_charts_consumption_cache,
    year_to_unix,
)

__all__ = (
    "API_REQUEST_TIMEOUT",
    "UNRESOLVED_IMPORT_TECHNOLOGY",
    "build_energy_charts_solver_inputs",
    "cleaning_data",
    "consumption_results_from_energy_charts",
    "data_processing",
    "get_data",
    "get_package_user_data_dir",
    "get_prod",
    "get_trade",
    "write_energy_charts_consumption_cache",
    "year_to_unix",
)
