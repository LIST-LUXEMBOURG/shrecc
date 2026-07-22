"""Compatibility facade for the former electricity-treatment module.

New code should use :mod:`shrecc.energy_charts` for source preparation and
:mod:`shrecc.solver` for consumption-mix calculations.
"""

from shrecc._legacy_treatment import (
    add_missing_elements,
    calculate_Z_cons,
    calculate_results,
    concatenate_results,
    create_block_diagonal_matrix,
    process_matrix,
    process_results_light,
    treating_data,
)
from shrecc.energy_charts import data_processing
from shrecc.result_store import (
    load_pickle as load_from_pickle,
    save_pickle as save_to_pickle,
)

__all__ = (
    "add_missing_elements",
    "calculate_Z_cons",
    "calculate_results",
    "concatenate_results",
    "create_block_diagonal_matrix",
    "data_processing",
    "load_from_pickle",
    "process_matrix",
    "process_results_light",
    "save_to_pickle",
    "treating_data",
)
