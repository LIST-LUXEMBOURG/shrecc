# Copyright © 2024-2026 Luxembourg Institute of Science and Technology
# Licensed under the MIT License (see LICENSE file for details).
# Authors: [Sabina Bednářová, Thomas Gibon]


"""shrecc"""

__all__ = (
    "__version__",
    "NewDatabase",
    "create_database",
    "filt_cutoff",
    "get_data",
    "data_processing",
)
from .database import create_database, filt_cutoff
from .energy_charts import data_processing, get_data
from .pipeline import NewDatabase

__version__ = "0.1.0.dev3"
