# Copyright © 2024 Luxembourg Institute of Science and Technology
# Licensed under the MIT License (see LICENSE file for details).
# Authors: [Sabina Bednářová, Thomas Gibon]


"""shrecc"""

__all__ = (
    "__version__",
    "create_database",
    "filt_cutoff",
    "get_data",
    "data_processing",
)
from .database import create_database, filt_cutoff
from .download import get_data
from .treatment import data_processing

__version__ = "0.0.2"
