# SHRECC

Simple Hourly Resolution Electricity Consumption Calculation

## Description

SHRECC package is a python package directly compatible with Brightway2 or Brightway2.5 to create time-aware electricity databases. For any given year and countries (check availability on https://api.energy-charts.info/), download and prepare data for low-voltage electricity consumption.

## Features

- **High-resolution electricity mixes** – Generates electricity life cycle inventories (LCIs) with **hourly** resolution, enhancing accuracy for life cycle assessment (LCA).
- **Brightway2/2.5 compatibility** – Seamlessly integrates with Brightway, allowing direct use in existing LCA models.
- **Dynamic temporal representation** – Users can select electricity mixes by **hour, month, or season**, addressing fluctuations in renewable energy generation and consumption.
- **Automated data retrieval** – Pulls electricity production, trade, and consumption data from the **Energy Charts API**, ensuring up-to-date datasets.
- **Ecoinvent matching** – Aligns with **ecoinvent classifications**, converting from ENTSO-E datasets.
- **User-controlled updates** – Enables **one-time or recurring** updates, allowing continuous tracking of electricity mix evolution over time.
- **Optimized impact assessments** – Helps reduce uncertainty and improve **decision-making for electricity-intensive technologies** by considering real-time electricity mix variations.

## Installation

### Clone the repository

For the code: git clone [https://git.list.lu/shrecc_project/shrecc](https://git.list.lu/shrecc_project/shrecc)
For the data: [https://git.list.lu/shrecc_project/shrecc_data](https://git.list.lu/shrecc_project/shrecc_data)

### Install dependencies

#### Using pip

If you are using a standard Python environment, install dependencies with:
` pip install -r requirements.txt`

#### Using Conda environment

If you prefer to use Conda, create an environment from the provided environment.yml file:
`conda env create -f environment.yml`
Then activate the environment: `conda activate shrecc`
Alternatively, if the environment already exists and you want to update it: `conda env update --file environment.yml --prune`

#### Using Conda environment (avoid Anaconda)

If you prefer to use Conda and meanwhile avoid using Anaconda, create an environment from the provided environment_clean.yml file:
`conda env create -f environment_clean.yml`

Or, if you have Mamba installed (a faster Conda alternative):

`mamba env create -f environment_clean.yml`
Then activate the environment: `conda activate shrecc_clean`
Alternatively, if the environment already exists and you want to update it: `conda env update --file environment_clean.yml --prune`

Or, with Mamba:

`mamba env update -n shrecc_clean -f environment.yml  `

## Usage

You can find usage examples in the Jupyter notebook: [notebooks/example.ipynb](notebooks/example.ipynb).

## License

Copyright © 2024 Luxembourg Institute of Science and Technology
Licensed under the MIT License.

## Authors

* Sabina Bednářová (<sabina.bednarova@list.lu>)
* Thomas Gibon (<thomas.gibon@list.lu>)
