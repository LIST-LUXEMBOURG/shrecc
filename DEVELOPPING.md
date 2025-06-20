# Developping SHRECC

## Versionning

The project uses [setuptools_scm](https://pypi.org/project/setuptools-scm/) to handle versioning, and it uses [semantic versioning](https://semver.org/).

### Version "bumping"

+ To increase the version, just add a git tag to the `main` branch.
  This can be done either directly from the CLI console, or through the web interface of gitlab.
  Only the commits pushed to the main branch with a "tag" get succesfully published to [pypi.org](https://pypi.org/project/shrecc/).
+ In general, prefer keeping a ".devX" tag on the "develop branch" and make it become a full semantic versionning compliant tag once a merge from develop to main is tone.
  For example, the current tag in main could be "0.1.0", and the tag on the develop branch of a commit after the latest one in main could be "0.2.0.dev1".
  This is useful to push "development" versions to pypi.

### Code linting and formatting with pre-commit

This repository has a working [pre-commit configuration](.pre-commit-config.yaml).
Please make sure you add [pre-commit](https://pre-commit.com/) to your environment when developping (tests, features, debugging, etc.) against this repository.

## Clone the source

For the code: 

```
git clone https://git.list.lu/shrecc_project/shrecc
```

For the data: 
```
https://git.list.lu/shrecc_project/shrecc_data
```

### Dependencies

To develop code against this repository, you could install the package in editable mode in a `pip` type of environment.
+ After having cloned the repository, change to the directory of the root of the repository
+ and then pip install the package in editable mode


```
git clone https://git.list.lu/shrecc_project/shrecc
cd shrecc
pip install -e .
```
This will only add the necessary dependencies to use shrecc. 
If you are planning on doing notebook writing and so on, you might need to add those dependencies yourself.
The `requirements.txt` file provides a list of dependencies that can be used for notebooks and a full brightway environment.


#### Using Conda environment
There is also a conda environment in the repository that can be used to setup a full environment to develop or run the notebooks.

If you prefer to use Conda, create an environment from the provided [environment.yml](environment.yml) file:
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

## Documentation

The documentation is automatically built using `sphinx` uppon pushing to the `main` branch of the repository, and published at [shrecc.readthedocs.io](https://shrecc.readthedocs.io/en/latest/)

The documentation of the package is under the [docs](docs) directory.
The directory contains some configuration files, and the index to the documentation.

+ Under this directory, there is a subdirectory [contents](docs/contents) where the actual documentation should go.
+ the [notebooks](docs/contents/notebooks/) directory can have full jupyter notebooks that get added to the documentation.

### Building the Documentation

You can build the documentation locally by installing the documentation Conda environment:

```bash
conda env create -f docs/environment.yml
```

activating the environment

```bash
conda activate sphinx_shrecc
```

and [running the build command](https://www.sphinx-doc.org/en/master/man/sphinx-build.html#sphinx-build):

```bash
sphinx-autobuild docs _build/html -a -j auto --ignore 'docs/content/api*' --open-browser
```

