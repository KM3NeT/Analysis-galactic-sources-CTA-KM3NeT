[![License](https://img.shields.io/badge/License-BSD_3--Clause-blueviolet.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/684463999.svg)](https://zenodo.org/badge/latestdoi/684463999)


# CTA and KM3NeT Common Source Search

This repository contains the prospects for combined analyses of hadronic emission from γ-ray sources in the Milky Way with CTA and KM3NeT/ARCA. It complements the publication "Prospects for combined analyses of hadronic emission from γ-ray sources in the Milky Way with CTA and KM3NeT" (link: tba), and for in-depth description of the analysis please refer to the paper.
The aim of this analysis is to simulate how well a combined analysis of CTA and KM3NeT data can differentiate between hadronic and leptonic emission scenarios of galactic gamma-ray sources. The focus is on the comparison of the combined analysis to the separate analysis of the two instruments within [Gammapy](https://docs.gammapy.org/0.17/index.html).
This content is only compatible with `gammapy v0.17`, later versions are not supported.  
It should be noted, that `gammapy v0.17` is not compatible with the M1 CPU. The only option to run this analysis with this CPU is to use a docker image.
This option will be provided in the next version of the repository.

## Content

* **Analysis/**: Notebooks to reproduce the analysis
* **data/**: Instrument Response Functions (IRFs) and flux model for the sources
* **envs/**: Configuration files for setting up the python environment
* **src/**: supplementary scripts

## Installation

### Download
First it is required to download the whole content of the repository, it can be done using `git`:
```sh
git clone git@github.com:KM3NeT/Analysis-galactic-sources-CTA-KM3NeT.git
```
or
```sh
git clone https://github.com/KM3NeT/Analysis-galactic-sources-CTA-KM3NeT.git
```
then
```sh
cd cta-and-km3net/
```

### Creating the environment
#### Using conda
In order to use conda to build the environment, conda has to be installed. To see how, use these [Installation instructions](https://docs.anaconda.com/free/anaconda/install/).

Build environment using `conda` from `environment.yml` file:
```sh
conda env create -f envs/environment.yml
conda activate km3net_cta_env
```
#### Using venv

It requires to build a dedicated environment.
Build environment using `pip`, first it requires to install manually `python3.8`, then install `virtualenv`:
```sh
pip install virtualenv
# for standard preinstalled python 3.8
virtualenv venv --python=python3.8
# or specify path
virtualenv venv --python=/path/to/python3.8
```
acitvate `venv`:
```sh
# on Windows
.\venv\Scripts\activate.ps1
# on Linux
source venv/bin/activate
```
Install necessary packages:
```sh
pip install cython numpy
pip install -r requirements.txt
```
#### Using Jupyter
In order to run the notebooks, you need to have Jupyter installed. You can install it using `pip install jupyter` or following the instructions at the [Juypter website](https://jupyter.org/install).

### Running the Jupyter kernel

Jupyter notebook kernel and launch your notebook:
```sh
python -m ipykernel install --user --name=km3net_cta
jupyter-notebook
```
And for `zsh` shell, you need to execute these lines first before installation of the kernel
```zsh
conda install -c conda-forge notebook
conda install -c conda-forge nb_conda_kernels
```

## Integration with REANA
Analysis can be run in [REANA](https://docs.reana.io/), for this purpose it needs to install `reana-client` inside virtual environment:
```sh
# inside venv or conda env
pip install reana-client
```
$\textcolor{red}{\text{Warning!}}$
`reana-client` is currently not compatible with Windows even inside a conda environment.


After installation of the client, it needs to set connection using a token. For convinience all REANA commands are specified in `run_reana.sh` script. Launch the script in terminal.
```sh
export REANA_SERVER_URL=https://reana.cern.ch
export REANA_ACCESS_TOKEN=*YOUR_TOKEN*
. run_reana.sh
# get the results of analysis
reana-client download
```