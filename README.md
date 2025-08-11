# Preparing for the worst: Long-term and short-term weather extremes in resource adequacy assessment

Code for preprint "Preparing for the worst: Long-term and short-term weather extremes in resource adequacy assessment" ([arXiv:2508.05163](https://arxiv.org/abs/2508.05163)). Additional data accompanying the paper can be found at the Zenodo repository [10.5281/zenodo.16753688](https://doi.org/10.5281/zenodo.16753688).

Abstract:
> Security of supply is a common and important concern when integrating renewables in net-zero power systems.
Extreme weather affects both demand and supply leading to power system stress; in Europe this stress spreads continentally beyond the meteorological root cause.
We use an approach based on shadow prices to identify periods of elevated stress called system-defining events and analyse their impact on the power system.
By classifying different types of system-defining events, we identify challenges to power system operation and planning.
Crucially, we find the need for sufficient resilience back-up (power) capacities whose financial viability is precarious due to weather variability.
Furthermore, we disentangle short- and long-term resilience challenges with distinct metrics and stress tests to incorporate both into future energy modelling assessments.
Our methodology and implementation in the open model PyPSA-Eur can be re-applied to other systems and help researchers and policymakers in building more resilient and adequate energy systems.

## Overview

This repository contains the code necessary to reproduce the results and plots. A fork of PyPSA-Eur which was used to run the optimisations is attached as a submodule. The network files as well as various data files (processing data, sensitivity analysis) are stored in the Zenodo repository [10.5281/zenodo.16753688](https://doi.org/10.5281/zenodo.16753688). Detailed descriptions can be found in the Zenodo readme.

The results are analysed in the Jupyter notebooks `workflow/notebooks/paper_plots.ipynb` and `workflow/notebooks/supplementary_material.ipynb`; they utilise data from the Zenodo repository. These data were generated with the scripts in `workflow/notebooks/`.


### Installation and set-up

1. Clone this git repository with the `--recurse-submodules` option in order to also pull the PyPSA-Eur submodule:
   ```bash
   git clone --recurse-submodules git@github.com:aleks-g/stressed-system.git```

2. Download the data from Zenodo and place it as described in the readme file, i.e. add the network files to `workflow/pypsa-eur/results/`.

3. Install the conda environments `envs/environment.yaml` (for running PyPSA-Eur) and `envs/notebook_analysis.yaml` (for running the notebooks). You can do this with the following commands:
   ```bash
   conda env create -f envs/environment.yaml
   conda env create -f envs/notebook_analysis.yaml
   ```

4. Run the following workflow with the following command to obtain the results of the capacity expansion and the filtered system-defining events (SDEs):
    ```bash
    cd workflow/pypsa-eur
    conda activate stressed-system
    snakemake -call all_difficult_periods --configfile config/stressful-weather.yaml -n #dry-run
    snakemake -call all_difficult_periods --configfile config/stressful-weather.yaml
    ```
    
    This will generate the necessary data files in the `processing_data/stressful-weather-sensitivities/` folder.

5. For the validation of the results, run the dispatch optimisation:
    ```bash
    snakemake -call all_operational_years --configfile config/stressful-weather.yaml -n #dry-run
    snakemake -call all_operational_years --configfile config/stressful-weather.yaml
    ```

6. For the sensitivity analysis, run:
    ```bash
    snakemake -call all_operational_years --configfile config/stressful-weather-sensitivities.yaml -n #dry-run
    snakemake -call all_operational_years --configfile config/stressful-weather-sensitivities.yaml
    ```
    This will generate the necessary data files in the `processing_data/stressful-weather-sensitivities/` folder.

### Necessary data and inputs

The data provided in the Zenodo repository are necessary to run the workflow.

With the current settings, a Gurobi license is needed, e.g. an academic one. Other solvers can be used to run PyPSA-Eur as well.

If rerunning from scratch, follow the PyPSA-Eur workflow (v.0.13.0) as documented in [as documented here](https://pypsa-eur.readthedocs.io/en/latest/) following [the codebase](https://github.com/PyPSA/pypsa-eur) Run the following scripts afterwards:
- `workflow/notebooks/_generate_data_for_analysis.py` to generate the data for analysis.
- `workflow/notebooks/_dashboard.py` to generate the dashboard data.
- `workflow/notebooks/_unserved_energy.py` to generate the unserved energy data.
- `workflow/notebooks/_cluster_sdes.py` to generate the clusters.

## Contributors

Aleksander Grochowicz, Hannah Bloomfield, Marta Victoria. The filtering of SDEs is based on previous work published in []() with the codebase available at [this repository](https://github.com/koen-vg/stressful-weather/tree/v0) and copyrights are acknowledged in the relevant scripts.

## Licenses
MIT license, unless specified otherwise.
