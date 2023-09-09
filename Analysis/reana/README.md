## Reproduce analysis with REANA

The content of this folder allows to easily reproduce the results of the current analysis with [REANA](https://reanahub.io/).
The main part of REANA workflow is a `yaml` file. All information and changes can be done inside an appropriate REANA `yaml` file.  
There are several `yaml` files here:
- `reana_cta_dataset.yml` (~5 min), it launches `flux_models.py` and `create_cta_dataset.py` as a serial workflow. As an output it produces plots and CTA dataset in `.fits` format.
- `reana_km3net_dataset.yml` (max 2-3 hours), it launches `create_km3net_dataset.py` and `flux_models.py` as a serial workflow. As an output it produces plots, KM3NeT dataset in `.fits` format and separated background for neutrino and muon contributions for each of datasets in the same `.fits` format.
- `reana_comb_analysis.yml` (> 20 hours), it launches `create_cta_dataset.py`, `create_km3net_dataset.py`, `flux_models.py` and `src/perform_scan.py` as a serial workflow. As an output it returns all previous results for both datasets plus results of one scan for three different cases
    - CTA only
    - KM3NeT only
    - CTA plus KM3NeT together
- `reana_plot_avg_limits.yml` (~5 min), it launches `plot_avg_limits.py` as a serial workflow. As an output it produces plots.
- `reana_plot_dTS_results.yml` (1-2 min), it launches `plot_dTS_results.py` as a serial workflow. As an output it produces plots. 

Changes between different REANA `yaml` files can be done in `run_reana.sh` by uncommenting the appropriate lines.  
Moreover all current python scripts can be executed locally using
```bash
# km3net_cta_env must be activated
python script_name.py
```
`create_cta_dataset.py`,`create_km3net_dataset.py`, `plot_avg_limits.py`, `plot_dTS_results.py` can be launched independently, however the script `src/perform_scan.py` will require datasets of both setups.
The output will appear in the new folder `results` and `data` in the root directory of the repository.
The output can be easialy downloaded using next command
```bash
reana-client download
```
it downloads the output of the last workflow of REANA in the current directory.