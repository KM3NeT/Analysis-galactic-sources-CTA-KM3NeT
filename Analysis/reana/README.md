## Reproduce analysis with REANA

The content of this folder allows to easily reproduce the results of the current analysis with [REANA](https://reanahub.io/).  
The main part of REANA workflow is a `yaml` file. There are several `yaml` files here:
- `reana_cta_dataset.yml`, it launches `create_cta_dataset.py` as a serial workflow. As an output it produces plots and CTA dataset in `.fits` format.
- `reana_km3net_dataset.yml`, it launches `create_km3net_dataset.py` as a serial workflow. As an output it produces plots, KM3NeT dataset in `.fits` format and separated background for neutrino and muon contributions for each of datasets in the same `.fits` format.
- `reana_comb_dataset.yml`, it launches `create_cta_dataset.py`, `create_km3net_dataset.py` and `src/perform_scan.py` as a serial workflow. As an output it returns all previous results for both datasets plus results of one scan for three different cases
    - CTA only
    - KM3NeT only
    - CTA plus KM3NeT together

Changes between different REANA `yaml` files can be done in `run_reana.sh` by uncommenting the appropriate lines.  
Moreover all current python scripts can be executed locally using
```bash
# km3net_cta_env must be activated
python script_name.py
```
`create_cta_dataset.py` and `create_km3net_dataset.py` can be launched independently, however the script `src/perform_scan.py` will require results of the previous two.
The output will appear in the folders `results` and `outcome` in the root directory of the repository.