# Source model data

The folder contains source catalog with sources names and their astronomical parameters.  

Flux points for each source are taken from:
- eHWC J1907+063 - https://arxiv.org/abs/1909.08609
- VelaX - https://www.aanda.org/articles/aa/full_html/2012/12/aa19919-12/aa19919-12.html
- RX J1713.7-3946 - https://www.aanda.org/articles/aa/full_html/2018/04/aa29790-16/aa29790-16.html
- Westerlund 1 - https://www.mpi-hd.mpg.de/hfm/HESS/pages/publications/auxiliary/AA537_A114.html  

Some of them have been modified for standardization and have the same structure for each source.

## Generating model fits

In order to generate model fits from the provided model data sets in the `FP_{SOURCE}.csv` files, the notebook _Fit_models_ has to be run. Fits are stored in the `/modelfits` folder.

## Plots

The plots from the same notebook _Fit_models_ are stored in this folder.