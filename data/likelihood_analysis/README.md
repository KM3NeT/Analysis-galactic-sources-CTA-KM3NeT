# Likelihood scan results

## Scan result data

These `csv` data files are the filtered outcome of the full likelihood analysis for each particular source.
They are used for the final plots production.
One iteration of the scan can be done by launching `/src/perform_scan.py` locally or using REANA. 
It takes around 18 hours.
- **0.0** means the leptonic input scenario
- **1.0** means the hadronic input scenario

In total it was done 100 iterations. `csv` files have next columns:
- **case** three cases are considered (CTA only, KM3NeT only, CTA and KM3NeT together)
- **seed** different values were used for each scan
- **f_value** the hadronic fraction which is scanned
- **stat_total** total statistics
- **stat_gamma** statistics for gamma datasets
- **stat_nu** statistics for neutrino datasets
- **int_PD** integral over PD
- **int_IC** integral over IC

## Plots

Output of the notebooks _Plot_avg_limits_ and _Plot_dTS_results_ are stored here.