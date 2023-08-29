# Reproducing the analysis

The aim of this analysis is to simulate how well a combined analysis of CTA and KM3NeT data can differentiate between hadronic and leptonic emission scenarios of galactic gamma-ray sources. The focus is on the comparison of the combined analysis to the separate analysis of the two instruments within [Gammapy](https://docs.gammapy.org/0.17/index.html).

## Execution of the notebooks

The execution of the analysis can be configured in `analysis_config.yml`, where parameters for the full analysis execution are stored. 
The notebooks can be run manually, with the selected sources defined in the configuration file. Before starting a notebook, set the source you want to execute the notebooks for in the configuration file.

## Generating source models

A fit to the source models can be created using the **Fit_models** notebook.

## Creating datasets

To create pseudo datasets from the IRFs, two notebooks are provided for KM3NeT and CTA respectively. The first creates pseudo data sets, the second plots the resulting maps and spectra:

* After running the generation of pseudo-datasets with the **Create_CTA_datasets** script, the resulting maps are also stored in `/cta`.
* The pseudo data and spectrum from flux models can be plotted using **Plot CTA datasets**.
* Equivalently the KM3NeT data sets can be generated with the **Create_KM3NeT_datasets**. As this is computationally intensive and the resulting datasets are large, see the `/data/km3net` folder for instructions how to download the preprocessed data sets.
* Plot the outcome of the KM3NeT data set generation with **Plot_KM3NeT_datasets**

## Performing likelihood analysis

### Analysis strategy
For this simulation the hadronic contribution of the models is calculated based on the integrals $I$ over the gamma-ray flux $\phi(E)$
$$I = \int_{0.1\mathrm{\,TeV}}^{100\mathrm{\,TeV}} \phi(E) \cdot dE$$

The value of the hadronic contribution $f$ follows as
$$f = \frac{I_H}{I_L + I_H}$$

In the analysis the results for three test scenarios will be compared with each other.  
In *Scenario 1* only knowledge of the CTA data is assumed and different combinations of the models are scanned using the prior function described below. For each value of the hadronic contribution $f_{0}$ both models are optimized and the $\small\Delta \mathrm{TS}$ values show how well CTA can differentiate between hadronic and leptonic emission scenarios by itself. 

In *Scenario 2* a scan is performed on KM3NeT data under the knowledge of the CTA spectrum including its uncertainties. This is implemented in a way that the hadronic pion-decay (PD) model is fitted to the CTA data including an error estimation. The parameters $P_\mathrm{CTA}$ of the best-fit proton spectrum are used to calculate the neutrino flux prediction.
A prior function is added to the $TS$-value of the KM3NeT data set for all parameters $P$ in $[ A, \space\Gamma,\space E_\mathrm{cut},\space\beta ]$:

$$\mathrm{TS_{total}} = \mathrm{TS} + \sum_P\frac{(P - P_\mathrm{CTA})^2}{({\small\Delta} P_{\mathrm{CTA}})^2}$$

In order to perform the scan over different hadronic contributions the value $A_\mathrm{CTA}$ of the best-fit amplitude and the corresponding uncertainty are scaled by $f$. This method allows for small variations of the neutrino spectrum within the uncertainty of the CTA spectrum. Note that for $f=0$ one cannot scale the error to 0 but instead a negligible small error is used.

For *Scenario 3* the prior functions of the KM3NeT data set are removed and the scan is performed including the CTA data set. Similar to scenario 1 the scan is performed using the prior function on $f$ while both models are optimized for both data sets simultaneously. This scenario corresponds to the combined fit where the low level data from both instruments is available.

### The prior function

In order to force the fit to maintain a certain ration $f$ of the leptonic and hadronic model, a penalty term is added to the total TS value of the data sets. This term looks like this $$ S \cdot \frac{(f - f_0)^2}{\Delta f^2} $$ where S is the prior_scale, $f_0$ is the ratio of the models which should be scanned and $\Delta f = 0.01$ is the uncertainty which the fit can use to optimize the total TS value. Note that for the evaluation of the profile likelihood scan (for obtaining confidence intervals) the contribution of the prior function is removed.

### The scan python script

The python script (`src/perform_scan.py`) for analysis can be run locally (not recommended, takes more than 18 hours) or using REANA. It can accept 3 arguments:
1. The source name (default **VelaX**)
2. A random seed (default 1)
3. The input hadronic contribution for the simulation ([0,1], where 0 refers to the leptonic case and 1 to the hadronic case. All scenarios in between can also be tested). (default 0)

However it can be launched without arguments (default values are provided automatically).
To run this script successfully, it requires to have generated
- CTA dataset
- KM3NeT dataset
- Flux models

Results of one performed scan will be stored in `outcome` folder as `.npy` files.

### Calculation of credible intervals

The TS value is defined as $$ \mathrm{TS} = - 2 \ln(\mathcal{L}) $$ 
so we can use the $\small\Delta\mathrm{TS}$ value to calculate a likelihood ratio: 
$$ \frac{\mathcal{L_\mathrm{M1}}}{\mathcal{L_\mathrm{M2}}} = \exp[-\small\Delta\mathrm{TS}/2] $$
We follow a Bayesian approach and treat this as our posterior probability distribution.

In the notebook _Plot_dTS_results_ the ${\small\Delta}  \mathrm{TS}$ values for each scan are averaged over the realizations and described by a continuous spline function. One can then integrate $\exp[-\small\Delta\mathrm{TS}(f)/2]$ and find the values of $f$ for which 90\% of the area under this curve are included. This yields the 90% credible interval for $f$.

### Interpretation of the results
On a general note one finds that the sensitivity of the combined analysis is very similar to the sensitivity one would get by combining the results of the separate analyses. This is also expected because in the cases tested here the analyses were consistent in a sense that the same spatial and spectral models were used for the emission region and the ratio of the fluxes were constrained by the same prior functions. For most of the sources the models are fitted to similar values (dominated by the high CTA statistics) for the separate and combined analyses so that the $\small\Delta\mathrm{TS}$-values simply add up. However this is not always the case as one can see in the case of leptonic emission of Vela X. There the Pion Decay model can not describe the shape of the Inverse Compton model really well because of its hard cutoff. When adding the KM3NeT data set to the analysis the shape of the models is constraint in a different way compared to the CTA-alone scan so that the combined scan would give stronger constraints on the hadronic contribution than the addition of the separate scans.