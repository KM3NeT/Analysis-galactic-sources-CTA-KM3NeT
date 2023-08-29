import numpy as np
import pandas as pd
import time
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

from gammapy.datasets import Datasets
from gammapy.modeling.models import (
    DiskSpatialModel,
    SkyModel,
    NaimaSpectralModel,
    GaussianSpatialModel,
)
from naima.models import ExponentialCutoffPowerLaw, InverseCompton

import warnings
from pathlib import Path

from os import makedirs
import sys


start_time = time.time()
current_dir = Path(__file__).resolve().parent
par_dir = current_dir.parent
# parpar_dir = par_dir.parent
sys.path.append(str(par_dir))
# sys.path.append(str(parpar_dir))

from src import PriorMapDataset, PriorMapDataset2, PriorDatasets, Fit_wp
from src import PionDecayKelner06

from configure_analysis import AnalysisConfig

analysisconfig = AnalysisConfig()
analysisconfig.set_datapath()


# path to generated CTA and KM3NeT datasets
# path_to_data = Path(par_dir / "data")
path_to_data = analysisconfig.datapath
print(path_to_data)

### Read inputs ###

# read in the command line arguments
try:
    source_name = sys.argv[1]
    seed = int(sys.argv[2])
    f_input = float(sys.argv[3])  # 0 -> leptonic scenario, 1 -> hadronic scenario
except IndexError:
    print("No args were provided, used defaults")
    source_name = analysisconfig.get_source()
    seed = 1
    f_input = 0

# create a folder to store the outcome of this scan
outcome_path = Path(
    path_to_data
    / "likelihood_analysis"
    / "numpy_files"
    / source_name
    / "{:.1f}".format(f_input)
)
makedirs(outcome_path, exist_ok=True)

# Some other settings
error_scale = 0.02  # these values were found
prior_scale = 0.1  # to work best

e_int_min = 0.1 * u.TeV  # integrate models within these
e_int_max = 100 * u.TeV  # bounds to compute the hadronic fraction f

# Read in source parameters
source_pos_dist = pd.read_csv(
    Path(path_to_data / "models" / "sources_catalog.csv"), index_col=0
)
src_pos = SkyCoord(
    *source_pos_dist.loc[source_name][["RA", "Dec"]], unit="deg", frame="icrs"
)

# read in the input model parameters
# before the files for PD and IC should be generated
model_pars_PD = np.loadtxt(
    Path(
        par_dir / "data" / "models" / "modelfits" / f"input_model_PD_{source_name}.txt"
    )
)
model_pars_IC = np.loadtxt(
    Path(
        par_dir / "data" / "models" / "modelfits" / f"input_model_IC_{source_name}.txt"
    )
)


# Read in the Datasets
# KM3NeT yaml files from KM3NeT datasets
fdata = Path(
    path_to_data
    / "km3net"
    / "pseudodata"
    / f"KM3NeT_{source_name}_10y"
    / f"KM3NeT_{source_name}_10y_datasets.yaml"
)
fmodel = Path(
    path_to_data
    / "km3net"
    / "pseudodata"
    / f"KM3NeT_{source_name}_10y"
    / f"KM3NeT_{source_name}_10y_models.yaml"
)
print(f"fdata: {fdata}\nfmodel: {fmodel}")

datasets_km3net = PriorDatasets.read(filedata=fdata, filemodel=fmodel)

dataset_cta = PriorMapDataset2.read(
    Path(path_to_data / "cta" / "pseudodata" / f"CTA_{source_name}_200h_p4.fits.gz")
)
### Create models ###

# Spatial model (same for all datasets)
if source_name != "HESSJ1908":
    spatial_model = DiskSpatialModel(
        lon_0=src_pos.ra,
        lat_0=src_pos.dec,
        r_0=source_pos_dist["Radius"][source_name] * u.deg,
        frame="icrs",
    )
else:
    spatial_model = GaussianSpatialModel(
        lon_0=src_pos.ra,
        lat_0=src_pos.dec,
        sigma=source_pos_dist["Radius"][source_name] * u.deg,
        frame="icrs",
    )

# Hadronic (PD) spectral model
ECPL_PD = ExponentialCutoffPowerLaw(
    amplitude=model_pars_PD[0] / u.eV,
    e_0=model_pars_PD[1] * u.TeV,
    alpha=model_pars_PD[2],
    e_cutoff=model_pars_PD[3] * u.TeV,
    beta=model_pars_PD[4],
)
gamma_ECPL_PD = PionDecayKelner06(ECPL_PD, particle_type="gamma")
nu_ECPL_PD = PionDecayKelner06(
    ECPL_PD, particle_type="muon_neutrino", oscillation_factor=0.5
)

gamma_spectral_model_PD = NaimaSpectralModel(
    gamma_ECPL_PD, distance=source_pos_dist["Dist"][source_name] * u.kpc
)
nu_spectral_model_PD = NaimaSpectralModel(
    nu_ECPL_PD, distance=source_pos_dist["Dist"][source_name] * u.kpc
)

gamma_model_PD = SkyModel(
    spectral_model=gamma_spectral_model_PD,
    spatial_model=spatial_model,
    name="gamma_PD",
    datasets_names=[dataset_cta.name],
)
nu_model_PD = SkyModel(
    spectral_model=nu_spectral_model_PD,
    spatial_model=spatial_model,
    name="nu_PD",
    datasets_names=datasets_km3net.names,
)

# Link all parameters of spectral models
gamma_model_PD.spectral_model.amplitude = nu_model_PD.spectral_model.amplitude
gamma_model_PD.spectral_model.e_0 = nu_model_PD.spectral_model.e_0
gamma_model_PD.spectral_model.alpha = nu_model_PD.spectral_model.alpha
gamma_model_PD.spectral_model.beta = nu_model_PD.spectral_model.beta
gamma_model_PD.spectral_model.e_cutoff = nu_model_PD.spectral_model.e_cutoff

# Freeze / constrain some parameters
nu_model_PD.spectral_model.beta.min = 0.1
nu_model_PD.spectral_model.beta.max = 2.0
nu_model_PD.spectral_model.e_cutoff.min = (
    0.0  # needed because of Case 2 fit (sometimes negative e_cutoff)
)
nu_model_PD.spectral_model.e_0.frozen = True
nu_model_PD.spatial_model.parameters.freeze_all()
gamma_model_PD.spatial_model.parameters.freeze_all()

# Leptonic (IC) spectral model
ECPL_IC = ExponentialCutoffPowerLaw(
    amplitude=model_pars_IC[0] / u.eV,
    e_0=model_pars_IC[1] * u.TeV,
    alpha=model_pars_IC[2],
    e_cutoff=model_pars_IC[3] * u.TeV,
    beta=model_pars_IC[4],
)
gamma_ECPL_IC = InverseCompton(ECPL_IC, seed_photon_fields="CMB")
gamma_spectral_model_IC = NaimaSpectralModel(
    gamma_ECPL_IC, distance=source_pos_dist["Dist"][source_name] * u.kpc
)

gamma_model_IC = SkyModel(
    spectral_model=gamma_spectral_model_IC,
    spatial_model=spatial_model,
    name="gamma_IC",
    datasets_names=[dataset_cta.name],
)

# Freeze / constrain some parameters
gamma_model_IC.spectral_model.beta.min = 0.1
gamma_model_IC.spectral_model.beta.max = 2
gamma_model_IC.spectral_model.e_0.frozen = True
gamma_model_IC.spatial_model.parameters.freeze_all()


def restore_param_values(f=None, freeze=False, par_IC=None, par_PD=None):
    # hadronic (PD)
    nu_model_PD.spectral_model.amplitude.value = model_pars_PD[0]
    nu_model_PD.spectral_model.e_0.value = model_pars_PD[1]
    nu_model_PD.spectral_model.alpha.value = model_pars_PD[2]
    nu_model_PD.spectral_model.e_cutoff.value = model_pars_PD[3]
    nu_model_PD.spectral_model.beta.value = model_pars_PD[4]
    if par_PD is not None:
        for p_model, p_set in zip(nu_model_PD.spectral_model.parameters, par_PD):
            p_model.value = p_set.value
            p_model.error = p_set.error * error_scale

    # leptonic (IC)
    gamma_model_IC.spectral_model.amplitude.value = model_pars_IC[0]
    gamma_model_IC.spectral_model.e_0.value = model_pars_IC[1]
    gamma_model_IC.spectral_model.alpha.value = model_pars_IC[2]
    gamma_model_IC.spectral_model.e_cutoff.value = model_pars_IC[3]
    gamma_model_IC.spectral_model.beta.value = model_pars_IC[4]
    if par_IC is not None:
        for p_model, p_set in zip(gamma_model_IC.spectral_model.parameters, par_IC):
            p_model.value = p_set.value
            p_model.error = p_set.error * error_scale

    if f is not None:
        nu_model_PD.spectral_model.amplitude.value *= f
        gamma_model_IC.spectral_model.amplitude.value *= 1 - f

    if freeze:
        if f == 0.0:
            nu_model_PD.spectral_model.parameters.freeze_all()

        elif f == 1.0:
            gamma_model_IC.spectral_model.parameters.freeze_all()


### Fake datasets ###
restore_param_values(f=f_input)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="invalid value encountered in divide"
)

datasets_km3net[0].models.append(nu_model_PD)
for ds in tqdm(datasets_km3net, "fake datasets"):
    ds.fake(random_state=seed)

dataset_cta.models.append(gamma_model_PD)
dataset_cta.models.append(gamma_model_IC)
dataset_cta.fake(random_state=seed)


### Analysis ###
scan_values = np.linspace(0.0, 1.0, 21)

prior_on_f = dict(
    f=0,
    f_err=0.01,
    scale=prior_scale,
    power=2,
    e_int_min=e_int_min,
    e_int_max=e_int_max,
)

warnings.resetwarnings()

## Case1: CTA alone ##
# path1 = path / "Case1"
path1 = Path(outcome_path / "Case1")
# path1.mkdir(exist_ok=True)
makedirs(path1, exist_ok=True)

datasets1 = Datasets([dataset_cta])
fit1 = Fit_wp(datasets1)

# pre-fit PD model
with fit1._parameters.restore_values:
    restore_param_values(f=1.0, freeze=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # first run the fit with scipy
        fit1.optimize(backend="scipy")
        # scipy does not respect bounds - so check if beta is not too big
        beta0 = gamma_model_PD.spectral_model.beta.value
        if beta0 >= 2.0:
            # with open(path1 / "check_bounds_{}.dat".format(seed), "a") as f:
            with open(Path(path1 / f"check_bounds_{seed}.dat"), "a") as f:
                f.write('Parameter "beta" (PD) was set from {} to 1.99'.format(beta0))
            gamma_model_PD.spectral_model.beta.value = 1.99
        elif beta0 <= 0.1:
            # with open(path1 / "check_bounds_{}.dat".format(seed), "a") as f:
            with open(Path(path1 / f"check_bounds_{seed}.dat"), "a") as f:
                f.write('Parameter "beta" (PD) was set from {} to 0.11'.format(beta0))
            gamma_model_PD.spectral_model.beta.value = 0.11

        # run again with iminuit
        result1_PD = fit1.run()

    # write results
    with open(Path(path1 / f"parameters_{seed}.dat"), "a") as f:
        f.write("Best-fit parameters for CTA Fit:\n")
        print(result1_PD.parameters.free_parameters.to_table(), "\n\n\n", file=f)

    # compute integral of result spectrum, store parameter values
    int_had = gamma_model_PD.spectral_model.integral(e_int_min, e_int_max)
    par_PD0 = gamma_model_PD.spectral_model.parameters.copy()

    nuisance = dict(
        amp=result1_PD.parameters.free_parameters["amplitude"].value,
        amp_err=result1_PD.parameters.free_parameters["amplitude"].error,
        amp_unit=result1_PD.parameters.free_parameters["amplitude"].unit,
        alpha=result1_PD.parameters.free_parameters["alpha"].value,
        alpha_err=result1_PD.parameters.free_parameters["alpha"].error,
        beta=result1_PD.parameters.free_parameters["beta"].value,
        beta_err=result1_PD.parameters.free_parameters["beta"].error,
        ecut=result1_PD.parameters.free_parameters["e_cutoff"].value,
        ecut_err=result1_PD.parameters.free_parameters["e_cutoff"].error,
        ecut_unit=result1_PD.parameters.free_parameters["e_cutoff"].unit,
    )
    amp_cta = result1_PD.parameters.free_parameters["amplitude"].value
    amp_err_cta = result1_PD.parameters.free_parameters["amplitude"].error

# pre-fit IC model
with fit1._parameters.restore_values:
    restore_param_values(f=0.0, freeze=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # first run the fit with scipy
        fit1.optimize(backend="scipy")
        # scipy does not respect bounds - so check if beta is not too big
        beta1 = gamma_model_IC.spectral_model.beta.value
        if beta1 >= 2.0:
            with open(Path(path1 / f"check_bounds_{seed}.dat"), "a") as f:
                f.write('Parameter "beta" (IC) was set from {} to 1.99'.format(beta1))
            gamma_model_IC.spectral_model.beta.value = 1.99
        elif beta1 <= 0.1:
            with open(Path(path1 / f"check_bounds_{seed}.dat"), "a") as f:
                f.write('Parameter "beta" (IC) was set from {} to 0.11'.format(beta1))
            gamma_model_IC.spectral_model.beta.value = 0.11

        # run again with iminuit
        result1_IC = fit1.run()

    # write results
    with open(Path(path1 / f"parameters_{seed}.dat"), "a") as f:
        f.write("Best-fit parameters for Lep Fit:\n")
        print(result1_IC.parameters.free_parameters.to_table(), "\n\n\n", file=f)

    # Store parameter values
    par_IC0 = gamma_model_IC.spectral_model.parameters.copy()

# now perform the scan
stats = []
stats_no_prior = []
integral_PD = []
integral_IC = []
dataset_cta.nuisance = prior_on_f

for f_val in tqdm(scan_values, "scan values fit1"):
    with fit1._parameters.restore_values:
        restore_param_values(f=f_val, freeze=True, par_IC=par_IC0, par_PD=par_PD0)
        prior_on_f["f"] = f_val
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fit1.optimize()
            # with open(path1 / "parameters_{}.dat".format(seed), "a") as f:
            with open(Path(path1 / f"parameters_{seed}.dat"), "a") as f:
                f.write("Best-fit parameters for {}:\n".format(f_val))
                print(result.parameters.free_parameters.to_table(), "\n\n\n", file=f)

            stat = result.total_stat
            stat_no_prior = dataset_cta.stat_sum_no_prior()
            int_PD = gamma_model_PD.spectral_model.integral(e_int_min, e_int_max)
            int_IC = gamma_model_IC.spectral_model.integral(e_int_min, e_int_max)

        stats.append(stat)
        stats_no_prior.append(stat_no_prior)
        integral_PD.append(int_PD.to_value("cm-2 s-1"))
        integral_IC.append(int_IC.to_value("cm-2 s-1"))

res_scan1 = dict(
    values=scan_values,
    stat=np.array(stats),
    stat_no_prior=np.array(stats_no_prior),
    int_PD=np.array(integral_PD),
    int_IC=np.array(integral_IC),
)

# write the results
# np.save(path1 / "res_case1_{}R.npy".format(seed), res_scan1)
np.save(Path(path1 / f"res_case1_{seed}R.npy"), res_scan1)

## Case2: KM3NeT alone ##
path2 = Path(outcome_path / "Case2")
makedirs(path2, exist_ok=True)

datasets_km3net.nuisance = nuisance
fit2 = Fit_wp(datasets_km3net)

stats = []
stats_with_prior = []
integral_PD = []

for f_val in tqdm(scan_values, "scan values fit2"):
    with fit2._parameters.restore_values:
        restore_param_values(f=f_val, freeze=True, par_IC=par_IC0, par_PD=par_PD0)
        nuisance["amp"] = amp_cta * f_val
        nuisance["amp_err"] = amp_err_cta * f_val + 1e20  # Error cannot be 0
        nu_model_PD.spectral_model.amplitude.value = amp_cta * f_val
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result3 = fit2.optimize()
            with open(Path(path2 / f"parameters_{seed}.dat"), "a") as f:
                f.write("Best-fit parameters for {}:\n".format(f_val))
                print(result3.parameters.free_parameters.to_table(), "\n\n\n", file=f)
            stat_with_prior = result3.total_stat
            stat = datasets_km3net.stat_sum_no_prior()
            int_PD = gamma_model_PD.spectral_model.integral(e_int_min, e_int_max)

        stats.append(stat)
        stats_with_prior.append(stat_with_prior)
        integral_PD.append(int_PD.to_value("cm-2 s-1"))

res_scan2 = dict(
    values=scan_values,
    stat=np.array(stats),
    stat_with_prior=np.array(stats_with_prior),
    int_PD=np.array(integral_PD),
    int_had=int_had.to_value("cm-2 s-1"),
)

# is there a reason why wP is used here???

# write the results
np.save(Path(path2 / f"res_case2_{seed}R.npy"), res_scan2)

# remove prior from data set
datasets_km3net.nuisance = None


## Case 3: combined ##
path3 = Path(outcome_path / "Case3")
makedirs(path3, exist_ok=True)

datasets_km3net.extend([dataset_cta])
fit3 = Fit_wp(datasets_km3net)

stats = []
stats_nu = []
stats_gamma = []
stats_gamma_no_prior = []
integral_PD = []
integral_IC = []

for f_val in tqdm(scan_values, "scan values fit3"):
    with fit3._parameters.restore_values:
        restore_param_values(f=f_val, freeze=True, par_IC=par_IC0, par_PD=par_PD0)
        prior_on_f["f"] = f_val
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result3 = fit3.optimize()
            with open(Path(path3 / f"parameters_{seed}.dat"), "a") as f:
                f.write("Best-fit parameters for {}:\n".format(f_val))
                print(result3.parameters.free_parameters.to_table(), "\n\n\n", file=f)

            stat = result3.total_stat
            stat_nu = 0
            for z in range(len(fit3.datasets) - 1):
                stat_nu += fit3.datasets[z].stat_sum()
            stat_gamma = fit3.datasets[-1].stat_sum()
            stat_gamma_no_prior = fit3.datasets[-1].stat_sum_no_prior()
            int_PD = gamma_model_PD.spectral_model.integral(e_int_min, e_int_max)
            int_IC = gamma_model_IC.spectral_model.integral(e_int_min, e_int_max)

        stats.append(stat)
        stats_nu.append(stat_nu)
        stats_gamma.append(stat_gamma)
        stats_gamma_no_prior.append(stat_gamma_no_prior)
        integral_PD.append(int_PD.to_value("cm-2 s-1"))
        integral_IC.append(int_IC.to_value("cm-2 s-1"))

res_scan3 = dict(
    values=scan_values,
    stat=np.array(stats),
    stat_nu=np.array(stats_nu),
    stat_gamma=np.array(stats_gamma),
    stat_gamma_no_prior=np.array(stats_gamma_no_prior),
    int_PD=np.array(integral_PD),
    int_IC=np.array(integral_IC),
)

# write the results
np.save(Path(path3 / f"res_case3_{seed}R.npy"), res_scan3)

print(f"--- {round((time.time() - start_time),3)} s ---")
