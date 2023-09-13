import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time
from tqdm import tqdm


from astropy.table import Table
import astropy.units as u

from gammapy.estimators import FluxPoints
from gammapy.modeling.models import NaimaSpectralModel
from naima.models import ExponentialCutoffPowerLaw, InverseCompton

from scipy.optimize import curve_fit

import sys
from pathlib import Path
from os import makedirs

start_time = time.time()

current_dir = Path(__file__).resolve().parent
par_dir = current_dir.parent
parpar_dir = par_dir.parent
sys.path.append(str(par_dir))
sys.path.append(str(parpar_dir))

from src.flux_utils import PionDecayKelner06
from src import plot_utils, AnalysisConfig, SourceModel

plt.rcdefaults()
plot_utils.mpl_settings()
analysis_conf = AnalysisConfig()

source_name = analysis_conf.get_source()
model = SourceModel(sourcename=source_name)
source_tuple = model.get_sourcetuple

# path where to store plots
output_path = Path(parpar_dir / "results" / "plots")
makedirs(output_path, exist_ok=True)

path_FP = Path(parpar_dir / "data" / "models" / f"{source_tuple[0]}.csv")
FP_df = pd.read_csv(path_FP)

# create astropy.table
FP_table = Table()
FP_table["e_ref"] = FP_df["e_ref"].values
FP_table["dnde"] = FP_df["dnde"].values
FP_table["dnde_err"] = FP_df["dnde_err"].values
FP_table["e_ref"].unit = "TeV"
FP_table["dnde"].unit = "TeV-1 cm-2 s-1"
FP_table["dnde_err"].unit = "TeV-1 cm-2 s-1"
FP_table.meta["SED_TYPE"] = "dnde"

FP = FluxPoints(FP_table)

# get source distance
source_dist = (
    pd.read_csv(
        Path(parpar_dir / "data" / "models" / "sources_catalog.csv"), index_col=0
    )["Dist"][source_tuple[1]]
    * u.kpc
)

# define a proton spectrum
ECPL_PD = ExponentialCutoffPowerLaw(
    amplitude=source_tuple[4][0] / u.eV,
    e_0=source_tuple[4][1] * u.TeV,
    alpha=source_tuple[4][2],
    e_cutoff=source_tuple[4][3] * u.TeV,
    beta=source_tuple[4][4],
)

gamma_ECPL_PD = NaimaSpectralModel(
    PionDecayKelner06(ECPL_PD, particle_type="gamma"), distance=source_dist
)

# define an elctron spectrum
ECPL_IC = ExponentialCutoffPowerLaw(
    amplitude=source_tuple[5][0] / u.eV,
    e_0=source_tuple[5][1] * u.TeV,
    alpha=source_tuple[5][2],
    e_cutoff=source_tuple[5][3] * u.TeV,
    beta=source_tuple[5][4],
)

gamma_ECPL_IC = NaimaSpectralModel(
    InverseCompton(ECPL_IC, seed_photon_fields="CMB"), distance=source_dist
)


# functions for predicted flux
def PD_pred(x, amp, alpha, e_cut):
    gamma_ECPL_PD.parameters["amplitude"].value = amp
    gamma_ECPL_PD.parameters["alpha"].value = alpha
    gamma_ECPL_PD.parameters["e_cutoff"].value = e_cut
    return gamma_ECPL_PD(FP.e_ref).value


def IC_pred(x, amp, alpha, e_cut):
    gamma_ECPL_IC.parameters["amplitude"].value = amp
    gamma_ECPL_IC.parameters["alpha"].value = alpha
    gamma_ECPL_IC.parameters["e_cutoff"].value = e_cut
    return gamma_ECPL_IC(FP.e_ref).value


# fit the proton model to the flux points
popt_ECPL_PD, cov_ECPL_PD = curve_fit(
    PD_pred,  # function that is fitted
    [1, 2, 3],  # arbitrary
    FP_df["dnde"],  # flux point values
    sigma=FP_df["dnde_err"],  # flux point errors
    p0=source_tuple[6],  # initial guess for parameters
)

chi2_PD = plot_utils.chi2(
    gamma_ECPL_PD,
    FP.e_ref,
    u.Quantity(FP_table["dnde"]),
    u.Quantity(FP_table["dnde_err"]),
) / (
    len(FP.table) - 3
)  # chi^2 / ndof

print("fit parameters: amplitude, index, cutoff energy")
print(popt_ECPL_PD)
print("their uncertainties")
print(np.sqrt(np.diag(cov_ECPL_PD)))
print(f"chi^2 / ndof = {chi2_PD.to('')}")

# fit the electron model to the flux points
popt_ECPL_IC, cov_ECPL_IC = curve_fit(
    IC_pred,  # function that is fitted
    [1, 2, 3],  # arbitrary
    FP_df["dnde"],  # flux point values
    sigma=FP_df["dnde_err"],  # flux point errors
    p0=source_tuple[7],  # initial guess for parameters
)

chi2_IC = plot_utils.chi2(
    gamma_ECPL_IC,
    FP.e_ref,
    u.Quantity(FP_table["dnde"]),
    u.Quantity(FP_table["dnde_err"]),
) / (
    len(FP.table) - 3
)  # chi^2 / ndof

print("fit parameters: amplitude, index, cutoff energy")
print(popt_ECPL_IC)
print("their uncertainties")
print(np.sqrt(np.diag(cov_ECPL_IC)))
print(f"chi^2 / ndof = {chi2_IC.to('')}")


# save output of fit
np.savetxt(
    Path(
        parpar_dir
        / "data"
        / "models"
        / "modelfits"
        / f"input_model_PD_{source_tuple[1]}.txt"
    ),
    gamma_ECPL_PD.parameters.values,
)

np.savetxt(
    Path(
        parpar_dir
        / "data"
        / "models"
        / "modelfits"
        / f"input_model_IC_{source_tuple[1]}.txt"
    ),
    gamma_ECPL_IC.parameters.values,
)

# produce plots and sotre them
fig_width = 8.8 / 2.54
ax_width = 0.8
fig_height = 2.2
ax_height = fig_width * ax_width / 1.618 / fig_height  # golden ratio

fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
ax = fig.add_axes([0.175, 0.18, ax_width, ax_height])
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="gammapy")
    FP.plot(energy_power=2, marker=".", markersize=5, label=source_tuple[3], ax=ax)
    gamma_ECPL_IC.plot(
        source_tuple[8] * u.TeV,
        energy_power=2,
        label=r"IC fit ($\chi^2/N_\mathrm{{dof}}={:.3f}$)".format(chi2_IC.value),
        ax=ax,
    )
    gamma_ECPL_PD.plot(
        source_tuple[8] * u.TeV,
        energy_power=2,
        label=r"PD fit ($\chi^2/N_\mathrm{{dof}}={:.3f}$)".format(chi2_PD.value),
        ax=ax,
    )

ax.set_xlabel("$E\,[\mathrm{TeV}]$")
ax.set_ylabel(
    r"$E^2\times\,\mathrm{d}N\,/\,\mathrm{d}E\,\,[\mathrm{TeV}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}]$"
)
ax.set_ylim(1e-13, 5e-11)
ax.text(0.97, 0.95, source_tuple[2], ha="right", va="top", transform=ax.transAxes)
ax.legend(loc="lower left")

plot_utils.format_log_axis(ax.xaxis)

for form in ["png", "pdf"]:
    fig.savefig(Path(output_path / f"fit_{source_tuple[0]}.{form}"), dpi=300)

print(f"--- {round((time.time() - start_time),3)} s ---")
