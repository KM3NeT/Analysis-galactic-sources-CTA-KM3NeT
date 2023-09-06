import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from astropy.table import Table
from scipy.integrate import quad
from scipy.optimize import curve_fit, newton, root
from scipy.interpolate import interp1d

from glob import glob
from pathlib import Path
import os
import sys

start_time = time.time()

current_dir = Path(__file__).resolve().parent
par_dir = current_dir.parent
parpar_dir = par_dir.parent
sys.path.append(str(par_dir))
sys.path.append(str(parpar_dir))
sys.path.append("../src")

output_folder = "results"
output_path = Path(parpar_dir / output_folder)

from src import AnalysisConfig
import src.plot_utils as plot_utils
from src.fill_table import *

analysis_conf = AnalysisConfig()

plot_utils.mpl_settings()

for i in [0.68, 0.90]:
    print(f"q = {i}")
    fill_table(["HESSJ1908", "RXJ1713", "VelaX", "Westerlund1"], q=i)

result_q68 = pd.read_csv(
    analysis_conf.get_file("likelihood_analysis/results_q68.csv"), index_col=0
)
result_q90 = pd.read_csv(
    analysis_conf.get_file("likelihood_analysis/results_q90.csv"), index_col=0
)


pos_dict = {"VelaX": 3, "RXJ1713": 2, "HESSJ1908": 0, "Westerlund1": 1}
ylabels = [r"Vela X", r"RX J1713.7$-$3946", r"eHWC J1907+063", r"Westerlund 1"]

plt.rcParams.update({"font.family": "serif", "font.size": 7.0})

fig_width = 8.8 / 2.54
ax_width = 0.86
fig_height = 2.6
ax_height = fig_width * ax_width / 1.5 / fig_height

fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
ax.set_position([0.13, 0.15, ax_width, ax_height])

for key in result_q68.index:
    fin, favg, fn68, fp68, _, _, _, fq68 = result_q68.loc[key]
    _, _, fn90, fp90, _, _, _, fq90 = result_q90.loc[key]

    sname = key.split("_")[0]
    if fin == 0:
        color = "blue"
        offset = 0.07
        left = 0
        fq68b = fq68
        fq90b = fq90 - fq68b
    elif fin == 1:
        color = "red"
        offset = -0.07
        left = 1
        fq68b = fq68 - 1
        fq90b = fq90 - 1 - fq68b
    else:
        color = "k"
    ax.errorbar(
        favg,
        pos_dict[sname] + offset,
        marker="*",
        markerfacecolor="white",
        xerr=[[fn68], [fp68]],
        capsize=4,
        color="k",
        alpha=1,
    )
    ax.barh(pos_dict[sname] + offset, fq68b, 0.4, alpha=0.5, left=left, color=color)
    ax.barh(pos_dict[sname] + offset, fq90b, 0.2, alpha=0.3, left=fq68, color=color)
    q_size = 0.2
    ax.vlines(
        fq68,
        pos_dict[sname] + offset - q_size,
        pos_dict[sname] + offset + q_size,
        color=color,
        alpha=1,
    )
    q_size = 0.1
    ax.vlines(
        fq90,
        pos_dict[sname] + offset - q_size,
        pos_dict[sname] + offset + q_size,
        color=color,
        alpha=1,
    )

ax.grid()
ax.set_xlabel(r"$f\,=\,I_\mathrm{had}\,/\,(I_\mathrm{had} + I_\mathrm{lep})$")
ax.set_yticks(list(pos_dict.values()))
ax.set_yticklabels(ylabels, rotation=25)
ax.text(0, 1.03, "f$_{in}$ = 0", transform=ax.transAxes, color="blue", fontsize=8)
ax.text(0.90, 1.03, "f$_{in}$ = 1", transform=ax.transAxes, color="red", fontsize=8)
ax.axvline(0, color="blue")
ax.axvline(1, color="red")


os.makedirs(Path(output_path / "plots"), exist_ok=True)


def save_fig(fig, file_name):
    for form in ["png", "pdf"]:
        fig.savefig(
            Path(output_path / "plots" / str(file_name + f".{form}")),
            bbox_inches="tight",
        )


save_fig(fig, "limit_avg_comb")

print(f"--- {round((time.time() - start_time),3)} s ---")
