import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


from scipy.integrate import quad
from scipy.optimize import curve_fit, newton
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

output_folder = "results"
output_path = Path(parpar_dir / output_folder)

from src import AnalysisConfig
import src.plot_utils as plot_utils
from src.fill_table import *


analysis_conf = AnalysisConfig()

plot_utils.mpl_settings()

data_path = analysis_conf.get_file("likelihood_analysis")

source_names = ["RXJ1713", "VelaX", "HESSJ1908", "Westerlund1"]

# option for model sets in analysis_config.yml
# leptonic = 0.0
# hadronic = 1.0

model = analysis_conf.get_value(value="dts_result")[0]


def process_data(data_path, source_name, model):
    file_path = os.path.join(data_path, f"{source_name}_{model}.csv")
    df = pd.read_csv(file_path)
    gdf = df.groupby(["seed", "case"])

    stat_tot = gdf["stat_total"].apply(lambda v: np.array(v)).unstack()
    TS1 = np.vstack(stat_tot[1])
    TS2 = np.vstack(stat_tot[2])
    TS3 = np.vstack(stat_tot[3])

    int_PD = gdf["int_PD"].apply(lambda v: np.array(v)).unstack()
    int_PD1 = np.vstack(int_PD[1])
    int_PD2 = np.vstack(int_PD[2])
    int_PD3 = np.vstack(int_PD[3])

    int_IC = gdf["int_IC"].apply(lambda v: np.array(v)).unstack()
    int_IC1 = np.vstack(int_IC[1])
    # int_IC2 = np.vstack(int_IC[2])
    int_IC3 = np.vstack(int_IC[3])

    TS_nu = gdf["stat_nu"].apply(lambda v: np.array(v)).unstack()
    TS3_nu = np.vstack(TS_nu[3])

    TS_gamma = gdf["stat_gamma"].apply(lambda v: np.array(v)).unstack()
    TS3_gamma = np.vstack(TS_gamma[3])

    TS1_avg = np.mean(TS1, axis=0)
    int_PD1_avg = np.mean(int_PD1, axis=0)
    int_IC1_avg = np.mean(int_IC1, axis=0)
    f1_avg = int_PD1_avg / (int_PD1_avg + int_IC1_avg)

    TS2_avg = np.mean(TS2, axis=0)
    f2_avg = np.mean(int_PD2, axis=0)

    TS3_avg = np.mean(TS3, axis=0)
    TS3_nu_avg = np.mean(TS3_nu, axis=0)
    TS3_gamma_avg = np.mean(TS3_gamma, axis=0)

    int_PD3_avg = np.mean(int_PD3, axis=0)
    int_IC3_avg = np.mean(int_IC3, axis=0)
    f3_avg = int_PD3_avg / (int_PD3_avg + int_IC3_avg)

    return (
        TS1,
        TS2,
        TS3,
        TS1_avg,
        TS2_avg,
        TS3_avg,
        TS3_avg,
        TS3_nu_avg,
        TS3_gamma_avg,
        f1_avg,
        f2_avg,
        f3_avg,
    )


ylim_dct = {
    source_names[0]: (-1.555, 17.866),
    source_names[1]: (-4.864, 84.895),
    source_names[2]: (-1.777, 17.615),
    source_names[3]: (-2.434, 30.003),
}


fig_width = 8.8 / 2.54
ax_width = 0.86
fig_height = 2.6
ax_height = fig_width * ax_width / 1.5 / fig_height

fig, axes = plt.subplots(
    nrows=2, ncols=2, figsize=(2.5 * fig_width, 2.5 * fig_height), dpi=200
)
plt.subplots_adjust(hspace=0.4)


for ax, source_name_it in zip(axes.flat, source_names):
    (
        TS1,
        TS2,
        TS3,
        TS1_avg,
        TS2_avg,
        TS3_avg,
        TS3_avg,
        TS3_nu_avg,
        TS3_gamma_avg,
        f1_avg,
        f2_avg,
        f3_avg,
    ) = process_data(data_path=data_path, source_name=source_name_it, model=model)

    TS3_quantiles95 = np.quantile(TS3, [0.05, 0.95], axis=0)
    TS3_quantiles68 = np.quantile(TS3, [1 - 0.68, 0.68], axis=0)

    TS12_quantiles95 = np.quantile(np.array(TS1) + TS2, [0.05, 0.95], axis=0)
    TS12_quantiles68 = np.quantile(np.array(TS1) + TS2, [1 - 0.68, 0.68], axis=0)
    ax.set_title(plot_utils.source_name_labels[source_name_it])
    ax.set_xlabel("$f\,=\,I_\mathrm{had}\,/\,(I_\mathrm{had} + I_\mathrm{lep})$")
    ax.set_ylabel("$\Delta \mathrm{TS}$")

    idx_norm = TS3_avg.argmin()

    ax.plot(
        f1_avg,
        TS1_avg - TS1_avg[idx_norm],
        "x",
        c="tab:red",
        label="CTA only",
        zorder=4,
    )
    ax.plot(
        f2_avg,
        TS2_avg - TS2_avg[idx_norm],
        "x",
        c="tab:blue",
        label="KM3NeT only",
        zorder=4,
    )
    ax.plot(
        f3_avg,
        TS3_avg - TS3_avg[idx_norm],
        "-",
        c="tab:green",
        label="Combined",
        zorder=5,
    )
    ax.plot(
        f3_avg,
        TS3_gamma_avg - TS3_gamma_avg[idx_norm],
        "--",
        c="tab:red",
        label="Combined (CTA)",
        zorder=3,
    )
    ax.plot(
        f3_avg,
        TS3_nu_avg - TS3_nu_avg[idx_norm],
        "--",
        c="tab:blue",
        label="Combined (KM3NeT)",
        zorder=3,
    )

    ax.fill_between(
        f3_avg,
        TS3_quantiles95[0, :] - TS3_avg[idx_norm],
        TS3_quantiles95[1, :] - TS3_avg[idx_norm],
        color="tab:green",
        linewidth=0.0,
        alpha=0.3,
        zorder=1,
    )
    ax.fill_between(
        f3_avg,
        TS3_quantiles68[0, :] - TS3_avg[idx_norm],
        TS3_quantiles68[1, :] - TS3_avg[idx_norm],
        color="tab:green",
        linewidth=0.0,
        alpha=0.3,
        zorder=1,
    )

    ax.hlines(0, -0.05, 1.05, color="0.7", ls="--", zorder=2)
    ax.axvline(model, ymax=0.27, color="k", ls="--", zorder=2)
    trans = plt.matplotlib.transforms.blended_transform_factory(
        ax.transData, ax.transAxes
    )
    ha_dict = {0.0: "left", 0.5: "center", 1.0: "right"}
    ax.text(
        model,
        0.28,
        "$f_\mathrm{{in}} = {:g}$".format(model),
        ha=ha_dict[model],
        va="bottom",
        color="k",
        transform=trans,
    )
    y1, y2 = ax.get_ylim()

    ax.set_xlim(-0.05, 1.05)
    if source_name_it in ylim_dct.keys():
        ax.set_ylim(ylim_dct[source_name_it])

    leg_loc = {0.0: "upper left", 0.5: "upper center", 1.0: "upper right"}
    ax.legend(loc=leg_loc.get(model, "upper left"))

os.makedirs(Path(output_path / "plots"), exist_ok=True)


def save_fig(fig, file_name):
    for form in ["png", "pdf"]:
        fig.savefig(Path(output_path / "plots" / str(file_name + f".{form}")))


save_fig(fig, f"all_sources_scan_{str(model)}")

print(f"--- {round((time.time() - start_time),3)} s ---")
