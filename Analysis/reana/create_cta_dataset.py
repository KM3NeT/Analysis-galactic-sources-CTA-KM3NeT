import numpy as np
import time
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm as cm
from matplotlib import colors as colors

from astropy.coordinates import SkyCoord
from astropy import units as u

from gammapy.datasets import MapDataset
from gammapy.data import Observation
from gammapy.maps import WcsGeom, MapAxis
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.irf import load_cta_irfs
from gammapy.modeling.models import (
    NaimaSpectralModel,
    DiskSpatialModel,
    GaussianSpatialModel,
    SkyModel,
)

from naima.models import ExponentialCutoffPowerLaw
from regions import CircleSkyRegion
from os import makedirs


import sys
from pathlib import Path

start_time = time.time()

current_dir = Path(__file__).resolve().parent
par_dir = current_dir.parent
parpar_dir = par_dir.parent
sys.path.append(str(par_dir))
sys.path.append(str(parpar_dir))


from src import PionDecayKelner06, AnalysisConfig, SourceModel
import src.plot_utils as plot_utils


plot_utils.mpl_settings()

analysis_conf = AnalysisConfig()

irfs = load_cta_irfs(analysis_conf.get_file("cta/irfs/irf_file_new_CTA.fits"))
wobble_offset = 1 * u.deg
livetime = 200 * u.h
source_name = analysis_conf.get_source()

output_folder = "results"
output_path = Path(parpar_dir / output_folder)

source_pos_dist = pd.read_csv(
    Path(parpar_dir / "data" / "models" / "sources_catalog.csv"), index_col=0
)
src_pos = SkyCoord(
    *source_pos_dist.loc[source_name][["RA", "Dec"]], unit="deg", frame="icrs"
)

# Get 4 symmetric pointings each with 1deg offset from the source position
pointings = src_pos.directional_offset_by([0, 90, 180, 270] * u.deg, wobble_offset)

# Define map geometry for binned simulation
e_edges = np.logspace(-1, 2.1875, 52) * u.TeV  # bkg not properly defined above ~154 TeV
energy_reco_axis = MapAxis.from_edges(e_edges, unit="TeV", name="energy", interp="log")

geom = WcsGeom.create(
    skydir=src_pos,
    binsz=0.02,
    width=(6, 6),
    frame="icrs",
    axes=[energy_reco_axis],
)

# 16 bins/decade is enough also for the true energy axis
energy_true_axis = MapAxis.from_edges(
    e_edges, unit="TeV", name="energy_true", interp="log"
)

# Generate datasets
stacked_dataset = MapDataset.create(
    geom, name="CTA-dataset-{}".format(source_name), energy_axis_true=energy_true_axis
)

dataset_maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])
maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg)

# create 4 observations for the 4 pointing positions, each with 1/4 of the total live time
# (this is not realistic, but does not matter since observations are stacked in the next step)

count = 0
for pointing in tqdm(pointings, "Fill staked dataset"):
    obs = Observation.create(pointing=pointing, livetime=livetime / 4, irfs=irfs)
    with np.errstate(divide="ignore", invalid="ignore"):
        dataset = dataset_maker.run(stacked_dataset.copy(name="P{}".format(count)), obs)
    dataset = maker_safe_mask.run(dataset, obs)

    # in case the background model contains infinites
    assert np.isfinite(dataset.background_model.map.data[dataset.mask_safe.data]).all()
    dataset.background_model.map.data[~dataset.mask_safe.data] = 0.0

    # stack the datasets
    with np.errstate(divide="ignore", invalid="ignore"):
        stacked_dataset.stack(dataset)

    count += 1


# makedirs(output_path, exist_ok=True)
# The data set is not included in the repo because of its size but can be written to disk using:
# stacked_dataset.write(
#     path.join(parentdir, results, "CTA_{}_200h_p4.fits.gz".format(source_name)),
#     overwrite=True,
# )

# stacked_dataset.write(
#     Path(output_path / f"CTA_{source_name}_{int(livetime.value)}h_p4.fits.gz"),
#     overwrite=True,
# )


if analysis_conf.get_value("write_CTA_pseudodata", "io"):
    outfilename = analysis_conf.get_file(
        "cta/pseudodata/CTA_{}_{}{}_p4.fits.gz".format(
            source_name, int(livetime.value), livetime.unit
        )
    )
    stacked_dataset.write(outfilename, overwrite=True)

# plot
model_pars_PD = np.loadtxt(
    Path(
        parpar_dir
        / "data"
        / "models"
        / "modelfits"
        / f"input_model_PD_{source_name}.txt"
    )
)

ECPL_PD = ExponentialCutoffPowerLaw(
    amplitude=model_pars_PD[0] / u.eV,
    e_0=model_pars_PD[1] * u.TeV,
    alpha=model_pars_PD[2],
    e_cutoff=model_pars_PD[3] * u.TeV,
    beta=model_pars_PD[4],
)
gamma_ECPL_PD = PionDecayKelner06(ECPL_PD, particle_type="gamma")
spectral_model = NaimaSpectralModel(
    gamma_ECPL_PD, distance=source_pos_dist["Dist"][source_name] * u.kpc
)

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
model = SkyModel(
    spectral_model=spectral_model, spatial_model=spatial_model, name=source_name
)

npred_bkg = stacked_dataset.npred()
stacked_dataset.models.append(model)
npred_sum = stacked_dataset.npred()
npred_src = npred_sum - npred_bkg
stacked_dataset.fake(random_state=137)
counts = stacked_dataset.counts
max_counts = counts.sum_over_axes().data.max()

src_region = CircleSkyRegion(
    src_pos, (source_pos_dist["Radius"][source_name] + 0.5) * u.deg
)
src_reg_mask = counts.geom.to_image().region_mask([src_region])

fig_width = 8.8 / 2.54
fig_height = 2.75

cmap = cm.afmhot
norm = colors.Normalize(vmin=0, vmax=max_counts)
data = counts.sum_over_axes().data

fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
fig, ax, cbar = counts.sum_over_axes().plot(add_cbar=True)
im = ax.imshow(data, cmap=cmap, norm=norm)
ax.scatter(
    src_pos.ra.value,
    src_pos.dec.value,
    marker="o",
    fc="None",
    ec="tab:blue",
    transform=ax.get_transform("world"),
)
ax.scatter(
    pointings.ra.value,
    pointings.dec.value,
    marker="x",
    color="tab:blue",
    transform=ax.get_transform("world"),
    zorder=4,
)
src_reg_pix = src_region.to_pixel(ax.wcs)
src_reg_pix.plot(ax=ax, ec="w", ls="--")

cbar.update_normal(cm.ScalarMappable(norm=norm, cmap=cmap))
cbar.set_label("Counts per pixel")
makedirs(Path(output_path / "plots"), exist_ok=True)
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.97)


def save_fig(fig, file_name):
    for form in ["png", "pdf"]:
        fig.savefig(Path(output_path / "plots" / str(file_name + f".{form}")))


save_fig(fig, f"map_CTA_counts_PD_{source_name}")


fig_width = 8.8 / 2.54
fig_height = 2.75

data = npred_sum.sum_over_axes().data

fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
fig, ax, cbar = npred_sum.sum_over_axes().plot(fig=fig, add_cbar=True)
im = ax.imshow(data, cmap=cmap, norm=norm)
ax.scatter(
    src_pos.ra.value,
    src_pos.dec.value,
    marker="o",
    fc="None",
    ec="tab:blue",
    transform=ax.get_transform("world"),
)
ax.scatter(
    pointings.ra.value,
    pointings.dec.value,
    marker="x",
    color="tab:blue",
    transform=ax.get_transform("world"),
    zorder=4,
)
src_reg_pix = src_region.to_pixel(ax.wcs)
src_reg_pix.plot(ax=ax, ec="w", ls="--")

cbar.update_normal(cm.ScalarMappable(norm=norm, cmap=cmap))
cbar.set_label("Counts per pixel")

plt.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.97)

save_fig(fig, f"map_CTA_npred_PD_{source_name}")

fig_width = 8.8 / 2.54
ax_width = 0.83
fig_height = 2.2
ax_height = fig_width * ax_width / 1.618 / fig_height  # golden ratio

fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
ax = fig.add_axes([0.16, 0.18, ax_width, ax_height])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$E\,[\mathrm{TeV}]$")
ax.set_ylabel("Counts in source region")

ax.plot(
    energy_reco_axis.center.value,
    npred_bkg.data[:, src_reg_mask].sum(axis=1),
    color="tab:blue",
    label="Background",
)
ax.plot(
    energy_reco_axis.center.value,
    npred_src.data[:, src_reg_mask].sum(axis=1),
    color="tab:red",
    label="Signal",
)
ax.plot(
    energy_reco_axis.center.value,
    npred_sum.data[:, src_reg_mask].sum(axis=1),
    color="tab:green",
    label="Sum",
)
ax.errorbar(
    energy_reco_axis.center.value,
    counts.data[:, src_reg_mask].sum(axis=1),
    xerr=None,
    yerr=np.sqrt(counts.data[:, src_reg_mask].sum(axis=1)),
    linestyle="None",
    marker="o",
    markersize=2,
    color="k",
    label="Fake data",
    zorder=8,
)

ax.text(
    0.97,
    0.95,
    plot_utils.source_name_labels[source_name],
    ha="right",
    va="top",
    transform=ax.transAxes,
)
ax.text(0.97, 0.87, "CTA", ha="right", va="top", transform=ax.transAxes)

plot_utils.format_log_axis(ax.xaxis)
plot_utils.format_log_axis(ax.yaxis)

ax.legend(loc="lower left")

save_fig(fig, f"counts_reg_CTA_PD_{source_name}")

print(f"--- {round((time.time() - start_time),3)} s ---")
