import numpy as np
import time
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from matplotlib import colors as colors

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time

from gammapy.data import Observation
from gammapy.datasets import MapDataset, Datasets
from gammapy.maps import WcsGeom, MapAxis, WcsNDMap
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.irf import PSF3D, EnergyDispersion2D, Background2D, EffectiveAreaTable2D
from gammapy.modeling.models import (
    BackgroundModel,
    NaimaSpectralModel,
    DiskSpatialModel,
    GaussianSpatialModel,
    SkyModel,
)

from naima.models import ExponentialCutoffPowerLaw

from regions import CircleSkyRegion
from pathlib import Path

# from os import path
from os import makedirs
import sys

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


start_time = time.time()

current_dir = Path(__file__).resolve().parent
par_dir = current_dir.parent
parpar_dir = par_dir.parent
sys.path.append(str(par_dir))
sys.path.append(str(parpar_dir))

from src import PionDecayKelner06

# from src import read_analysis_config, get_source
import src.plot_utils as plot_utils

output_folder = "results"
output_path = Path(parpar_dir / output_folder)

plot_utils.mpl_settings()

# Westerlund1, HESSJ1908, RXJ1713, VelaX
source_name = "VelaX"

irf_path = Path(parpar_dir / "data" / "km3net")

years = 10  # exposition time


source_pos_dist = pd.read_csv(
    Path(parpar_dir / "data" / "models" / "sources_catalog.csv"), index_col=0
)
src_pos = SkyCoord(
    *source_pos_dist.loc[source_name][["RA", "Dec"]], unit="deg", frame="icrs"
)

# Visibility
pos_arca = EarthLocation.from_geodetic(
    lat="36° 16'", lon="16° 06'", height=-3500
)  # ARCA pos

# Set a time grid with hourly binning over one year
hours = 1
time_step = hours * 3600  # in seconds hourly
start = Time("2019-01-01T00:00:00", format="isot").unix
end = Time("2020-01-01T00:00:00", format="isot").unix
times = np.linspace(start, end - time_step, int(365 * 24 * 3600 / time_step))
obstimes = Time(times, format="unix")

frames = AltAz(obstime=obstimes, location=pos_arca)

# transform source position into local coordinates for each time step
local_coords = src_pos.transform_to(frames)

zen_angles = local_coords.zen.value
print("zen_min =", zen_angles.min())
print("zen_max =", zen_angles.max())

# zenith angle binning
cos_zen_bins = np.linspace(-1, 1, 13)
# cos_zen_bins = np.linspace(-1, 1, 25) # test with finer binning
cos_zen_binc = 0.5 * (cos_zen_bins[:-1] + cos_zen_bins[1:])
zen_bins = np.arccos(cos_zen_bins) * 180 / np.pi
zen_binc = np.arccos(cos_zen_binc) * 180 / np.pi

# compute visibility time for each zenith angle bin
# vis_hist = np.histogram(np.cos(zen_angles * np.pi / 180), bins=cos_zen_bins)
# vis_times = vis_hist[0] / vis_hist[0].sum() * u.yr

vis_hist, _ = np.histogram(np.cos(zen_angles * np.pi / 180), bins=cos_zen_bins)
vis_times = (vis_hist / vis_hist.sum()) * u.yr


# determine which zenith angle bins are going to be used
# bin_mask = (zen_binc > 80) & (vis_hist[0] > 0)
bin_mask = (zen_binc > 80) & (vis_hist > 0)
# how many datasets will be used
n_datasets = bin_mask.sum()

# compute dataset indices for all time bins
dataset_idx = np.digitize(zen_angles, bins=zen_bins)
# for what it was done?!
dataset_idx -= dataset_idx.min()

# compute masks for each zenith angle bin
# zen_masks = []
# for i in range(n_datasets):
#     z1 = zen_bins[1:][bin_mask][i]
#     z2 = zen_bins[:-1][bin_mask][i]
#     zen_masks.append((zen_angles > z1) & (zen_angles < z2))


zen_masks = [
    (zen_angles > z1) & (zen_angles < z2)
    for z1, z2 in zip(zen_bins[1:][bin_mask], zen_bins[:-1][bin_mask])
]


# compute average zenith angle for each zenith angle bin
zen_mean = [zen_angles[m].mean() for m in zen_masks]
# for zen in zen_mean:
#     print(zen)

# CREATE DATASETS

# Set up geometries

# set the true and reconstructed energy axes
print("# set the true and reconstructed energy axes")
energy_axis = MapAxis.from_bounds(
    1e2, 1e6, nbin=16, unit="GeV", name="energy", interp="log"
)
# energy_axis = MapAxis.from_bounds(1e2, 1e6, nbin=32, unit='GeV', name='energy', interp='log') # test with finer binning
energy_axis_true = MapAxis.from_bounds(
    1e2, 1e7, nbin=80, unit="GeV", name="energy_true", interp="log"
)

# also set the energy migration axis for the energy dispersion and the theta axis for the PSF
migra_axis = MapAxis.from_edges(np.logspace(-5, 2, 57), name="migra")
theta_axis = MapAxis.from_edges(np.linspace(0, 8, 101), name="theta", unit="deg")

# create the geometries
geom = WcsGeom.create(
    binsz=0.1 * u.deg,
    # binsz  = 0.05*u.deg, # test with finer binning
    width=30 * u.deg,
    # width  = 15*u.deg, # test with finer binning
    skydir=src_pos,
    frame="icrs",
    axes=[energy_axis],
)
geom_true = WcsGeom.create(
    binsz=0.1 * u.deg,
    # binsz  = 0.05*u.deg, # test with finer binning
    width=30 * u.deg,
    # width  = 15*u.deg, # test with finer binning
    skydir=src_pos,
    frame="icrs",
    axes=[energy_axis_true],
)


livetime_pointings = vis_times[bin_mask] * years
pointings = src_pos.directional_offset_by(0 * u.deg, np.array(zen_mean) * u.deg)

# transform all map coordinates into local coordinate system
# this takes a lot of RAM memory
map_coord = geom_true.to_image().get_coord()
sky_coord = map_coord.skycoord

# create an empty ndarray
map_coord_zeniths = np.ndarray(
    shape=(len(obstimes), geom_true._shape[0], geom_true._shape[1]), dtype=np.float16
)

for i in tqdm(range(len(obstimes)), "Map coordinates"):
    map_coord_zeniths[i] = sky_coord.transform_to(frames[i]).zen.value


# Load KM3NeT IRFs files
aeff = EffectiveAreaTable2D.read(Path(irf_path / "aeff.fits"))
edisp = EnergyDispersion2D.read(Path(irf_path / "edisp.fits"))
psf = PSF3D.read(Path(irf_path / "psf.fits"))


irfs = dict(aeff=aeff, psf=psf, edisp=edisp)

# Also the neutrino / muon background
bkg_nu = Background2D.read(Path(irf_path / "bkg_nu.fits"))
bkg_mu = Background2D.read(Path(irf_path / "bkg_mu.fits"))

# Create observations
# obs_list = [
#     Observation.create(pointing=pointings[i], livetime=livetime_pointings[i], irfs=irfs)
#     for i in tqdm(range(n_datasets), "Create obs list")
# ]

# # Make the MapDatasets
# empty = MapDataset.create(
#     geom=geom,
#     energy_axis_true=energy_axis_true,
#     migra_axis=migra_axis,
#     rad_axis=theta_axis,
#     binsz_irf=1,
# )

# dataset_maker = MapDatasetMaker(selection=["exposure", "psf", "edisp"])

# datasets = []
# for i, obs in enumerate(obs_list):
#     dataset = dataset_maker.run(empty.copy(name="nu{}".format(i + 1)), obs)
#     datasets.append(dataset)


# Create observations and map datasets
# datasets = []

# for i in tqdm(range(n_datasets), "Create datasets"):
#     # Create observation
#     obs = Observation.create(
#         pointing=pointings[i], livetime=livetime_pointings[i], irfs=irfs
#     )

#     # Create empty map dataset
#     empty = MapDataset.create(
#         geom=geom,
#         energy_axis_true=energy_axis_true,
#         migra_axis=migra_axis,
#         rad_axis=theta_axis,
#         binsz_irf=1,
#     )

#     # Make the map dataset
#     dataset_maker = MapDatasetMaker(selection=["exposure", "psf", "edisp"])
#     dataset = dataset_maker.run(empty.copy(name="nu{}".format(i + 1)), obs)

#     datasets.append(dataset)


# Create observations and map datasets
datasets = []

# Create empty map dataset
empty = MapDataset.create(
    geom=geom,
    energy_axis_true=energy_axis_true,
    migra_axis=migra_axis,
    rad_axis=theta_axis,
    binsz_irf=1,
)

for i in tqdm(range(n_datasets), "Create datasets"):
    # Create observation
    obs = Observation.create(
        pointing=pointings[i], livetime=livetime_pointings[i], irfs=irfs
    )

    # Make the map dataset
    dataset_maker = MapDatasetMaker(selection=["exposure", "psf", "edisp"])
    dataset = dataset_maker.run(empty.copy(name="nu{}".format(i + 1)), obs)

    datasets.append(dataset)


# of the considered field-of-view
edge_coord_pixel_pos = [
    (299, 0),
    (299, 150),
    (299, 299),
    (150, 0),
    (150, 150),
    (150, 299),
    (0, 0),
    (0, 150),
    (0, 299),
]


makedirs(output_path, exist_ok=True)
bkg_plot_dir = Path(output_path / "plots" / "km3net_bkg" / source_name)
makedirs(bkg_plot_dir, exist_ok=True)


def save_fig(fig, file_name):
    for form in ["png", "pdf"]:
        fig.savefig(Path(output_path / "plots" / str(file_name + f".{form}")))


# Plot zenith angle distribution for all edge coordinates defined above, for each zenith angle bin
# for i in range(n_datasets):
#     z1 = zen_bins[1:][bin_mask][i]
#     z2 = zen_bins[:-1][bin_mask][i]

#     fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True)
#     for j, pp in enumerate(edge_coord_pixel_pos):
#         ax = axes[j // 3][j % 3]
#         zen_vals = map_coord_zeniths[:, pp[0], pp[1]][zen_masks[i]]
#         ax.hist(zen_vals, bins=np.linspace(60, 180, 61), histtype="step")
#         ylim = ax.get_ylim()
#         ax.vlines(zen_bins[7], 0, ylim[1], color="tab:green", ls="--")
#         ax.vlines(zen_vals.mean(), 0, ylim[1], color="tab:blue", ls="--")
#         ax.fill_between(
#             [z1, z2], [0, 0], [ylim[1], ylim[1]], color="k", alpha=0.2, zorder=-4
#         )
#         ax.set_ylim(0, ylim[1])
#         ax.set_title(
#             "R.A.: ${:.1f}^\circ$, Dec.: ${:.1f}^\circ$".format(
#                 sky_coord[pp[0], pp[1]].ra.value, sky_coord[pp[0], pp[1]].dec.value
#             )
#         )
#         ax.grid(ls="--")
#         if j // 3 == 2:
#             ax.set_xlabel(r"$\theta\,[\mathrm{deg}]$")

#     fig.text(
#         0.5,
#         0.99,
#         r"Zenith angle distribution for $\theta\in [{:.1f},{:.1f}]$".format(z1, z2),
#         ha="center",
#         va="top",
#         fontsize="x-large",
#         transform=fig.transFigure,
#     )

#     plt.subplots_adjust(top=0.94)

#     # plt.show()
#     for form in ["png", "pdf"]:
#         fig.savefig(
#             "{}/KM3NeT_zenith_angle_dist_fov_edges_bin_{:02d}.{}".format(
#                 bkg_plot_dir, i, form
#             )
#         )


def bkg_2d_eval_patched(self, fov_lon, fov_lat, energy_reco, method="linear", **kwargs):
    return self.data.evaluate(
        offset=fov_lon, energy=energy_reco, method="linear", **kwargs
    )


Background2D.evaluate = bkg_2d_eval_patched

# Similarly, we need a patched method to evaluate the exposure correctly


# First, define a small helper function
def calc_exposure(offset, aeff, geom):
    energy = geom.get_axis_by_name("energy_true").center
    exposure = aeff.data.evaluate(
        offset=offset, energy_true=energy[:, np.newaxis, np.newaxis]
    )
    return exposure


# This method does the same as Gammapy's `make_map_exposure_true_energy`,
# except that it takes an offset angle (or, in our case, the zenith angle)
# instead of a pointing position.
# def make_map_exposure_true_energy_patched(offset, livetime, aeff, geom):
#     exposure = calc_exposure(offset, aeff, geom)
#     exposure = (exposure * livetime).to("m2 s")
#     return WcsNDMap(geom, exposure.value.reshape(geom.data_shape), unit=exposure.unit)


def make_map_exposure_true_energy_patched(offsets, livetime, aeff, geom):
    energy = geom.get_axis_by_name("energy_true").center
    offsets = np.asarray(offsets)
    exposure = calc_exposure(offsets, aeff, geom)
    exposure = (exposure * livetime).to("m2 s")
    exposure_map = WcsNDMap(
        geom, exposure.value[..., np.newaxis, np.newaxis], unit=exposure.unit
    )
    exposure_map = exposure_map.resample_axis(energy, "energy_true")
    return exposure_map


d_omega = geom_true.to_image().solid_angle()

# Compute the predicted background and the exposure for all edge pixels and all time bins
nu_background_edge_pixels = []
mu_background_edge_pixels = []
exposure_edge_pixels = []
for pp in tqdm(edge_coord_pixel_pos, "Pixels computation"):
    # print(f"pp[0] = {pp[0]}")
    # print(f"pp[1] = {pp[1]}")
    zen_vals = map_coord_zeniths[:, pp[0], pp[1]]
    # check if zen_vals is zero
    if np.any(np.isnan(zen_vals)) or np.any(zen_vals == 0):
        continue

    # compute neutrino background for every zenith angle value
    bkg_nu_de = bkg_nu.evaluate_integrate(
        fov_lon=zen_vals * u.deg,
        fov_lat=None,  # does not matter
        energy_reco=geom.get_axis_by_name("energy").edges[:, np.newaxis],
    )
    bkg_nu_int = (bkg_nu_de * d_omega[pp[0], pp[1]]).to_value("s-1").sum(axis=0)
    nu_background_edge_pixels.append(bkg_nu_int)

    # compute muon background for every zenith angle value
    bkg_mu_de = bkg_mu.evaluate_integrate(
        fov_lon=zen_vals * u.deg,
        fov_lat=None,  # does not matter
        energy_reco=geom.get_axis_by_name("energy").edges[:, np.newaxis],
    )
    bkg_mu_int = (bkg_mu_de * d_omega[pp[0], pp[1]]).to_value("s-1").sum(axis=0)
    mu_background_edge_pixels.append(bkg_mu_int)

    # compute exposure for every zenith angle value
    # (sum over all energies - just for this test)
    exp = (
        calc_exposure(zen_vals * u.deg, aeff, geom_true).sum(axis=(0, 1))
        * time_step
        * u.s
    ).to_value("m2 s")
    exposure_edge_pixels.append(exp)


# compute background for all edge pixels and mean zenith angle
nu_background_edge_pixels_zen_mean = []
mu_background_edge_pixels_zen_mean = []
exposure_edge_pixels_zen_mean = []

for i in tqdm(range(n_datasets), "Background"):
    nu_background_edge_pixels_zen_mean.append([])
    mu_background_edge_pixels_zen_mean.append([])
    exposure_edge_pixels_zen_mean.append([])

    for pp in edge_coord_pixel_pos:
        zen_vals = map_coord_zeniths[:, pp[0], pp[1]][zen_masks[i]]

        # neutrino background
        bkg_nu_de = bkg_nu.evaluate_integrate(
            fov_lon=zen_vals.mean() * u.deg,
            fov_lat=None,  # does not matter
            energy_reco=geom.get_axis_by_name("energy").edges[:, np.newaxis],
        )
        bkg_nu_int = (bkg_nu_de * d_omega[pp[0], pp[1]]).to_value("s-1").sum(axis=0)
        nu_background_edge_pixels_zen_mean[-1].append(bkg_nu_int[0])

        # muon background
        bkg_mu_de = bkg_mu.evaluate_integrate(
            fov_lon=zen_vals.mean() * u.deg,
            fov_lat=None,  # does not matter
            energy_reco=geom.get_axis_by_name("energy").edges[:, np.newaxis],
        )
        bkg_mu_int = (bkg_mu_de * d_omega[pp[0], pp[1]]).to_value("s-1").sum(axis=0)
        mu_background_edge_pixels_zen_mean[-1].append(bkg_mu_int[0])

        # compute exposure for every zenith angle value
        # (sum over all energies - just for this test)
        exp = (
            calc_exposure(zen_vals.mean() * u.deg, aeff, geom_true).sum(axis=(0, 1))
            * time_step
            * u.s
        ).to_value("m2 s")
        exposure_edge_pixels_zen_mean[-1].append(exp[0])


# Plot neutrino background rate prediction distribution for all edge coordinates defined above,
# for each zenith angle bin
# for i in range(n_datasets):
#     z1 = zen_bins[1:][bin_mask][i]
#     z2 = zen_bins[:-1][bin_mask][i]

#     fig, axes = plt.subplots(3, 3, figsize=(10, 10))
#     for j, pp in enumerate(edge_coord_pixel_pos):
#         ax = axes[j // 3][j % 3]
#         bkg_vals = nu_background_edge_pixels[j][zen_masks[i]]
#         ax.hist(
#             np.log10(bkg_vals),
#             bins=np.linspace(np.log10(bkg_vals.min()), np.log10(bkg_vals.max()), 51),
#             histtype="step",
#         )
#         ylim = ax.get_ylim()
#         ax.vlines(np.log10(bkg_vals.mean()), 0, ylim[1], color="tab:blue", ls="--")
#         ax.vlines(
#             np.log10(nu_background_edge_pixels_zen_mean[i][j]),
#             0,
#             ylim[1],
#             color="tab:red",
#             ls="--",
#         )
#         ax.set_ylim(0, ylim[1])
#         ax.set_title(
#             "R.A.: ${:.1f}^\circ$, Dec.: ${:.1f}^\circ$".format(
#                 sky_coord[pp[0], pp[1]].ra.value, sky_coord[pp[0], pp[1]].dec.value
#             )
#         )
#         ax.text(
#             0.95,
#             0.95,
#             "${:.3g}\%$".format(
#                 100
#                 * (nu_background_edge_pixels_zen_mean[i][j] - bkg_vals.mean())
#                 / bkg_vals.mean()
#             ),
#             ha="right",
#             va="top",
#             bbox=dict(ec="k", fc="w", alpha=0.5),
#             transform=ax.transAxes,
#         )
#         ax.grid(ls="--")
#         if j // 3 == 2:
#             ax.set_xlabel("$\log_{10}(\mathrm{Rate}\,/\,\mathrm{s}^{-1})$")

#     fig.text(
#         0.5,
#         0.99,
#         r"Atmospheric neutrino background for $\theta\in [{:.1f},{:.1f}]$".format(
#             z1, z2
#         ),
#         ha="center",
#         va="top",
#         fontsize="x-large",
#         transform=fig.transFigure,
#     )

#     plt.subplots_adjust(top=0.94)

#     # plt.show()
#     for form in ["png", "pdf"]:
#         fig.savefig(
#             "{}/KM3NeT_nu_bkg_rate_dist_fov_edges_bin_{:02d}.{}".format(
#                 bkg_plot_dir, i, form
#             )
#         )

# Plot muon background rate prediction distribution for all edge coordinates defined above,
# for each zenith angle bin
# for i in range(n_datasets):
#     z1 = zen_bins[1:][bin_mask][i]
#     z2 = zen_bins[:-1][bin_mask][i]

#     fig, axes = plt.subplots(3, 3, figsize=(10, 10))
#     for j, pp in enumerate(edge_coord_pixel_pos):
#         ax = axes[j // 3][j % 3]
#         bkg_vals = mu_background_edge_pixels[j][zen_masks[i]]
#         if (bkg_vals > 0).sum() > 1:
#             h = ax.hist(
#                 np.log10(bkg_vals[bkg_vals > 0]),
#                 bins=np.linspace(
#                     np.log10(bkg_vals[bkg_vals > 0].min()), np.log10(bkg_vals.max()), 51
#                 ),
#                 histtype="step",
#             )[0]
#             ylim = ax.get_ylim()
#             ax.vlines(np.log10(bkg_vals.mean()), 0, ylim[1], color="tab:blue", ls="--")
#             if mu_background_edge_pixels_zen_mean[i][j] > 0:
#                 ax.vlines(
#                     np.log10(mu_background_edge_pixels_zen_mean[i][j]),
#                     0,
#                     ylim[1],
#                     color="tab:red",
#                     ls="--",
#                 )
#                 ax.text(
#                     0.95,
#                     0.95,
#                     "${:.3g}\%$".format(
#                         100
#                         * (mu_background_edge_pixels_zen_mean[i][j] - bkg_vals.mean())
#                         / bkg_vals.mean()
#                     ),
#                     ha="right",
#                     va="top",
#                     bbox=dict(ec="k", fc="w", alpha=0.5),
#                     transform=ax.transAxes,
#                 )
#             ax.set_ylim(0, ylim[1])
#         ax.set_title(
#             "R.A.: ${:.1f}^\circ$, Dec.: ${:.1f}^\circ$".format(
#                 sky_coord[pp[0], pp[1]].ra.value, sky_coord[pp[0], pp[1]].dec.value
#             )
#         )
#         ax.grid(ls="--")
#         if j // 3 == 2:
#             ax.set_xlabel("$\log_{10}(\mathrm{Rate}\,/\,\mathrm{s}^{-1})$")

#     fig.text(
#         0.5,
#         0.99,
#         r"Atmospheric muon background for $\theta\in [{:.1f},{:.1f}]$".format(z1, z2),
#         ha="center",
#         va="top",
#         fontsize="x-large",
#         transform=fig.transFigure,
#     )

#     plt.subplots_adjust(top=0.94)

#     # plt.show()
#     for form in ["png", "pdf"]:
#         fig.savefig(
#             "{}/KM3NeT_mu_bkg_rate_dist_fov_edges_bin_{:02d}.{}".format(
#                 bkg_plot_dir, i, form
#             )
#         )

# Plot exposure distribution for all edge coordinates defined above,
# for each zenith angle bin
# for i in range(n_datasets):
#     z1 = zen_bins[1:][bin_mask][i]
#     z2 = zen_bins[:-1][bin_mask][i]

#     fig, axes = plt.subplots(3, 3, figsize=(10, 10))
#     for j, pp in enumerate(edge_coord_pixel_pos):
#         ax = axes[j // 3][j % 3]
#         exp = exposure_edge_pixels[j][zen_masks[i]]
#         ax.hist(
#             np.log10(exp),
#             bins=np.linspace(np.log10(exp.min()), np.log10(exp.max()), 51),
#             histtype="step",
#         )
#         ylim = ax.get_ylim()
#         ax.vlines(np.log10(exp.mean()), 0, ylim[1], color="tab:blue", ls="--")
#         ax.vlines(
#             np.log10(exposure_edge_pixels_zen_mean[i][j]),
#             0,
#             ylim[1],
#             color="tab:red",
#             ls="--",
#         )
#         ax.set_ylim(0, ylim[1])
#         ax.set_title(
#             "R.A.: ${:.1f}^\circ$, Dec.: ${:.1f}^\circ$".format(
#                 sky_coord[pp[0], pp[1]].ra.value, sky_coord[pp[0], pp[1]].dec.value
#             )
#         )
#         ax.text(
#             0.95,
#             0.95,
#             "${:.3g}\%$".format(
#                 100 * (exposure_edge_pixels_zen_mean[i][j] - exp.mean()) / exp.mean()
#             ),
#             ha="right",
#             va="top",
#             bbox=dict(ec="k", fc="w", alpha=0.5),
#             transform=ax.transAxes,
#         )
#         ax.grid(ls="--")
#         if j // 3 == 2:
#             ax.set_xlabel("$\log_{10}(\mathrm{Exposure}\,/\,\mathrm{m}^2\,\mathrm{s})$")

#     fig.text(
#         0.5,
#         0.99,
#         r"Exposure for $\theta\in [{:.1f},{:.1f}]$".format(z1, z2),
#         ha="center",
#         va="top",
#         fontsize="x-large",
#         transform=fig.transFigure,
#     )

#     plt.subplots_adjust(top=0.94)

#     # plt.show()
#     for form in ["png", "pdf"]:
#         fig.savefig(
#             "{}/KM3NeT_exposure_dist_fov_edges_bin_{:02d}.{}".format(
#                 bkg_plot_dir, i, form
#             )
#         )

# !!! this takes several hours !!!

nu_background_maps = np.zeros((n_datasets, *geom.data_shape))
mu_background_maps = np.zeros((n_datasets, *geom.data_shape))
exposure_maps = np.zeros((n_datasets, *geom_true.data_shape))
for i in tqdm(range(len(obstimes)), "Fill datasets"):
    # skip this time bin if the zenith angle of the source position is below 80 deg
    if dataset_idx[i] >= n_datasets:
        assert zen_angles[i] <= zen_bins[7]
        continue

    # zenith angle map for this time bin
    zen_vals = map_coord_zeniths[i]
    if np.any(np.isnan(zen_vals)) or np.any(zen_vals == 0):
        continue

    # compute neutrino background map for this time bin
    bkg_nu_de = bkg_nu.evaluate_integrate(
        fov_lon=zen_vals * u.deg,
        fov_lat=None,  # does not matter
        energy_reco=geom.get_axis_by_name("energy").edges[:, np.newaxis, np.newaxis],
    )
    bkg_nu_de = (bkg_nu_de * d_omega).to_value("s-1")

    # compute muon background map for this time bin
    bkg_mu_de = bkg_mu.evaluate_integrate(
        fov_lon=zen_vals * u.deg,
        fov_lat=None,  # does not matter
        energy_reco=geom.get_axis_by_name("energy").edges[:, np.newaxis, np.newaxis],
    )
    bkg_mu_de = (bkg_mu_de * d_omega).to_value("s-1")

    # compute exposure for this time bin
    exp = (calc_exposure(zen_vals * u.deg, aeff, geom_true) * time_step * u.s).to_value(
        "m2 s"
    )

    # add neutrino background rate for the correct dataset
    np.add.at(nu_background_maps, dataset_idx[i], bkg_nu_de)

    # add muon background rate for the correct dataset
    np.add.at(mu_background_maps, dataset_idx[i], bkg_mu_de)

    # add exposure for the correct dataset
    np.add.at(exposure_maps, dataset_idx[i], exp)


# Multiply exposure maps by 10, for 10 years of data
exposure_maps *= years

# Replace the exposure attribute for each dataset
for i in tqdm(range(n_datasets), "Replace"):
    exp_map = WcsNDMap(geom_true.copy(), data=exposure_maps[i], unit="m2 s")
    datasets[i].exposure = exp_map

# divide the summed-up background rates for each dataset by the number of time bins
# that contribute to this dataset --> obtain an averaged rate
# then, multiply with the total livetime for each dataset
for i in range(n_datasets):
    n_times = (dataset_idx == i).sum()
    nu_background_maps[i] /= n_times
    mu_background_maps[i] /= n_times

    nu_background_maps[i] *= livetime_pointings[i].to_value("s")
    mu_background_maps[i] *= livetime_pointings[i].to_value("s")


# Create the atmospheric neutrino background model
bkg_models = []
for i in tqdm(range(n_datasets), "atmospheric nu model"):
    # We already apply the energy dispersion when we create the IRF
    # file, so the following is obsolete now.
    #
    # # Create energy dispersion matrix for each observation because
    # # the background model is expected in reconstructed energy
    # edisp_matrix = edisp.to_energy_dispersion(zen_mean[i]*u.deg,
    #                                           e_true = energy_axis_true.edges,
    #                                           e_reco = energy_axis.edges)
    #
    # # create the model map on the true geometry
    # bkg_model_map = WcsNDMap(geom_true.copy(), data=nu_background_maps[i])
    # bkg_model_map.geom.axes[0].name = 'energy_true'
    #
    # # apply the energy dispersion
    # bkg_model_map = bkg_model_map.apply_edisp(edisp_matrix)

    # create map
    bkg_model_map = WcsNDMap(geom.copy(), data=nu_background_maps[i])

    # generate BackgroundModel instance
    bkg_model = BackgroundModel(
        bkg_model_map, name=datasets[i].name + "-bkg", datasets_names=[datasets[i].name]
    )

    # store model
    bkg_models.append(bkg_model)

# Same for the atmospheric muon background
bkg_models_mu = []
for i in tqdm(range(n_datasets), "atmospheric mu model"):
    # Create map
    bkg_model_map = WcsNDMap(geom.copy(), data=mu_background_maps[i])

    # generate BackgroundModel instance
    bkg_model = BackgroundModel(
        bkg_model_map,
        name=datasets[i].name + "bkg-mu",
        datasets_names=[datasets[i].name],
    )

    # store model
    bkg_models_mu.append(bkg_model)

# First attach only neutrino background model to datasets, to compute npred
npred_bkg_nu = []
for i in range(n_datasets):
    datasets[i].models = bkg_models[i]
    npred_bkg_nu.append(datasets[i].npred())

# Neutrino background, first zenith angle bin (upgoing)
npred_bkg_nu[0].sum_over_axes().plot(add_cbar=True)

# Neutrino background, intermediate zenith angle bin
npred_bkg_nu[3].sum_over_axes().plot(add_cbar=True)

# Neutrino background, last zenith angle bin (horizon)
npred_bkg_nu[-1].sum_over_axes().plot(add_cbar=True)

# Now add the muon background (store npred for both the sum and the muon background only)
npred_bkg_mu = []
npred_bkg_sum = []
for i in range(n_datasets):
    datasets[i].background_model.stack(bkg_models_mu[i])
    bkg_sum = datasets[i].npred()
    bkg_mu = bkg_sum - npred_bkg_nu[i]
    npred_bkg_mu.append(bkg_mu)
    npred_bkg_sum.append(bkg_sum)

# Muon background, first zenith angle bin (upgoing)
npred_bkg_mu[0].sum_over_axes().plot(add_cbar=True)

# Muon background, intermediate zenith angle bin
npred_bkg_mu[3].sum_over_axes().plot(add_cbar=True)

# Muon background, last zenith angle bin (horizon)
npred_bkg_mu[-1].sum_over_axes().plot(add_cbar=True)

# Write the datasets to disk
# (note: datasets are not included in the repo because of their large size)
ds_dir = f"KM3NeT_{source_name}_{years}y"
makedirs(Path(output_path / ds_dir), exist_ok=True)
Datasets(datasets).write(Path(output_path / ds_dir), ds_dir, overwrite=True)

makedirs(Path(output_path / "npred_bkg_km3net"), exist_ok=True)

# Also store npred maps for nu and mu background
for i in range(n_datasets):
    npred_bkg_nu[i].write(
        Path(
            output_path
            / "npred_bkg_km3net"
            / "{}_npred_nu_{:02d}.fits".format(source_name, i + 1)
        ),
        overwrite=True,
    )
    npred_bkg_mu[i].write(
        Path(
            output_path
            / "npred_bkg_km3net"
            / "{}_npred_mu_{:02d}.fits".format(source_name, i + 1)
        ),
        overwrite=True,
    )

# PLOT

cmap = cm.afmhot

model_pars_PD = np.loadtxt(
    Path(parpar_dir / "data" / "models" / f"input_model_PD_{source_name}.txt")
)

ECPL_PD = ExponentialCutoffPowerLaw(
    amplitude=model_pars_PD[0] / u.eV,
    e_0=model_pars_PD[1] * u.TeV,
    alpha=model_pars_PD[2],
    e_cutoff=model_pars_PD[3] * u.TeV,
    beta=model_pars_PD[4],
)
nu_ECPL_PD = PionDecayKelner06(
    ECPL_PD, particle_type="muon_neutrino", oscillation_factor=0.5
)
spectral_model = NaimaSpectralModel(
    nu_ECPL_PD, distance=source_pos_dist["Dist"][source_name] * u.kpc
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

# Append source model to all datasets
for i in range(n_datasets):
    datasets[i].models.append(model)

# Compute npred for source model
npred_sum = []
npred_src = []
for i in range(n_datasets):
    bkg_src = datasets[i].npred()
    src = bkg_src - npred_bkg_sum[i]
    npred_sum.append(bkg_src)
    npred_src.append(src)

rs = np.random.RandomState(seed=314)
for i in range(n_datasets):
    datasets[i].fake(rs)

# fig, axes = plt.subplots(
#     2, 3, figsize=(10, 6), subplot_kw=dict(projection=geom.to_image().wcs)
# )

# # norm = colors.Normalize(vmax=0.01)
# for i, ds_idx in enumerate([0, 3, -1]):
#     npred_sum[ds_idx].sum_over_axes().plot(ax=axes[0][i], stretch="sqrt")
#     data = npred_sum[ds_idx].sum_over_axes().data
#     im = axes[0][i].imshow(data, cmap=cmap, vmax=0.1)
#     datasets[ds_idx].counts.smooth(0.25 * u.deg).sum_over_axes().plot(ax=axes[1][i])
#     data = datasets[ds_idx].counts.smooth(0.25 * u.deg).sum_over_axes().data
#     im = axes[1][i].imshow(data, cmap=cmap, vmax=0.1)

# add up npred and counts of all datasets
npred_bkg_nu_all = WcsNDMap(geom)
npred_bkg_mu_all = WcsNDMap(geom)
npred_src_all = WcsNDMap(geom)
npred_sum_all = WcsNDMap(geom)
counts_all = WcsNDMap(geom)
for i in range(n_datasets):
    npred_bkg_nu_all += npred_bkg_nu[i]
    npred_bkg_mu_all += npred_bkg_mu[i]
    npred_src_all += npred_src[i]
    npred_sum_all += npred_sum[i]
    counts_all += datasets[i].counts

# Source region
src_region = CircleSkyRegion(
    src_pos, (source_pos_dist["Radius"][source_name] + 0.5) * u.deg
)
src_reg_mask = geom.to_image().region_mask([src_region])

# # Predicted counts, summed for all datasets
# fig, ax, cbar = npred_sum_all.sum_over_axes().plot(add_cbar=True, stretch="sqrt")
# data = npred_sum_all.sum_over_axes().data
# im = ax.imshow(data, cmap=cmap, vmin=0.0)
# src_reg_pix = src_region.to_pixel(ax.wcs)
# src_reg_pix.plot(ax=ax, ls="--")

# # Fake counts, summed for all datasets
# counts_all.sum_over_axes().smooth(0.25 * u.deg).plot(
#     add_cbar=True, vmin=0
# )  # Add vmin=0


# For the paper
fig_width = 8.8 / 2.54
fig_height = 2.75

fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
fig, ax, cbar = counts_all.sum_over_axes().smooth(0.25 * u.deg).plot(add_cbar=True)

data = counts_all.sum_over_axes().smooth(0.25 * u.deg).data
im = ax.imshow(data, cmap=cmap, vmin=0.0)
src_reg_pix = src_region.to_pixel(ax.wcs)
src_reg_pix.plot(ax=ax, ec="w", ls="--")


norm = colors.Normalize(vmin=0.0)
cbar.update_normal(cm.ScalarMappable(norm=norm, cmap=cmap))
cbar.set_label("Counts per pixel")

plt.subplots_adjust(left=0.15, right=0.95, bottom=0.17, top=0.97)

# for form in ["png", "pdf"]:
#     fig.savefig(
#         path.join(
#             results_path,
#             "plots/map_KM3NeT_counts_summed_PD_{}.{}".format(source_name, form),
#         )
#     )

save_fig(fig, f"map_KM3NeT_counts_summed_PD_{source_name}")

# Counts spectrum
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

e = energy_axis.center.value / 1e3
ax.plot(
    e,
    npred_bkg_nu_all.data[:, src_reg_mask].sum(axis=1),
    color="tab:blue",
    label=r"$\nu$ background",
)
ax.plot(
    e,
    npred_bkg_mu_all.data[:, src_reg_mask].sum(axis=1),
    color="tab:blue",
    ls="--",
    label=r"$\mu$ background",
)
ax.plot(
    e, npred_src_all.data[:, src_reg_mask].sum(axis=1), color="tab:red", label="Signal"
)
ax.plot(
    e, npred_sum_all.data[:, src_reg_mask].sum(axis=1), color="tab:green", label="Sum"
)
cts = counts_all.data[:, src_reg_mask].sum(axis=1)
e = e[cts > 0]
cts = cts[cts > 0]
ax.errorbar(
    e,
    cts,
    xerr=None,
    yerr=plot_utils.feldman_cousins_errors(cts).T,
    linestyle="None",
    marker="o",
    markersize=3,
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
ax.text(0.97, 0.87, "KM3NeT", ha="right", va="top", transform=ax.transAxes)

plot_utils.format_log_axis(ax.xaxis)
plot_utils.format_log_axis(ax.yaxis)

ax.legend(loc="lower left", ncol=2, columnspacing=1)

# for form in ["png", "pdf"]:
#     fig.savefig(
#         path.join(
#             results_path, "plots/counts_reg_KM3NeT_PD_{}.{}".format(source_name, form)
#         )
#     )

save_fig(fig, f"counts_reg_KM3NeT_PD_{source_name}")


print(f"--- {round((time.time() - start_time),3)} s ---")