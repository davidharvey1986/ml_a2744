"""
get_model_probabilities.py
--------------------------
Core inference helpers for the dark-matter classification pipeline.

Public API
~~~~~~~~~~
args                       – global configuration namespace (class)
get_probabilities(...)     – run a list of checkpoints over a test set
get_threshold_for_cross(…) – summarise per-cross-section mean probabilities
get_probs_from_results(…)  – extract means/errors from an all_results dict
get_variable_redshift_dist – build a spatially-resolved photo-z map
get_latent_space(…)        – extract latent representations
plot_predictions(…)        – diagnostic probability histograms
plot_observations(…)       – overlay observed data on a plot
get_dataset_meta(…)        – load a simulation pkl by dataset / cross-section
get_kappa(…)               – Kaiser-Squires convergence map
get_nsigma(…)              – distance from a distribution in sigma units
get_direct_prob(…)         – CDF-based probability relative to median
get_mass_cut(…)            – mass threshold from Harvey / Jauzac measurements
get_massfunction_weights(…)– reweighting factors for different mass functions
get_gaussian_process()     – build a sklearn GP with a Matern kernel
gp_invert(…)               – invert a GP prediction to recover sigma
thresh_to_cross(…)         – map a probability threshold to a cross-section
get_density(…)             – 2-D KDE on a grid
curly_brace(…)             – draw a curly brace annotation on a matplotlib axis
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import glob
import pickle as pkl

# ---------------------------------------------------------------------------
# Numerical / scientific
# ---------------------------------------------------------------------------
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter, zoom
from scipy.optimize import curve_fit
from scipy.special import lambertw, logsumexp
from scipy.stats import (
    binned_statistic,
    binned_statistic_2d,
    cauchy,
    chi,
    gaussian_kde,
    ks_2samp,
    lognorm,
    norm,
)

# ---------------------------------------------------------------------------
# Astronomy
# ---------------------------------------------------------------------------
from astropy import units
from astropy.cosmology import Planck18
from astropy.io import fits
import lenspack
import lenspack.utils
import lenspack.image.inversion

# ---------------------------------------------------------------------------
# PyTorch / torchvision
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    ResNet34_Weights,
    SqueezeNet1_1_Weights,
    mobilenet_v3_small,
    resnet18,
    resnet34,
    squeezenet1_1,
)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib import gridspec, pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter, LinearLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------
import umap

# ---------------------------------------------------------------------------
# Misc third-party
# ---------------------------------------------------------------------------
import wandb
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# ---------------------------------------------------------------------------
# Project-local
# ---------------------------------------------------------------------------
from add_shear_to_data import (
    bin_obs_data,
    get_obs_data,
    get_source_redshift,
    ra_dec_to_simulation_image_pos,
    return_error_in_mean,
)
from dataset import (
    MAP_ZL,
    MAP_ZS,
    apply_intrinsic_ell,
    get_cross_section_from_filename,
    prepare_dataloaders,
    rescale_lens_source_configuration,
    sigma_critical,
)
from getColorFromRange import colourFromRange
from mass_weighting import a2744_log_mass, gaussian_mass_weights
from model import create_model
from refactor_parameter_space import *          # domain-specific parameter space helpers
from RRGtools import run_match
from pyRRG.calc_shear import calc_shear
from train import evaluate, train_epoch
from utils import calculate_class_weights, parse_args, set_seed, setup_wandb


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Colour map for cross-section labels used in diagnostic plots.
COLOR_BY_CROSS = {
    "-1.00": "grey",
    "0.00":  "b",
    "0.01":  "g",
    "0.05":  "r",
    "0.07":  "k",
    "0.10":  "y",
    "0.20":  "c",
    "0.30":  "purple",
    "1.00":  "grey",
}

#: Per-simulation colour scheme used in publication figures.
COLOUR_SCHEME = {
    "darkskies": "#E69F00",   # orange
    "bahamas":   "#56B4E9",   # sky blue
    "tng":       "#009E73",   # bluish green
    "flamingo":  "#D55E00",   # vermilion
}

# Pre-computed radial grid (kpc) for mass-integration steps.
_vec = np.arange(100) - 49.5
_xg, _yg = np.meshgrid(_vec * 20.0, _vec * 20.0)
rgrid = np.sqrt(_xg ** 2 + _yg ** 2)


def integrated_log_mass(kappa, critical_kappa, aperture=750.0):
    """log10 projected mass [Msun] inside *aperture* kpc of (N, 100, 100) kappa maps."""
    return np.log10(
        (np.sum(kappa[:, rgrid < aperture], axis=-1) * critical_kappa * (20 * units.kpc) ** 2).value
    )


# ---------------------------------------------------------------------------
# Global configuration namespace
# ---------------------------------------------------------------------------

class args:
    """
    Flat namespace that carries all runtime hyper-parameters.

    This mirrors the argument object produced by ``parse_args()`` so that
    scripts can import a ready-to-use default configuration without having
    to go through the CLI.  Individual attributes may be overridden in-place
    before passing the class to any function.
    """
    # Model
    pretrained           = True
    in_channels          = 2
    model                = "squeezenet1_1"
    cnn_base_channels    = 32
    num_avgpool_head     = 1
    domain_discriminator = None

    # Domain adaptation
    adaptation           = "cdan"
    use_mixup            = False
    mixup_strategy       = False
    source_domain        = "a2744"
    target_domain        = "darkskies_obs"

    # Training / data
    weighting_scheme     = "inverse_frequency"
    train_split          = 0.8
    batch_size           = 32
    num_workers          = 0
    device               = "MPS"
    seed                 = 10

    # Augmentation
    aug_h_flip_prob          = 0.5
    aug_v_flip_prob          = 0.5
    aug_rotation_degrees     = 360
    aug_rotation_prob        = 1.0
    aug_crop_prob            = 0.5
    aug_crop_scale_min       = 0.9
    aug_crop_scale_max       = 1.1

    # Image
    image_size           = 100
    use_log_transform    = False
    use_normalization    = False

    # Data / features
    data_dir             = "../data/"
    dtypes               = ["image"]
    meta_names           = []
    mass_index           = 0
    log_mass_cut         = 0
    downsample           = 1

    # Lensing / physics
    zl                   = 0.305
    default_zl           = 0.305
    apply_intrinsic_ell  = 1
    jwst_filter          = "concat"
    med_norm             = -1
    shape_measurement_bias = {"e1": {"c": 0, "m": 0}, "e2": {"c": 0, "m": 0}}

    # Miscellaneous
    verbose              = False
    ignore_dataset       = [""]
    unbalance            = False
    cluster_member_contamination = 0.0

    # Source redshift – initialised from the observed catalogue
    zs          = get_source_redshift(jwst_filter, data_dir="../data/100/a2744")
    default_zs  = zs
    print(f"Source redshift: {zs}")


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def get_probabilities(
    target_domain,
    list_of_models,
    args=None,
    data_loaders=None,
    test_set="target_test",
    quiet=False,
):
    """
    Run one or more model checkpoints over a test set and return raw
    softmax probabilities together with useful diagnostics.

    Parameters
    ----------
    target_domain : str
        Name of the domain to evaluate on (e.g. ``"darkskies_obs"``).
    list_of_models : list[str]
        Paths to model checkpoint files.
    args : class, optional
        Configuration namespace.  Defaults to the module-level ``args``.
    data_loaders : dict, optional
        Pre-built data-loader dict.  Built automatically when *None*.
    test_set : str
        Key inside ``data_loaders`` to evaluate on.
    quiet : bool
        Suppress progress bars and informational prints.

    Returns
    -------
    dict with keys:
        all_cross_sections, all_binary_labels, probabilities, indexes,
        very_good, very_bad, very_good_idx, very_bad_idx, data_loaders
    """
    if args is None:
        args = get_temp_args()

    args.target_domain = target_domain

    # Build model objects from checkpoints
    models = []
    for checkpoint in list_of_models:
        args.checkpoint = checkpoint
        models.append(create_model(args))

    if not quiet:
        print(f"Found {len(models)} models")

    device = "mps"
    CONF_LIMIT = 0.2   # probability threshold for "very good / very bad" examples

    all_cross_sections = []
    all_binary_labels  = []
    probabilities      = []
    very_good          = []
    very_good_idx      = []
    very_bad           = []
    very_bad_idx       = []
    indexes            = []

    with torch.no_grad():
        for imodel in tqdm(models, disable=quiet):
            this_probabilities = []
            this_cross         = []

            args.seed = imodel.args.seed
            dataloaders = (
                prepare_dataloaders(args) if data_loaders is None else data_loaders
            )

            # Flatten all batches once so we can re-index by idx below
            all_data = [[j[0] for j in i] for i in dataloaders[test_set]]

            for idx, batch_data in enumerate(dataloaders[test_set][0]):

                data, cross_sections, binary_labels, file_idx, image_idx = batch_data
                data = [i[idx] for i in all_data]

                if data[0].shape[0] == 1:
                    if args.verbose:
                        print("Batch size 1 — skipping to avoid batch-norm issues")
                    continue

                if args.verbose:
                    print(f"Evaluating {data[0].shape[0]} samples in batch")

                try:
                    outputs_dict = imodel(data)
                except Exception as exc:
                    raise ValueError(
                        f"Forward pass failed — input shape: {data[0].shape}"
                    ) from exc

                binary_labels = binary_labels.to(device)
                this_cross.append(cross_sections.cpu())
                all_binary_labels.append(binary_labels.cpu())

                prob = torch.softmax(outputs_dict["classification"], dim=1)
                this_probabilities.append(prob)
                indexes.append(image_idx)

                # Collect confidently-classified examples for qualitative review
                sidm_mask = cross_sections == 0.05
                very_good.append(
                    data[0][np.where((prob[:, 0] < CONF_LIMIT) & sidm_mask)[0]]
                )
                very_bad.append(
                    data[0][np.where((prob[:, 0] > 1 - CONF_LIMIT) & sidm_mask)[0]]
                )
                very_good_idx.append(
                    image_idx[np.where((prob[:, 0] < CONF_LIMIT) & sidm_mask)[0]]
                )
                very_bad_idx.append(
                    image_idx[np.where((prob[:, 0] > 1 - CONF_LIMIT) & sidm_mask)[0]]
                )

            probabilities.append(torch.cat(this_probabilities))
            all_cross_sections.append(torch.cat(this_cross))

    return {
        "all_cross_sections": torch.stack(all_cross_sections),
        "all_binary_labels":  torch.cat(all_binary_labels),
        "probabilities":      torch.stack(probabilities),
        "indexes":            np.concatenate(indexes),
        "very_good":          np.concatenate(very_good),
        "very_bad":           np.concatenate(very_bad),
        "very_good_idx":      np.concatenate(very_good_idx),
        "very_bad_idx":       np.concatenate(very_bad_idx),
        "data_loaders":       dataloaders,
    }


# ---------------------------------------------------------------------------
# Probability / threshold helpers
# ---------------------------------------------------------------------------

def get_threshold_for_cross(
    results_list,
    dataset=None,
    function=np.mean,
    mass_cut=None,
    quiet=True,
    integrated_mass=False,
    mass_weights=None,
    h=0.7,
    ncomponents=None,
    mass_gaussian=None,
):
    """
    Summarise per-cross-section mean probabilities over one or more result
    dicts (e.g. multiple Monte-Carlo draws).

    Parameters
    ----------
    results_list : dict or list[dict]
        Output(s) of :func:`get_probabilities`.
    dataset : str, optional
        Simulation name — required when *mass_cut* or *ncomponents* is set.
    function : callable
        Aggregation function applied to the probability vector (default: mean).
    mass_cut : tuple[float, float], optional
        ``(log_mass_min, log_mass_max)`` selection in solar masses.
    quiet : bool
        Suppress informational prints inside ``get_threshold_for_cross``.
    integrated_mass : bool
        Compute mass by integrating the convergence map rather than using
        the stored metadata mass.
    mass_weights : dict, optional
        ``{"x": bin_edges, "y": weights}`` for mass-function reweighting.
    h : float
        Hubble parameter (used for unit conversions).
    ncomponents : int, optional
        Keep clusters with more than this many sub-structure components.
    mass_gaussian : tuple[float, float], optional
        ``(mu, sigma)`` in log10 Msun, e.g. ``mass_weighting.a2744_mass_gaussian()``.
        Instead of a hard mass cut, weight each cluster by
        N(log M; mu, sigma) / n(log M), with n the mass density of the clusters
        averaged (see mass_weighting.py). Use with integrated_mass=True for the
        A2744 (<750 kpc) mass; not combinable with mass_weights.

    Returns
    -------
    dict with keys:
        thresholds, threshold_err, cross_sections, probabilities, indexes
    """
    need_mass = (mass_cut is not None) or (mass_gaussian is not None)
    if (need_mass or ncomponents is not None) and (dataset is None):
        raise ValueError("A dataset name is required for mass or component selections.")
    if (mass_gaussian is not None) and (mass_weights is not None):
        raise ValueError("Use either mass_gaussian or mass_weights, not both.")

    if not isinstance(results_list, list):
        results_list = [results_list]

    # Pre-compute lensing critical density if we need projected masses
    critical_kappa = None
    if integrated_mass and need_mass:
        # The stored kappa is at the map redshifts, not the observed n(z)
        critical_kappa = (
            lenspack.utils.sigma_critical(MAP_ZL, MAP_ZS, Planck18)
            .to(units.Msun / units.kpc / units.kpc)
        )

    all_thresholds     = []
    all_threshold_err  = []
    all_probs          = []
    all_indexes        = []

    mass_data  = {}
    components = {}

    for results in results_list:
        unique_cross = torch.unique(results["all_cross_sections"])
        thresholds    = []
        threshold_err = []
        these_probs   = []
        these_indexes = []

        with torch.no_grad():
            for icross in unique_cross:
                icross_lab = str(icross)
                cross_mask = icross == results["all_cross_sections"][0]

                # ---- Build selection mask --------------------------------
                idx = results["indexes"][cross_mask]
                sel = np.ones(idx.shape[0], dtype=bool)
                if need_mass or ncomponents is not None:
                    if icross_lab not in components:
                        meta, data = get_dataset_meta(dataset, icross)
                        components[icross_lab] = meta["ncomponents"]
                        if need_mass:
                            mass_data[icross_lab] = (
                                integrated_log_mass(data[:, 2], critical_kappa)
                                if integrated_mass else meta["mass"]
                            )

                    if ncomponents is not None:
                        sel &= components[icross_lab][idx] > ncomponents
                    if mass_cut is not None:
                        mass = mass_data[icross_lab][idx]
                        sel &= (mass > mass_cut[0]) & (mass < mass_cut[1])

                # ---- Probability for selected objects --------------------
                prob = results["probabilities"][
                    icross == results["all_cross_sections"], :
                ][sel, :]
                probs = prob[:, 0].detach().numpy()

                if mass_gaussian is not None:
                    w = gaussian_mass_weights(mass_data[icross_lab][idx][sel], *mass_gaussian)
                    threshold = np.nansum(w * probs) / np.sum(w)
                elif mass_weights is None:
                    threshold = function(probs)
                else:
                    mass = mass_data[icross_lab][idx][sel]
                    # mass_weights must be built from the same mass definition
                    # (integrated_mass) as the selection; digitize - 1 is the bin index
                    ibin = np.clip(
                        np.digitize(mass, mass_weights["x"]) - 1,
                        0, len(mass_weights["y"]) - 1,
                    )
                    w = mass_weights["y"][ibin]
                    threshold = np.nansum(w * probs) / np.sum(w)

                thresholds.append(threshold)
                threshold_err.append(
                    np.nanquantile(probs, [0.16, 0.84]) / np.sqrt(probs.shape[0])
                )
                these_probs.append(prob)
                these_indexes.append(idx[sel])

        all_thresholds.append(thresholds)
        all_threshold_err.append(threshold_err)
        all_probs.append(these_probs)
        all_indexes.append(these_indexes)

    n_results = len(results_list)
    mean_thresholds = np.mean(np.array(all_thresholds), axis=0)
    mean_threshold_err = np.mean(all_threshold_err, axis=0) / np.sqrt(n_results)

    # Pool probabilities and indexes across Monte-Carlo draws
    n_cross = len(unique_cross)
    pooled_probs   = [
        np.concatenate([all_probs[j][i]   for j in range(n_results)])
        for i in range(n_cross)
    ]
    pooled_indexes = [
        np.concatenate([all_indexes[j][i] for j in range(n_results)])
        for i in range(n_cross)
    ]

    return {
        "thresholds":    mean_thresholds,
        "threshold_err": mean_threshold_err,
        "cross_sections": unique_cross.detach().numpy(),
        "probabilities": pooled_probs,
        "indexes":       pooled_indexes,
    }


def get_probs_from_results(all_results, target="tgt", **params):
    """
    Extract per-cross-section mean probabilities and uncertainties from a
    dict of seed-keyed result lists (as produced by ``get_all_model_results``).

    Parameters
    ----------
    all_results : dict
        Keys of the form ``"seed_N"``, each mapping to a dict with ``"tgt"``
        and ``"src"`` lists of :func:`get_probabilities` outputs.
    target : str
        Which domain to summarise (``"tgt"`` or ``"src"``).
    **params
        Forwarded to :func:`get_threshold_for_cross`.

    Returns
    -------
    cross_sections : np.ndarray
    means : np.ndarray
    errors : np.ndarray
    """
    thresholds = []
    for key, value in all_results.items():
        if "seed" not in key:
            continue

        result_list = value[target] if isinstance(value, dict) else value
        tgt = get_threshold_for_cross(result_list, quiet=False, **params)
        thresholds.append(1 - tgt["thresholds"])

    thresholds = np.array(thresholds)
    means  = np.mean(thresholds, axis=0)
    errors = return_error_in_mean(thresholds)

    return tgt["cross_sections"], means, errors


# ---------------------------------------------------------------------------
# Data / dataset helpers
# ---------------------------------------------------------------------------

def get_temp_args(noisy=True):
    """Return the global ``args`` namespace (optionally zeroing intrinsic ell)."""
    if not noisy:
        args.apply_intrinsic_ell = 0.0
    return args


def get_dataset_meta(dataset, icross):
    """
    Load the simulation metadata and image array for a given dataset and
    cross-section value.

    Parameters
    ----------
    dataset : str
        Simulation name, e.g. ``"bahamas"``, ``"darkskies"``, ``"tng"``,
        ``"flamingo"``, or a vd-prefixed variant.
    icross : float or torch.Tensor
        Cross-section value.

    Returns
    -------
    tuple[dict, np.ndarray]
        ``(meta_dict, image_array)`` as stored in the pkl file.
    """
    if "vd" in dataset:
        path = f"../data/100/shear/{dataset}.pkl"
    elif icross != 0:
        path = f"../data/100/shear/{dataset}_{icross:0.1g}.pkl"
    elif dataset in ("flamingo", "tng"):
        path = f"../data/100/shear/{dataset}.pkl"
    else:
        path = f"../data/100/shear/{dataset}_cdm.pkl"

    return pkl.load(open(path, "rb"))


def get_variable_redshift_dist(data_dir="../data/100/a2744/", ifilter="concat"):
    """
    Build a 2-D spatially-resolved source-redshift map from the photo-z
    catalogue.  Pixels with no observed galaxy are set to the global default
    redshift and returned as ``NaN`` so the caller can fill them stochastically.

    Parameters
    ----------
    data_dir : str
        Directory containing the observed catalogue.
    ifilter : str
        JWST filter combination to load.

    Returns
    -------
    np.ndarray, shape (image_size, image_size)
    """
    cuts = {
        "f115w": {"signal_noise_cut": 0, "stat_type": "snr"},
        "f150w": {"signal_noise_cut": 5, "stat_type": "snr"},
    }
    obs_data = get_obs_data(ifilter, data_dir=data_dir, photoz=True, cuts=cuts)
    delta_ra, delta_dec = ra_dec_to_simulation_image_pos(obs_data)

    half = args.image_size // 2
    # binned_statistic_2d(x=ra, y=dec) is indexed [ra, dec]; transpose to [dec, ra]
    # like the shear maps (bin2d), so each pixel gets its own source redshift
    n_z = binned_statistic_2d(
        delta_ra,
        delta_dec,
        obs_data["redshift"],
        range=[[-half, half], [-half, half]],
        statistic=np.median,
        bins=args.image_size,
    ).statistic.T

    n_z[~np.isfinite(n_z)] = args.default_zs
    return n_z


def get_mass_cut(ifilter, zl=0.305, zs=1.6, thresh=0.0, nsigma=2, study="harvey"):
    """
    Return the log10 mass threshold (and confidence interval) consistent with
    published mass measurements of Abell 2744 (numbers in mass_weighting.py).

    Parameters
    ----------
    study : str
        ``"harvey"`` or ``"jauzac"``.

    Returns
    -------
    tuple[float, list[float]]
        ``(central_log_mass, [lower_log_mass, upper_log_mass])``
    """
    centre, bounds = a2744_log_mass(nsigma=nsigma, study=study)
    return centre, list(bounds)


def get_massfunction_weights(dataset, ifilter="concat", integrated_mass=True):
    """
    Compute per-mass-bin weights that account for the different mass functions
    across simulations: the inverse of the CDM mass function.

    integrated_mass : bool
        Use the projected mass inside 750 kpc, as get_threshold_for_cross does
        with integrated_mass=True; otherwise the total mass in the stamp.
        Must match the integrated_mass passed to get_threshold_for_cross.

    Returns
    -------
    dict with keys ``"x"`` (bin edges) and ``"y"`` (inverse counts).
    """
    # The stored kappa is at the map redshifts, not the observed n(z)
    critical_density = sigma_critical(MAP_ZL, MAP_ZS, Planck18).to(
        units.Msun / units.kpc / units.kpc
    )

    pkl_path = (
        f"../data/100/shear/{dataset}_cdm.pkl"
        if dataset in ("bahamas", "darkskies")
        else f"../data/100/shear/{dataset}.pkl"
    )
    meta, data = pkl.load(open(pkl_path, "rb"))

    if integrated_mass:
        mass = integrated_log_mass(data[:, 2], critical_density)
    else:
        mass = np.log10(
            (np.sum(data[:, 2], axis=(-1, -2)) * critical_density * (20 * units.kpc) ** 2).value
        )

    mass_function, x = np.histogram(mass, bins=np.linspace(14, 15.5, 30))
    mass_function[mass_function == 0] = 1
    return {"x": x, "y": 1.0 / mass_function}


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def get_nsigma(value, dist, statistic="median"):
    """
    Express *value* in units of the one-sided width of *dist*.

    Parameters
    ----------
    statistic : str
        Central tendency estimator: ``"median"``, ``"mean"``, or ``"maxlike"``.
    """
    xcum = np.sort(dist)
    cumsum = np.cumsum(xcum) / np.sum(xcum)

    if statistic == "maxlike":
        y, x = np.histogram(dist, 50)
        xc = (x[:-1] + x[1:]) / 2.0
        centre = xc[np.argmax(y)]
    elif statistic == "median":
        centre = np.median(dist)
    elif statistic == "mean":
        centre = np.mean(dist)
    else:
        raise ValueError(f"Unknown statistic: {statistic!r}")

    lo = centre - xcum[np.argmin(np.abs(0.16 - cumsum))]
    hi = xcum[np.argmin(np.abs(0.84 - cumsum))] - centre

    result = []
    for v in np.atleast_1d(value):
        result.append((v - centre) / hi if v > centre else (centre - v) / lo)
    return np.stack(result)


def get_direct_prob(value, dist):
    """
    Return the tail probability of *value* relative to the median of *dist*,
    normalised so that the median maps to 1.
    """
    xcum    = np.sort(dist)
    cumsum  = np.cumsum(xcum) / np.sum(xcum)
    median  = np.median(dist)

    result = []
    for v in np.atleast_1d(value):
        cdf_v = cumsum[np.argmin(np.abs(xcum - v))]
        result.append(cdf_v / 0.5 if v < median else (1 - cdf_v) / 0.5)
    return np.stack(result)


# ---------------------------------------------------------------------------
# Gaussian process helpers
# ---------------------------------------------------------------------------

def get_gaussian_process():
    """Return a GP regressor with a Matérn-3/2 kernel plus white noise."""
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=0.3, length_scale_bounds=(1e-2, 10), nu=1.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    )
    return GaussianProcessRegressor(kernel=kernel, alpha=1e-4)


def gp_invert(gp, threshold):
    """
    Evaluate ``gp`` on a fine log-sigma grid and return a likelihood over
    that grid for the observed *threshold* value(s).

    Returns
    -------
    sigma_grid : np.ndarray
    loglike : np.ndarray
    """
    sigma_grid = np.logspace(-6, 2, 10_000)
    X_grid = np.log10(sigma_grid).reshape(-1, 1)
    y_pred, y_std = gp.predict(X_grid, return_std=True)

    if isinstance(threshold, float):
        loglike = norm.logpdf(threshold, loc=y_pred, scale=y_std * 2)
    else:
        loglike = np.array(
            [norm.logpdf(t, loc=y_pred, scale=y_std * 2) for t in threshold.flatten()]
        )

    return sigma_grid, loglike


def thresh_to_cross(thresh, ifilter="concat"):
    """
    Map a model probability threshold to a dark-matter self-interaction
    cross-section via a Gaussian process fit to calibration probabilities.
    """
    _, probs = pkl.load(open("pickles/probs_for_cross_concat_nob1.pkl", "rb"))
    _, probabilities, _ = pkl.load(open("pickles/model_on_data.pkl", "rb"))
    nmodels = probabilities[ifilter].shape[0]

    # Mean threshold per cross-section, averaged over models
    all_thresholds = np.array([
        [1.0 - np.mean(probs[cs][i]) for cs in probs]
        for i in range(nmodels)
    ])
    thresholds = np.mean(all_thresholds, axis=0)
    err        = np.std(all_thresholds, axis=0)

    cross = np.array([float(cs) for cs in probs])
    cdm_thresh = thresholds[cross == 0]

    # Exclude CDM point before fitting
    mask = cross != 0
    err        = err[mask]
    thresholds = thresholds[mask]
    cross      = np.log10(cross[mask])

    gp = get_gaussian_process()
    gp.fit(cross.reshape(-1, 1), thresholds)
    sigma_grid, loglike = gp_invert(gp, thresh)

    if isinstance(thresh, float):
        return sigma_grid[np.argmax(loglike)]
    else:
        return np.array(
            [sigma_grid[np.argmax(row)] for row in loglike]
        ).reshape(thresh.shape)


# ---------------------------------------------------------------------------
# Lensing / kappa helpers
# ---------------------------------------------------------------------------

def get_kappa(all_cats, smooth=1, extent=None, correct_for_ngal=False):
    """
    Compute a Kaiser-Squires convergence map from a shear catalogue.

    Parameters
    ----------
    all_cats : dict
        Must contain keys ``"x"``, ``"y"``, ``"gamma1"``, ``"gamma2"``.
    smooth : float
        Gaussian smoothing scale in pixels (0 → no smoothing).
    extent : array-like, optional
        Spatial extent passed to ``lenspack.utils.bin2d``.
    correct_for_ngal : bool
        Up-weight shear in cells with more galaxies.

    Returns
    -------
    kappa_e, kappa_b : np.ndarray
    """
    npix = 100
    e1, e2 = lenspack.utils.bin2d(
        all_cats["x"], all_cats["y"],
        v=(2.0 * all_cats["gamma1"], 2.0 * all_cats["gamma2"]),
        npix=npix, extent=extent,
    )
    ngal = lenspack.utils.bin2d(
        all_cats["x"], all_cats["y"], v=None, npix=npix, extent=extent
    )

    if correct_for_ngal:
        med_ngal = np.median(ngal[ngal != 0])
        e1 *= ngal / med_ngal
        e2 *= ngal / med_ngal

    ke, kb = lenspack.image.inversion.ks93(e1, e2)

    if smooth > 0:
        return gaussian_filter(ke, smooth), gaussian_filter(kb, smooth)
    return ke, kb


def get_latent_space(
    all_models,
    bias=(0, 0),
    quiet=False,
    args=None,
    targets=("source_val", "target_test"),
):
    """
    Extract the penultimate-layer latent representations for both observed
    data and simulation test sets.

    Parameters
    ----------
    all_models : list
        Loaded model objects.
    bias : tuple[float, float]
        Additive bias applied to the two observed-data channels.
    targets : sequence[str]
        Data-loader keys to extract representations for.

    Returns
    -------
    dict with keys ``"all_cross_sections"`` and ``"latent_space"``.
    """
    if args is None:
        args = get_temp_args()

    args.source_domain        = "bahamas_obs"
    args.target_domain        = "darkskies_obs"
    args.apply_intrinsic_ell  = 1.0
    args.train_split          = 0.05
    args.verbose              = False

    obs_meta, obs_data = pkl.load(
        open("../data/100/a2744/obs_data_concat.pkl", "rb")
    )
    obs_data[0, 0, :, :] += bias[0]
    obs_data[0, 1, :, :] += bias[1]

    all_cross    = []
    latent_spaces = []

    # ---------- Observed data (no ground-truth cross-section) ----------
    obs_latents = []
    obs_crosses = []
    for imodel in tqdm(all_models, disable=quiet):
        args.seed = imodel.args.seed
        x = imodel.dynamic_pool(
            torch.tensor(obs_data[0][None, :, :, :], dtype=torch.float32)
        )
        z       = imodel.backbone(x)[:, :, 0, 0]
        latent  = torch.softmax(imodel.classification_head(z), dim=-1)[:, 1:2]
        obs_latents.append(latent)
        obs_crosses.append(torch.tensor([-1]))

    latent_spaces.append(torch.cat(obs_latents))
    all_cross.append(torch.cat(obs_crosses))

    # ---------- Simulation test sets -----------------------------------
    dataloaders = prepare_dataloaders(args)
    with torch.no_grad():
        for imodel in tqdm(all_models, disable=quiet):
            this_latent = []
            this_cross  = []
            args.seed   = imodel.args.seed

            for test_set in targets:
                target_latent = []
                target_cross  = []
                for batch_data in dataloaders[test_set][0]:
                    data, cross_sections, *_ = batch_data
                    x      = imodel.dynamic_pool(data)
                    z      = imodel.backbone(x)[:, :, 0, 0]
                    latent = torch.softmax(
                        imodel.classification_head(z), dim=-1
                    )[:, 1:2]
                    target_latent.append(latent)
                    target_cross.append(cross_sections.cpu())

                this_latent.append(torch.cat(target_latent))
                this_cross.append(torch.cat(target_cross))

            latent_spaces.append(this_latent)
            all_cross.append(this_cross)

    return {"all_cross_sections": all_cross, "latent_space": latent_spaces}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_predictions(
    results,
    calibrate_mean=None,
    samplesizes=(1, 10, 50, 100, 200),
    return_calibrate_mean=False,
    function=np.mean,
    gs=None,
    return_gs=False,
    hist_kwargs=None,
    line_kwargs=None,
):
    """
    Plot per-cross-section probability histograms and detection significance
    as a function of sample size.

    Parameters
    ----------
    results : dict
        Output of :func:`get_probabilities`.
    calibrate_mean : float, optional
        CDM reference mean.  Computed from the CDM probabilities if *None*.
    samplesizes : sequence[int]
        Number of clusters to stack when computing significance.
    return_calibrate_mean : bool
        If *True*, return only the calibrated mean (no plot).
    function : callable
        Aggregation function (default: mean).
    gs, return_gs : GridSpec / bool
        Pass an existing GridSpec or get the new one back.
    """
    hist_kwargs = hist_kwargs or {}
    line_kwargs = line_kwargs or {}

    unique_cross = torch.unique(results["all_cross_sections"])

    if not return_calibrate_mean:
        if gs is None:
            fig = plt.figure(figsize=(len(unique_cross) * 4.0, 3.0))
            gs  = GridSpec(1, len(unique_cross) + 12)
        final_ax  = plt.subplot(gs[-5:])
        single_ax = plt.subplot(gs[-11:-6])

    with torch.no_grad():
        for iax, icross in enumerate(unique_cross):
            nsigma_not_cdm     = []
            var_nsigma_not_cdm = []

            prob = results["probabilities"][
                icross == results["all_cross_sections"], :
            ]

            if icross == 0:
                if calibrate_mean is None:
                    calibrate_mean = function(prob[:, 0].detach().numpy())
                    print(f"Calibrated mean: {calibrate_mean:.4f}")
                if return_calibrate_mean:
                    return calibrate_mean

            ax = plt.subplot(gs[iax])
            hist_kwargs["color"] = COLOR_BY_CROSS[f"{icross:0.2f}"]
            ax.hist(prob[:, 0], bins=np.linspace(0, 1, 20), **hist_kwargs)

            for isamplesize in samplesizes:
                isamplesize = min(isamplesize, prob.shape[0])
                if isamplesize == 1:
                    nsigma = (
                        (calibrate_mean - prob[:, 0].detach().numpy())
                        / np.std(prob[:, 0].detach().numpy())
                    )
                else:
                    nsigma = [
                        (calibrate_mean - function(
                            np.random.choice(prob[:, 0], isamplesize, replace=False)
                        )) / (
                            np.std(np.random.choice(prob[:, 0], isamplesize, replace=False))
                            / np.sqrt(isamplesize)
                        )
                        for _ in range(100)
                    ]

                nsigma_not_cdm.append(np.mean(nsigma))
                var_nsigma_not_cdm.append(np.std(nsigma))

            ylim = ax.get_ylim()[1]
            mean_p = torch.mean(prob[:, 0]).item()
            ax.plot([mean_p, mean_p], [0, 2 * ylim], "k--")
            ax.plot([0.5,   0.5],     [0, 2 * ylim], "k:")
            print(f"{icross:.2f}: mean p = {mean_p:.4f}")
            ax.set_ylim(0, ylim)
            ax.set_yticklabels([])

            line_kwargs.setdefault("fmt", "o-")
            line_kwargs["color"] = COLOR_BY_CROSS[f"{icross:0.2f}"]
            final_ax.errorbar(
                samplesizes, nsigma_not_cdm, yerr=var_nsigma_not_cdm,
                capsize=2, **line_kwargs,
            )
            single_ax.errorbar(
                np.log10(1 + icross), nsigma_not_cdm[0],
                var_nsigma_not_cdm[0], fmt="o-", capsize=2,
            )

    for ref_sigma in (0, 1, 3, 5):
        final_ax.plot([0, 210], [ref_sigma, ref_sigma], "--", color="grey")
    final_ax.set_xlim(-10.0, 210)

    if return_gs:
        return gs


def plot_observations(
    filename,
    ifilter,
    ax=None,
    error_index=0.38,
    noise=False,
    uncertainty=(68,),
    legend=False,
    plot_args=None,
    fill_args=None,
    plotpdf=False,
):
    """
    Overlay observed-data probability constraints on a cross-section plot.

    Parameters
    ----------
    filename : str or list[str]
        Path(s) to pkl files produced by the inference pipeline.
    ifilter : str
        JWST filter key used to index the probability dict.
    noise : bool
        Use the noise-marginalised probabilities.
    uncertainty : sequence[int]
        Confidence levels (68, 95, 99) to shade.
    plotpdf : bool
        Plot a Gaussian PDF instead of a horizontal band.
    """
    plot_args = plot_args or {}
    fill_args = fill_args or {}
    plot_args.setdefault("color", "black")

    if isinstance(filename, list):
        key = 2 if noise else 1
        data = np.concatenate(
            [pkl.load(open(f, "rb"))[key][ifilter] for f in filename]
        )
        if noise:
            data = data.flatten()
        nmodels = data.shape[0] / len(filename)
    else:
        models, probabilities, probabilities_noise = pkl.load(open(filename, "rb"))
        data    = probabilities_noise[ifilter].flatten() if noise else probabilities[ifilter]
        nmodels = data.shape[0]

    if ax is None:
        ax = plt.gca()

    means = 1 - np.mean(data)
    error = return_error_in_mean(data)

    if plotpdf:
        xpdf = np.linspace(0.45, 0.7, 1000)
        ypdf = norm.pdf(xpdf, means, error)
        ax.plot(xpdf, ypdf, color=plot_args["color"])
        ax.fill_between(xpdf, ypdf, np.zeros(1000), **fill_args)
        return

    x_range = np.logspace(-3, 1, 100)
    ax.plot(x_range, np.full(100, means), **plot_args)
    print(f"Observed mean: {means:.4f}  error: {error:.4f}")

    for iunc in uncertainty:
        nsigma_map = {68: 1, 95: 2, 99: 3}
        ns  = nsigma_map.get(iunc, 1)
        lo  = means - error * ns
        hi  = means + error * ns
        print(f"  {iunc}% band: [{lo:.4f}, {hi:.4f}]")

        if ifilter == "concat" and not noise:
            ax.text(0.002, means, "A2744 UNCOVER DATA", ha="left", va="bottom", fontsize=12)
            ax.text(0.002, lo, f"{iunc}\\% Uncertainty",  ha="left", va="bottom", fontsize=12)

        if not noise:
            ax.fill_between(x_range, np.full(100, hi), np.full(100, lo), **fill_args)
            ax.plot(x_range, np.full(100, hi), "--", color=plot_args["color"])
            ax.plot(x_range, np.full(100, lo), "--", color=plot_args["color"])

    if legend:
        ax.legend()


# ---------------------------------------------------------------------------
# Geometry / visualisation utilities
# ---------------------------------------------------------------------------

def get_density(x, y):
    """Evaluate a 2-D KDE on a 200×200 grid covering the data extent."""
    kde = gaussian_kde(np.vstack([x, y]))
    xx, yy = np.mgrid[x.min():x.max():200j, y.min():y.max():200j]
    density = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xx, yy, density


def curly_brace(ax, x1, x2, y, height, upward=True):
    """
    Draw a curly brace between *x1* and *x2* at height *y*.

    Parameters
    ----------
    upward : bool
        If *True* the brace opens upward; otherwise it opens downward.
    """
    mid  = (x1 + x2) / 2
    sign = 1 if upward else -1
    h    = height

    verts = [
        (x1,  y),
        (x1,  y + sign * h / 2),
        (mid, y + sign * h / 2),
        (mid, y + sign * h),
        (mid, y + sign * h / 2),
        (x2,  y + sign * h / 2),
        (x2,  y),
    ]
    codes = [Path.MOVETO] + [Path.CURVE3] * 6
    ax.add_patch(patches.PathPatch(Path(verts, codes), fill=False, lw=2))
