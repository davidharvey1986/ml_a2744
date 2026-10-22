#!/usr/bin/env python
"""
run_and_plot.py
---------------
1. Run get_all_model_results for the tweakedalign_ta models (seeds 1-8)
   into pickles/all_models_concat_nz_alignbest_ta_results.pkl
2. Produce figure2 using the wb1 results (seeds 1-30, tweakedalign_best)
   and save to plots/final_model_weighting_with_data.pdf
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import pickle as pkl
import numpy as np
import scienceplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use(["science", "grid"])

from scipy.stats import norm
from tqdm import tqdm

from add_shear_to_data import get_obs_data, get_model_names, return_error_in_mean
from get_model_probabilities import (
    args, get_variable_redshift_dist, get_probabilities,
    get_threshold_for_cross, get_massfunction_weights,
    get_mass_cut, plot_observations, curly_brace, COLOUR_SCHEME,
)

DATA_DIR   = "../data/100/a2744/"
FILTER_CUTS = {
    "f115w": {"signal_noise_cut": 0, "stat_type": "snr"},
    "f150w": {"signal_noise_cut": 5, "stat_type": "snr"},
}
DOMAIN_MAP = {"tgt": "darkskies_obs", "src": "bahamas_obs"}

# -------------------------------------------------------------------------
# Step 1 – run ta models
# -------------------------------------------------------------------------

def run_ta_models(imonte=5):
    obs_data = get_obs_data("concat", data_dir=DATA_DIR, photoz=True, cuts=FILTER_CUTS)
    fid_n_z  = get_variable_redshift_dist(data_dir=DATA_DIR)
    args.zs            = fid_n_z
    args.source_domain = "darkskies_obs"

    ifilter     = "concat"
    output_name = f"pickles/all_models_{ifilter}_nz_alignbest_ta_results.pkl"

    all_models, seed_index = get_model_names(
        model="tweakedalign_ta_best", include_str="nob1", ifilter=ifilter
    )

    all_results = pkl.load(open(output_name, "rb")) if os.path.isfile(output_name) else {}

    for imodel in tqdm(all_models, desc="ta seeds"):
        seed     = imodel.split("_")[seed_index]
        seed_key = f"seed_{seed}"

        if seed_key not in all_results:
            all_results[seed_key] = {domain: [] for domain in DOMAIN_MAP}

        args.jwst_filter        = ifilter
        args.apply_intrinsic_ell = 1.0
        args.ignore_dataset      = [""]
        args.unbalance           = True

        ndone = len(all_results[seed_key]["src"])

        for _ in range(ndone, imonte):
            missing_mask       = ~np.isfinite(fid_n_z.flatten())
            candidate_zs       = obs_data["redshift"][
                np.abs(obs_data["redshift"] - args.default_zs) > 1e-3
            ]
            fill_zs = candidate_zs[
                np.random.randint(0, candidate_zs.shape[0], size=missing_mask.sum())
            ]
            n_z = fid_n_z.copy()
            n_z[~np.isfinite(fid_n_z)] = fill_zs
            args.zs = n_z

            for domain_key, target_domain in DOMAIN_MAP.items():
                results = get_probabilities(target_domain, [imodel], args, quiet=True)
                results.pop("data_loaders", None)
                all_results[seed_key][domain_key].append(results)

        pkl.dump(all_results, open(output_name, "wb"))

    print(f"Saved ta results → {output_name}")
    return output_name


# -------------------------------------------------------------------------
# Step 2 – figure 2 (main result plot)
# -------------------------------------------------------------------------

def figure2(results_file="pickles/all_models_concat_nz_alignbest_wb1_results.pkl"):
    cdm = 3e-3
    ifilter = "concat"

    fig, axarr = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    postive_mass, mass_cut = get_mass_cut(ifilter, nsigma=2, study="harvey")
    mass_cut = list(mass_cut)

    domain = {"tgt": "darkskies_obs", "src": "bahamas_obs"}

    all_results = pkl.load(open(results_file, "rb"))

    cdm_vals    = []
    cdm_val_err = []
    npts        = 10000
    xpdf        = np.linspace(0.35, 0.7, npts)
    marginalised_pdf = np.zeros(npts)

    for target in ["src", "tgt"]:
        all_thresholds = []
        domain_name    = domain[target].split("_")[0]

        mass_weights = get_massfunction_weights(domain[target].split("_")[0])

        for imodel in all_results.keys():
            if "seed" not in imodel:
                continue
            tgt = get_threshold_for_cross(
                all_results[imodel][target],
                mass_cut=mass_cut,
                integrated_mass=True,
                function=np.mean,
                mass_weights=mass_weights,
                dataset=domain[target].split("_")[0],
                ncomponents=1,
                quiet=True,
            )
            all_thresholds.append(tgt["thresholds"])

        all_thresholds = np.array(all_thresholds)
        means  = np.mean(all_thresholds, axis=0)
        errors = return_error_in_mean(all_thresholds)

        cdm_vals.append(1 - means[0])
        cdm_val_err.append(errors[0])

        colour = COLOUR_SCHEME[domain_name.lower()]

        if len(tgt["cross_sections"]) > 1:
            tgt["cross_sections"][0] += cdm
            axarr[0].errorbar(
                tgt["cross_sections"], 1 - means, errors,
                fmt="-o", capsize=2, markersize=8, color=colour,
            )
            axarr[0].plot(
                tgt["cross_sections"], 1 - means, "o",
                label=domain_name.upper(), markersize=8, color=colour,
            )
            axarr[0].errorbar(
                cdm, 1 - means[0], errors[0],
                fmt="*", capsize=2, markersize=15, color=colour,
            )
            axarr[0].plot(cdm, 1 - means[0], "*", markersize=15, color=colour)

        ypdf = norm.pdf(xpdf, 1 - means[0], errors[0])
        marginalised_pdf += ypdf
        axarr[1].plot(xpdf, ypdf, color=colour)
        axarr[1].fill_between(
            xpdf, ypdf, np.zeros(npts),
            color=colour, alpha=0.2, label=domain_name.upper(),
        )

    marginalised_pdf /= np.sum(marginalised_pdf) * (xpdf[1] - xpdf[0])
    axarr[1].plot(xpdf, marginalised_pdf, lw=2, ls="--", color="k", label="Marginalised CDM")
    axarr[1].fill_between(xpdf, marginalised_pdf, np.zeros(npts), color="k", alpha=0.2)

    axarr[0].set_xscale("log")
    axarr[0].set_xlabel(r"Self-interaction cross-section $[\mathrm{cm}^2\,\mathrm{g}^{-1}]$", fontsize=15)
    axarr[0].set_ylabel("Model output", fontsize=15)
    axarr[0].set_xlim(1e-3, 1.5)

    plot_observations(
        "pickles/model_on_data.pkl", ifilter, ax=axarr[0],
        plot_args={"lw": 2, "color": "k"},
        fill_args={"alpha": 0.2, "color": "k"},
        uncertainty=[68, 95],
    )
    axarr[0].legend(loc=2, ncols=2)

    plot_observations(
        "pickles/model_on_data.pkl", ifilter, ax=axarr[1],
        plot_args={"lw": 2, "color": "k"},
        fill_args={"alpha": 0.1, "color": "k"},
        plotpdf=True,
    )

    axarr[1].set_xlabel("Model output", fontsize=15)
    axarr[1].set_ylabel("Probability distribution", fontsize=15)
    axarr[1].set_xlim(0.4, 0.7)
    axarr[1].set_ylim(0, 40)
    axarr[1].grid(False)

    edges = [np.min(cdm_vals) - 0.02, np.max(cdm_vals) + 0.02]
    curly_brace(axarr[1], edges[0], edges[1], 22, 3, upward=True)
    axarr[1].text(np.mean(edges), 25, "Collisionless\nDark Matter", ha="center", fontsize=12)

    text_center = 1 - np.mean(pkl.load(open("pickles/model_on_data.pkl", "rb"))[1]["concat"])
    axarr[1].text(text_center, 20, "Observation\n(A2744 UNCOVER)", ha="center", fontsize=12)

    axarr[1].legend(ncols=2)
    fig.align_xlabels()

    if not os.path.isdir("plots"):
        os.makedirs("plots")
    outfile = "plots/final_model_weighting_with_data_wb1.pdf"
    plt.savefig(outfile, dpi=200)
    print(f"Saved figure → {outfile}")

    # Print significance
    models, prob, noise = pkl.load(open("pickles/model_on_data.pkl", "rb"))
    obsprob = 1 - prob["concat"]
    obsstd  = return_error_in_mean(obsprob)
    peak    = xpdf[np.argmax(marginalised_pdf)]
    margstd = np.sqrt(
        np.sum((xpdf - peak) ** 2 * marginalised_pdf) / np.sum(marginalised_pdf)
    )
    total_sigma  = np.sqrt(margstd ** 2 + obsstd ** 2)
    significance = (np.mean(obsprob) - peak) / total_sigma
    ot_prob      = (1 - norm.cdf(xpdf, peak, total_sigma))[
        np.argmin(np.abs(xpdf - (1 - np.mean(prob["concat"]))))
    ]
    sigmas     = np.linspace(1, 6, 1000)
    sig_formal = sigmas[np.argmin(np.abs(norm.sf(sigmas) - ot_prob))]
    print(f"Significance: {sig_formal:.2f} sigma  (one-tailed p = {ot_prob:.4f})")
    print(f"Inter-CDM variance: {np.std(cdm_vals):.4f},  mean intra-CDM error: {np.mean(cdm_val_err):.4f}")
    print(f"Discrepancy: {np.mean(obsprob) - peak:.4f}")

    return outfile


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-ta", action="store_true", help="Skip running ta models")
    ap.add_argument("--imonte", type=int, default=5)
    opts = ap.parse_args()

    if not opts.skip_ta:
        print("=== Running ta models ===")
        run_ta_models(imonte=opts.imonte)

    print("\n=== Generating figure 2 ===")
    outfile = figure2()
    print(f"\nDone. Plot at: {outfile}")
