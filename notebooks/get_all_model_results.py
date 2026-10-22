#!/usr/bin/env python
"""
get_all_model_results.py
------------------------
Run all trained models over observed and simulated data and cache the
probability outputs to disk.

Relies on helpers defined in get_model_probabilities.py:
  - get_variable_redshift_dist
  - get_probabilities
  - args  (global configuration class)
"""

import os
import pickle as pkl

import numpy as np
import scienceplots
import matplotlib.pyplot as plt
from tqdm import tqdm

from add_shear_to_data import get_obs_data, get_model_names
from get_model_probabilities import args, get_variable_redshift_dist, get_probabilities

plt.style.use(["science", "grid"])


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

DATA_DIR = "../data/100/a2744/"
FILTER_CUTS = {
    "f115w": {"signal_noise_cut": 0,  "stat_type": "snr"},
    "f150w": {"signal_noise_cut": 5,  "stat_type": "snr"},
}
DOMAIN_MAP = {
    "tgt": "darkskies_obs",
    "src": "bahamas_obs",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(imonte: int = 5) -> None:
    """
    For every trained model (identified by seed) run *imonte* Monte-Carlo
    draws of the source-redshift distribution and evaluate the model on both
    target and source domains.  Results are accumulated in a per-filter
    pickle that is updated after every seed so progress is never lost.

    Parameters
    ----------
    imonte : int
        Number of Monte-Carlo realisations of the source-redshift map to run
        per model.  Draws that are already present in the output file are
        skipped so the script is safe to re-run.
    """

    # ------------------------------------------------------------------
    # Observed catalogue (used only to sample photo-z for empty pixels)
    # ------------------------------------------------------------------
    obs_data = get_obs_data(
        "concat",
        data_dir=DATA_DIR,
        photoz=True,
        cuts=FILTER_CUTS,
    )

    # Fiducial spatially-resolved redshift map (pixels without a galaxy
    # are NaN and will be filled by random draws from the photo-z catalogue)
    fid_n_z = get_variable_redshift_dist(data_dir=DATA_DIR)

    # Propagate the redshift distribution and domain to the global args
    args.zs = fid_n_z
    args.source_domain = "darkskies_obs"

    # ------------------------------------------------------------------
    # Loop over filters
    # ------------------------------------------------------------------
    for ifilter in ["concat"]:

        all_models, seed_index = get_model_names(
            model="tweakedalign_best",
            include_str="nob1",
            ifilter=ifilter,
        )

        output_name = (
            f"pickles/all_models_{ifilter}_nz_alignbest_wb1_results.pkl"
        )

        # Load existing results so we can resume interrupted runs
        all_results: dict = (
            pkl.load(open(output_name, "rb"))
            if os.path.isfile(output_name)
            else {}
        )

        # ------------------------------------------------------------------
        # Loop over individual model checkpoints
        # ------------------------------------------------------------------
        for imodel in tqdm(all_models, desc=f"[{ifilter}] seeds"):

            seed = imodel.split("_")[seed_index]
            seed_key = f"seed_{seed}"

            # Initialise entry for this seed if not already present
            if seed_key not in all_results:
                all_results[seed_key] = {domain: [] for domain in DOMAIN_MAP}

            # Configure filter and intrinsic ellipticity flag
            args.jwst_filter = ifilter
            args.apply_intrinsic_ell = 1.0
            args.ignore_dataset = [""]   # evaluate on all datasets at test time
            args.unbalance = True

            # Number of MC draws already completed for this seed
            ndone = len(all_results[seed_key]["src"])

            # ------------------------------------------------------------------
            # Monte-Carlo loop over redshift realisations
            # ------------------------------------------------------------------
            for _ in range(ndone, imonte):

                # Sample photo-zs for pixels that have no observed galaxy
                missing_mask = ~np.isfinite(fid_n_z.flatten())
                candidate_redshifts = obs_data["redshift"][
                    np.abs(obs_data["redshift"] - args.default_zs) > 1e-3
                ]
                fill_redshifts = candidate_redshifts[
                    np.random.randint(
                        low=0,
                        high=candidate_redshifts.shape[0],
                        size=missing_mask.sum(),
                    )
                ]

                n_z = fid_n_z.copy()
                n_z[~np.isfinite(fid_n_z)] = fill_redshifts
                args.zs = n_z

                # Evaluate on each domain (target + source)
                for domain_key, target_domain in DOMAIN_MAP.items():
                    results = get_probabilities(
                        target_domain,
                        [imodel],
                        args,
                        quiet=True,
                    )
                    # Drop the data-loader reference — it holds GPU tensors
                    # and cannot be pickled reliably
                    results.pop("data_loaders", None)
                    all_results[seed_key][domain_key].append(results)

            # Persist after every seed so progress survives interruption
            pkl.dump(all_results, open(output_name, "wb"))


if __name__ == "__main__":
    main()
