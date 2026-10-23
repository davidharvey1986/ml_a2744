#!/usr/bin/env python
"""
Build the inputs for simulation-based inference of sigma/m from the trained CNNs.

For every model (default: the 30 ``tweakedalign_best`` seeds) and every simulated
cluster in data/100/obs/<filter>/{bahamas,darkskies}*.pkl this saves

    latent  : the 512-d backbone features fed to the classification head
    softmax : P(SIDM) = softmax(classification)[:, 1]

together with the parameters of each cluster

    cross_section    : sigma/m [cm^2/g]
    log10_mass       : background-subtracted projected mass in the 2 Mpc stamp [Msun]
    log10_mass_750   : the same inside a 750 kpc aperture (the published A2744 definition)
    ncomponents      : number of merging components (mass ratio < 10)

and the same latent / softmax for A2744 (obs_data_<filter>.pkl: index 0 is the
data, the rest are random-rotation noise maps).

Rows are identical for every model: all clusters are used (including bahamas_1,
which no model was trained on) and the shape noise is drawn once per realisation
and shared by all models. ``held_out[m]`` flags the clusters model m did not see
in either training stage, reconstructed from create_balanced_test_set (this
assumes glob returns the files in the same order as at training time).

Outputs (--out-dir, default sbi/summaries/)
-------
params_<domain>.npz   cross_section, log10_mass, log10_mass_750, ncomponents,
                      catalogue_mass, filename, image_idx  (N,)
                      held_out (n_models, N), seeds (n_models,)
softmax_<domain>.npy  (n_models, n_noise, N) float32
latents_<domain>.npy  (n_models, n_noise, N, 512) float32   [large, not in git]
obs.npz               softmax (n_models, n_obs), latents (n_models, n_obs, 512)
info.json             models, settings and input files

With --legacy-dir the same numbers are also written in the format read by
Dark_Matter_DA/sbi (latents.pkl, theta.pkl, obs_latents.pkl, latent_softmax.pkl,
latent_meta_fixed.pkl), using noise realisation 0.

Usage
-----
    cd ml_check && astro-env/bin/python sbi/make_summaries.py
    astro-env/bin/python sbi/make_summaries.py --nmodels 2 --out-dir sbi/test   # quick check
"""

import argparse
import datetime
import glob
import json
import os
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks"
sys.path[:0] = [str(REPO), str(NOTEBOOKS)]
LAUNCH_DIR = Path.cwd()   # command-line paths are relative to where the script was run
# get_model_probabilities (and the default args) use paths relative to notebooks/
os.chdir(NOTEBOOKS)

from add_shear_to_data import (  # noqa: E402
    get_background_sigma,
    get_model_names,
    get_num_merging_components,
    get_surface_density_norms,
)
from dataset import (  # noqa: E402
    BalancedClusterDataset,
    create_balanced_test_set,
    get_cross_section_from_filename,
    get_test_transform,
)
from get_model_probabilities import args as default_args, get_variable_redshift_dist  # noqa: E402
from model import create_model  # noqa: E402

DOMAINS = ["bahamas", "darkskies"]        # same order as the old latent_space[m][d]
PIXEL_PC = 20e3                           # 20 kpc pixels
APERTURE_KPC = 750.
_vec = (np.arange(100) - 49.5) * 20.
RGRID = np.hypot(*np.meshgrid(_vec, _vec))  # kpc from the stamp centre


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tweakedalign_best", help="checkpoint suffix passed to get_model_names")
    p.add_argument("--include-str", default="nob1")
    p.add_argument("--nmodels", type=int, default=None)
    p.add_argument("--filter", default="concat")
    p.add_argument("--n-noise", type=int, default=1, help="shape-noise realisations per cluster")
    p.add_argument("--noise-seed", type=int, default=1234)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "summaries"))
    p.add_argument("--legacy-dir", default=None, help="also write the old Dark_Matter_DA/sbi pickles here")
    return p.parse_args()


def setup_args(ifilter):
    args = default_args
    args.jwst_filter = ifilter
    args.apply_intrinsic_ell = 1.0
    args.ignore_dataset = [""]
    args.verbose = False
    args.num_workers = 0
    # spatially varying source redshift, as for the published results
    args.zs = get_variable_redshift_dist(data_dir="../data/100/a2744/", ifilter=ifilter)
    return args


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_models(opts, args, device):
    names, _ = get_model_names(
        ifilter=opts.filter, model_dir=str(REPO / "models"), model=opts.model,
        include_str=opts.include_str, nmodels=opts.nmodels,
    )
    models = []
    for name in sorted(names):
        args.checkpoint = name
        imodel = create_model(args).to(device).eval()
        assert getattr(imodel, "num_meta_info", 0) == 0, "image-only models expected"
        imodel.checkpoint_name = os.path.basename(name)
        models.append(imodel)
    models.sort(key=lambda m: m.args.seed)
    return models


@torch.no_grad()
def embed(imodel, x):
    '''Same as model.forward but without the squeeze that breaks batches of one.'''
    if imodel.num_avgpool_head > 0:
        x = imodel.dynamic_pool(x)
    features = imodel.backbone(x).flatten(1)
    logits = imodel.classification_head(features)
    return features, torch.softmax(logits, dim=1)[:, 1]


def run_models(models, images, device, batch_size):
    '''images (N, 2, 100, 100) -> latents (n_models, N, 512), p_sidm (n_models, N)'''
    latents, p_sidm = [], []
    for imodel in tqdm(models, desc="models", leave=False):
        out_z, out_p = [], []
        for start in range(0, len(images), batch_size):
            z, p = embed(imodel, images[start:start+batch_size].to(device))
            out_z.append(z.cpu())
            out_p.append(p.cpu())
        latents.append(torch.cat(out_z))
        p_sidm.append(torch.cat(out_p))
    return torch.stack(latents).numpy(), torch.stack(p_sidm).numpy()


def noisy_images(dataset, seed):
    '''Apply the test transform (shape noise, z-rescaling) once, reproducibly.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.stack([dataset[i][0] for i in tqdm(range(len(dataset)), desc="noise", leave=False)])


def cluster_params(filenames, image_idx, ifilter):
    '''Mass, aperture mass and n-components for each row, from the convergence maps.'''
    n = len(filenames)
    out = {k: np.full(n, np.nan) for k in ["cross_section", "log10_mass", "log10_mass_750",
                                           "ncomponents", "catalogue_mass"]}
    for fname in np.unique(filenames):
        rows = np.where(filenames == fname)[0]
        idx = image_idx[rows]
        conv = f"../data/100/convergence/{fname}"
        meta, data = pkl.load(open(conv, "rb"))
        if 'redshift' not in meta:
            meta['redshift'] = np.full(data.shape[0], 0.305)

        # Same surface density as add_shear_to_data.main(sample_data=True)
        norms = get_surface_density_norms(meta, conv)[:, 0] * 1e-12       # Msun / pc^2
        background = get_background_sigma(meta, conv, meta['redshift']).value
        sigma = data[idx, 0] * norms[idx, None, None] - background[idx, None, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            out["log10_mass"][rows] = np.log10(sigma.sum((-1, -2)) * PIXEL_PC**2)
            out["log10_mass_750"][rows] = np.log10(sigma[:, RGRID < APERTURE_KPC].sum(-1) * PIXEL_PC**2)

        ncomp = get_num_merging_components(conv, components_dir=str(NOTEBOOKS / "pickles"))
        assert len(ncomp) == data.shape[0], f"{fname}: {len(ncomp)} components vs {data.shape[0]} maps"
        out["ncomponents"][rows] = ncomp[idx]
        out["cross_section"][rows] = get_cross_section_from_filename(fname)
        if 'mass' in meta:
            out["catalogue_mass"][rows] = np.asarray(meta['mass'], dtype=float)[idx]
    return out


class _Labels:
    '''Stand-in for a dataset in create_balanced_test_set (only needs labels).'''
    def __init__(self, cross_sections):
        self.cross_sections = np.asarray(cross_sections)

    def __len__(self):
        return len(self.cross_sections)


def training_test_set(pattern, counts, train_args):
    '''(filename, image_idx) pairs in the test split for one training stage.'''
    files = [os.path.basename(f) for f in glob.glob(os.path.join(default_args.data_dir, pattern))]
    files = [f for f in files if f not in getattr(train_args, "ignore_dataset", [""])]
    pairs = [(f, i) for f in files for i in range(counts[f])]
    labels = [get_cross_section_from_filename(f) for f, _ in pairs]
    test = create_balanced_test_set(
        _Labels(labels), test_size=1 - getattr(train_args, "train_split", 0.8),
        random_state=train_args.seed, verbose=False,
        unbalance=getattr(train_args, "unbalance", False),
    )
    return {pairs[i] for i in test}


def held_out_mask(models, domain, filenames, image_idx, ifilter):
    '''
    True where a cluster was in the test split of both the base and fine-tune
    stage, or in a file excluded from training (ignore_dataset, e.g. bahamas_1).
    '''
    counts = {f: int(np.sum(filenames == f)) for f in np.unique(filenames)}
    rows = list(zip(filenames, image_idx))
    mask = np.zeros((len(models), len(rows)), dtype=bool)
    for m, imodel in enumerate(models):
        ignored = set(getattr(imodel.args, "ignore_dataset", []))
        base = training_test_set(f"100/shear/{domain}*", counts, imodel.args)
        tune = training_test_set(f"100/obs/{ifilter}/{domain}*", counts, imodel.args)
        mask[m] = [r[0] in ignored or (r in base and r in tune) for r in rows]
    return mask


def run_domain(domain, models, args, opts, out_dir, device):
    dataset = BalancedClusterDataset(
        args.data_dir, f"{args.image_size}/obs/{opts.filter}/{domain}*",
        transform=get_test_transform(args), args=args, dtype="image",
    )
    filenames = np.array([dataset.files_data[f]["filename"] for f, _ in dataset.file_indices])
    image_idx = np.array([i for _, i in dataset.file_indices])
    n = len(dataset)
    print(f"{domain}: {n} clusters from {sorted(set(filenames))}")

    params = cluster_params(filenames, image_idx, opts.filter)
    assert np.allclose(params["cross_section"], dataset.cross_sections)
    params["held_out"] = held_out_mask(models, domain, filenames, image_idx, opts.filter)
    print(f"  held out per model: {params['held_out'].sum(1).min()}-{params['held_out'].sum(1).max()} of {n}")

    softmax = np.zeros((len(models), opts.n_noise, n), dtype=np.float32)
    latents = np.lib.format.open_memmap(
        out_dir / f"latents_{domain}.npy", mode="w+", dtype=np.float32,
        shape=(len(models), opts.n_noise, n, 512),
    )
    for r in range(opts.n_noise):
        images = noisy_images(dataset, opts.noise_seed + r)
        latents[:, r], softmax[:, r] = run_models(models, images, device, opts.batch_size)
    latents.flush()

    np.save(out_dir / f"softmax_{domain}.npy", softmax)
    np.savez(out_dir / f"params_{domain}.npz", filename=filenames, image_idx=image_idx,
             seeds=np.array([imodel.args.seed for imodel in models]), **params)
    return params, softmax


def run_obs(models, opts, out_dir, device):
    obs_file = f"../data/100/a2744/obs_data_{opts.filter}.pkl"
    _, obs = pkl.load(open(obs_file, "rb"))
    latents, softmax = run_models(models, torch.tensor(obs, dtype=torch.float32), device, opts.batch_size)
    np.savez(out_dir / "obs.npz", latents=latents, softmax=softmax)
    print(f"A2744: ensemble P(SIDM) = {softmax[:, 0].mean():.4f} +/- {softmax[:, 0].std():.4f} (model scatter)")
    return obs_file


def write_legacy(legacy_dir, out_dir, models):
    '''Old Dark_Matter_DA/sbi inputs, rows aligned between all files.'''
    legacy_dir.mkdir(parents=True, exist_ok=True)
    P = {d: dict(np.load(out_dir / f"params_{d}.npz")) for d in DOMAINS}
    S = {d: np.load(out_dir / f"softmax_{d}.npy")[:, 0] for d in DOMAINS}
    Z = {d: np.load(out_dir / f"latents_{d}.npy", mmap_mode="r")[:, 0] for d in DOMAINS}
    obs = np.load(out_dir / "obs.npz")
    nm = len(models)

    dump = lambda obj, name: pkl.dump(obj, open(legacy_dir / name, "wb"))
    dump([[np.asarray(Z[d][m]) for d in DOMAINS] for m in range(nm)], "latents.pkl")
    dump([[P[d]["cross_section"][:, None] for d in DOMAINS] for m in range(nm)], "theta.pkl")
    dump([[obs["latents"][m, 0] for d in DOMAINS] for m in range(nm)], "obs_latents.pkl")
    # latent_space[0][d] is the ensemble-mean P(SIDM) of A2744 (previously one model's value)
    obs_mean = np.array([obs["softmax"][:, 0].mean()], dtype=np.float32)
    dump({
        "latent_space": [[obs_mean for d in DOMAINS]] + [[S[d][m][:, None] for d in DOMAINS] for m in range(nm)],
        "all_cross_sections": [[np.array([-1.]) for d in DOMAINS]] + [[P[d]["cross_section"] for d in DOMAINS]] * nm,
    }, "latent_softmax.pkl")
    dump({d: {k: P[d][k] for k in ["image_idx", "log10_mass", "log10_mass_750", "ncomponents", "cross_section"]}
          for d in DOMAINS}, "latent_meta_fixed.pkl")
    print(f"wrote legacy pickles to {legacy_dir}")


def main():
    opts = parse()
    out_dir = LAUNCH_DIR / opts.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    args = setup_args(opts.filter)
    models = load_models(opts, args, device)
    print(f"{len(models)} models, seeds {[m.args.seed for m in models]}, device {device}")

    for domain in DOMAINS:
        run_domain(domain, models, args, opts, out_dir, device)
    obs_file = run_obs(models, opts, out_dir, device)

    info = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "models": [m.checkpoint_name for m in models],
        "seeds": [m.args.seed for m in models],
        "options": vars(opts),
        "obs_file": obs_file,
        "softmax": "P(SIDM) = softmax(classification)[:, 1]",
        "mass": "log10 sum(Sigma - background) * (20 kpc)^2, from data/100/convergence",
        "inputs": {f: datetime.datetime.fromtimestamp(os.path.getmtime(f)).isoformat(timespec="seconds")
                   for f in sorted(glob.glob(f"../data/100/obs/{opts.filter}/*.pkl")) + [obs_file]},
    }
    json.dump(info, open(out_dir / "info.json", "w"), indent=2)

    if opts.legacy_dir:
        write_legacy(LAUNCH_DIR / opts.legacy_dir, out_dir, models)


if __name__ == "__main__":
    main()
