#!/usr/bin/env python
"""
data_sensitivity_map.py
-----------------------
Script version of data_sensitivity_map.ipynb: where in the map are the CNNs sensitive?

A window of segment_size x segment_size pixels is slid (stride 1) over a map and filled
with the median of that map, channel by channel. For every window and model

    Delta = P(SIDM | full map) - P(SIDM | window filled)

so Delta > 0 means that region pushes the output towards SIDM. As in the notebook:

  * A2744: map 0 of ../data/100/a2744/obs_data_concat.pkl
  * the A2744 analogue (pickles/a2744_analogue.pkl, a CDM and an SIDM cluster) and the
    first --nclusters clusters of each BAHAMAS cross-section, forward modelled with the
    test transform with no shape noise (apply_intrinsic_ell = 0) and zs = 1.65
  * Figure 1 (plots/sensitivity_map.pdf): mean Delta over models in units of its standard
    deviation, for the analogue and A2744, over the convergence (Kaiser-Squires for A2744,
    true kappa for the analogue)
  * Figure 2 (plots/Sensitivty_relation.pdf): fraction of pixels with Delta > 0 as a
    function of convergence (true kappa for the simulations, the Kaiser-Squires map of the
    data for A2744)

Fixes relative to the notebook:
  * one model group for every map (--model; the notebook mixed tweakedalign_best,
    'tweaked_final', which no longer matches any file, and 'final', which matches all
    *_final groups at once), models sorted so --nmodels picks the same ones every run
  * simulated maps normalised as in training (BalancedClusterDataset: subtract the min,
    divide by the max): the BAHAMAS maps were not normalised at all and the analogue was
    divided by its range
  * same reference for A2744 and the simulations in Figure 2 (the unoccluded output; the
    notebook used it for A2744 but the median over the map for the simulations), and the
    same fill (per-channel median of the map itself; the notebook's A2744 fill was the
    median over the data and a noise-only map together)
  * same segment size for A2744 and the simulations in Figure 2 (the notebook read a
    2-pixel non-overlapping A2744 map against 1-pixel simulated ones) and one Delta
    normalisation in Figure 1 (the simulated panels were divided by std / 2)
  * moving-window maps (size 101 - s) are drawn at the window centres and padded onto the
    100 x 100 grid instead of being stretched with zoom
  * binomial errors sqrt(f (1 - f) / n) (the notebook had sqrt(f) / sqrt(n)), and the
    A2744 permutation errors are computed on the fraction, not on the mean of +/-1
  * bahamas_1 (the notebook asked for bahamas_1.0, which does not exist), n_comp from the
    component files (meta has no 'ncomponents'), and the A2744 catalogue read with the
    shear cuts that obs_data_concat.pkl was made with (get_obs_data also rewrites the
    filtered catalogue, so the default cuts overwrote it)

Outputs: pickles/sensitivity_<model>/*.npz (cached; --recompute to redo) and the two
figures in plots/.

Usage (from anywhere):
    astro-env/bin/python notebooks/data_sensitivity_map.py
    astro-env/bin/python notebooks/data_sensitivity_map.py --model tweakedalign_ba_best --nclusters 100
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)                                  # the helpers use paths relative to notebooks/
sys.path.insert(0, str(HERE.parent))

from get_model_probabilities import *          # noqa: E402,F401,F403  (as the notebook)
from scipy.stats import binned_statistic      # noqa: E402
from add_shear_to_data import SHEAR_CUTS, get_model_names, get_num_merging_components  # noqa: E402
from dataset import get_test_transform        # noqa: E402
import scienceplots                            # noqa: E402,F401
from pathlib import Path                       # noqa: E402  (the * import brings matplotlib's Path)

plt.style.use(["science", "grid"])

DATASETS = ["bahamas_cdm", "bahamas_0.1", "bahamas_0.3", "bahamas_1"]
H = 0.7                                         # the analogue kappa is divided by h


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tweakedalign_best", help="model group, models/concat/*<model>.pth")
    p.add_argument("--include-str", default="nob1")
    p.add_argument("--nmodels", type=int, default=5)
    p.add_argument("--map-segment", type=int, default=4, help="window size for the sensitivity maps (Figure 1)")
    p.add_argument("--relation-segment", type=int, default=1, help="window size for Figure 2")
    p.add_argument("--nclusters", type=int, default=40, help="first n clusters of each BAHAMAS file")
    p.add_argument("--min-ncomp", type=int, default=0, help="Figure 2 uses clusters with n_comp > this")
    p.add_argument("--analogue", default="pickles/a2744_analogue.pkl")
    p.add_argument("--zs", type=float, default=1.65)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--recompute", action="store_true")
    return p.parse_args()


# ── models ─────────────────────────────────────────────────────────────────────
def load_models(opts, device):
    names, _ = get_model_names("concat", model=opts.model, include_str=opts.include_str)
    names = sorted(names)[:opts.nmodels]
    models = []
    for name in names:
        args.checkpoint = name
        models.append(create_model(args).to(device).eval())
    print(f"Found {len(models)} models ({opts.model})")
    return models


@torch.no_grad()
def p_sidm(models, x):
    '''P(SIDM) = 1 - softmax[:, 0] for each model, (n_models, n). Model.forward without its squeeze.'''
    out = []
    for m in models:
        y = m.dynamic_pool(x) if m.num_avgpool_head > 0 else x
        logits = m.classification_head(m.backbone(y).flatten(1))
        out.append(torch.softmax(logits, dim=1)[:, 1])
    return torch.stack(out).cpu().numpy()


def occlusion(models, image, segment_size, device, batch_size):
    '''
    P(SIDM) for the full map, (n_models,), and with each segment_size window filled
    with the per-channel median of the map, (n, n, n_models) with n = 101 - segment_size.
    Rows of the output follow the rows of the image, as in the notebook.
    '''
    image = torch.as_tensor(image, dtype=torch.float32).to(device)
    fill = image.flatten(1).median(dim=1).values[:, None, None]
    n = image.shape[-1] - segment_size + 1
    corners = [(i, j) for i in range(n) for j in range(n)]
    out = []
    for start in range(0, len(corners), batch_size):
        chunk = corners[start:start + batch_size]
        batch = image.repeat(len(chunk), 1, 1, 1)
        for k, (i, j) in enumerate(chunk):
            batch[k, :, i:i + segment_size, j:j + segment_size] = fill
        out.append(p_sidm(models, batch))
    occluded = np.concatenate(out, axis=1).T.reshape(n, n, len(models))
    return p_sidm(models, image[None])[:, 0], occluded


def delta_map(p_full, p_occ):
    '''Mean over models of P(SIDM | full) - P(SIDM | occluded), (n, n).'''
    return np.mean(p_full - p_occ, axis=-1)


def to_grid(m, segment_size, npix=100):
    '''A moving-window map (n = npix + 1 - segment_size) onto the npix grid, each value at its window centre.'''
    lo = (segment_size - 1) // 2
    return np.pad(m, ((lo, segment_size - 1 - lo),) * 2, mode="edge")


def window_extent(segment_size, half=50):
    '''imshow extent that puts each window value at its centre.'''
    c = half - segment_size / 2.
    return [-c, c, -c, c]


# ── inputs ─────────────────────────────────────────────────────────────────────
def normalise(t):
    '''As BalancedClusterDataset.__getitem__: subtract the min, divide by the (original) max.'''
    mx = torch.max(t)
    return (t - torch.min(t)) / mx


def forward_model(shear, transform):
    return normalise(transform(torch.tensor(shear, dtype=torch.float32))).numpy()


def obs_kappa(correct_for_ngal):
    obs_data = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS)
    obs_cat = bin_obs_data(obs_data)
    obs_data["x"], obs_data["y"] = obs_cat["delta_ra"], obs_cat["delta_dec"]
    ke, _ = get_kappa(obs_data, smooth=1.5, correct_for_ngal=correct_for_ngal, extent=[-50, 50, -50, 50])
    return ke


def cached(path, recompute, make):
    if path.exists() and not recompute:
        return dict(np.load(path))
    res = make()
    np.savez(path, **res)
    return res


# ── figures ────────────────────────────────────────────────────────────────────
def annotate(ax, label):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.errorbar(15 - 50, 5 - 50, xerr=10, capsize=2, color="white", lw=2, capthick=2)
    ax.text(15 - 50, 5 - 50, "200kpc", ha="center", va="bottom", color="white", weight="bold")
    ax.plot([80 - 50, 90 - 50], [85 - 50, 85 - 50], "-", color="white", lw=2)
    ax.plot([90 - 50, 90 - 50], [95 - 50, 85 - 50], "-", color="white", lw=2)
    ax.text(85 - 50, 84 - 50, "E", ha="center", va="top", color="white", weight="bold")
    ax.text(91 - 50, 90 - 50, "N", ha="left", va="center", color="white", weight="bold")
    ax.text(5 - 50, 40, label, color="white", weight="bold", fontsize=20)


def save(fig, fname):
    fig.savefig(fname)
    if shutil.which("pdfcrop"):
        os.system(f"pdfcrop {fname} {fname} > /dev/null")
    print(f"wrote {fname}")


def plot_sensitivity_map(analogue, analogue_res, obs_res, obs_kappa_e, segment_size, fname):
    cmap_contour, cmap_image = "YlOrRd", "viridis"
    kappa_bins = np.linspace(0.1, 0.8, 8)
    vmin, vmax = 2, 5
    extent = window_extent(segment_size)

    fig, axarr = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.1)

    for ax, key, label in zip(axarr, ["cdm", "sidm"], ["Collisionless", "SIDM0.2"]):
        s = delta_map(analogue_res[f"{key}_full"], analogue_res[f"{key}_occ"])
        s /= s.std()
        ax.imshow(s, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap_image)
        ax.contour(gaussian_filter(analogue[key]["kappa"] / H, 1.5), origin="lower", extent=[-50, 50, -50, 50],
                   cmap=cmap_contour, levels=np.linspace(0., 1.0, 20))
        annotate(ax, label)

    ax = axarr[-1]
    s = delta_map(obs_res["full"], obs_res["occ"])
    s /= s.std()
    prob = ax.imshow(s, origin="lower", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap_image)
    kappa = ax.contourf(zoom(obs_kappa_e, 2), origin="lower", extent=[-50, 50, -50, 50], cmap=cmap_contour,
                        levels=kappa_bins, alpha=0.01)
    ax.contour(zoom(obs_kappa_e, 2), origin="lower", extent=[-50, 50, -50, 50], cmap=cmap_contour, levels=kappa_bins)
    annotate(ax, "Observation")

    fraction = 0.015
    cbar = fig.colorbar(prob, ax=axarr, orientation="vertical", fraction=fraction * 0.96, pad=0.025)
    cbar.ax.set_title("$S$")
    cbar = fig.colorbar(kappa, ax=axarr, orientation="vertical", fraction=fraction, pad=0.01)
    cbar.ax.set_title(r"$\kappa$")
    cbar.solids.set_alpha(1)
    save(fig, fname)


def sidm_fraction(kappa, delta, bins):
    '''Fraction of pixels with Delta > 0 per kappa bin, binomial error, count (pixels with Delta = 0 dropped).'''
    keep = np.isfinite(kappa) & (delta != 0)
    k, positive = kappa[keep], (delta[keep] > 0).astype(float)
    frac = binned_statistic(k, positive, bins=bins, statistic="mean")[0]
    count = binned_statistic(k, positive, bins=bins, statistic="count")[0]
    return frac, np.sqrt(frac * (1 - frac) / count), count, k, positive


def plot_relation(sims, obs_kappa_e, obs_delta, fname, rng):
    fig = plt.figure(figsize=(6, 4))
    ax = plt.gca()

    bins = np.linspace(0., 0.5, 15)
    xc = (bins[1:] + bins[:-1]) / 2.
    for dataset, (kappa, delta) in sims.items():
        frac, err, _, _, _ = sidm_fraction(kappa, delta, bins)
        ax.errorbar(xc, frac, err, fmt="o-", capsize=2, lw=2, label=" ".join(dataset.split("_")).upper())

    bins = np.linspace(0., 0.5, 20)
    xc = (bins[1:] + bins[:-1]) / 2.
    yobs, _, _, k, positive = sidm_fraction(obs_kappa_e.ravel(), obs_delta.ravel(), bins)
    # scatter of the binned fraction when kappa is shuffled across pixels
    monte = []
    for _ in range(1000):
        monte.append(binned_statistic(rng.permutation(k), positive, bins=bins, statistic="mean")[0])
    yobsstd = np.std(np.array(monte), axis=0)
    ax.errorbar(xc, yobs, yobsstd, fmt="o", color="k", label="Observation", capsize=2)

    leg = ax.legend()
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.0)
    ax.set_xlabel(r"Weak lensing convergence, $\kappa$")
    ax.fill_between([0.0, 1.0], [0.5] * 2, [1] * 2, color="c", alpha=0.2)
    ax.fill_between([0.0, 1.0], [0] * 2, [0.5] * 2, color="k", alpha=0.2)
    ax.text(0.05, 0.9, "Regions sensitive to SIDM", transform=ax.transAxes, fontsize=15)
    ax.text(0.4, 0.1, "Regions sensitive to CDM", transform=ax.transAxes, fontsize=15)
    ax.set_ylabel("Fraction of pixels that are sensitive to SIDM")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0., 1)
    save(fig, fname)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    opts = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    out = Path("pickles") / f"sensitivity_{opts.model}"
    out.mkdir(parents=True, exist_ok=True)
    Path("plots").mkdir(exist_ok=True)

    args.source_domain = args.target_domain = "a2744"
    args.apply_intrinsic_ell = 0.
    args.zs = opts.zs
    args.verbose = False
    transform = get_test_transform(args)
    models = load_models(opts, device)
    occ = lambda img, s: occlusion(models, img, s, device, opts.batch_size)       # noqa: E731

    # A2744
    _, data = pkl.load(open("../data/100/a2744/obs_data_concat.pkl", "rb"))
    obs = {}
    for s in sorted({opts.map_segment, opts.relation_segment}):
        print(f"A2744, segment {s}")
        obs[s] = cached(out / f"obs_{s}.npz", opts.recompute,
                        lambda: dict(zip(["full", "occ"], occ(data[0], s))))

    # A2744 analogue
    analogue = pkl.load(open(opts.analogue, "rb"))

    def run_analogue():
        res = {}
        for key in ["cdm", "sidm"]:
            res[f"{key}_full"], res[f"{key}_occ"] = occ(forward_model(analogue[key]["shear"], transform), opts.map_segment)
        return res

    print(f"A2744 analogue, segment {opts.map_segment}")
    analogue_res = cached(out / f"analogue_{opts.map_segment}.npz", opts.recompute, run_analogue)

    plot_sensitivity_map(analogue, analogue_res, obs[opts.map_segment], obs_kappa(correct_for_ngal=True),
                         opts.map_segment, "plots/sensitivity_map.pdf")

    # BAHAMAS clusters against their true convergence
    s = opts.relation_segment
    sims = {}
    for dataset in DATASETS:
        _, maps = pkl.load(open(f"../data/100/obs/concat/{dataset}.pkl", "rb"))

        def run_sims():
            full, occluded = [], []
            for i in tqdm(range(opts.nclusters), desc=dataset):
                f, o = occ(forward_model(maps[i, :2], transform), s)
                full.append(f)
                occluded.append(o)
            return {"full": np.array(full), "occ": np.array(occluded)}

        res = cached(out / f"{dataset}_{s}_n{opts.nclusters}.npz", opts.recompute, run_sims)
        ncomp = get_num_merging_components(f"../data/100/convergence/{dataset}.pkl", components_dir="pickles")
        _, shear = pkl.load(open(f"../data/100/shear/{dataset}.pkl", "rb"))
        idx = [i for i in range(opts.nclusters) if ncomp[i] > opts.min_ncomp]
        delta = np.array([to_grid(delta_map(res["full"][i], res["occ"][i]), s) for i in idx])
        sims[dataset] = (shear[idx, 2].ravel(), delta.ravel())

    obs_delta = to_grid(delta_map(obs[s]["full"], obs[s]["occ"]), s)
    plot_relation(sims, obs_kappa(correct_for_ngal=False), obs_delta, "plots/Sensitivty_relation.pdf",
                  np.random.default_rng(0))


if __name__ == "__main__":
    main()
