#!/usr/bin/env python
'''
Is the CNN responding to A2744's lensing signal? Rotate every shear by alpha (positions
fixed, gamma -> gamma exp(2i alpha)) in A2744 and in mass-matched simulated clusters
(n_comp >= 3, before the shape noise is added) and follow P(SIDM). alpha = 45 deg turns
the E-mode into a pure B-mode, alpha = 90 deg makes the pattern radial (a negative mass).
A network reading the lensing signal should lose the CDM / SIDM difference at 45 deg;
if A2744 stays where it was, its score is not coming from the E-mode.

    # median-centred stamp (current pipeline)
    astro-env/bin/python sbi/eb_swap_test.py
    # stamp centred on A2744's main halo
    astro-env/bin/python sbi/eb_swap_test.py --centred --model tweakedalign_k0p9_ctr_ba_best \
        --summaries $PWD/sbi/summaries_k0p9_ctr_best_n5
'''
import argparse
import json
import pickle as pkl
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lenspack.utils import bin2d

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import make_summaries as ms  # noqa: E402  (chdirs to notebooks/)
import sensitivity as sens  # noqa: E402
from add_shear_to_data import SHEAR_CUTS, galaxy_distortion, get_obs_data, normalise_map, ra_dec_to_simulation_image_pos  # noqa: E402
from dataset import get_test_transform  # noqa: E402
from mass_weighting import a2744_mass_gaussian  # noqa: E402
from profile_matching import O2_PX, source_redshift_map  # noqa: E402

ALPHAS = [0, 15, 30, 45, 60, 75, 90]     # deg
RANDOM = -1                              # independent random angle per galaxy (A2744) / pixel (simulations)


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tweakedalign_ba_best")
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_ba_best_n5"))
    p.add_argument("--centred", action="store_true", help="stamp centred on A2744's main halo (concat_centred files)")
    p.add_argument("--nmodels", type=int, default=10)
    p.add_argument("--n", type=int, default=100, help="mass-matched clusters per cross-section")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "results" / "eb_swap_test"))
    return p.parse_args()


def rotate(c1, c2, alpha, rng=None):
    if alpha == RANDOM:          # keep |chi|, scramble every orientation independently
        a = rng.uniform(0, 2 * np.pi, np.shape(c1))
        return np.hypot(c1, c2) * np.cos(a), np.hypot(c1, c2) * np.sin(a)
    a = np.deg2rad(2 * alpha)
    return c1 * np.cos(a) - c2 * np.sin(a), c1 * np.sin(a) + c2 * np.cos(a)


def main():
    opts = parse()
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = ms.get_device()
    args = ms.setup_args("concat")
    tag = "concat_centred" if opts.centred else "concat"
    if opts.centred:
        # shape-noise geometry (ngal_concat_centred.pkl) and source redshifts about O2
        args.jwst_filter = tag
        args.zs = source_redshift_map()
        centre = dict(ra_0=O2_PX[0], dec_0=O2_PX[1])
    else:
        centre = {}
    transform = get_test_transform(args)
    models = ms.load_models(SimpleNamespace(filter="concat", model=opts.model, include_str="nob1",
                                            nmodels=opts.nmodels), args, device)

    cat = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS)
    ra, dec = ra_dec_to_simulation_image_pos(cat, **centre)
    c1, c2 = galaxy_distortion(cat)
    _, stored = pkl.load(open(f"../data/100/a2744/obs_data_{tag}.pkl", "rb"))
    raw = np.array(bin2d(ra, dec, v=(c1, c2), npix=100, extent=[-50, 50, -50, 50]))
    assert np.allclose(np.array(normalise_map(*raw)), stored[0], atol=1e-5), f"A2744 map differs from obs_data_{tag}.pkl"

    mu, sig = a2744_mass_gaussian()
    sel = sens.pick_clusters(opts.summaries, "bahamas", mu, sig, opts.n, np.random.default_rng(opts.seed))
    keep = np.isin(sel["cross_section"], [0, sel["cross_section"].max()])
    sel = {k: v[keep] for k, v in sel.items()}
    shears = []
    for f in np.unique(sel["filename"]):
        _, obs = pkl.load(open(f"../data/100/obs/{tag}/{f}", "rb"))
        shears += [(r, obs[sel["image_idx"][r], :2].astype(np.float32)) for r in np.where(sel["filename"] == f)[0]]
    shears = [g for _, g in sorted(shears, key=lambda x: x[0])]

    res = {"model": opts.model, "stamp": tag, "n_models": len(models), "alphas": ALPHAS + ["random"], "rows": []}
    for alpha in ALPHAS + [RANDOM]:
        rng = np.random.default_rng(opts.seed)
        n_a = 20 if alpha == RANDOM else 1
        a_img = torch.stack([torch.tensor(np.array(normalise_map(*bin2d(ra, dec, v=rotate(c1, c2, alpha, rng), npix=100,
                                                                     extent=[-50, 50, -50, 50]))), dtype=torch.float32)
                             for _ in range(n_a)])
        p_a = float(sens.model_outputs(models, a_img, device, 8).mean())
        np.random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        imgs = torch.stack([sens.normalise(transform(torch.tensor(np.array(rotate(g[0], g[1], alpha, rng)), dtype=torch.float32)))[0] for g in shears])
        p_s = sens.model_outputs(models, imgs, device, 64).mean(0)
        row = {"alpha": alpha, "a2744": p_a}
        for s in np.unique(sel["cross_section"]):
            v = p_s[sel["cross_section"] == s]
            row[f"bahamas_{s:g}"] = [float(v.mean()), float(v.std() / np.sqrt(len(v))), float((v >= p_a).mean())]
        res["rows"].append(row)
        print(("random per galaxy/pixel" if alpha == RANDOM else f"alpha {alpha:3d} deg") + f"  A2744 {p_a:.3f}  " + "  ".join(
            f"σ={s:g}: {row[f'bahamas_{s:g}'][0]:.3f}±{row[f'bahamas_{s:g}'][1]:.3f} (≥A2744 {row[f'bahamas_{s:g}'][2]:.2f})"
            for s in np.unique(sel["cross_section"])))
    name = f"{opts.model}_{tag}"
    json.dump(res, open(out / f"{name}.json", "w"), indent=1)

    plt = sens.style()
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    for s, c in zip(np.unique(sel["cross_section"]), ["#56B4E9", "#0b3d6e"]):
        m = np.array([x[f"bahamas_{s:g}"][:2] for x in res["rows"][:-1]])
        ax.errorbar(ALPHAS, m[:, 0], m[:, 1], color=c, marker="o", ms=4, label=f"BAHAMAS σ/m={s:g}" if s else "BAHAMAS CDM")
    rows, rnd = res["rows"][:-1], res["rows"][-1]
    for s, c in zip(np.unique(sel["cross_section"]), ["#56B4E9", "#0b3d6e"]):
        ax.axhline(rnd[f"bahamas_{s:g}"][0], color=c, ls="--", lw=1)
    ax.axhline(rnd["a2744"], color="k", ls="--", lw=1, label="random angle per galaxy / pixel")
    ax.plot(ALPHAS, [x["a2744"] for x in rows], "o-", color="k", lw=2, label="A2744")
    ax.axvline(45, color="#6b6b6b", lw=1, ls=":")
    ax.set_xticks(ALPHAS)
    ax.set_xlabel("shear rotation α [deg]  (45: E → B, 90: tangential → radial)")
    ax.set_ylabel("ensemble P(SIDM)")
    ax.set_title(f"{opts.model}, {tag} stamp: {len(models)} models, {opts.n} mass-matched clusters per σ/m, "
                 f"n_comp ≥ {min(sens.A2744_NCOMP)}", fontsize=8)
    ax.legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=150)


if __name__ == "__main__":
    main()
