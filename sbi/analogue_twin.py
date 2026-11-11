#!/usr/bin/env python
'''
A BAHAMAS SIDM analogue of A2744 next to the same halo in the CDM run (same initial conditions), as observed: each
map forward-modelled with A2744's galaxy positions, shape noise and photo-z (sensitivity.simulated_maps, three noise
draws), Kaiser-Squires inverted and smoothed by --smooth pixels, contours at S/N 2,3,4,6,8. The CDM twin is the CDM
map (same redshift and projection) whose convergence, with the radial profile removed, correlates best with the SIDM
one outside 100 kpc.

    astro-env/bin/python sbi/analogue_twin.py --sidm-image 2609
'''
import argparse
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]
import sensitivity as sens  # noqa: E402  (chdirs to notebooks/)
import make_summaries as ms  # noqa: E402
from lenspack.image.inversion import ks93  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sidm-file", default="bahamas_1.pkl")
    p.add_argument("--cdm-file", default="bahamas_cdm.pkl")
    p.add_argument("--sidm-image", type=int, default=2609)
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_wb1_final_n5"))
    p.add_argument("--draws", type=int, default=3)
    p.add_argument("--smooth", type=float, default=2.)
    p.add_argument("--out", default=str(REPO / "sbi" / "results" / "a2744_analogue_cdm_twin"))
    return p.parse_args()


def find_twin(opts):
    s, ms_ = pkl.load(open(f"../data/100/shear/{opts.sidm_file}", "rb"))
    c, mc = pkl.load(open(f"../data/100/shear/{opts.cdm_file}", "rb"))
    i = opts.sidm_image
    y, x = np.indices(ms_.shape[-2:]) - (ms_.shape[-1] / 2 - 0.5)
    R = np.hypot(x, y).astype(int)

    def resid(k):
        k = gaussian_filter(np.log10(np.clip(k, 1e-3, None)), 1.5)
        prof = np.bincount(R.ravel(), k.ravel()) / np.bincount(R.ravel())
        return (k - prof[R])[R > 5]
    a = resid(ms_[i, 2])
    cand = np.where((c["redshift"] == s["redshift"][i]) & (c["projection"] == s["projection"][i]))[0]
    r = sorted([(np.corrcoef(a, resid(mc[j, 2]))[0, 1], j) for j in cand], reverse=True)
    j = r[0][1]
    print(f"SIDM {opts.sidm_file} image {i} (ID {s['clusterID'][i]}, mass {s['mass'][i]:.3f}, n_comp {s['ncomponents'][i]}) -> "
          f"CDM image {j} (ID {c['clusterID'][j]}, mass {c['mass'][j]:.3f}, n_comp {c['ncomponents'][j]}), residual corr "
          f"{r[0][0]:.3f} (next {r[1][0]:.3f}, median {np.median([t[0] for t in r]):.3f})")
    return j, r[0][0]


def main():
    opts = parse()
    j_cdm, corr = find_twin(opts)
    P = np.load(Path(opts.summaries) / "params_bahamas.npz", allow_pickle=True)
    soft = np.load(Path(opts.summaries) / "softmax_bahamas.npy").mean(0)          # (n_noise, N)
    obs_p = np.load(Path(opts.summaries) / "obs.npz")["softmax"][:, 0]
    rows = {}
    for f, img in ((opts.sidm_file, opts.sidm_image), (opts.cdm_file, j_cdm)):
        r = int(np.where((P["filename"] == f) & (P["image_idx"] == img))[0][0])
        rows[(f, img)] = (r, soft[:, r], float(P["log10_mass_750"][r]))

    args = ms.setup_args("concat")
    raw_obs, *_ = sens.a2744_map()

    def snr(raw):
        ke, kb = (gaussian_filter(k, opts.smooth) for k in ks93(raw[0] / 2., raw[1] / 2.))
        return ke / np.std(kb)
    cols = [("A2744 (data)", [snr(raw_obs)], f"P(SIDM) = {obs_p.mean():.3f} ± {obs_p.std():.3f}")]
    for (f, img), (r, p, m) in rows.items():
        sel = {"filename": np.array([f]), "image_idx": np.array([img])}
        maps = [snr(np.asarray(sens.simulated_maps(sel, args, seed)[0][0])) for seed in range(opts.draws)]
        lab = "SIDM σ/m=1" if f == opts.sidm_file else "CDM twin"
        cols.append((f"{lab}: row {r} (image {img})", maps,
                     f"log M750 {m:.2f}, P(SIDM) = {p.mean():.3f} [{len(p)} draws {p.min():.2f}–{p.max():.2f}]"))

    plt = sens.style()
    fig, axes = plt.subplots(opts.draws, len(cols), figsize=(4.2 * len(cols), 4 * opts.draws), squeeze=False,
                             constrained_layout=True)
    E = [-1000, 1000, -1000, 1000]
    for k, (title, maps, sub) in enumerate(cols):
        for d in range(opts.draws):
            ax = axes[d, k]
            if d >= len(maps):
                ax.axis("off")
                continue
            im = ax.imshow(maps[d], origin="lower", extent=E, cmap="magma", vmin=-1, vmax=8)
            ax.contour(maps[d], origin="lower", extent=E, levels=[2, 3, 4, 6, 8], colors="w", linewidths=0.7)
            if d == 0:
                ax.set_title(f"{title}\n{sub}", fontsize=9)
            if k > 0:
                ax.text(-950, -950, f"noise draw {d + 1}", color="w", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.5, label="KS κ_E S/N")
    fig.suptitle(f"A2744 SIDM analogue and the same halo in BAHAMAS CDM (same initial conditions; residual-κ correlation "
                 f"{corr:.2f}), as observed: A2744 shape noise + photo-z, KS, Gaussian σ = {opts.smooth:g} px; "
                 f"P(SIDM) from {Path(opts.summaries).name}", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(f"{opts.out}.{e}", dpi=130)
    print(f"wrote {opts.out}.png")


if __name__ == "__main__":
    main()
