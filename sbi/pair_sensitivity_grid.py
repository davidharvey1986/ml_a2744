#!/usr/bin/env python
'''
Grid of the occlusion-sensitivity maps of the SIDM / CDM-twin pairs (sbi/pair_sensitivity.py outputs
pair<N>_sensitivity_w<W>s<S>.npz): each pair as two panels (SIDM left, CDM twin right), --per-row pairs per row,
one colour scale, true-kappa contours and P(SIDM) of each map.

    astro-env/bin/python sbi/pair_sensitivity_grid.py --window 4 --stride 1
'''
import argparse
import json
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]
import sensitivity as sens  # noqa: E402  (chdirs to notebooks/)

RES = REPO / "sbi" / "results"


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs-json", default=str(RES / "a2744_analogue_pairs_nc4_all.json"))
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--per-row", type=int, default=3)
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    opts = parse()
    J = json.load(open(opts.pairs_json))
    tag = f"w{opts.window}s{opts.stride}"
    have = [(n, q) for n, q in enumerate(J["pairs"], 1) if (RES / f"pair{n}_sensitivity_{tag}.npz").exists()]
    print(f"{len(have)} of {len(J['pairs'])} pairs have {tag} maps")
    kap = {f: pkl.load(open(f"../data/100/shear/{f}", "rb"))[1] for f in (J["settings"]["sidm_file"], J["settings"]["cdm_file"])}
    Z = {n: np.load(RES / f"pair{n}_sensitivity_{tag}.npz") for n, _ in have}
    vmax = np.percentile(np.abs(np.concatenate([np.r_[Z[n]["sidm_image_delta"].ravel(), Z[n]["cdm_image_delta"].ravel()]
                                                for n, _ in have])), 99)
    centres = Z[have[0][0]]["centres"]
    half = opts.stride / 2.
    extent = [centres[0] - half, centres[-1] + half] * 2

    nrow = int(np.ceil(len(have) / opts.per_row))
    plt = sens.style()
    fig, axes = plt.subplots(nrow, 2 * opts.per_row, figsize=(2.6 * 2 * opts.per_row, 2.75 * nrow), squeeze=False,
                             constrained_layout=True)
    for ax in axes.flat:
        ax.axis("off")
    for k, (n, q) in enumerate(have):
        r, c = divmod(k, opts.per_row)
        for j, (key, f, lab) in enumerate((("sidm_image", J["settings"]["sidm_file"], "SIDM"),
                                           ("cdm_image", J["settings"]["cdm_file"], "CDM"))):
            ax = axes[r, 2 * c + j]
            ax.axis("on")
            d = Z[n][f"{key}_delta"]
            im = ax.imshow(np.ma.masked_equal(d, 0), origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.contour(np.log10(np.clip(gaussian_filter(kap[f][q[key], 2], 1), 1e-2, None)), origin="lower",
                       extent=[-50, 50, -50, 50], levels=[-0.8, -0.6, -0.4, -0.2, 0], colors="k", linewidths=0.5, linestyles="solid")
            p = float(np.mean(Z[n][f"{key}_p"]))
            nc = q["ncomp_sidm" if j == 0 else "ncomp_cdm"]
            ax.set_title(f"{'pair ' + str(n) + ': ' if j == 0 else ''}{lab} {q[key]}, P = {p:.2f}, n$_c$ = {nc}",
                         fontsize=7.5)
            ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, shrink=0.3, pad=0.01)
    cb.set_label(r"$\Delta P$(SIDM) when removed (red: pushes to SIDM)")
    fig.suptitle(f"Occlusion sensitivity ({opts.window}×{opts.window} px, stride {opts.stride}; 5 models × 3 noise draws) "
                 f"of merging σ/m = 1 halos with high P(SIDM) (left) and their CDM twins (right); contours: true log κ "
                 f"-0.8 … 0", fontsize=11)
    out = opts.out or RES / f"pair_sensitivity_grid_{tag}"
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=110)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
