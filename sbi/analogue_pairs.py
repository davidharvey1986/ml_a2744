#!/usr/bin/env python
'''
Merging BAHAMAS SIDM clusters that the network calls SIDM while the same halo in the CDM run (same initial
conditions) looks CDM. Every SIDM map (n_comp >= --min-ncomp, within --mass-window of the A2744 mass) is matched to
its CDM twin: the CDM map at the same redshift and projection whose log convergence, with its radial profile
removed, correlates best outside 100 kpc. Pairs are ranked by P(SIDM)_SIDM - P(SIDM)_CDM (ensemble mean over noise
draws, --summaries) among those with P_SIDM >= --p-sidm-min, P_CDM <= --p-cdm-max and a clear match. The top --show
pairs are drawn one per row: true kappa and an observed KS S/N map (A2744 shape noise + photo-z) for each version.

    astro-env/bin/python sbi/analogue_pairs.py
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
import make_summaries as ms  # noqa: E402
from lenspack.image.inversion import ks93  # noqa: E402
from mass_weighting import a2744_mass_gaussian  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sidm-file", default="bahamas_1.pkl")
    p.add_argument("--cdm-file", default="bahamas_cdm.pkl")
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_wb1_final_n5"))
    p.add_argument("--min-ncomp", type=int, default=3)
    p.add_argument("--twin-min-ncomp", type=int, default=1, help="n_comp of the CDM twin at least this")
    p.add_argument("--mass-window", type=float, default=0.3, help="dex about the Harvey A2744 mass (log M750)")
    p.add_argument("--p-sidm-min", type=float, default=0.6)
    p.add_argument("--p-cdm-max", type=float, default=0.45)
    p.add_argument("--min-corr", type=float, default=0.6, help="residual-kappa correlation of the twin")
    p.add_argument("--show", type=int, default=4)
    p.add_argument("--smooth", type=float, default=2.)
    p.add_argument("--out", default=str(REPO / "sbi" / "results" / "a2744_analogue_pairs"))
    return p.parse_args()


def residuals(maps):
    '''Log kappa minus its azimuthal mean, outside 5 pixels, standardised: (N, n_pix).'''
    n = maps.shape[-1]
    y, x = np.indices((n, n)) - (n / 2 - 0.5)
    R = np.hypot(x, y).astype(int)
    cnt = np.bincount(R.ravel())
    out = []
    for k in maps:
        k = gaussian_filter(np.log10(np.clip(k, 1e-3, None)), 1.5)
        prof = np.bincount(R.ravel(), k.ravel()) / cnt
        v = (k - prof[R])[R > 5]
        out.append((v - v.mean()) / v.std())
    return np.array(out)


def main():
    opts = parse()
    P = np.load(Path(opts.summaries) / "params_bahamas.npz", allow_pickle=True)
    p_ens = np.load(Path(opts.summaries) / "softmax_bahamas.npy").mean(0).mean(0)       # (N,) mean over models, draws
    row = {(f, int(i)): r for r, (f, i) in enumerate(zip(P["filename"], P["image_idx"]))}
    mu, _ = a2744_mass_gaussian()

    s, ms_ = pkl.load(open(f"../data/100/shear/{opts.sidm_file}", "rb"))
    c, mc = pkl.load(open(f"../data/100/shear/{opts.cdm_file}", "rb"))
    rs = np.array([row[(opts.sidm_file, i)] for i in range(len(s["mass"]))])
    cand = np.where((P["ncomponents"][rs] >= opts.min_ncomp) & (np.abs(P["log10_mass_750"][rs] - mu) < opts.mass_window)
                    & (p_ens[rs] >= opts.p_sidm_min))[0]
    print(f"{len(cand)} {opts.sidm_file} maps with n_comp >= {opts.min_ncomp}, |log M750 - {mu:.2f}| < {opts.mass_window}, "
          f"P(SIDM) >= {opts.p_sidm_min}")
    res_c = residuals(mc[:, 2])
    res_s = residuals(ms_[cand, 2])
    pairs = []
    for k, i in enumerate(cand):
        pool = np.where((c["redshift"] == s["redshift"][i]) & (c["projection"] == s["projection"][i]))[0]
        corr = res_c[pool] @ res_s[k] / res_s.shape[1]
        o = np.argsort(corr)[::-1]
        j = int(pool[o[0]])
        rj, ri = row[(opts.cdm_file, j)], rs[i]
        pairs.append({"sidm_image": int(i), "cdm_image": j, "sidm_row": int(ri), "cdm_row": int(rj),
                      "corr": float(corr[o[0]]), "corr_next": float(corr[o[1]]),
                      "p_sidm": float(p_ens[ri]), "p_cdm": float(p_ens[rj]),
                      "logm_sidm": float(P["log10_mass_750"][ri]), "logm_cdm": float(P["log10_mass_750"][rj]),
                      "ncomp_sidm": int(P["ncomponents"][ri]), "ncomp_cdm": int(P["ncomponents"][rj]),
                      "z": float(s["redshift"][i]), "projection": int(s["projection"][i])})
    good = [q for q in pairs if q["corr"] >= opts.min_corr and q["corr"] - q["corr_next"] > 0.02 and q["p_cdm"] <= opts.p_cdm_max
            and q["ncomp_cdm"] >= opts.twin_min_ncomp]
    good.sort(key=lambda q: q["p_cdm"] - q["p_sidm"])
    print(f"{len(good)} pairs with a clear twin (corr >= {opts.min_corr}) and P_CDM <= {opts.p_cdm_max}:")
    for q in good[:15]:
        print(f"  SIDM image {q['sidm_image']:4d} (row {q['sidm_row']}) P {q['p_sidm']:.3f} n_comp {q['ncomp_sidm']} "
              f"logM {q['logm_sidm']:.2f} | CDM image {q['cdm_image']:4d} (row {q['cdm_row']}) P {q['p_cdm']:.3f} "
              f"n_comp {q['ncomp_cdm']} logM {q['logm_cdm']:.2f} | corr {q['corr']:.2f} (next {q['corr_next']:.2f}) "
              f"z {q['z']:g} proj {q['projection']}")
    json.dump({"settings": vars(opts), "pairs": good}, open(f"{opts.out}.json", "w"), indent=1)
    top = good[:opts.show]
    if not top:
        return

    args = ms.setup_args("concat")

    def snr(raw):
        ke, kb = (gaussian_filter(k, opts.smooth) for k in ks93(raw[0] / 2., raw[1] / 2.))
        return ke / np.std(kb)
    obs = {}
    for f, key in ((opts.sidm_file, "sidm_image"), (opts.cdm_file, "cdm_image")):
        sel = {"filename": np.array([f] * len(top)), "image_idx": np.array([q[key] for q in top])}
        raws, _ = sens.simulated_maps(sel, args, 0)
        obs[f] = [snr(np.asarray(r)) for r in raws]

    plt = sens.style()
    fig, axes = plt.subplots(len(top), 4, figsize=(16, 4 * len(top)), squeeze=False, constrained_layout=True)
    E = [-1000, 1000, -1000, 1000]
    for r, q in enumerate(top):
        for k, (f, key, kappa, lab) in enumerate(((opts.sidm_file, "sidm_image", ms_, "SIDM σ/m=1"),
                                                  (opts.cdm_file, "cdm_image", mc, "CDM twin"))):
            pk = q["p_sidm"] if k == 0 else q["p_cdm"]
            nc = q["ncomp_sidm"] if k == 0 else q["ncomp_cdm"]
            ax = axes[r, 2 * k]
            ax.imshow(np.log10(np.clip(gaussian_filter(kappa[q[key], 2], 1), 1e-2, None)), origin="lower", extent=E,
                      cmap="magma", vmin=-1.3, vmax=0.3)
            ax.set_title(f"{lab}: image {q[key]}, P(SIDM) = {pk:.3f}\nn_comp {nc}, true log κ", fontsize=9)
            ax = axes[r, 2 * k + 1]
            m = obs[f][r]
            ax.imshow(m, origin="lower", extent=E, cmap="magma", vmin=-1, vmax=8)
            ax.contour(m, origin="lower", extent=E, levels=[2, 3, 4, 6, 8], colors="w", linewidths=0.7)
            ax.set_title(f"{lab}: observed KS S/N (one noise draw)", fontsize=9)
        axes[r, 0].set_ylabel(f"pair {r + 1}: ΔP = {q['p_sidm'] - q['p_cdm']:.2f}, twin corr {q['corr']:.2f}\n"
                              f"log M750 {q['logm_sidm']:.2f} / {q['logm_cdm']:.2f}, z {q['z']:g}", fontsize=9)
    fig.suptitle(f"Merging (n_comp ≥ {opts.min_ncomp}; twin ≥ {opts.twin_min_ncomp}) BAHAMAS σ/m = 1 clusters with high P(SIDM) whose CDM twin (same "
                 f"halo, same initial conditions) has low P(SIDM); P from {Path(opts.summaries).name}", fontsize=11)
    for e in ("png", "pdf"):
        fig.savefig(f"{opts.out}.{e}", dpi=120 if len(top) <= 8 else 70)
    print(f"wrote {opts.out}.png")


if __name__ == "__main__":
    main()
