#!/usr/bin/env python
'''
Response curves matched in the shear ratio instead of mass: ensemble P(SIDM) (mean over models
and noise draws) against sigma/m for every simulated cluster whose
R = <chi_t>(200-400 kpc) / <chi_t>(50-200 kpc) (shear_ratio.py, noise-free) is within --dR of
A2744's, at any mass, for all clusters and for n_comp >= 2; and the unmatched response of all
clusters for reference. One panel per simulation.

    astro-env/bin/python sbi/response_ratio.py --summaries sbi/summaries_spin2_best_n5
'''
import argparse
import json
import pickle as pkl
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import shear_ratio as sr  # noqa: E402  (chdirs to notebooks/)
import sensitivity as sens  # noqa: E402
from response_plot import X_CDM  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_spin2_best_n5"))
    p.add_argument("--ratio-json", default=str(REPO / "sbi" / "results" / "shear_ratio" / "shear_ratio.json"),
                   help="shear_ratio.py output, for A2744's R")
    p.add_argument("--dR", type=float, default=0.07)
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None, help="default <summaries>/../results/<summaries name>/response_ratio.png")
    return p.parse_args()


def main():
    opts = parse()
    S = Path(opts.summaries).resolve()
    out = Path(opts.out) if opts.out else S.parent / "results" / S.name / "response_ratio.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    R_a = json.load(open(opts.ratio_json))["a2744"]["adopted"]["R"]
    obs = np.load(S / "obs.npz")["softmax"]
    p_a = float(obs[:, 0].mean()) if obs.ndim == 2 else float(obs[:, 0, 0].mean())

    res = {"a2744_R": R_a, "a2744_p_sidm": p_a, "dR": opts.dR}
    plt = sens.style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True, sharey=True)
    for ax, dom in zip(axes, ("bahamas", "darkskies")):
        P = dict(np.load(S / f"params_{dom}.npz"))
        p = np.load(S / f"softmax_{dom}.npy").mean((0, 1))
        R = np.full(len(p), np.nan)
        for f in np.unique(P["filename"]):
            rows = np.where(P["filename"] == f)[0]
            _, maps = pkl.load(open(f"../data/100/obs/concat/{f}", "rb"))
            R[rows] = sr.ratio(sr.to_chi(maps[P["image_idx"][rows], :2]))
        cs, nc = P["cross_section"], np.minimum(P["ncomponents"], 6)
        near = np.abs(R - R_a) < opts.dR
        res[dom] = {}
        for lab, sel, fmt, col in [("all clusters", np.isfinite(R), "s:", "0.5"),
                                   (f"R within {opts.dR} of A2744", near, "o-", "#0072B2"),
                                   (f"R within {opts.dR}, n_comp ≥ 2", near & (nc >= 2), "^--", "#D55E00")]:
            sig = np.unique(cs)
            m = [p[sel & (cs == c)] for c in sig]
            y = np.array([v.mean() for v in m])
            e = np.array([v.std() / np.sqrt(len(v)) for v in m])
            q = np.array([np.percentile(v, [16, 84]) for v in m])
            x = np.where(sig == 0, X_CDM, sig)
            ax.fill_between(x, q[:, 0], q[:, 1], color=col, alpha=0.08, lw=0)
            ax.errorbar(x, y, e, fmt=fmt, color=col, ms=5, capsize=2, lw=1.6,
                        label=f"{lab}: n {min(map(len, m))}-{max(map(len, m))}")
            res[dom][lab] = {"sigma": sig.tolist(), "mean": y.tolist(), "err": e.tolist(), "q16_84": q.tolist(),
                             "n": [len(v) for v in m], "frac_ge_a2744": [float(np.mean(v >= p_a)) for v in m]}
            print(f"{dom:9s} {lab:28s} " + " | ".join(f"σ={s:g} {a:.3f} (n {len(v)}, ≥A2744 {np.mean(v >= p_a):.2f})"
                                                       for s, a, v in zip(sig, y, m)))
        ax.axhline(p_a, color="k", lw=2, alpha=0.5, label=f"A2744: {p_a:.3f} (R = {R_a:.2f})")
        ax.set_xscale("log")
        ax.set_xlim(0.003, 1.4)
        ax.set_xticks([X_CDM, 0.01, 0.05, 0.1, 0.2, 0.3, 1], ["CDM", "0.01", "0.05", "0.1", "0.2", "0.3", "1"])
        ax.minorticks_off()
        ax.set_xlabel("σ/m [cm²/g]")
        ax.set_title(f"{dom.upper()}: no mass matching (16-84% of clusters shaded)", fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("ensemble P(SIDM)")
    fig.suptitle(f"{opts.label or S.name}: response matched in the shear ratio R = "
                 r"$\langle\chi_t\rangle$(200-400) / $\langle\chi_t\rangle$(50-200 kpc)", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + e), dpi=150)
    json.dump(res, open(out.with_suffix(".json"), "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
