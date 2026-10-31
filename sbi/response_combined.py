#!/usr/bin/env python
'''
BAHAMAS and DARKSKIES response curves on one plot: the mass-weighted mean ensemble P(SIDM)
against sigma/m (n_comp >= --min-ncomp, noise draw 0, as response_plot.curves), 16-84% of
clusters shaded, with A2744 at the adopted centre and its centring spread (--centre-json).

    astro-env/bin/python sbi/response_combined.py --summaries sbi/summaries_spin2cx_best_n5 \
        --centre-json sbi/results/centre_uncertainty/tweakedalign_k0p9_ctr_ba_best_spin2_cx_ft_.json
'''
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import sensitivity as sens  # noqa: E402
from mass_weighting import a2744_mass_gaussian  # noqa: E402
from response_plot import X_CDM, curves  # noqa: E402

STYLES = {"bahamas": ("#0072B2", "o-"), "darkskies": ("#D55E00", "s--")}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_spin2cx_best_n5"))
    p.add_argument("--centre-json", default=str(REPO / "sbi" / "results" / "centre_uncertainty" /
                                                "tweakedalign_k0p9_ctr_ba_best_spin2_cx_ft_.json"))
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--label", default="spin2 re-fine-tuned, corrected noise (30 models)")
    p.add_argument("--out", default=str(REPO / "sbi" / "results" / "chifix" / "response_combined"))
    return p.parse_args()


def main():
    opts = parse()
    S = REPO / opts.summaries
    mu, sig = a2744_mass_gaussian()
    cen = json.load(open(REPO / opts.centre_json))          # relative to the repo (make_summaries chdirs)
    p_a = cen["p_sidm_adopted_centre"][0]
    lo, _, hi = cen["p_sidm_over_centre_draws"]["percentiles_16_50_84"]

    plt = sens.style()
    fig, ax = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    ax.axhspan(lo, hi, color="0.6", alpha=0.25, lw=0, label=f"A2744 centring, 16-84%: {lo:.3f}-{hi:.3f}")
    ax.axhline(p_a, color="k", lw=2, alpha=0.7, label=f"A2744 (adopted centre): {p_a:.3f}")
    for dom, (col, fmt) in STYLES.items():
        c = curves(S, dom, opts.min_ncomp, mu, sig)[0]
        x = np.where(np.array(c["sigma"]) == 0, X_CDM, c["sigma"])
        ax.fill_between(x, c["q16"], c["q84"], color=col, alpha=0.12, lw=0)
        ax.errorbar(x, c["all"], c["all_err"], fmt=fmt, color=col, ms=6, capsize=2, lw=2,
                    label=f"{dom.upper()}: CDM {c['all'][0]:.3f} → σ={c['sigma'][-1]:g} {c['all'][-1]:.3f}")
        print(f"{dom:9s} " + " | ".join(f"σ={s:g}: {a:.3f}±{e:.3f} [16-84% {q1:.3f}-{q2:.3f}]"
                                        for s, a, e, q1, q2 in zip(c["sigma"], c["all"], c["all_err"], c["q16"], c["q84"])))
    ax.set_xscale("log")
    ax.set_xlim(0.003, 1.4)
    ax.set_xticks([X_CDM, 0.01, 0.05, 0.1, 0.2, 0.3, 1], ["CDM", "0.01", "0.05", "0.1", "0.2", "0.3", "1"])
    ax.minorticks_off()
    ax.set_xlabel("σ/m [cm²/g]")
    ax.set_ylabel("ensemble P(SIDM)")
    ax.set_title(f"{opts.label}: mass-matched, n_comp ≥ {opts.min_ncomp} (16-84% of clusters shaded)", fontsize=9)
    ax.legend(fontsize=8, loc="lower right")
    out = f"{opts.out}_nc{opts.min_ncomp}"
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=150)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
