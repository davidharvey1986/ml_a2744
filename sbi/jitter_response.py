#!/usr/bin/env python
'''
Response of one model group when the simulated clusters are moved off centre
(make_summaries.py --centre-jitter-kpc), against the centred response: mass-matched P(SIDM)
against sigma/m (n_comp >= --min-ncomp, noise draw 0), one panel per simulation. A2744 is not
shifted; its line is from the first summaries directory.

    astro-env/bin/python sbi/jitter_response.py \
        --summaries sbi/summaries_spin2_best_n5 sbi/summaries_spin2_best_jit35 sbi/summaries_spin2_best_jit500 \
        --labels centred "35 kpc jitter" "500 kpc jitter"
'''
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "sbi")]
from mass_weighting import a2744_mass_gaussian  # noqa: E402
from response_plot import X_CDM, curves  # noqa: E402

STYLES = [("#000000", "o-"), ("#0072B2", "s--"), ("#D55E00", "^:"), ("#009E73", "d-.")]


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--title", default="spin2 best")
    p.add_argument("--out", default=str(REPO / "sbi" / "results" / "jitter_response" / "jitter_response.png"))
    return p.parse_args()


def main():
    opts = parse()
    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mu, sig = a2744_mass_gaussian()
    p_a = []                                   # A2744 per set of models (the data map has no added noise)
    for S in opts.summaries:
        obs = np.load(Path(S).resolve() / "obs.npz")["softmax"]
        p_a.append(float(obs[:, 0].mean()) if obs.ndim == 2 else float(obs[:, 0, 0].mean()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True, sharey=True)
    res = {"a2744_p_sidm": p_a, "min_ncomp": opts.min_ncomp}
    for ax, dom in zip(axes, ("bahamas", "darkskies")):
        res[dom] = {}
        for S, lab, (col, fmt) in zip(opts.summaries, opts.labels, STYLES):
            c, *_ = curves(Path(S).resolve(), dom, opts.min_ncomp, mu, sig)
            res[dom][lab] = c
            x = np.where(np.array(c["sigma"]) == 0, X_CDM, c["sigma"])
            ax.fill_between(x, c["q16"], c["q84"], color=col, alpha=0.07, lw=0)
            ax.errorbar(x, c["all"], c["all_err"], fmt=fmt, color=col, ms=5, capsize=2, lw=1.8,
                        label=f"{lab}: CDM {c['all'][0]:.3f} → {c['all'][-1]:.3f}")
            print(f"{dom:9s} {lab:16s} " + " | ".join(f"σ={s:g}: {a:.3f}±{e:.3f} (held-out {h:.3f})"
                                                    for s, a, e, h in zip(c["sigma"], c["all"], c["all_err"], c["held"])))
        drawn = set()
        for p, lab, (col, _) in zip(p_a, opts.labels, STYLES):
            if round(p, 3) not in drawn:
                drawn.add(round(p, 3))
                ax.axhline(p, color=col, lw=2, alpha=0.6, ls="-", label=f"A2744 (not shifted), {lab}: {p:.3f}")
        ax.set_xscale("log")
        ax.set_xlim(0.003, 1.4)
        ax.set_xticks([X_CDM, 0.01, 0.05, 0.1, 0.2, 0.3, 1], ["CDM", "0.01", "0.05", "0.1", "0.2", "0.3", "1"])
        ax.minorticks_off()
        ax.set_xlabel("σ/m [cm²/g]")
        ax.set_title(f"{dom.upper()}: simulated clusters shifted off centre, mass-matched, n_comp ≥ {opts.min_ncomp}",
                     fontsize=9)
        ax.legend(fontsize=7.5, loc="upper left")
    axes[0].set_ylabel("ensemble P(SIDM)")
    fig.suptitle(f"{opts.title}: response with the simulated maps jittered (16-84% of clusters shaded)", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + e), dpi=150)
    json.dump(res, open(out.with_suffix(".json"), "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
