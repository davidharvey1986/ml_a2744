#!/usr/bin/env python
"""
Split the cluster-to-cluster scatter of the CNN output into shape noise and
intrinsic cluster variety, using summaries made with make_summaries.py --n-noise > 1.

For each simulation and cross-section, with merging clusters weighted to the A2744
mass (as compare_model_groups.py), on x = logit of the ensemble-mean P(SIDM):

    noise variance      mean over clusters of the variance across noise draws
    intrinsic variance  variance over clusters of the draw-averaged x, minus
                        noise variance / n_draws
    noise fraction      noise / (noise + intrinsic) for a single realisation
    d' single           (mean - CDM mean) / CDM scatter of one realisation (as A2744)
    d' noise-free       the same with only the intrinsic CDM scatter: the limit if
                        shape noise could be removed

Usage
-----
    cd ml_check && astro-env/bin/python sbi/noise_variance.py sbi/summaries_ba_best_n5 sbi/summaries_ba_final_n5
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sbi"))
from compare_model_groups import COLOUR, DOMAINS, INK, MUTED, logit, weights  # noqa: E402
from mass_weighting import a2744_mass_gaussian  # noqa: E402


def wvar(x, w):
    m = np.average(x, weights=w)
    return np.average((x - m) ** 2, weights=w)


def decompose(S, mu, sig):
    out = {}
    for d in DOMAINS:
        P = dict(np.load(S / f"params_{d}.npz"))
        soft = np.load(S / f"softmax_{d}.npy")                  # (n_models, n_noise, N)
        n_draws = soft.shape[1]
        x = logit(soft.mean(0))                                 # (n_noise, N)
        w = weights(P, mu, sig)
        cs = P["cross_section"]
        res = {}
        for c in np.unique(cs):
            sel = (cs == c) & (w > 0)
            xs, ws = x[:, sel], w[sel]
            noise = float(np.average(xs.var(axis=0, ddof=1), weights=ws))
            intrinsic = float(wvar(xs.mean(0), ws) - noise / n_draws)
            res[f"{c:g}"] = {"mean": float(np.average(xs.mean(0), weights=ws)),
                             "noise_var": noise, "intrinsic_var": intrinsic,
                             "noise_fraction": noise / (noise + intrinsic)}
        cdm = res["0"]
        for k, r in res.items():
            shift = r["mean"] - cdm["mean"]
            r["dprime_single"] = shift / np.sqrt(cdm["noise_var"] + cdm["intrinsic_var"])
            r["dprime_noise_free"] = shift / np.sqrt(cdm["intrinsic_var"])
        out[d] = res
        out[f"{d}_n_draws"] = n_draws
    return out


def plot(R, out):
    plt.rcParams.update({"axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.labelcolor": INK, "axes.grid": True, "grid.color": "#e6e6e6",
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "font.size": 10})
    fig, axes = plt.subplots(len(R), 2, figsize=(11, 3.6 * len(R)), constrained_layout=True, squeeze=False)
    for row, (name, res) in zip(axes, R.items()):
        ax = row[0]
        labels, ticks, k = [], [], 0
        for d in DOMAINS:
            for s, r in res[d].items():
                ax.bar(k, r["intrinsic_var"], color=COLOUR[d], width=0.7)
                ax.bar(k, r["noise_var"], bottom=r["intrinsic_var"], color=COLOUR[d], alpha=0.35, width=0.7,
                       hatch="///", edgecolor="white")
                ax.annotate(f"{100 * r['noise_fraction']:.0f}%", (k, r["intrinsic_var"] + r["noise_var"]),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK)
                labels.append(("CDM" if s == "0" else s) + ("\nB" if d == "bahamas" else "\nD"))
                ticks.append(k)
                k += 1
            k += 0.6
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("variance of logit P(SIDM)")
        ax.set_title(f"{name}: intrinsic (solid) + shape noise (hatched)", color=INK)

        ax = row[1]
        for d in DOMAINS:
            sig = np.array([float(s) for s in res[d] if s != "0"])
            single = [res[d][f"{s:g}"]["dprime_single"] for s in sig]
            free = [res[d][f"{s:g}"]["dprime_noise_free"] for s in sig]
            ax.plot(sig, single, "-o", color=COLOUR[d], lw=2, ms=7, mec="white", label=f"{d.upper()} single realisation")
            ax.plot(sig, free, "--o", color=COLOUR[d], lw=2, ms=7, mfc="white", label=f"{d.upper()} noise-free")
        ax.set_xscale("log")
        ax.set_xlabel(r"$\sigma/m$ [cm$^2$ g$^{-1}$]")
        ax.set_ylabel("separation from CDM / CDM scatter")
        ax.set_title(f"{name}: single-cluster separation", color=INK)
        ax.legend(fontsize=8, loc="upper left")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"noise_variance.{ext}", dpi=180)


def main():
    dirs = [Path(p).resolve() for p in sys.argv[1:]]
    mu, sig = a2744_mass_gaussian()
    R = {}
    for S in dirs:
        R[S.name] = decompose(S, mu, sig)
        print(f"\n{S.name} ({R[S.name]['bahamas_n_draws']} noise draws)")
        print(f"  {'':9s} {'sigma/m':>7s} {'noise var':>9s} {'intr var':>9s} {'noise %':>7s} {'dp single':>9s} {'dp no-noise':>11s}")
        for d in DOMAINS:
            for s, r in R[S.name][d].items():
                print(f"  {d:9s} {s:>7s} {r['noise_var']:9.4f} {r['intrinsic_var']:9.4f} "
                      f"{100 * r['noise_fraction']:6.0f}% {r['dprime_single']:9.2f} {r['dprime_noise_free']:11.2f}")
    out = REPO / "sbi" / "results"
    json.dump(R, open(out / "noise_variance.json", "w"), indent=2)
    plot(R, out)
    print(f"\nwrote {out}/noise_variance.json and noise_variance.pdf/.png")


if __name__ == "__main__":
    main()
