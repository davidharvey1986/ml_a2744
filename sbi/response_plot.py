#!/usr/bin/env python
'''
Response of one model group: mass-matched ensemble P(SIDM) against sigma/m for BAHAMAS and
DARKSKIES on one axis (all clusters, and held-out only: each model on the clusters it did not
train on, then averaged over models), with A2744 at the adopted centre, at Lenstool O2 and its
centring 16-84% range from sbi/centre_uncertainty.py.

    astro-env/bin/python sbi/response_plot.py --summaries sbi/summaries_spin2_best_n5 \
        --centre-json sbi/results/centre_uncertainty/tweakedalign_k0p9_ctr_ba_best_spin2.json --label "spin2 best"
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
from mass_weighting import a2744_mass_gaussian, gaussian_mass_weights  # noqa: E402

COLOUR = {"bahamas": "#0072B2", "darkskies": "#E69F00"}
X_CDM = 0.004


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", required=True)
    p.add_argument("--centre-json", default=None, help="output of centre_uncertainty.py")
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--label", default=None)
    p.add_argument("--out", default=None, help="default <summaries>/../results/<summaries name>/response.png")
    return p.parse_args()


def wmean(v, w):
    return np.sum(w * v) / w.sum()


def curves(S_dir, domain, min_ncomp, mu, sig):
    P = dict(np.load(Path(S_dir) / f"params_{domain}.npz"))
    S = np.load(Path(S_dir) / f"softmax_{domain}.npy")[:, 0]          # (models, clusters), noise draw 0
    sm, ho = S.mean(0), P["held_out"]
    cs = np.unique(P["cross_section"])
    sel = (P["ncomponents"] >= min_ncomp) & np.isfinite(P["log10_mass_750"])
    out = {k: [] for k in ("sigma", "all", "all_err", "held", "held_err", "q16", "q84", "n", "n_eff")}
    for s in cs:
        k = sel & (P["cross_section"] == s)
        w = gaussian_mass_weights(P["log10_mass_750"][k], mu, sig)
        v = sm[k]
        m = wmean(v, w)
        n_eff = w.sum() ** 2 / np.sum(w ** 2)
        o = np.argsort(v)
        cw = np.cumsum(w[o]) / w.sum()
        per = np.array([wmean(S[i][k & ho[i]], gaussian_mass_weights(P["log10_mass_750"][k & ho[i]], mu, sig))
                        for i in range(S.shape[0]) if (k & ho[i]).sum() >= 5])
        for key, val in [("sigma", float(s)), ("all", m), ("all_err", np.sqrt(wmean((v - m) ** 2, w) / n_eff)),
                         ("held", per.mean()), ("held_err", per.std() / np.sqrt(len(per))),
                         ("q16", v[o][np.searchsorted(cw, 0.16)]), ("q84", v[o][np.searchsorted(cw, 0.84)]),
                         ("n", int(k.sum())), ("n_eff", n_eff)]:
            out[key].append(float(val))
    return out, sm, P, sel


def main():
    opts = parse()
    mu, sig = a2744_mass_gaussian()
    S_dir = Path(opts.summaries).resolve()
    out = Path(opts.out) if opts.out else S_dir.parent / "results" / S_dir.name / "response.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    obs = np.load(S_dir / "obs.npz")["softmax"]
    p_a = float(obs[:, 0].mean()) if obs.ndim == 2 else float(obs[:, 0, 0].mean())
    cu = json.load(open(opts.centre_json)) if opts.centre_json else None

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    if cu:
        lo, _, hi = cu["p_sidm_over_centre_draws"]["percentiles_16_50_84"]
        ax.axhspan(lo, hi, color="k", alpha=0.10, lw=0, label=f"A2744 centring 16-84%: {lo:.3f}-{hi:.3f}")
        ax.axhline(cu["p_sidm_lenstool_o2"][0], color="#D55E00", lw=1.5, ls="--",
                   label=f"A2744, Lenstool O2: {cu['p_sidm_lenstool_o2'][0]:.3f}")
    ax.axhline(p_a, color="k", lw=2, label=f"A2744, adopted centre: {p_a:.3f}")
    res = {"a2744_p_sidm": p_a, "min_ncomp": opts.min_ncomp}
    for domain in ("bahamas", "darkskies"):
        c, sm, P, sel = curves(S_dir, domain, opts.min_ncomp, mu, sig)
        x = np.where(np.array(c["sigma"]) == 0, X_CDM, c["sigma"])
        col = COLOUR[domain]
        ax.fill_between(x, c["q16"], c["q84"], color=col, alpha=0.12, lw=0, label=f"{domain.upper()}: 16-84% of clusters")
        ax.errorbar(x, c["all"], c["all_err"], fmt="o-", color=col, ms=6, capsize=2, lw=2, label=f"{domain.upper()}: all clusters")
        ax.errorbar(x * 1.06, c["held"], c["held_err"], fmt="s--", color=col, ms=5, capsize=2, mfc="white", lw=1.2,
                    label=f"{domain.upper()}: held-out only")
        frac = []
        for s in c["sigma"]:
            k = sel & (P["cross_section"] == s)
            w = gaussian_mass_weights(P["log10_mass_750"][k], mu, sig)
            frac.append(float(wmean((sm[k] >= p_a).astype(float), w)))
        c["fraction_ge_a2744"] = frac
        res[domain] = c
        print(f"{domain}: " + " | ".join(f"σ={s:g}: {a:.3f} (held-out {h:.3f}), ≥A2744 {f:.2f}"
                                        for s, a, h, f in zip(c["sigma"], c["all"], c["held"], frac)))
    ax.set_xscale("log")
    ax.set_xlim(0.003, 1.4)
    ax.set_xticks([X_CDM, 0.01, 0.05, 0.1, 0.2, 0.3, 1], ["CDM", "0.01", "0.05", "0.1", "0.2", "0.3", "1"])
    ax.minorticks_off()
    ax.set_xlabel("σ/m [cm²/g]")
    ax.set_ylabel("ensemble P(SIDM)")
    ax.set_title(f"{opts.label or S_dir.name}: mass-matched, n_comp ≥ {opts.min_ncomp}", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), dpi=150)
    json.dump(res, open(out.with_suffix(".json"), "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
