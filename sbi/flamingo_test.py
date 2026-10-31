#!/usr/bin/env python
'''
Independent CDM test on FLAMINGO (fiducial, high and low AGN feedback) and TNG-Cluster (never trained on):
where A2744 sits among FLAMINGO clusters of its mass and merger class, next to the BAHAMAS
and DARKSKIES CDM clusters through the same models.

For each simulation the clusters with n_comp >= --min-ncomp are weighted to the A2744 mass
(mass_weighting.gaussian_mass_weights), over all noise draws. Reported: the weighted mean
ensemble P(SIDM), its 16-84% over clusters, and the weighted fraction scoring >= A2744
(one-sided significance, floored at 1/N_eff).

    astro-env/bin/python sbi/flamingo_test.py --summaries sbi/summaries_spin2cx_best_n5 \
        --flamingo sbi/summaries_spin2cx_best_n5_flamingo
'''
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde, norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from mass_weighting import a2744_mass_gaussian, effective_sample_size, gaussian_mass_weights  # noqa: E402

SIMS = [("bahamas", "BAHAMAS CDM", "#56B4E9"), ("darkskies", "DARKSKIES CDM", "#E69F00"),
        ("flamingo", "FLAMINGO", "#009E73"), ("flamingo_lo", "FLAMINGO low AGN", "#8FD175"),
        ("flamingo_hi", "FLAMINGO high AGN", "#00593F"), ("tng", "TNG-Cluster", "#CC79A7")]
SIDM = [("bahamas", "BAHAMAS σ/m=0.3", "#7B68EE", 0.3), ("bahamas", "BAHAMAS σ/m=1", "#0072B2", 1),
        ("vd_bahamas", "BAHAMAS vdSIDM", "#E41A1C", None)]   # dashed (DARKSKIES stops at 0.2); None: every map


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", required=True, help="BAHAMAS/DARKSKIES summaries (and obs.npz)")
    p.add_argument("--flamingo", required=True, help="FLAMINGO summaries")
    p.add_argument("--min-ncomp", type=int, nargs="+", default=[2, 3])
    p.add_argument("--label", default="")
    p.add_argument("--hist", action="store_true", help="step histograms instead of weighted KDEs")
    p.add_argument("--out", default=None, help="default <flamingo dir>/flamingo_test")
    return p.parse_args()


def load(d, sim, sigma=0):
    P = np.load(d / f"params_{sim}.npz")
    soft = np.load(d / f"softmax_{sim}.npy").mean(0)               # (n_noise, N) ensemble P(SIDM)
    keep = np.ones(len(P["cross_section"]), bool) if sigma is None else np.isclose(P["cross_section"], sigma)
    return soft[:, keep], P["log10_mass_750"][keep], P["ncomponents"][keep]


def stats(p, m, nc, p_obs, mu, sig, min_nc):
    sel = (nc >= min_nc) & np.isfinite(m)
    w = gaussian_mass_weights(m[sel], mu, sig)
    x = p[:, sel]
    wx = np.broadcast_to(w, x.shape).ravel()
    xr = x.ravel()
    o = np.argsort(xr)
    c = np.cumsum(wx[o]) / wx.sum()
    frac = float(np.sum(wx * (xr >= p_obs)) / wx.sum())
    n_eff = effective_sample_size(w)
    return {"n": int(sel.sum()), "n_eff": float(n_eff), "mean": float(np.sum(wx * xr) / wx.sum()),
            "q16_50_84": [float(np.interp(q, c, xr[o])) for q in (0.16, 0.5, 0.84)],
            "frac_ge_a2744": frac, "z": float(norm.isf(max(frac, 1 / n_eff))), "lower_limit": bool(frac < 1 / n_eff),
            "x": xr, "w": wx}


def label(name, r):
    return f"{name}: {r['mean']:.3f}, ≥A2744 {100 * r['frac_ge_a2744']:.1f}% ({'>' if r['lower_limit'] else ''}{r['z']:.2f}σ)"


def main():
    opts = parse()
    S, F = REPO / opts.summaries, REPO / opts.flamingo
    out = Path(opts.out) if opts.out else F / "flamingo_test"
    mu, sig = a2744_mass_gaussian()
    p_obs = float(np.load(S / "obs.npz")["softmax"][:, 0].mean())
    print(f"A2744 ensemble P(SIDM) = {p_obs:.3f}; A2744 log M750 = {mu:.3f} +/- {sig:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(opts.min_ncomp), figsize=(6.5 * len(opts.min_ncomp), 4.6),
                             constrained_layout=True, squeeze=False)
    res = {"a2744_p_sidm": p_obs}
    bins, grid = np.linspace(0, 1, 41), np.linspace(0, 1, 400)
    for ax, k in zip(axes[0], opts.min_ncomp):
        res[f"ncomp>={k}"] = {}
        for sim, name, col, sigma in [(*s, 0) for s in SIMS] + SIDM:
            d = S if sim in ("bahamas", "darkskies") else F
            if not (d / f"params_{sim}.npz").exists():
                continue
            r = stats(*load(d, sim, sigma), p_obs, mu, sig, k)
            x, w = r.pop("x"), r.pop("w")
            kw = dict(lw=2, color=col, ls="-" if sigma == 0 else "--")
            if opts.hist:
                ax.hist(x, bins, weights=w, density=True, histtype="step", **kw, label=label(name, r))
            else:
                ax.plot(grid, gaussian_kde(x, weights=w)(grid), **kw, label=label(name, r))
            res[f"ncomp>={k}"][sim + (f"_sigma{sigma:g}" if sigma else "")] = r
            print(f"n_comp>={k}  {name:18s} N={r['n']:5d} (N_eff {r['n_eff']:6.0f})  mean {r['mean']:.3f} "
                  f"[16-84% {r['q16_50_84'][0]:.3f}-{r['q16_50_84'][2]:.3f}]  >=A2744 {100 * r['frac_ge_a2744']:.2f}% "
                  f"({'>' if r['lower_limit'] else ''}{r['z']:.2f} sigma)")
        ax.axvline(p_obs, color="k", lw=2, label=f"A2744 {p_obs:.3f}")
        ax.set_xlabel("ensemble P(SIDM)")
        ax.set_xlim(0.15, 0.9)
        ax.set_ylabel("density (mass-weighted to A2744)")
        ax.set_title(f"CDM clusters (dashed: SIDM), n_comp ≥ {k}" + (f" | {opts.label}" if opts.label else ""), fontsize=10)
        ax.legend(fontsize=7.5, loc="upper left", frameon=False)
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=150)
    json.dump(res, open(f"{out}.json", "w"), indent=2)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
