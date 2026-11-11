#!/usr/bin/env python
'''
Photo-z member leak against the convergence: the fraction of selected sources at 0.455 < z_ph < 0.6 (just above the
member cut) in bins of kappa, either the weak-lensing Kaiser-Squires map of the data (--kappa ks) or the Lenstool model
(--kappa lenstool, z_s = 2.28, galaxies with kappa >= 0.7 drawn open: the analysis removes them). A weighted straight
line q = q_0 + c kappa is fitted; the excess over q_0 is the leaked-member fraction f_cl = c kappa. The contamination
systematic dilutes each simulated map pixel by pixel as e -> (1 - f_cl) e with f_cl = c kappa
(dataset.cluster_member_contamination). plots_for_paper.figureA1 draws the KS version as a panel (draw_leakage).

    astro-env/bin/python sbi/leakage_vs_kappa.py --kappa ks --nbins 12
'''
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks")]
import add_shear_to_data as asd  # noqa: E402
from lenspack.image.inversion import ks93  # noqa: E402
from lenspack.utils import bin2d  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

LENSTOOL_EDGES = [0.15, 0.2, 0.3, 0.4, 0.55, 0.7, 1.2]
LEAK_LO, LEAK_HI = 0.455, 0.6
MIN_PER_BIN = 20


def ks_kappa(cat, smooth=1.5):
    '''Kaiser-Squires E-mode convergence at each galaxy, as get_model_probabilities.get_kappa (100x100 stamp pixels,
    Gaussian smoothing of 1.5 pixels), without importing that module (it rebuilds the shared catalogue).'''
    x, y = asd.ra_dec_to_simulation_image_pos(cat)
    e1, e2 = bin2d(x, y, v=(2.0 * cat["gamma1"], 2.0 * cat["gamma2"]), npix=100, extent=[-50, 50, -50, 50])
    ke = gaussian_filter(ks93(e1, e2)[0], smooth)
    ix = np.clip(np.floor(x + 50).astype(int), 0, 99)
    iy = np.clip(np.floor(y + 50).astype(int), 0, 99)
    return ke[iy, ix]


def leakage(cat, kappa="ks", nbins=12, kmin=-0.3, kmax=0.9):
    '''Binned leak fraction against kappa and the straight-line fit. cat needs 'redshift' (get_obs_data photoz=True).
    Returns a dict: kappa (bin medians), f_cl (excess over the fitted q_0), err, n, use (inside the analysis cut),
    q0, c, c_err, xlabel, kcut.'''
    z = np.asarray(cat["redshift"], float)
    vals, cnt = np.unique(z, return_counts=True)
    has = z != vals[np.argmax(cnt)]                         # galaxies without a photo-z carry the padded value
    leak = has & (z > LEAK_LO) & (z < LEAK_HI)
    if kappa == "lenstool":
        k, edges = asd.strong_lensing_kappa(cat), LENSTOOL_EDGES
        xlabel, kcut = r"Lenstool convergence, $\kappa$ ($z_s = 2.28$)", 0.7
    else:
        k, edges = ks_kappa(cat), np.linspace(kmin, kmax, nbins + 1)
        xlabel, kcut = r"Weak lensing convergence, $\kappa$", np.inf
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (k >= lo) & (k < hi)
        if m.sum() < MIN_PER_BIN:
            continue
        q = leak[m].mean()
        rows.append((float(np.median(k[m])), q, float(np.sqrt(q * (1 - q) / m.sum())), int(m.sum())))
    kc, q, e, n = (np.array(c) for c in zip(*rows))
    use = kc < kcut
    A = np.c_[np.ones(use.sum()), kc[use]] / e[use, None]
    cov = np.linalg.inv(A.T @ A)
    q0, c = cov @ A.T @ (q[use] / e[use])
    return {"kappa": kc, "f_cl": q - q0, "err": e, "n": n, "use": use, "q0": float(q0), "c": float(c),
            "c_err": float(np.sqrt(cov[1, 1])), "xlabel": xlabel, "kcut": kcut, "kappa_map": kappa}


def draw_leakage(ax, L, fontsize=15, legend_size=9, refs=((0.2, ":", "systematics range"),)):
    '''The leak points, the fit with its 1-sigma band and the reference lines f_cl = c kappa on ax.'''
    kc, f, e, use, kcut = L["kappa"], L["f_cl"], L["err"], L["use"], L["kcut"]
    ax.axhline(0, color="k", lw=0.8)
    hi = 1.2 if L["kappa_map"] == "lenstool" else kc.max() + 0.05
    if np.isfinite(kcut):
        ax.axvspan(kcut, hi, color="k", alpha=0.08, lw=0)
        ax.text(kcut + 0.02, 10.6, r"removed ($\kappa\geq0.7$)", fontsize=10)
    ax.errorbar(kc[use], 100 * f[use], 100 * e[use], fmt="o", color="k", capsize=2, ms=6, label=r"Photo-$z$ leak (A2744)")
    if (~use).any():
        ax.errorbar(kc[~use], 100 * f[~use], 100 * e[~use], fmt="o", color="k", mfc="white", capsize=2, ms=6)
    xx = np.linspace(min(0, kc.min() - 0.05), hi, 50)
    c, ce = L["c"], L["c_err"]
    ax.plot(xx, 100 * c * xx, "-", color="#D55E00", lw=2, label=rf"Fit: $f_{{\rm cl}} = ({c:.3f}\pm{ce:.3f})\,\kappa$")
    ax.fill_between(xx, 100 * (c - ce) * xx, 100 * (c + ce) * xx, color="#D55E00", alpha=0.2, lw=0)
    for cr, ls, lab in refs:
        ax.plot(xx, 100 * cr * xx, ls, color="#0072B2", lw=1.5, label=rf"$f_{{\rm cl}} = {cr:g}\,\kappa$ ({lab})")
    ax.set_xlim(xx[0], xx[-1])
    ax.set_ylim(-6, 12)
    ax.set_xlabel(L["xlabel"], fontsize=fontsize)
    ax.set_ylabel(r"Leaked members, $f_{\rm cl}$ [\%]", fontsize=fontsize)
    ax.legend(fontsize=legend_size, loc="upper left")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(REPO / "data" / "100" / "a2744_ratio_copy"),
                   help="A2744 data directory (a copy, so the shared catalogue is not rewritten)")
    p.add_argument("--kappa", choices=["lenstool", "ks"], default="ks")
    p.add_argument("--nbins", type=int, default=12, help="KS: bins between --kmin and --kmax")
    p.add_argument("--kmin", type=float, default=-0.3)
    p.add_argument("--kmax", type=float, default=0.9)
    p.add_argument("--out", default=None, help="default notebooks_jauzac/plots/leakage_vs_kappa[_ks]")
    opts = p.parse_args()
    opts.out = opts.out or str(REPO / "notebooks_jauzac" / "plots" / ("leakage_vs_kappa" + ("_ks" if opts.kappa == "ks" else "")))

    cat = asd.get_obs_data("concat", data_dir=opts.data_dir, photoz=True, cuts=asd.SHEAR_CUTS)
    L = leakage(cat, opts.kappa, opts.nbins, opts.kmin, opts.kmax)
    print(f"{opts.kappa}: q_0 {100 * L['q0']:.2f}%; f_cl:", ", ".join(
        f"{a:+.2f}: {100 * b:+.1f} ± {100 * c:.1f}% (N {d})" for a, b, c, d in zip(L["kappa"], L["f_cl"], L["err"], L["n"])))
    print(f"fit: c = {L['c']:.3f} ± {L['c_err']:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(5.6, 4.2), constrained_layout=True)
    draw_leakage(ax, L)
    for ext in ("png", "pdf"):
        fig.savefig(f"{opts.out}.{ext}", dpi=200)
    json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in L.items() if k != "xlabel"},
              open(f"{opts.out}.json", "w"), indent=1, default=float)
    print(f"wrote {opts.out}.png")


if __name__ == "__main__":
    main()
