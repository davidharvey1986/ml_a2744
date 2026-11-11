#!/usr/bin/env python
'''
Scan of the shear ratio R_X = <chi_t>(numerator annulus, --num-kpc, default 50-100 kpc) /
<chi_t>(X to X + 100 kpc) for X from the numerator's outer edge to 900 kpc, measured as shear_ratio.py (mean tangential distortion about the stamp centre over occupied
pixels; simulations noise-free at the A2744 galaxies, A2744 from the data with a galaxy bootstrap).
For each X: how R_X depends on sigma/m (median per cross-section), whether it predicts the
ensemble P(SIDM) within a cross-section (Spearman, and the slope of P ~ R + offset(sigma)), and
where A2744 falls among mass-matched clusters (n_comp >= --min-ncomp). Also a grid of P(SIDM)
against R_X scatter plots.

    astro-env/bin/python sbi/ratio_scan.py --summaries sbi/summaries_spin2_best_n5_chifix
'''
import argparse
import json
import pickle as pkl
import sys
from pathlib import Path

import numpy as np
from lenspack.utils import bin2d
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import shear_ratio as sr  # noqa: E402  (chdirs to notebooks/)
import sensitivity as sens  # noqa: E402
from add_shear_to_data import galaxy_distortion  # noqa: E402
from mass_weighting import a2744_mass_gaussian  # noqa: E402

NUM = (50, 100)
DEN = []
COLOURS = sr.COLOURS


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_spin2_best_n5_chifix"))
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--mass-window", type=float, default=0.15)
    p.add_argument("--nboot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-kpc", type=int, nargs=2, default=[50, 100])
    p.add_argument("--out-dir", default=None, help="default sbi/results/ratio_scan_<num>")
    return p.parse_args()


def annuli(maps):
    '''Mean chi_t over occupied pixels in the numerator and each denominator annulus, (..., 1 + len(DEN)).'''
    c2, s2 = np.cos(2 * sr.PHI), np.sin(2 * sr.PHI)
    t = -(maps[..., 0, :, :] * c2 + maps[..., 1, :, :] * s2)
    occ = np.any(maps != 0, axis=-3)
    out = []
    for lo, hi in [NUM] + DEN:
        m = occ & (sr.RG >= lo / 20.) & (sr.RG < hi / 20.)
        out.append(np.sum(t * m, axis=(-2, -1)) / np.maximum(m.sum(axis=(-2, -1)), 1))
    return np.stack(out, -1)


def main():
    global NUM, DEN
    opts = parse()
    NUM = tuple(opts.num_kpc)
    DEN = [(x, x + 100) for x in range(int(np.ceil(NUM[1] / 100) * 100), 1000, 100)]
    out = Path(opts.out_dir or REPO / "sbi" / "results" / f"ratio_scan_{NUM[0]}-{NUM[1]}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(opts.seed)
    S = Path(opts.summaries).resolve()
    mu, _ = a2744_mass_gaussian()

    raw_a, cat, ra, dec = sens.a2744_map()
    c1, c2 = galaxy_distortion(cat)

    def raw_of(idx):
        return np.array(bin2d(ra[idx], dec[idx], v=(c1[idx], c2[idx]), npix=sr.NPIX, extent=[-50, 50, -50, 50]))
    a = annuli(raw_a)
    boot = np.array([annuli(raw_of(rng.integers(0, len(ra), len(ra)))) for _ in range(opts.nboot)])
    occ = np.any(raw_a != 0, axis=0)
    n_num = int((occ & (sr.RG >= NUM[0] / 20.) & (sr.RG < NUM[1] / 20.)).sum())
    print(f"A2744: {n_num} occupied pixels in the {NUM[0]}-{NUM[1]} kpc annulus")
    obs = np.load(S / "obs.npz")["softmax"]
    p_a = float(obs[:, 0].mean()) if obs.ndim == 2 else float(obs[:, 0, 0].mean())
    R_a = a[0] / a[1:]
    R_a_boot = boot[:, :1] / boot[:, 1:]

    sims = {}
    for dom in ("bahamas", "darkskies"):
        P = dict(np.load(S / f"params_{dom}.npz"))
        p = np.load(S / f"softmax_{dom}.npy").mean((0, 1))
        A = np.full((len(p), 1 + len(DEN)), np.nan)
        for f in np.unique(P["filename"]):
            rows = np.where(P["filename"] == f)[0]
            _, maps = pkl.load(open(f"../data/100/obs/concat/{f}", "rb"))
            A[rows] = annuli(sr.to_chi(maps[P["image_idx"][rows], :2]))
        keep = (np.minimum(P["ncomponents"], 6) >= opts.min_ncomp) & np.isfinite(P["log10_mass_750"])
        sims[dom] = {"R": A[:, :1] / A[:, 1:], "p": p, "cs": P["cross_section"],
                     "matched": keep & (np.abs(P["log10_mass_750"] - mu) < opts.mass_window)}

    res = {"num_kpc": NUM, "den_kpc": DEN, "a2744_p_sidm": p_a, "a2744_R": R_a.tolist(),
           "a2744_R_boot_16_84": np.percentile(R_a_boot, [16, 84], axis=0).T.tolist(), "n_pix_num": n_num, "sims": {}}
    for dom, s in sims.items():
        cs = np.unique(s["cs"])
        rows = []
        for j, (lo, hi) in enumerate(DEN):
            R, m = s["R"][:, j], s["matched"] & np.isfinite(s["R"][:, j])
            med = {f"{c:g}": float(np.median(R[m & (s["cs"] == c)])) for c in cs}
            rho = {f"{c:g}": float(spearmanr(R[m & (s["cs"] == c)], s["p"][m & (s["cs"] == c)])[0]) for c in cs}
            # robust to outliers: slope of P against the rank-normalised R, given sigma
            z = np.argsort(np.argsort(R[m])) / m.sum()
            X = np.c_[z, (s["cs"][m][:, None] == cs[None]).astype(float)]
            slope = float(np.linalg.lstsq(X, s["p"][m], rcond=None)[0][0])
            pct = {f"{c:g}": float(np.mean(R[m & (s["cs"] == c)] >= R_a[j])) for c in (cs[0], cs[-1])}
            rows.append({"den_kpc": [lo, hi], "median": med, "spearman": rho, "dP_per_rank": slope,
                         "frac_ge_a2744": pct})
            print(f"{dom:9s} /{lo:4d}-{hi:4d}: A2744 {R_a[j]:6.2f} | median " +
                  " ".join(f"σ={c} {v:.2f}" for c, v in med.items()) +
                  f" | Spearman(R,P) {np.mean(list(rho.values())):+.2f} | ΔP over R range {slope:+.3f}"
                  f" | frac ≥ A2744: CDM {pct[f'{cs[0]:g}']:.2f}, σ={cs[-1]:g} {pct[f'{cs[-1]:g}']:.2f}")
        res["sims"][dom] = rows
    json.dump(res, open(out / "ratio_scan.json", "w"), indent=1)
    plot_summary(res, sims, out / "ratio_scan")
    plot_grid(res, sims, R_a, p_a, out / "ratio_scan_scatter")
    print(f"wrote {out}")


def plot_summary(res, sims, out):
    plt = sens.style()
    x = [lo for lo, _ in DEN]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for dom, ls in (("bahamas", "-"), ("darkskies", "--")):
        rows = res["sims"][dom]
        for c in rows[0]["median"]:
            axes[0].plot(x, [r["median"][c] for r in rows], ls, color=COLOURS.get(float(c)), marker="o", ms=3,
                         label=f"{dom} σ={c}")
        axes[1].plot(x, [np.mean(list(r["spearman"].values())) for r in rows], ls, color="k", marker="o",
                     label=f"{dom}: mean within-σ Spearman(R, P)")
        axes[1].plot(x, [r["dP_per_rank"] for r in rows], ls, color="#D55E00", marker="s",
                     label=f"{dom}: ΔP(SIDM) from lowest to highest R")
        c0, c1 = list(rows[0]["frac_ge_a2744"])
        axes[2].plot(x, [r["frac_ge_a2744"][c0] for r in rows], ls, color="k", marker="o", label=f"{dom} CDM")
        axes[2].plot(x, [r["frac_ge_a2744"][c1] for r in rows], ls, color="#D55E00", marker="o", label=f"{dom} σ={c1}")
    lo, hi = np.array(res["a2744_R_boot_16_84"]).T
    axes[0].errorbar(x, res["a2744_R"], [np.array(res["a2744_R"]) - lo, hi - np.array(res["a2744_R"])],
                     fmt="*", color="#E69F00", mec="k", ms=12, capsize=2, label="A2744", zorder=5)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(f"median R = ⟨χt⟩({NUM[0]}-{NUM[1]}) / ⟨χt⟩(X to X+100 kpc)")
    axes[1].axhline(0, color="0.6", lw=0.8)
    axes[1].set_ylabel("does R predict P(SIDM) at fixed σ/m?")
    axes[2].set_ylabel("fraction of mass-matched clusters with R ≥ A2744")
    for ax in axes:
        ax.set_xlabel("X [kpc]")
        ax.legend(fontsize=6.5)
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=150)


def plot_grid(res, sims, R_a, p_a, out):
    plt = sens.style()
    nr = int(np.ceil(len(DEN) / 3))
    fig, axes = plt.subplots(nr, 3, figsize=(14, 4 * nr), constrained_layout=True, sharey=True, squeeze=False)
    s = sims["bahamas"]
    for ax in axes.ravel()[len(DEN):]:
        ax.axis("off")
    for j, ax in zip(range(len(DEN)), axes.ravel()):
        R = s["R"][:, j]
        for c in np.unique(s["cs"]):
            m = s["matched"] & (s["cs"] == c) & np.isfinite(R)
            ax.scatter(R[m], s["p"][m], s=4, color=COLOURS.get(float(c)), alpha=0.4, lw=0, label=f"σ/m = {c:g}")
        ax.plot(R_a[j], p_a, "*", color="#E69F00", mec="k", ms=14, zorder=5, label="A2744")
        lo, hi = np.nanpercentile(R[s["matched"]], [1, 99])
        ax.set_xlim(min(lo, R_a[j]) - 0.1 * abs(hi - lo), max(hi, R_a[j]) + 0.1 * abs(hi - lo))
        pass  # no panel titles (the x-label names the annulus)
        ax.set_xlabel(f"R = ⟨χt⟩({NUM[0]}-{NUM[1]}) / ⟨χt⟩({DEN[j][0]}-{DEN[j][1]} kpc)", fontsize=8)
    axes[0, 0].legend(fontsize=7, markerscale=2)
    for a in axes[:, 0]:
        a.set_ylabel("ensemble P(SIDM)")
    pass  # no suptitle
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=130)


if __name__ == "__main__":
    main()
