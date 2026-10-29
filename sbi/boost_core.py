#!/usr/bin/env python
'''
Does A2744's high P(SIDM) come from its low central shear? Add a constant to the tangential
distortion (about the stamp centre, ~45 kpc from Lenstool O2) of every A2744 galaxy in each
50 kpc annulus inside R, keeping its cross component and noise, and rerun the CNN:

    scale f     the pixel-mean chi_t of each annulus (as shear_ratio.py measures it) times f
    CDM / σ=1   each annulus raised to the median chi_t profile of mass-matched BAHAMAS clusters
                (noise-free, n_comp >= 2) of that cross-section

Distortions are capped at |chi| < 0.999. The chi_t profile of each boosted map is re-measured and
printed against the target, and the A2744 P(SIDM) is compared with the simulated ones from the
summaries.

    astro-env/bin/python sbi/boost_core.py --include-str spin2_ft_
'''
import argparse
import json
import pickle as pkl
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from lenspack.utils import bin2d

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import make_summaries as ms  # noqa: E402  (chdirs to notebooks/)
import sensitivity as sens  # noqa: E402
import shear_ratio as sr  # noqa: E402
from add_shear_to_data import galaxy_distortion  # noqa: E402
from mass_weighting import a2744_mass_gaussian  # noqa: E402

EDGES = np.arange(0, 301, 50)                       # kpc


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="tweakedalign_k0p9_ctr_ba_best")
    p.add_argument("--include-str", default="spin2_ft_")
    p.add_argument("--nmodels", type=int, default=None)
    p.add_argument("--factors", type=float, nargs="+", default=[1.25, 1.5, 2., 2.5])
    p.add_argument("--radii-kpc", type=float, nargs="+", default=[100, 150])
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_spin2_best_n5_chifix"))
    p.add_argument("--out-dir", default=str(REPO / "sbi" / "results" / "boost_core"))
    return p.parse_args()


def profile(raw):
    '''Pixel-mean chi_t about the stamp centre in the EDGES annuli, over occupied pixels.'''
    c2, s2 = np.cos(2 * sr.PHI), np.sin(2 * sr.PHI)
    t = -(raw[..., 0, :, :] * c2 + raw[..., 1, :, :] * s2)
    occ = np.any(raw != 0, axis=-3)
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = occ & (sr.RG >= lo / 20.) & (sr.RG < hi / 20.)
        out.append(np.sum(t * m, axis=(-2, -1)) / np.maximum(m.sum(axis=(-2, -1)), 1))
    return np.stack(out, -1)


def main():
    opts = parse()
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = ms.get_device()
    models = ms.load_models(SimpleNamespace(filter="concat", model=opts.model, include_str=opts.include_str,
                                            nmodels=opts.nmodels), ms.setup_args("concat"), device)
    print(f"{len(models)} models: {opts.model} containing '{opts.include_str}'")

    raw_a, cat, ra, dec = sens.a2744_map()
    c1, c2 = galaxy_distortion(cat)
    r_kpc = 20 * np.hypot(ra, dec)
    phi = np.arctan2(dec, ra)
    cs2, sn2 = np.cos(2 * phi), np.sin(2 * phi)
    chi_t, chi_x = -(c1 * cs2 + c2 * sn2), c1 * sn2 - c2 * cs2
    prof_a = profile(raw_a)

    def run(delta):
        '''P(SIDM) and chi_t profile with chi_t raised by delta[bin] for galaxies in each annulus.'''
        b = np.digitize(r_kpc, EDGES) - 1
        add = np.where((b >= 0) & (b < len(delta)), np.r_[delta, 0][np.clip(b, 0, len(delta))], 0.)
        t = chi_t + add
        e1, e2 = -t * cs2 + chi_x * sn2, -t * sn2 - chi_x * cs2
        mod = np.hypot(e1, e2)
        cap = np.where(mod > 0.999, 0.999 / mod, 1.)
        raw = np.array(bin2d(ra, dec, v=(e1 * cap, e2 * cap), npix=sr.NPIX, extent=[-50, 50, -50, 50]),
                       dtype=np.float32)
        p = float(sens.model_outputs(models, sens.normalise(torch.tensor(raw))[0][None], device, 1).mean())
        return p, profile(raw), int((cap < 1).sum())

    # simulated profiles and P(SIDM), mass-matched, n_comp >= 2
    mu, _ = a2744_mass_gaussian()
    S = Path(opts.summaries)
    P = dict(np.load(S / "params_bahamas.npz"))
    p_sim = np.load(S / "softmax_bahamas.npy").mean((0, 1))
    m = (np.minimum(P["ncomponents"], 6) >= 2) & (np.abs(P["log10_mass_750"] - mu) < 0.15)
    sims = {}
    for f in np.unique(P["filename"][m]):
        rows = np.where(m & (P["filename"] == f))[0]
        _, maps = pkl.load(open(f"../data/100/obs/concat/{f}", "rb"))
        c = float(P["cross_section"][rows[0]])
        sims[c] = {"profile": np.median(profile(sr.to_chi(maps[P["image_idx"][rows], :2])), 0),
                   "p_median": float(np.median(p_sim[rows])), "p_84": float(np.percentile(p_sim[rows], 84)),
                   "p_95": float(np.percentile(p_sim[rows], 95))}

    edges_txt = " ".join(f"{lo}-{hi}" for lo, hi in zip(EDGES[:-1], EDGES[1:]))
    print(f"\nchi_t profile ({edges_txt} kpc)")
    print("A2744 data      " + " ".join(f"{v:6.3f}" for v in prof_a))
    for c, s in sims.items():
        print(f"BAHAMAS σ={c:<4g}   " + " ".join(f"{v:6.3f}" for v in s["profile"]) +
              f"   P(SIDM) median {s['p_median']:.3f}, 84% {s['p_84']:.3f}, 95% {s['p_95']:.3f}")

    p0 = run(np.zeros(1))[0]
    res = {"a2744_p_sidm": p0, "edges_kpc": EDGES.tolist(), "a2744_profile": prof_a.tolist(),
           "sims": {f"{c:g}": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
                    for c, s in sims.items()}, "runs": []}
    print(f"\nA2744 unchanged: P(SIDM) {p0:.3f}")
    for R in opts.radii_kpc:
        nb = int(R // 50)
        for f in opts.factors:
            p, prof, ncap = run((f - 1) * prof_a[:nb])
            res["runs"].append({"kind": "scale", "R_kpc": R, "factor": f, "p_sidm": p, "profile": prof.tolist()})
            print(f"scale χt × {f:<4g} inside {R:3g} kpc: P(SIDM) {p:.3f}   profile " +
                  " ".join(f"{v:6.3f}" for v in prof) + (f"   ({ncap} capped)" if ncap else ""))
    for c, s in sims.items():
        for R in opts.radii_kpc:
            nb = int(R // 50)
            p, prof, ncap = run(s["profile"][:nb] - prof_a[:nb])
            res["runs"].append({"kind": f"match σ={c:g}", "R_kpc": R, "p_sidm": p, "profile": prof.tolist()})
            print(f"match BAHAMAS σ={c:<4g} inside {R:3g} kpc: P(SIDM) {p:.3f}   profile " +
                  " ".join(f"{v:6.3f}" for v in prof) + (f"   ({ncap} capped)" if ncap else ""))
    json.dump(res, open(out / f"{opts.model}_{opts.include_str}.json", "w"), indent=1)
    plot(res, out / f"{opts.model}_{opts.include_str}")


def plot(res, out):
    plt = sens.style()
    mid = 0.5 * (np.array(res["edges_kpc"][:-1]) + np.array(res["edges_kpc"][1:]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    ax = axes[0]
    for c, s in res["sims"].items():
        ax.plot(mid, s["profile"], "--", color=sr.COLOURS.get(float(c)), label=f"BAHAMAS σ={c} median")
    ax.plot(mid, res["a2744_profile"], "o-", color="k", lw=2, label="A2744 data")
    for r in res["runs"]:
        if r["kind"] == "scale" and r["R_kpc"] == max(x["R_kpc"] for x in res["runs"]):
            ax.plot(mid, r["profile"], ":", marker=".", color="#E69F00", alpha=0.4 + 0.2 * r["factor"] / 2.5,
                    label=f"A2744, χt × {r['factor']:g} inside {r['R_kpc']:g} kpc")
    ax.set_xlabel("radius from the stamp centre [kpc]")
    ax.set_ylabel("⟨χt⟩ (pixel mean)")
    ax.legend(fontsize=7)
    ax = axes[1]
    for R, mk in zip(sorted({r["R_kpc"] for r in res["runs"]}), ["o", "s"]):
        rr = [r for r in res["runs"] if r["kind"] == "scale" and r["R_kpc"] == R]
        ax.plot([1] + [r["factor"] for r in rr], [res["a2744_p_sidm"]] + [r["p_sidm"] for r in rr], mk + "-",
                color="k", label=f"A2744, χt scaled inside {R:g} kpc")
    for c, s in res["sims"].items():
        col = sr.COLOURS.get(float(c))
        ax.axhline(s["p_median"], color=col, ls="--", lw=1, label=f"BAHAMAS σ={c}: median P")
        for r in res["runs"]:
            if r["kind"] == f"match σ={c}":
                ax.plot(1, r["p_sidm"], "*" if r["R_kpc"] == 150 else "D", color=col, mec="k", ms=11)
    ax.set_xlabel("factor on A2744's central χt")
    ax.set_ylabel("ensemble P(SIDM)")
    ax.set_title("markers at 1: A2744 with its core raised to each σ/m's median profile (★ 150 kpc, ◆ 100 kpc)",
                 fontsize=8)
    ax.legend(fontsize=7)
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=150)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
