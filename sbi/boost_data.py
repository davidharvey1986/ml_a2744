#!/usr/bin/env python
'''
Boost-factor correction of the A2744 data, Schrabback et al. (2025, Euclid ERO A2390, Sect. 5.4) form:

    B(r) = 1 + A exp(1 - r / r_s),   r_s = 350 kpc,  r from the stamp centre

Each A2744 galaxy's distortion chi is multiplied by B(r) (|chi| capped at 0.999), the map is binned as in
systematics.obs_raw_map and run through the models; the simulations are left as they are (the fiducial
systematics.py curves from --fiducial-from), so each A is paired with the fiducial. The forward-model
counterpart (diluting the simulations by 1/B(kappa)) is boost_kappa.py. The plot shows A2744's tangential
distortion profile for each A, labelled with its P(SIDM), and P(SIDM) against A next to the simulated curves.
--plot-only redraws it from boost_data.json.

    astro-env/bin/python sbi/boost_data.py --model tweakedalign_k0p7_ctr_ba_final --include-str spin2_cx_pgc_ft_ \
        --summaries sbi/summaries_spin2cx_final_n5 --out-dir sbi/results/boost_data_final \
        --fiducial-from sbi/results/systematics_spin2cx_final
'''
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sbi"))
import systematics as sy  # noqa: E402  (chdirs to notebooks/)

R_S = 350.
A_VALUES = [0.05, 0.08, 0.1, 0.2, 0.27, 0.3, 0.5, 0.75, 1.0]   # 0.08, 0.27: injection-test fits (global, r < 250 kpc)
MARK_A = {350.: (0.08, 0.27), 102.: (0.88,)}   # injection-test fits marked on the plot, per r_s (r_s free, r < 900 kpc: 0.88, 102 kpc)
CHI_MAX = 0.999
KPC_PER_PX = 20.
EDGES = np.geomspace(20, 1000, 13)       # kpc


def boosted_chi(d, ra, dec, A):
    g = np.sqrt(d["gamma1"] ** 2 + d["gamma2"] ** 2)
    c1, c2 = 2 * d["gamma1"] / (1 + g ** 2), 2 * d["gamma2"] / (1 + g ** 2)
    B = 1 + A * np.exp(1 - KPC_PER_PX * np.hypot(ra, dec) / R_S)
    c1, c2 = c1 * B, c2 * B
    mod = np.hypot(c1, c2)
    s = np.where(mod > CHI_MAX, CHI_MAX / np.maximum(mod, 1e-12), 1.)
    return c1 * s, c2 * s, int((mod > CHI_MAX).sum())


def boosted_map(d, ra, dec, A):
    c1, c2, ncap = boosted_chi(d, ra, dec, A)
    e1, e2 = sy.bin2d(ra, dec, v=(c1, c2), npix=sy.NPIX, extent=[-sy.NPIX // 2, sy.NPIX // 2, -sy.NPIX // 2, sy.NPIX // 2])
    return np.array([e1, e2], dtype=np.float32), ncap


def tangential_profile(d, ra, dec, A):
    '''Mean tangential distortion and its error in the EDGES radial bins about the stamp centre.'''
    c1, c2, _ = boosted_chi(d, ra, dec, A)
    r, phi = KPC_PER_PX * np.hypot(ra, dec), np.arctan2(dec, ra)
    chi_t = -(c1 * np.cos(2 * phi) + c2 * np.sin(2 * phi))
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (r >= lo) & (r < hi)
        if m.sum() >= 2:
            rows.append((np.sqrt(lo * hi), chi_t[m].mean(), chi_t[m].std() / np.sqrt(m.sum())))
    return [list(map(float, c)) for c in zip(*rows)]


def plot(res, path):
    '''A2744's tangential profile for each A (labelled with its P(SIDM)) and P(SIDM) against A with the curves.'''
    plt = sy.style()
    rows = [{"A": 0., "a2744_p_sidm": res["fiducial"]["a2744_p_rebuilt"], "dp_paired_err": 0.,
             "profile": res["fiducial"]["profile"]}] + res["boost"]
    rows = [r for r in rows if r["A"] <= (0.5 if res["r_s_kpc"] >= 350 else 1.5)]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.3, 1]})
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(rows):
        rc, t, te = r["profile"]
        c = sy.INK if r["A"] == 0 else cmap(0.15 + 0.75 * (i - 1) / max(len(rows) - 2, 1))
        ax.errorbar(rc, t, te, fmt="o-", color=c, ms=4, lw=2 if r["A"] == 0 or r["A"] in MARK_A.get(res["r_s_kpc"], ()) else 1.2, capsize=2,
                    label=f"A = {r['A']:g}" + (" (fiducial)" if r["A"] == 0 else "") + f":  P(SIDM) = {r['a2744_p_sidm']:.3f}")
    ax.set_xscale("log")
    ax.axhline(0, color=sy.MUTED, lw=0.8)
    ax.set_xlabel("radius from the A2744 stamp centre [kpc]")
    ax.set_ylabel(r"tangential distortion $\langle\chi_t\rangle$")
    ax.set_title(f"A2744 boosted by B(r) = 1 + A exp(1 − r / {res['r_s_kpc']:g} kpc)", fontsize=11)
    ax.legend(fontsize=8)
    A = np.array([r["A"] for r in res["boost"]])
    p = np.array([r["a2744_p_sidm"] for r in res["boost"]])
    pe = np.array([r["dp_paired_err"] for r in res["boost"]])
    A, p, pe = np.r_[0, A], np.r_[res["fiducial"]["a2744_p_rebuilt"], p], np.r_[0, pe]
    curve = res["fiducial"]["metrics"]["bahamas"]["curve"]
    for (k, v), c in zip(curve.items(), ["#222222", "#56B4E9", "#0072B2", "#003f6b"]):
        bx.axhline(v, color=c, ls="--", lw=1)
        bx.annotate(f"BAHAMAS σ/m = {k}" if k != "0" else "BAHAMAS CDM", (1, v), xycoords=("axes fraction", "data"),
                    xytext=(-4, 2), textcoords="offset points", ha="right", va="bottom", fontsize=8, color=c)
    marks = MARK_A.get(res["r_s_kpc"], ())
    for a in marks:
        bx.axvline(a, color=sy.FID_COLOUR, ls=":", lw=1)
    bx.errorbar(A, p, pe, fmt="o-", color=sy.INK, capsize=2, label="A2744 (paired error vs A = 0)")
    bx.set_xlabel(f"boost amplitude A, r_s = {res['r_s_kpc']:g} kpc" + (f"  (dotted: injection-test fit{'s' * (len(marks) > 1)} "
                  + ", ".join(f"{a:g}" for a in marks) + ")" if marks else ""))
    bx.set_ylabel("A2744 ensemble P(SIDM)")
    bx.legend(loc="lower left", fontsize=8)
    fig.savefig(path.with_suffix(".png"), dpi=170)
    fig.savefig(path.with_suffix(".pdf"))


def main():
    global R_S
    p = sy.argparse.ArgumentParser(description=__doc__, formatter_class=sy.argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--include-str", required=True)
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--summaries", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fiducial-from", required=True)
    p.add_argument("--A", type=float, nargs="+", default=A_VALUES)
    p.add_argument("--plot-only", action="store_true", help="redraw from <out-dir>/boost_data.json")
    p.add_argument("--nmodels", type=int, default=None, help="only the first n models (quick look)")
    p.add_argument("--r-s", type=float, default=R_S, help="boost scale radius [kpc] (injection test, r_s free, r < 900 kpc: 102)")
    opts = p.parse_args()
    R_S = opts.r_s
    if opts.plot_only:
        out = sy.LAUNCH_DIR / opts.out_dir
        plot(json.load(open(out / "boost_data.json")), out / "boost_data")
        return
    opts.redo, opts.from_cache, opts.max_clusters, opts.n_noise, opts.noise_seed, opts.batch_size = \
        False, False, None, 1, 4321, 256
    sy.A2744_NCOMP = tuple(range(opts.min_ncomp, 7))
    out = sy.LAUNCH_DIR / opts.out_dir
    out.mkdir(parents=True, exist_ok=True)
    opts.summaries = str((sy.LAUNCH_DIR / opts.summaries).resolve())
    shutil.copy(sy.LAUNCH_DIR / opts.fiducial_from / "fiducial.npz", out / "fiducial.npz")

    R = sy.Runner(opts, out)
    fid, n_fid = R.cached("fiducial", lambda d: None)
    if opts.nmodels:                          # the first n models (load order, as in the fiducial cache)
        R.models = R.models[:opts.nmodels]
        fid = {dom: (s[:opts.nmodels], o[:opts.nmodels]) for dom, (s, o) in fid.items()}
    data = R.cat.full
    ra, dec = R.cat.positions(data)
    keep = sy.strong_lensing_kappa(data) < sy.KAPPA_THRESH
    d, ra, dec = data[keep], ra[keep], dec[keep]
    p0 = sy.observe(R.models, sy.obs_raw_map(d, ra, dec), 1, R.device)
    res = {"r_s_kpc": R_S, "fiducial": {"metrics": R.summarise(fid), "a2744_p_rebuilt": float(p0.mean()),
                                        "profile": tangential_profile(d, ra, dec, 0.)}, "boost": []}
    sy.print_row("fiducial", res["fiducial"]["metrics"])
    print(f"  A2744 P(SIDM) rebuilt here {p0.mean():.4f} (fiducial cache {fid['bahamas'][1].mean():.4f})")
    for A in opts.A:
        raw, ncap = boosted_map(d, ra, dec, A)
        p_obs = sy.observe(R.models, raw, 1, R.device)
        r = {dom: (fid[dom][0], p_obs) for dom in sy.DOMAINS}
        res["boost"].append({"A": A, "a2744_p_sidm": float(p_obs.mean()), "dp_paired": float((p_obs - p0).mean()),
                             "dp_paired_err": float((p_obs - p0).std() / np.sqrt(len(p0))), "n_capped": ncap,
                             "profile": tangential_profile(d, ra, dec, A),
                             "metrics": R.summarise(r)})
        sy.print_row(f"A = {A:g}", res["boost"][-1]["metrics"], res["fiducial"]["metrics"])
        print(f"  A2744 P(SIDM) {p_obs.mean():.4f} ({(p_obs - p0).mean():+.4f} ± {res['boost'][-1]['dp_paired_err']:.4f} paired); "
              f"{ncap} galaxies capped at |chi| = {CHI_MAX}")
    json.dump(res, open(out / "boost_data.json", "w"), indent=2)
    plot(res, out / "boost_data")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
