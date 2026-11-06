#!/usr/bin/env python
'''
Cluster-member contamination with the Schrabback et al. (2025, Euclid ERO A2390, Sect. 5.4) boost-factor shape,
forward-modelled in the simulations instead of boosting the A2744 data:

    B(r) = n(r) / n_field = 1 + A exp(1 - r / r_s),   r_s = 350 kpc

recast as a function of convergence so it can be applied to any simulated cluster: each pixel's kappa is mapped to
the radius at which A2744's azimuthally averaged Lenstool kappa (zs = 2.28, about the stamp centre) takes that value,

    B(kappa) = 1 + A exp(1 - r_A2744(kappa) / r_s),

and the simulated shear is diluted by 1 / B (the contaminating fraction is 1 - 1/B). kappa above the A2744 profile's
inner bin maps to its innermost radius, kappa <= 0 to B = 1. Uses the systematics.py machinery (same models, noise
seed and A2744 map); the fiducial is reused from --fiducial-from so every A is paired with it.

    astro-env/bin/python sbi/boost_kappa.py --model tweakedalign_k0p7_ctr_ba_final --include-str spin2_cx_pgc_ft_ \
        --min-ncomp 2 --summaries sbi/summaries_spin2cx_final_n5 --out-dir sbi/results/boost_kappa_final \
        --fiducial-from sbi/results/systematics_spin2cx_final
'''
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sbi"))
import systematics as sy  # noqa: E402  (chdirs to notebooks/)
from pyrrg_sweep import LenstoolShear  # noqa: E402

R_S = 350.                                           # kpc
A_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
EDGES = np.geomspace(10, 2500, 41)                   # kpc, for the A2744 kappa profile


def a2744_kappa_profile(cat, ra, dec):
    '''Azimuthal mean Lenstool kappa (zs = 2.28) vs radius [kpc] about the stamp centre, made monotonic.'''
    L = LenstoolShear()
    fit = np.c_[ra, dec, np.ones(len(ra))]
    c_ra = np.linalg.lstsq(fit, np.asarray(cat["RA"]), rcond=None)[0][2]       # sky position of stamp (0, 0)
    c_dec = np.linalg.lstsq(fit, np.asarray(cat["DEC"]), rcond=None)[0][2]
    kpc_per_deg = sy.ms.PIXEL_PC / 1e3 * np.median(np.hypot(ra, dec) / np.maximum(
        np.hypot((np.asarray(cat["RA"]) - c_ra) * np.cos(np.deg2rad(c_dec)), np.asarray(cat["DEC"]) - c_dec), 1e-12))
    jj, ii = np.indices(L.k.shape)
    w_ra, w_dec = L.wcs.pixel_to_world_values(ii.ravel(), jj.ravel())
    r = kpc_per_deg * np.hypot((w_ra - c_ra) * np.cos(np.deg2rad(c_dec)), w_dec - c_dec)
    k = L.k.ravel()
    rc, kb = [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = (r >= lo) & (r < hi) & np.isfinite(k)
        if s.sum() > 3:
            rc.append(np.sqrt(lo * hi)); kb.append(k[s].mean())
    rc, kb = np.array(rc), np.minimum.accumulate(np.array(kb))                # non-increasing with radius
    return rc, kb


class SchrabbackContamination:
    '''Dilutes (e1, e2) by 1 / B(kappa); kappa comes in as the third channel and is always dropped.'''

    def __init__(self, A, r_tab, k_tab):
        self.A = A
        self.k_tab = torch.tensor(k_tab[::-1].copy(), dtype=torch.float32)   # increasing kappa
        self.r_tab = torch.tensor(r_tab[::-1].copy(), dtype=torch.float32)

    def r_of_kappa(self, kappa):
        k = kappa.clamp(min=float(self.k_tab[0]), max=float(self.k_tab[-1]))
        i = torch.searchsorted(self.k_tab, k.contiguous()).clamp(1, len(self.k_tab) - 1)
        k0, k1, r0, r1 = self.k_tab[i - 1], self.k_tab[i], self.r_tab[i - 1], self.r_tab[i]
        lr = torch.log(r0) + (torch.log(r1) - torch.log(r0)) * (k - k0) / (k1 - k0).clamp(min=1e-12)
        return torch.exp(lr)

    def boost(self, kappa):
        B = 1 + self.A * torch.exp(1 - self.r_of_kappa(kappa) / R_S)
        return torch.where(kappa > 0, B, torch.ones_like(B))

    def __call__(self, sample):
        kappa = sample[2] if sample.shape[0] == 3 else None
        sample = sample[:2]
        if kappa is None:
            raise ValueError("needs the convergence as a third channel")
        return sample / self.boost(kappa)


def main():
    p = sy.argparse.ArgumentParser(description=__doc__, formatter_class=sy.argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--include-str", required=True)
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--summaries", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fiducial-from", default=None, help="systematics out-dir whose fiducial.npz to reuse")
    p.add_argument("--A", type=float, nargs="+", default=A_VALUES)
    p.add_argument("--n-noise", type=int, default=1)
    p.add_argument("--noise-seed", type=int, default=4321)
    p.add_argument("--batch-size", type=int, default=256)
    opts = p.parse_args()
    opts.redo, opts.from_cache, opts.max_clusters = False, False, None
    sy.A2744_NCOMP = tuple(range(opts.min_ncomp, 7))
    out = sy.LAUNCH_DIR / opts.out_dir
    out.mkdir(parents=True, exist_ok=True)
    opts.summaries = str((sy.LAUNCH_DIR / opts.summaries).resolve())
    if opts.fiducial_from and not (out / "fiducial.npz").exists():
        shutil.copy(sy.LAUNCH_DIR / opts.fiducial_from / "fiducial.npz", out / "fiducial.npz")

    R = sy.Runner(opts, out)
    rc, kb = a2744_kappa_profile(R.cat.full, R.fid_ra, R.fid_dec)
    print("A2744 Lenstool kappa(r): " + ", ".join(f"{r:.0f} kpc {k:.3f}" for r, k in zip(rc[::4], kb[::4])))

    original = sy.cluster_member_contamination
    current = {}

    def contamination(a):                             # sy.sim_transforms builds this per configuration
        return SchrabbackContamination(current["A"], rc, kb) if "A" in current else original(a)
    sy.cluster_member_contamination = contamination

    res = {"r_s_kpc": R_S, "a2744_profile": {"r_kpc": rc.tolist(), "kappa": kb.tolist()}}
    fid, n_fid = R.cached("fiducial", lambda d: R.evaluate(d))
    res["fiducial"] = {"metrics": R.summarise(fid), "n_gal": n_fid}
    sy.print_row("fiducial", res["fiducial"]["metrics"])
    res["boost"] = []
    for A in opts.A:
        current["A"] = A
        # a non-zero cluster_member_contamination makes SimMaps.sample add the kappa channel
        r, n = R.cached(f"boost_A{A:g}", lambda d: R.evaluate(d, args_over={"cluster_member_contamination": 1e-3}))
        res["boost"].append({"A": A, "B_100kpc": 1 + A * np.exp(1 - 100 / R_S), "B_300kpc": 1 + A * np.exp(1 - 300 / R_S),
                             "metrics": R.summarise(r)})
        sy.print_row(f"A = {A:g}", res["boost"][-1]["metrics"], res["fiducial"]["metrics"])
    current.clear()
    json.dump(res, open(out / "boost_kappa.json", "w"), indent=2)
    plot(res, out / "boost_kappa")
    print(f"wrote {out}")


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rc, kb = np.array(res["a2744_profile"]["r_kpc"]), np.array(res["a2744_profile"]["kappa"])
    A = np.array([b["A"] for b in res["boost"]])
    get = lambda k: np.array([b["metrics"]["combined"][k] for b in res["boost"]])
    f = res["fiducial"]["metrics"]["combined"]
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4), constrained_layout=True)
    r = np.geomspace(20, 2000, 200)
    for a in (0.1, 0.3, 1.0):
        ax[0].plot(r, 1 + a * np.exp(1 - r / R_S), label=f"A = {a:g}")
    ax[0].set(xscale="log", xlabel="r [kpc]", ylabel="B(r)", title=f"Schrabback+25 boost, r_s = {R_S:g} kpc")
    ax[0].legend()
    ax[1].plot(rc, kb, "k")
    ax[1].set(xscale="log", yscale="log", xlabel="r [kpc]", ylabel="κ (Lenstool, zs = 2.28)",
              title="A2744 azimuthal κ(r): maps κ → r")
    ax[2].plot(np.r_[0, A], np.r_[f["log_sigma"], get("log_sigma")], "o-")
    lo = np.r_[f["log_sigma"] - f["log_sigma_err"][0], get("log_sigma") - np.array([b["metrics"]["combined"]["log_sigma_err"][0] for b in res["boost"]])]
    hi = np.r_[f["log_sigma"] + f["log_sigma_err"][1], get("log_sigma") + np.array([b["metrics"]["combined"]["log_sigma_err"][1] for b in res["boost"]])]
    ax[2].fill_between(np.r_[0, A], lo, hi, alpha=0.2)
    ax[2].set(xlabel="A", ylabel="log10 σ/m (A2744, 16–84%)", title="inferred cross-section")
    ax[3].plot(np.r_[0, A], np.r_[f["z_cdm"], get("z_cdm")], "o-")
    ax[3].set(xlabel="A", ylabel="z_CDM [σ]", title="significance against CDM")
    for a in ax[2:]:
        sec = a.secondary_xaxis("top", functions=(lambda x: x * np.exp(1 - 100 / R_S), lambda y: y / np.exp(1 - 100 / R_S)))
        sec.set_xlabel("B(100 kpc) − 1")
    fig.suptitle("Cluster-member dilution of the simulations with B(κ) = 1 + A exp(1 − r_A2744(κ)/r_s)  (BAHAMAS + DARKSKIES)")
    for e in ("png", "pdf"):
        fig.savefig(f"{path}.{e}", dpi=150)


if __name__ == "__main__":
    main()
