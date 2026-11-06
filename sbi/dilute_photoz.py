#!/usr/bin/env python
'''
Cluster-member dilution of the simulations at the level of the photo-z leak check
(sbi/results/injection_dilution/photoz_leak_check.json): the fraction of selected sources at 0.455 < z_ph < 0.6 (just
above the member cut) in radial bins about the A2744 stamp centre, minus its field level (the N-weighted mean beyond
250 kpc), is taken as the leaked-member fraction f_cl(r). Each simulated map's shear is multiplied by 1 - f_cl(r)
(r about the stamp centre, 20 kpc pixels) before the shape noise is added; A2744's map is left as it is. --scale
multiplies f_cl (the leak check is a lower bound: members with catastrophic photo-z do not show up in it).
Uses the systematics.py machinery; the fiducial is reused from --fiducial-from so each scale is paired with it.

    astro-env/bin/python sbi/dilute_photoz.py --model tweakedalign_k0p7_ctr_wb1_ba_final --include-str spin2_cx_pgc_ft_ \
        --nmodels 5 --summaries sbi/summaries_wb1_final_n5 --out-dir sbi/results/dilute_photoz_wb1_final_5models \
        --fiducial-from sbi/results/systematics_wb1_final
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

LEAK = REPO / "sbi" / "results" / "injection_dilution" / "photoz_leak_check.json"
KPC_PER_PX = 20.
FIELD_R = 250.                                       # kpc: bins beyond this set the field leak level


def leak_profile():
    '''Radial bins [kpc] and the leaked-member fraction above the field level in each (rows: r_lo, r_hi, N, frac, ...).'''
    t = np.array(json.load(open(LEAK)))
    lo, hi, n, frac = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    field = np.sum((n * frac)[lo >= FIELD_R]) / np.sum(n[lo >= FIELD_R])
    return lo, hi, np.clip(frac - field, 0, None), field


class RadialDilution:
    '''Multiplies (e1, e2) by 1 - scale * f_cl(r); a kappa third channel, if present, is dropped.'''

    def __init__(self, lo, hi, f, scale, npix=sy.NPIX):
        x = (np.arange(npix) - (npix - 1) / 2) * KPC_PER_PX
        r = np.hypot(*np.meshgrid(x, x))
        fmap = np.zeros_like(r)
        for a, b, v in zip(lo, hi, f):
            fmap[(r >= a) & (r < b)] = v
        self.factor = torch.tensor(1 - np.clip(scale * fmap, 0, 1), dtype=torch.float32)

    def __call__(self, sample):
        return sample[:2] * self.factor


def main():
    p = sy.argparse.ArgumentParser(description=__doc__, formatter_class=sy.argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--include-str", required=True)
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--summaries", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fiducial-from", required=True, help="systematics out-dir whose fiducial.npz to reuse")
    p.add_argument("--scale", type=float, nargs="+", default=[1., 2.])
    p.add_argument("--nmodels", type=int, default=None, help="only the first n models (quick look)")
    p.add_argument("--n-noise", type=int, default=1)
    p.add_argument("--noise-seed", type=int, default=4321)
    p.add_argument("--batch-size", type=int, default=256)
    opts = p.parse_args()
    opts.redo, opts.from_cache, opts.max_clusters = False, False, None
    sy.A2744_NCOMP = tuple(range(opts.min_ncomp, 7))
    out = sy.LAUNCH_DIR / opts.out_dir
    out.mkdir(parents=True, exist_ok=True)
    opts.summaries = str((sy.LAUNCH_DIR / opts.summaries).resolve())
    shutil.copy(sy.LAUNCH_DIR / opts.fiducial_from / "fiducial.npz", out / "fiducial.npz")

    R = sy.Runner(opts, out)
    fid, n_fid = R.cached("fiducial", lambda d: None)
    if opts.nmodels:                                  # the first n models (load order, as in the fiducial cache)
        R.models = R.models[:opts.nmodels]
        fid = {d: (s[:opts.nmodels], o[:opts.nmodels]) for d, (s, o) in fid.items()}
    lo, hi, f, field = leak_profile()
    print(f"photo-z leak: field level {field:.3f}; f_cl above it: " +
          ", ".join(f"{a:.0f}-{b:.0f} kpc {v:.3f}" for a, b, v in zip(lo, hi, f)))

    original = sy.cluster_member_contamination
    current = {}

    def contamination(a):                             # sy.sim_transforms builds this per configuration
        return RadialDilution(lo, hi, f, current["scale"]) if "scale" in current else original(a)
    sy.cluster_member_contamination = contamination

    res = {"f_cl": {"r_lo": lo.tolist(), "r_hi": hi.tolist(), "f": f.tolist(), "field": field},
           "fiducial": {"metrics": R.summarise(fid)}, "dilution": []}
    sy.print_row("fiducial", res["fiducial"]["metrics"])
    for s in opts.scale:
        current["scale"] = s
        # a non-zero cluster_member_contamination makes SimMaps.sample add the kappa channel (dropped here)
        r, n = R.cached(f"photoz_x{s:g}" + (f"_n{opts.nmodels}" if opts.nmodels else ""),
                        lambda d: R.evaluate(d, args_over={"cluster_member_contamination": 1e-3}))
        res["dilution"].append({"scale": s, "f_cl_core": float(s * f[0]), "metrics": R.summarise(r)})
        sy.print_row(f"f_cl x {s:g}", res["dilution"][-1]["metrics"], res["fiducial"]["metrics"])
        c = res["dilution"][-1]["metrics"]["bahamas"]["curve"]
        print("  BAHAMAS curve: " + ", ".join(f"{k} {v:.3f}" for k, v in c.items()) +
              f"; A2744 P(SIDM) {res['dilution'][-1]['metrics']['combined']['a2744_p_sidm']:.3f}")
    current.clear()
    c = res["fiducial"]["metrics"]["bahamas"]["curve"]
    print("  fiducial BAHAMAS curve: " + ", ".join(f"{k} {v:.3f}" for k, v in c.items()))
    json.dump(res, open(out / "dilute_photoz.json", "w"), indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
