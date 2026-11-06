#!/usr/bin/env python
'''
Compare pyRRG runs of the A2744 mosaics (different moment weight multipliers `mult` and webbPSF PSF libraries,
/Users/davidharvey/Work/webbPSF/cats_sweep/<tag>/<filter>/, made by cats_sweep/run_config.sh) through the data
pipeline: add_shear_to_data.get_obs_data (SHEAR_CUTS with the lower size cut scanned, 2-D G1(size, S/N), cluster
members removed, filters combined, UNCOVER photo-z's) in a scratch data_dir per run, kappa < KAPPA_THRESH.

Per run and size cut, from the galaxies' shear estimates gamma about the stamp centre (100x100 stamp):
  n        galaxies within 1.4 Mpc
  snr_t    tangential S/N, sqrt(sum_bins (<g_t> / sigma_bin)^2), 100 kpc - 1.4 Mpc
  chi2_x   cross-shear chi^2 over the same bins (null test; ~ number of bins)
  A        amplitude of the measured g_t relative to the Bergamini+23 Lenstool model at the same galaxies
           (reduced shear from its deflection field, scaled to each galaxy's photo-z), 90 kpc - 1 Mpc, with error
           and the chi^2 of that one-parameter fit

    astro-env/bin/python sbi/pyrrg_sweep.py --tags m2_p2.0mar26 m1_p1.0 ...
'''
import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import add_shear_to_data as asd  # noqa: E402

SWEEP = Path("/Users/davidharvey/Work/webbPSF/cats_sweep")
DPL = Path("/private/tmp/claude-502/-Users-davidharvey-Applications-Imperial-Job/2fd953fc-3740-40a9-b0d1-91ed4c06ee8a/scratchpad/gt_sl/lenstool")
ROOT = "abell2744clu-grizli-v5.4-{}-clear_drc_sci_clean.shears"
EDGES = np.geomspace(40, 1500, 19)
ZL, ZS_MAP = 0.3072, 2.28
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", required=True, help="cats_sweep/<tag>; 'live' = data/100/a2744 (the 28/10 run)")
    p.add_argument("--size-lo", type=float, nargs="+", default=[2.0, 2.5, 2.75, 3.0, 3.5])
    p.add_argument("--work", default=str(SWEEP / "eval"))
    p.add_argument("--out", default=str(REPO / "sbi" / "results" / "pyrrg_sweep" / "pyrrg_sweep.json"))
    return p.parse_args()


class LenstoolShear:
    '''Reduced shear of the Lenstool model at galaxy positions and photo-z's (from the zs = 2.28 deflection maps).'''

    def __init__(self):
        hx, hy = fits.open(DPL / "dplx_zs2.28.fits")[0], fits.open(DPL / "dply_zs2.28.fits")[0]
        ax, ay = hx.data.astype(float), hy.data.astype(float)
        step = 940. / (ax.shape[0] - 1)
        dxax, dyax = np.gradient(ax, step, axis=1), np.gradient(ax, step, axis=0)
        dxay, dyay = np.gradient(ay, step, axis=1), np.gradient(ay, step, axis=0)
        self.k, self.g1, self.g2 = 0.5 * (dxax + dyay), 0.5 * (dxax - dyay), 0.5 * (dyax + dxay)
        self.wcs = WCS(hx.header)

    @staticmethod
    def beta(z):
        z = np.asarray(z, float); out = np.zeros_like(z); ok = z > ZL + 0.01
        out[ok] = (cosmo.angular_diameter_distance_z1z2(ZL, z[ok]) / cosmo.angular_diameter_distance(z[ok])).value
        return out

    def gt(self, cat, phi_cat):
        '''Model g_t at the galaxies, in the catalogue frame (spin-2 rotated from the map frame).'''
        pi, pj = self.wcs.world_to_pixel_values(np.asarray(cat["RA"]), np.asarray(cat["DEC"]))
        s = lambda m: map_coordinates(m, [pj, pi], order=1, mode="constant", cval=np.nan)
        w = self.beta(cat["redshift"]) / self.beta(np.array([ZS_MAP]))[0]
        k, g1, g2 = s(self.k) * w, s(self.g1) * w, s(self.g2) * w
        g = (g1 + 1j * g2) / (1 - k)
        A = np.c_[cat["x"], cat["y"], np.ones(len(cat["x"]))]
        J = np.array([np.linalg.lstsq(A, pi, rcond=None)[0][:2], np.linalg.lstsq(A, pj, rcond=None)[0][:2]])
        M = J / np.sqrt(abs(np.linalg.det(J)))
        S = np.array([[g.real, g.imag], [g.imag, -g.real]]).transpose(2, 0, 1)
        Sc = np.einsum("ki,nkl,lj->nij", M, S, M)
        return -(Sc[:, 0, 0] * np.cos(2 * phi_cat) + Sc[:, 0, 1] * np.sin(2 * phi_cat))


def catalogue(tag, size_lo, work):
    '''get_obs_data on the run's catalogues in a scratch data_dir (photo-z's, strong-lensing galaxies kept).'''
    d = Path(work) / tag
    d.mkdir(parents=True, exist_ok=True)
    src = REPO / "data" / "100" / "a2744" if tag == "live" else SWEEP / tag
    for f in ("f115w", "f150w"):
        link, target = d / ROOT.format(f), (src / ROOT.format(f) if tag == "live" else src / f / ROOT.format(f))
        if not link.exists():
            link.symlink_to(target)
    unc = d / "UNCOVER_DR4_SPS_catalog.fits"
    if not unc.exists():
        unc.symlink_to(REPO / "data" / "100" / "a2744" / "UNCOVER_DR4_SPS_catalog.fits")
    cuts = {f: {**c, "size_cut": [size_lo, c["size_cut"][1]]} for f, c in asd.SHEAR_CUTS.items()}
    cwd = os.getcwd(); os.chdir(d)                     # run_match writes its matched_*.fits to the cwd
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return asd.get_obs_data("concat", data_dir=str(d), cuts=cuts, photoz=True, drop_strong_lensing=False)
    finally:
        os.chdir(cwd)


def binned(v, r, m):
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = m & (r >= lo) & (r < hi) & np.isfinite(v)
        out.append((np.sqrt(lo * hi), s.sum(), v[s].mean() if s.sum() > 1 else np.nan,
                    v[s].std() / np.sqrt(s.sum()) if s.sum() > 1 else np.nan))
    return np.array(out).T


def evaluate(cat, model):
    x, y = asd.ra_dec_to_simulation_image_pos(cat)
    r, phi = 20. * np.hypot(x, y), np.arctan2(y, x)
    keep = (asd.strong_lensing_kappa(cat) < asd.KAPPA_THRESH) & (np.abs(x) < 50) & (np.abs(y) < 50)
    gt = -(cat["gamma1"] * np.cos(2 * phi) + cat["gamma2"] * np.sin(2 * phi))
    gx = cat["gamma1"] * np.sin(2 * phi) - cat["gamma2"] * np.cos(2 * phi)
    gm = model.gt(cat, phi)
    rc, n, t, te = binned(gt, r, keep)
    _, _, xx, xe = binned(gx, r, keep)
    _, _, mt, _ = binned(gm, r, keep)
    sel = (rc > 100) & np.isfinite(t)
    fit = (rc > 90) & (rc < 1000) & np.isfinite(t) & np.isfinite(mt)
    wA = mt[fit] ** 2 / te[fit] ** 2
    A = np.sum(t[fit] * mt[fit] / te[fit] ** 2) / wA.sum()
    return {"n": int((keep & (r < 1400)).sum()),
            "snr_t": float(np.sqrt(np.sum((t[sel] / te[sel]) ** 2))),
            "chi2_x": float(np.sum((xx[sel] / xe[sel]) ** 2)), "nbins": int(sel.sum()),
            "A": float(A), "A_err": float(1 / np.sqrt(wA.sum())),
            "chi2_A": float(np.sum(((t[fit] - A * mt[fit]) / te[fit]) ** 2)), "nbins_A": int(fit.sum()),
            "median_size": float(np.median(cat["SIZE"]))}


def main():
    opts = parse()
    model = LenstoolShear()
    out = Path(opts.out); out.parent.mkdir(parents=True, exist_ok=True)
    res = json.load(open(out)) if out.exists() else {}
    print(f"{'run':16s} {'size>':>5s} {'n':>6s} {'S/N_t':>6s} {'chi2_x/nb':>9s} {'A':>11s} {'chi2_A/nb':>9s}")
    for tag in opts.tags:
        for s in opts.size_lo:
            m = evaluate(catalogue(tag, s, opts.work), model)
            res[f"{tag}|{s:g}"] = m
            print(f"{tag:16s} {s:5g} {m['n']:6d} {m['snr_t']:6.1f} {m['chi2_x']:5.1f}/{m['nbins']:<3d} "
                  f"{m['A']:.3f}±{m['A_err']:.3f} {m['chi2_A']:5.1f}/{m['nbins_A']:<3d}", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
