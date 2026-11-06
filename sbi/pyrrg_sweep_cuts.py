#!/usr/bin/env python
'''
Optimise the shear cuts for each pyRRG run of the A2744 mosaics (moment weight `mult` x webbPSF PSF library,
/Users/davidharvey/Work/webbPSF/cats_sweep/<tag>/<filter>/; 'live' = data/100/a2744, the 28/10 run), as
sbi/optimise_cuts.py but with the current data pipeline: per filter and set of cuts, pyRRG calc_shear (G1 refitted
on the selection) and then the 2-D G1(size, S/N) of shear_susceptibility (fixed bandwidths, the CV choice for the
fiducial catalogue), UNCOVER cluster members removed, the filters combined (galaxies in both get the mean shear),
strong-lensing kappa < KAPPA_THRESH, the 100x100 stamp.

The per-galaxy quantities that do not depend on the cuts (members, photo-z's, kappa, the Lenstool model g_t, the
f115w-f150w pairs, one-to-one within 2" on the uncut catalogues) are computed once per run, so a set of cuts costs
milliseconds. Per run: coordinate ascent on the tangential S/N (each filter's cuts in turn), recording for every
set evaluated
  snr_t   sqrt(sum_bins (<g_t> / sigma_bin)^2), 100 kpc - 1.4 Mpc (shear estimates gamma, not chi)
  chi2_x  cross-shear chi^2 over the same bins (null)
  A       measured g_t / Lenstool-model g_t at the same galaxies (90 kpc - 1 Mpc), its error and fit chi^2

    astro-env/bin/python sbi/pyrrg_sweep_cuts.py --tags live m2_p2.0mar26 m1_p1.5 ...
'''
import argparse
import contextlib
import io
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.io import fits

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "sbi")]
import add_shear_to_data as asd  # noqa: E402
from optimise_cuts import FID_CUTS, MEMBER_Z, coords, mutual_match  # noqa: E402
from pyrrg_sweep import EDGES, SWEEP, ROOT, LenstoolShear  # noqa: E402
from pyRRG.calc_shear import calc_shear  # noqa: E402
from shear_susceptibility import RHODES_FACTOR, local_linear  # noqa: E402

FILTERS = ["f115w", "f150w"]
H = {"f115w": (0.22, 0.30), "f150w": (0.22, 0.45)}        # 2-D G1 bandwidths (CV on the fiducial catalogue)
GRID = {"signal_noise_cut": [0, 2, 4, 6, 8, 10], "size_lo": [2, 2.5, 2.75, 3, 3.5, 4], "size_hi": [200],
        "mag_hi": [27, 27.5, 28, 29]}


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--out", default=str(REPO / "sbi" / "results" / "pyrrg_sweep" / "pyrrg_sweep_cuts.json"))
    return p.parse_args()


def raw(tag, f):
    path = (REPO / "data" / "100" / "a2744" / ROOT.format(f)) if tag == "live" else SWEEP / tag / f / ROOT.format(f)
    return fits.open(path)[1].data


def g1_2d(cat, f, ngrid=40):
    '''G1(size, snr) by local-linear regression of g1_gal with the fixed bandwidths, evaluated at the galaxies.'''
    snr, size, g = (np.asarray(cat[k], float) for k in ("snr", "gal_size", "g1_gal"))
    ok = np.isfinite(snr * size * g) & (snr > 0) & (size > 0)
    X = np.c_[np.log10(size[ok]), np.log10(snr[ok])]
    mu, sd = X.mean(0), X.std(0)
    gz, gs = (np.linspace(X[:, i].min(), X[:, i].max(), ngrid) for i in (0, 1))
    Q = np.array(np.meshgrid(gz, gs, indexing="ij")).reshape(2, -1).T
    grid = local_linear((X - mu) / sd, g[ok], (Q - mu) / sd, np.array(H[f])).reshape(ngrid, ngrid)
    from scipy.interpolate import RegularGridInterpolator
    it = RegularGridInterpolator((gz, gs), grid)
    q = np.c_[np.log10(np.clip(size, 10 ** gz[0], 10 ** gz[-1])), np.log10(np.clip(snr, 10 ** gs[0], 10 ** gs[-1]))]
    return it(q)


class Run:
    def __init__(self, tag, model, uncover):
        self.tag = tag
        self.raw = {f: raw(tag, f) for f in FILTERS}
        cu = coords(uncover, "ra", "dec")
        self.gal = {}
        for f in FILTERS:
            r = self.raw[f]
            i, d, _ = coords(r).match_to_catalog_sky(cu)
            matched = d < 0.5 * u.arcsec
            z = np.where(matched, uncover["z_ml"][i], np.nan)
            member = matched & (z < MEMBER_Z)
            self.gal[f] = {"member": member, "z": z, "number": np.asarray(r["NUMBER"])}
        zs = np.concatenate([self.gal[f]["z"][~self.gal[f]["member"]] for f in FILTERS])
        zdef = asd.sig_mean(zs[np.isfinite(zs)])
        for f in FILTERS:
            g, r = self.gal[f], self.raw[f]
            z = np.where(np.isfinite(g["z"]), g["z"], zdef)
            d = {"x": np.asarray(r["x"]), "y": np.asarray(r["y"]), "RA": np.asarray(r["RA"]),
                 "DEC": np.asarray(r["DEC"]), "redshift": z}
            px, py = asd.ra_dec_to_simulation_image_pos(d)
            g.update(r=20. * np.hypot(px, py), phi=np.arctan2(py, px),
                     use=(~g["member"]) & (asd.strong_lensing_kappa(d) < asd.KAPPA_THRESH) & (np.abs(px) < 50) & (np.abs(py) < 50))
            g["gt_model"] = model.gt(d, g["phi"])
        a, b = self.raw["f115w"], self.raw["f150w"]
        self.ia, self.ib = mutual_match(coords(a), coords(b), 2 * u.arcsec)
        self.shear = {f: {} for f in FILTERS}

    def calibrate(self, f, c, tmp):
        snr, lo, hi, mag = c
        kw = {**FID_CUTS, "signal_noise_cut": snr, "size_cut": [lo, hi], "mag_cut": [0, mag], "stat_type": "snr",
              "verbose": False}
        with contextlib.redirect_stdout(io.StringIO()):
            calc_shear(self.raw[f], tmp, **kw)
        s = fits.open(tmp)[1].data
        G1 = g1_2d(s, f)
        gam = np.full((len(self.raw[f]), 2), np.nan)
        idx = np.searchsorted(self.gal[f]["number"], s["NUMBER"], sorter=np.argsort(self.gal[f]["number"]))
        idx = np.argsort(self.gal[f]["number"])[idx]
        gam[idx, 0] = s["e1"] / G1 / RHODES_FACTOR
        gam[idx, 1] = s["e2"] / G1 / RHODES_FACTOR
        self.shear[f][c] = gam

    def evaluate(self, c115, c150):
        A_, B_ = self.shear["f115w"][c115], self.shear["f150w"][c150]
        ga, gb = self.gal["f115w"], self.gal["f150w"]
        sa, sb = np.isfinite(A_[:, 0]) & ~ga["member"], np.isfinite(B_[:, 0]) & ~gb["member"]
        both = sa[self.ia] & sb[self.ib]
        pa, pb = self.ia[both], self.ib[both]
        only_a = sa.copy(); only_a[pa] = False
        only_b = sb.copy(); only_b[pb] = False
        g = np.r_[(A_[pa] + B_[pb]) / 2, A_[only_a], B_[only_b]]
        cat = {k: np.r_[ga[k][pa], ga[k][only_a], gb[k][only_b]] for k in ("r", "phi", "use", "gt_model")}
        m = cat["use"]
        r, phi = cat["r"][m], cat["phi"][m]
        gt = -(g[m, 0] * np.cos(2 * phi) + g[m, 1] * np.sin(2 * phi))
        gx = g[m, 0] * np.sin(2 * phi) - g[m, 1] * np.cos(2 * phi)
        gm = cat["gt_model"][m]
        b = np.digitize(r, EDGES) - 1
        nb = len(EDGES) - 1
        n = np.bincount(b[(b >= 0) & (b < nb)], minlength=nb)
        def mean_err(v):
            ok = (b >= 0) & (b < nb) & np.isfinite(v)
            s1 = np.bincount(b[ok], v[ok], nb); s2 = np.bincount(b[ok], v[ok] ** 2, nb); k = np.bincount(b[ok], minlength=nb)
            mu = s1 / np.maximum(k, 1)
            return mu, np.sqrt(np.maximum(s2 / np.maximum(k, 1) - mu ** 2, 0) / np.maximum(k - 1, 1))
        t, te = mean_err(gt); xx, xe = mean_err(gx); mt, _ = mean_err(gm)
        rc = np.sqrt(EDGES[:-1] * EDGES[1:])
        sel = (rc > 100) & (n > 1)
        fit = (rc > 90) & (rc < 1000) & (n > 1)
        w = mt[fit] ** 2 / te[fit] ** 2
        A = np.sum(t[fit] * mt[fit] / te[fit] ** 2) / w.sum()
        return {"n": int((r < 1400).sum()), "snr_t": float(np.sqrt(np.sum((t[sel] / te[sel]) ** 2))),
                "chi2_x": float(np.sum((xx[sel] / xe[sel]) ** 2)), "nbins": int(sel.sum()),
                "A": float(A), "A_err": float(1 / np.sqrt(w.sum())),
                "chi2_A": float(np.sum(((t[fit] - A * mt[fit]) / te[fit]) ** 2)), "nbins_A": int(fit.sum())}


def main():
    opts = parse()
    model = LenstoolShear()
    uncover = fits.open(REPO / "data" / "100" / "a2744" / "UNCOVER_DR4_SPS_catalog.fits")[1].data
    out = Path(opts.out); out.parent.mkdir(parents=True, exist_ok=True)
    res = json.load(open(out)) if out.exists() else {}
    combos = [c for c in itertools.product(*GRID.values()) if c[1] < c[2]]
    base = {f: (asd.SHEAR_CUTS[f]["signal_noise_cut"], *asd.SHEAR_CUTS[f]["size_cut"], asd.SHEAR_CUTS[f]["mag_cut"][1])
            for f in FILTERS}
    tmp = tempfile.NamedTemporaryFile(suffix=".fits", delete=False).name
    for tag in opts.tags:
        run = Run(tag, model, uncover)
        for f in FILTERS:
            for c in set(combos) | {base[f]}:
                try:
                    run.calibrate(f, c, tmp)
                except Exception as e:                 # e.g. too few galaxies for the G1 fit
                    print(f"   {tag} {f} {c}: {type(e).__name__}")
        seen = {}

        def row(c115, c150):
            if (c115, c150) not in seen:
                seen[(c115, c150)] = {"f115w": list(c115), "f150w": list(c150), **run.evaluate(c115, c150)}
            return seen[(c115, c150)]

        b = row(base["f115w"], base["f150w"])
        cur = dict(base)
        for _ in range(4):
            prev = dict(cur)
            cur["f115w"] = max(run.shear["f115w"], key=lambda c: row(c, cur["f150w"])["snr_t"])
            cur["f150w"] = max(run.shear["f150w"], key=lambda c: row(cur["f115w"], c)["snr_t"])
            if cur == prev:
                break
        best = row(cur["f115w"], cur["f150w"])
        rows = list(seen.values())
        res[tag] = {"baseline": b, "best_snr": best, "rows": rows}
        f = lambda r: (f"n {r['n']:5d} S/N_t {r['snr_t']:5.1f} chi2_x {r['chi2_x']:5.1f}/{r['nbins']} "  # noqa: E731
                       f"A {r['A']:.3f}±{r['A_err']:.3f} (chi2 {r['chi2_A']:.1f}/{r['nbins_A']})")
        print(f"{tag:14s} SHEAR_CUTS: {f(b)}", flush=True)
        print(f"{'':14s} best S/N:   {f(best)}  f115w {best['f115w']} f150w {best['f150w']}", flush=True)
        json.dump(res, open(out, "w"), indent=1)
    os.remove(tmp)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
