#!/usr/bin/env python
'''
Radial profiles of the A2744 main halo about Lenstool O2 with and without the Lenstool excess (excess_mass.py):
  left   convergence (zs = 2.28): the Lenstool model (no excess) and the model plus the excess S/N >= --min-snr part
         (KAPPA_E image-plane and KAPPA_E_SRCPOS maps, times the KS-map dilution, scaled from <Dls/Ds> of the
         sources to zs = 2.28); the srcpos map is binned at source-plane positions and drawn as if image-plane
  right  mean tangential distortion of the catalogue galaxies (kappa < KAPPA_THRESH) as observed and with the excess
         shear subtracted (as sbi/excess_removed.py), and the difference

    astro-env/bin/python sbi/excess_profile.py
'''
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]
os.chdir(REPO / "notebooks")
from excess_removed import COSMO, LT, LT_FIELD, LT_REF, Z_LENS, Z_MAP, beta, kappa_to_gamma  # noqa: E402
from excess_boost import O2  # noqa: E402
from add_shear_to_data import (  # noqa: E402
    KAPPA_THRESH, SHEAR_CUTS, SL_KAPPA_MAP, galaxy_distortion, get_obs_data, strong_lensing_kappa,
)

KPC_PER_ARCSEC = COSMO.kpc_proper_per_arcmin(Z_LENS).to(u.kpc / u.arcsec).value


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-snr", type=float, default=2.)
    p.add_argument("--edges", type=float, nargs="+", default=list(np.geomspace(20, 1200, 19)))
    p.add_argument("--out", default=str(REPO / "sbi" / "results" / "main_halo_profile" / "excess_profile"))
    return p.parse_args()


def radial(r, v, edges, w=None):
    w = np.ones_like(v) if w is None else w
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi) & np.isfinite(v)
        n = m.sum()
        rows.append((np.sqrt(lo * hi), np.average(v[m], weights=w[m]) if n else np.nan,
                     v[m].std() / np.sqrt(n) if n >= 2 else np.nan, n))
    return np.array(rows).T


def main():
    opts = parse()
    edges = np.array(opts.edges)
    cat = get_obs_data("concat", data_dir="../data/100/a2744/", photoz=True, cuts=SHEAR_CUTS)
    keep = strong_lensing_kappa(cat) < KAPPA_THRESH
    x = -(np.asarray(cat["RA"]) - LT_REF[0]) * np.cos(np.deg2rad(LT_REF[1])) * 3600.
    y = (np.asarray(cat["DEC"]) - LT_REF[1]) * 3600.
    b = beta(np.asarray(cat["redshift"]))
    to_228 = beta(2.28)[0] / b[np.asarray(cat["redshift"]) > Z_LENS + 0.1].mean()

    with fits.open(LT / "excess_kappa.fits") as h:
        hd = h["KAPPA_E"].header
        sig = h["SIGMA"].data.astype(float)
        km = {nm: h[nm].data.astype(float) for nm in ("KAPPA_E", "KAPPA_E_SRCPOS", "KAPPA_E_RAW")}
    mx = hd["CRVAL1"] + hd["CDELT1"] * (np.arange(sig.shape[1]) + 1 - hd["CRPIX1"])
    my = hd["CRVAL2"] + hd["CDELT2"] * (np.arange(sig.shape[0]) + 1 - hd["CRPIX2"])
    X, Y = np.meshgrid(mx, my)

    def shear_at(k, px, py):
        g1, g2 = kappa_to_gamma(k)
        return np.array([RegularGridInterpolator((my, mx), g, bounds_error=False, fill_value=0.)(np.c_[py, px]) for g in (g1, g2)])
    t = shear_at(km["KAPPA_E_RAW"], x, y)
    inside = keep & (x > mx[0]) & (x < mx[-1]) & (y > my[0]) & (y < my[-1])
    M = np.linalg.lstsq(t[:, inside].T, np.c_[cat["gamma1"][inside], cat["gamma2"][inside]], rcond=None)[0].T
    dil = float(np.mean(np.diag(M)))
    print(f"KS-map dilution {dil:.2f}; kappa(<Dls/Ds> of the sources) -> kappa(zs = 2.28): x{to_228:.3f}")

    def excess_region(nm):
        snr = km[nm] / sig
        lab, _ = label(snr >= opts.min_snr)
        near = (np.hypot(X - O2[0], Y - O2[1]) < 60) & (lab > 0)
        pk = np.unravel_index(np.argmax(np.where(near, snr, -np.inf)), snr.shape)
        return np.where(lab == lab[pk], km[nm], 0.)

    # convergence profiles
    with fits.open(SL_KAPPA_MAP) as h:
        k_sl, w = h[0].data.astype(float), WCS(h[0].header)
    jj, ii = np.indices(k_sl.shape)
    sra, sdec = w.pixel_to_world_values(ii.ravel(), jj.ravel())
    sx = -(sra - LT_REF[0]) * np.cos(np.deg2rad(LT_REF[1])) * 3600.
    sy = (sdec - LT_REF[1]) * 3600.
    r_sl = KPC_PER_ARCSEC * np.hypot(sx - O2[0], sy - O2[1])
    rc, k_lt, _, _ = radial(r_sl, k_sl.ravel(), edges)
    r_ex = KPC_PER_ARCSEC * np.hypot(X - O2[0], Y - O2[1]).ravel()
    exc = {}
    for nm, lab_ in (("KAPPA_E", "image-plane excess"), ("KAPPA_E_SRCPOS", "source-plane excess")):
        e = dil * to_228 * excess_region(nm).ravel()
        exc[lab_] = radial(r_ex, e, edges)[1]
        print(f"{lab_}: mean kappa added (zs 2.28) " +
              ", ".join(f"{lo:.0f}-{hi:.0f} kpc {np.mean(e[(r_ex >= lo) & (r_ex < hi)]):.3f}" for lo, hi in ((0, 100), (100, 200), (200, 400))))

    # tangential shear with and without the excess
    ax_, ay_ = (fits.getdata(LT / f).astype(float) for f in ("dx_z2.fits", "dy_z2.fits"))
    xg = np.linspace(LT_FIELD[0], LT_FIELD[1], ax_.shape[1])
    yg = np.linspace(LT_FIELD[2], LT_FIELD[3], ax_.shape[0])
    s = b / beta(Z_MAP)[0]
    dfl = [np.nan_to_num(RegularGridInterpolator((yg, xg), a, bounds_error=False, fill_value=0.)(np.c_[y, x])) for a in (ax_, ay_)]
    xs, ys = x - s * dfl[0], y - s * dfl[1]
    removed = {"source-plane excess": M @ shear_at(excess_region("KAPPA_E_SRCPOS"), xs, ys),
               "image-plane excess": M @ shear_at(excess_region("KAPPA_E"), x, y)}
    dx, dy = KPC_PER_ARCSEC * (x - O2[0]), KPC_PER_ARCSEC * (y - O2[1])
    r_g, phi = np.hypot(dx, dy), np.arctan2(dy, dx)

    def chi_t(g1, g2):
        c1, c2 = galaxy_distortion({"gamma1": g1, "gamma2": g2})
        return -(c1 * np.cos(2 * phi) + c2 * np.sin(2 * phi))
    g1, g2 = np.asarray(cat["gamma1"]), np.asarray(cat["gamma2"])
    prof = {"observed": radial(r_g[keep], chi_t(g1, g2)[keep], edges)}
    for nm, g in removed.items():
        prof[f"{nm} removed"] = radial(r_g[keep], chi_t(g1 - g[0], g2 - g[1])[keep], edges)

    import sensitivity as sens
    plt = sens.style()
    fig, (axk, axt, axd) = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    cols = {"source-plane excess": "#D55E00", "image-plane excess": "#0072B2"}
    ok = np.isfinite(k_lt)
    axk.plot(rc[ok], k_lt[ok], "k-", lw=2.2, label="Lenstool model (excess removed)")
    for nm, e in exc.items():
        axk.plot(rc[ok], k_lt[ok] + np.nan_to_num(e[ok]), "--", color=cols[nm], lw=1.8, label=f"Lenstool + {nm} (S/N ≥ {opts.min_snr:g}, ×{dil:.1f})")
        axk.plot(rc, np.where(np.nan_to_num(e) > 0, e, np.nan), ":", color=cols[nm], lw=1.2, label=f"{nm} alone")
    axk.axhline(KAPPA_THRESH, color="0.5", ls=":", lw=1)
    axk.text(rc[ok][-1], KAPPA_THRESH * 1.05, f"κ = {KAPPA_THRESH:g} (strong-lensing cut)", ha="right", fontsize=8, color="0.4")
    axk.set_xscale("log"); axk.set_yscale("log")
    axk.set_ylim(5e-3, 5)
    axk.set_ylabel(r"azimuthal mean $\kappa$ about O2 ($z_s$ = 2.28)")
    axk.set_title("Convergence of the main halo", fontsize=10)

    st = {"observed": ("k", "o-"), "source-plane excess removed": (cols["source-plane excess"], "s--"),
          "image-plane excess removed": (cols["image-plane excess"], "^--")}
    for i, (nm, (r_, v, e, n)) in enumerate(prof.items()):
        f = 1 + 0.02 * (i - 1)
        axt.errorbar(r_ * f, v, e, fmt=st[nm][1], color=st[nm][0], capsize=2, ms=4, lw=1.6, label=nm)
        if nm != "observed":
            axd.plot(r_, v - prof["observed"][1], st[nm][1], color=st[nm][0], ms=4, lw=1.6, label=f"{nm} − observed")
    axt.axhline(0, color="0.5", lw=0.8)
    axt.set_ylabel(r"mean tangential distortion $\chi_t$ about O2")
    axt.set_title(f"Catalogue galaxies with κ < {KAPPA_THRESH:g} (what the CNN sees)", fontsize=10)
    axd.axhline(0, color="0.5", lw=0.8)
    axd.set_ylabel(r"$\Delta\chi_t$")
    axd.set_title("Change in the tangential profile", fontsize=10)
    for a in (axk, axt, axd):
        a.set_xscale("log")
        a.set_xlim(edges[0], edges[-1])
        a.set_xlabel("radius from Lenstool O2 [kpc]")
        a.legend(fontsize=8)
    for e in ("png", "pdf"):
        fig.savefig(f"{opts.out}.{e}", dpi=150)
    print(f"wrote {opts.out}.png")


if __name__ == "__main__":
    main()
