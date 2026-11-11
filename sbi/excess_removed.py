#!/usr/bin/env python
'''
A2744 P(SIDM) with the Lenstool excess shear removed from the galaxy shapes. The excess convergence maps of
Dark_Matter_DA/lenstool/concat/delens_best/excess_mass.py (Kaiser-Squires of the delensed shapes) are turned back
into shear (forward KS on the same grid) and subtracted from each catalogue galaxy's gamma1, gamma2:
  srcpos      KAPPA_E_SRCPOS (binned at source-plane positions), evaluated at each galaxy's source-plane position
              (Lenstool deflection dx_z2/dy_z2 scaled by Dls/Ds of its photo-z)
  srcpos_sn2  the same with the map zeroed where its S/N < --min-snr
  image       KAPPA_E (binned at image positions), evaluated at the image positions
  image_sn2   the same, S/N >= --min-snr only
  total       KAPPA_E_RAW (all the mass, raw shapes): a check that removing the cluster signal moves P(SIDM)
The conversion from the maps' shear to gamma1/gamma2 is fitted by regressing the catalogue shears on the forward-KS
shear of KAPPA_E_RAW (M ~ 2 I: the galaxy ellipticities agree in units, matched galaxy by galaxy, but binning onto
5 arcsec pixels, half of them empty, and smoothing dilute the KS maps by ~2), and applied to every map. Each map is observed by the models of systematics.py (strong-lensing cut as the
fiducial), and summarised against the cached fiducial simulations of sbi/results/systematics_wb1_<tag>.

    astro-env/bin/python sbi/excess_removed.py --tag final
'''
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "sbi")]
LT = Path("/Users/davidharvey/Work/Dark_Matter_DA/lenstool/concat/delens_best")
LT_REF = (3.494741, -30.339158)              # Lenstool frame (best.par)
LT_FIELD = (-550., 170., -390., 170.)         # deflection-map extent (best.par)
Z_LENS, Z_MAP = 0.308, 2.0
COSMO = FlatLambdaCDM(H0=67., Om0=0.3)        # as the .par cosmology


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="final")
    p.add_argument("--min-snr", type=float, default=2.)
    p.add_argument("--out", default=None)
    return p.parse_args()


def kappa_to_gamma(k, pad=2):
    '''Forward Kaiser-Squires on the excess_mass.py grid ([y, x]); the inverse of its ks_map kernel.'''
    ny, nx = k.shape
    K = np.fft.fft2(np.pad(k, ((ny * pad, ny * pad), (nx * pad, nx * pad))))
    ky = np.fft.fftfreq(K.shape[0])[:, None]
    kx = np.fft.fftfreq(K.shape[1])[None, :]
    k2 = kx**2 + ky**2
    k2[0, 0] = 1
    g = np.fft.ifft2(K * ((kx**2 - ky**2) + 2j * kx * ky) / k2)[ny * pad:ny * (pad + 1), nx * pad:nx * (pad + 1)]
    return g.real, g.imag


def beta(z):
    z = np.atleast_1d(np.asarray(z, float))
    b = np.zeros_like(z)
    ok = z > Z_LENS
    b[ok] = (COSMO.angular_diameter_distance_z1z2(Z_LENS, z[ok]) / COSMO.angular_diameter_distance(z[ok])).value
    return b


def ks_emode(x, y, g1, g2, mx, my, pix, smooth=2.0):
    '''excess_mass.py ks_map: bin a shear onto the Lenstool grid, Kaiser-Squires, smooth; (E, B).'''
    from scipy.ndimage import gaussian_filter
    ny, nx = len(my), len(mx)
    rng_ = [[my[0] - pix / 2, my[0] - pix / 2 + ny * pix], [mx[0] - pix / 2, mx[0] - pix / 2 + nx * pix]]
    n = np.histogram2d(y, x, bins=(ny, nx), range=rng_)[0]
    e1 = np.histogram2d(y, x, bins=(ny, nx), range=rng_, weights=g1)[0]
    e2 = np.histogram2d(y, x, bins=(ny, nx), range=rng_, weights=g2)[0]
    g = np.where(n > 0, (e1 + 1j * e2) / np.maximum(n, 1), 0)
    pad = 2
    G = np.fft.fft2(np.pad(g, ((ny * pad, ny * pad), (nx * pad, nx * pad))))
    ky = np.fft.fftfreq(G.shape[0])[:, None]
    kx = np.fft.fftfreq(G.shape[1])[None, :]
    k2 = kx**2 + ky**2
    k2[0, 0] = 1
    K = np.fft.ifft2(G * ((kx**2 - ky**2) - 2j * kx * ky) / k2)[ny * pad:ny * (pad + 1), nx * pad:nx * (pad + 1)]
    return gaussian_filter(K.real, smooth), gaussian_filter(K.imag, smooth)


def emode_figure(emaps, x, y, keep, S, mx, my, sig_lt, hd, out, nrand=100):
    '''KS E-mode S/N of the catalogue shears (kappa < KAPPA_THRESH, in the Lenstool frame) before and after the excess
    is removed, and their difference; noise from random rotations of the fiducial shears.'''
    from sensitivity import LENSTOOL_HALOS, style
    pix = hd["CDELT1"]
    to_lt = lambda d: S.T @ np.array([np.asarray(d["gamma1"])[keep], np.asarray(d["gamma2"])[keep]])
    xk, yk = x[keep], y[keep]
    g0 = to_lt(emaps["fiducial"])
    rng = np.random.default_rng(1)
    gc = g0[0] + 1j * g0[1]
    sig = np.std([ks_emode(xk, yk, *(lambda r: (r.real, r.imag))(gc * np.exp(2j * np.pi * rng.random(len(gc)))), mx, my, pix)[0]
                  for _ in range(nrand)], 0)
    E = {nm: ks_emode(xk, yk, *to_lt(d), mx, my, pix)[0] for nm, d in emaps.items()}
    names = [nm for nm in emaps if nm != "fiducial"]
    panels = [(E["fiducial"] / sig, "catalogue E-mode (total mass), S/N")] + \
             [(E[nm] / sig, f"excess removed ({nm}), S/N") for nm in names] + \
             [((E["fiducial"] - E[names[0]]) / sig, f"removed: fiducial − {names[0]}, S/N")]
    plt = style()
    fig, axs = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    ext = [mx[0] - pix / 2, mx[-1] + pix / 2, my[0] - pix / 2, my[-1] + pix / 2]
    for axp, (img, title) in zip(axs.flat, panels):
        im = axp.imshow(img, origin="lower", extent=ext, cmap="RdBu_r", vmin=-5, vmax=5)
        axp.contour(img, levels=[2, 3, 4, 5], colors="k", linewidths=0.6, origin="lower", extent=ext)
        for nm, (hx, hy) in LENSTOOL_HALOS.items():
            axp.plot(hx, hy, "x", c="lime", ms=9, mew=2)
            axp.text(hx + 6, hy + 6, nm, color="lime", fontsize=8)
        axp.set_title(title, fontsize=10)
        axp.set_xlabel("x [arcsec]  (E ← → W)")
        axp.set_ylabel("y [arcsec]")
    fig.colorbar(im, ax=axs, shrink=0.8, label=r"$\kappa / \sigma_\kappa$")
    fig.suptitle("A2744 shear catalogue (κ < 0.7, as the CNN map), Kaiser-Squires on the Lenstool grid", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=130)
    print(f"wrote {out}.png")


def main():
    opts = parse()
    import systematics as sy
    from add_shear_to_data import KAPPA_THRESH, strong_lensing_kappa

    cache = REPO / "sbi" / "results" / f"systematics_wb1_{opts.tag}"
    sys.argv = ["systematics.py", "--model", f"tweakedalign_k0p7_ctr_wb1_ba_{opts.tag}", "--include-str",
                "spin2_cx_pgc_ft_", "--min-ncomp", "2", "--summaries", str(REPO / "sbi" / f"summaries_wb1_{opts.tag}_n5"),
                "--out-dir", str(cache)]
    so = sy.parse()
    sy.A2744_NCOMP = tuple(range(so.min_ncomp, 7))
    R = sy.Runner(so, cache)
    fid, _ = R.cached("fiducial", lambda d: None)

    data = R.cat.full
    ra, dec = R.cat.positions(data)
    keep = strong_lensing_kappa(data) < KAPPA_THRESH
    x = -(np.asarray(data["RA"]) - LT_REF[0]) * np.cos(np.deg2rad(LT_REF[1])) * 3600.
    y = (np.asarray(data["DEC"]) - LT_REF[1]) * 3600.

    ax_, ay_ = (fits.getdata(LT / f) .astype(float) for f in ("dx_z2.fits", "dy_z2.fits"))
    xg = np.linspace(LT_FIELD[0], LT_FIELD[1], ax_.shape[1])
    yg = np.linspace(LT_FIELD[2], LT_FIELD[3], ax_.shape[0])
    s = beta(np.asarray(data["redshift"])) / beta(Z_MAP)[0]
    defl = [np.nan_to_num(RegularGridInterpolator((yg, xg), a, bounds_error=False, fill_value=0.)(np.c_[y, x])) for a in (ax_, ay_)]
    xs, ys = x - s * defl[0], y - s * defl[1]
    print(f"source-plane shift: median {np.median(np.hypot(xs - x, ys - y)):.1f} arcsec, max {np.max(np.hypot(xs - x, ys - y)):.1f}")

    with fits.open(LT / "excess_kappa.fits") as h:
        hd = h["KAPPA_E"].header
        sig = h["SIGMA"].data.astype(float)
        maps = {nm: h[nm].data.astype(float) for nm in ("KAPPA_E", "KAPPA_E_SRCPOS", "KAPPA_E_RAW")}
    mx = hd["CRVAL1"] + hd["CDELT1"] * (np.arange(sig.shape[1]) + 1 - hd["CRPIX1"])
    my = hd["CRVAL2"] + hd["CDELT2"] * (np.arange(sig.shape[0]) + 1 - hd["CRPIX2"])

    def shear_at(k, px, py):
        g1, g2 = kappa_to_gamma(k)
        return [RegularGridInterpolator((my, mx), g, bounds_error=False, fill_value=0.)(np.c_[py, px]) for g in (g1, g2)]

    # sign / axis convention: catalogue shears against the forward-KS shear of the total-mass map
    t1, t2 = shear_at(maps["KAPPA_E_RAW"], x, y)
    inside = keep & (x > mx[0]) & (x < mx[-1]) & (y > my[0]) & (y < my[-1])
    A = np.c_[t1[inside], t2[inside]]
    M = np.linalg.lstsq(A, np.c_[data["gamma1"][inside], data["gamma2"][inside]], rcond=None)[0].T
    print(f"catalogue (gamma1, gamma2) = M (g1_LT, g2_LT), M =\n{np.round(M, 3)}  ({inside.sum()} galaxies)")
    S = np.sign(np.round(M / np.abs(M).max(), 0))                          # signed permutation
    print(f"adopted convention:\n{S}")

    def subtracted(k, px, py):
        g = np.array(shear_at(k, px, py))
        g1, g2 = M @ g                       # M, not just its signs: the binned+smoothed KS maps carry ~half the shear
        d = data.copy()
        d["gamma1"] = np.asarray(data["gamma1"]) - g1
        d["gamma2"] = np.asarray(data["gamma2"]) - g2
        return d, np.hypot(g1, g2)

    sn = {nm: np.where(maps[nm] / sig >= opts.min_snr, maps[nm], 0.) for nm in ("KAPPA_E", "KAPPA_E_SRCPOS")}
    variants = {"fiducial": (None, None, None),
                "srcpos": (maps["KAPPA_E_SRCPOS"], xs, ys),
                f"srcpos_sn{opts.min_snr:g}": (sn["KAPPA_E_SRCPOS"], xs, ys),
                "image": (maps["KAPPA_E"], x, y),
                f"image_sn{opts.min_snr:g}": (sn["KAPPA_E"], x, y),
                "total": (maps["KAPPA_E_RAW"], x, y)}
    out, emaps = {}, {}
    for name, (k, px, py) in variants.items():
        d, amp = (data, None) if k is None else subtracted(k, px, py)
        p_obs = sy.observe(R.models, sy.obs_raw_map(d[keep], ra[keep], dec[keep]), 1, R.device)
        res = {dm: (fid[dm][0], p_obs) for dm in sy.DOMAINS}
        m = R.summarise(res)["combined"]
        out[name] = {"p_sidm": float(p_obs.mean()), "p_sidm_std_models": float(p_obs.std()),
                     "metrics": m, "mean_removed_shear": None if amp is None else float(amp[keep].mean())}
        rm = "" if amp is None else f"  removed |g| mean {amp[keep].mean():.4f}, max {amp[keep].max():.3f}"
        lo, hi = m["log_sigma_err"]
        print(f"{name:12s} P(SIDM) {p_obs.mean():.3f} ± {p_obs.std() / np.sqrt(len(p_obs)):.3f}  "
              f"log sigma/m {m['log_sigma']:+.2f} (-{lo:.2f} +{hi:.2f})  z_CDM {m['z_cdm']:.2f}" + rm)
        if name in ("fiducial", "srcpos", f"srcpos_sn{opts.min_snr:g}"):
            emaps[name] = d
    emode_figure(emaps, x, y, keep, S, mx, my, sig, hd, Path(opts.out or REPO / "sbi" / "results" / "main_halo_profile" /
                                                           f"excess_removed_emode_{opts.tag}").with_suffix(""))
    path = Path(opts.out or REPO / "sbi" / "results" / "main_halo_profile" / f"excess_removed_{opts.tag}.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
