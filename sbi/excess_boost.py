#!/usr/bin/env python
'''
Mass of the Lenstool excess at O2 and A2744's P(SIDM) with that mass put into the inner core.

Mass: the excess convergence (KAPPA_E_SRCPOS of excess_mass.py, times the KS-map dilution fitted in
sbi/excess_removed.py) summed over its S/N >= --min-snr region connected to the peak nearest O2, times Sigma_crit for
<Dls/Ds> of the sources behind the cluster. For scale, the Lenstool (zs = 2.28) mass within --core-check kpc of O2.

Boost: a circular Gaussian convergence of that mass and width --widths kpc centred on Lenstool O2, as shear on every
catalogue galaxy (gamma_t = <kappa>(<r) - kappa(r), each galaxy's own Dls/Ds), added to gamma1/gamma2; also the excess
removed (S/N >= --min-snr part) and the same mass put into the core ("relocate"). Observed with the models of
systematics.py and summarised against the cached fiducial simulations, as sbi/excess_removed.py.

    astro-env/bin/python sbi/excess_boost.py --tag final
'''
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.constants import G, c as C
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "sbi")]
from excess_removed import COSMO, LT, LT_FIELD, LT_REF, Z_LENS, Z_MAP, beta, kappa_to_gamma  # noqa: E402

KPC_PER_ARCSEC = COSMO.kpc_proper_per_arcmin(Z_LENS).to(u.kpc / u.arcsec).value
O2 = (-290.369958, -228.819861)                       # Lenstool x, y [arcsec]


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="final")
    p.add_argument("--min-snr", type=float, default=2.)
    p.add_argument("--widths", type=float, nargs="+", default=[25., 50., 100.], help="Gaussian sigma [kpc]")
    p.add_argument("--core-check", type=float, default=100., help="kpc")
    p.add_argument("--out", default=None)
    return p.parse_args()


def sigma_crit_inf():
    '''c^2 / (4 pi G D_l) in Msun / kpc^2: Sigma_crit for a source with Dls/Ds = 1.'''
    return (C**2 / (4 * np.pi * G * COSMO.angular_diameter_distance(Z_LENS))).to(u.Msun / u.kpc**2).value


def gaussian_shear(x, y, mass, s_kpc, b):
    '''Shear (Lenstool frame) and convergence at (x, y) [arcsec] of a Gaussian of mass [Msun], width s_kpc, at O2.'''
    dx, dy = KPC_PER_ARCSEC * (x - O2[0]), KPC_PER_ARCSEC * (y - O2[1])
    r2 = dx**2 + dy**2
    k0 = mass / (2 * np.pi * s_kpc**2) / sigma_crit_inf() * b          # per galaxy, through its Dls/Ds
    q = r2 / (2 * s_kpc**2)
    kappa = k0 * np.exp(-q)
    kbar = k0 * np.where(q > 1e-8, (1 - np.exp(-q)) / np.maximum(q, 1e-8), 1.)
    gt = kbar - kappa
    phi = np.arctan2(dy, dx)
    return -gt * np.cos(2 * phi), -gt * np.sin(2 * phi), kappa


def main():
    opts = parse()
    import systematics as sy
    from add_shear_to_data import KAPPA_THRESH, SL_KAPPA_MAP, strong_lensing_kappa

    # ── the excess mass ──
    with fits.open(LT / "excess_kappa.fits") as h:
        hd = h["KAPPA_E"].header
        sig = h["SIGMA"].data.astype(float)
        kmaps = {nm: h[nm].data.astype(float) for nm in ("KAPPA_E", "KAPPA_E_SRCPOS", "KAPPA_E_RAW")}
    mx = hd["CRVAL1"] + hd["CDELT1"] * (np.arange(sig.shape[1]) + 1 - hd["CRPIX1"])
    my = hd["CRVAL2"] + hd["CDELT2"] * (np.arange(sig.shape[0]) + 1 - hd["CRPIX2"])
    pix_kpc2 = (hd["CDELT1"] * KPC_PER_ARCSEC) ** 2

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
    z = np.asarray(data["redshift"])
    b = beta(z)
    b_mean = b[z > Z_LENS + 0.1].mean()
    sc = sigma_crit_inf() / b_mean
    print(f"Sigma_crit for <Dls/Ds> = {b_mean:.3f} of {np.sum(z > Z_LENS + 0.1)} sources: {sc:.3g} Msun/kpc^2")

    def shear_at(k, px, py):
        g1, g2 = kappa_to_gamma(k)
        return np.array([RegularGridInterpolator((my, mx), g, bounds_error=False, fill_value=0.)(np.c_[py, px]) for g in (g1, g2)])
    t = shear_at(kmaps["KAPPA_E_RAW"], x, y)
    inside = keep & (x > mx[0]) & (x < mx[-1]) & (y > my[0]) & (y < my[-1])
    M = np.linalg.lstsq(t[:, inside].T, np.c_[data["gamma1"][inside], data["gamma2"][inside]], rcond=None)[0].T
    dil = float(np.mean(np.diag(M)))
    print(f"KS-map dilution (catalogue shear / map shear): {dil:.2f}")

    masses = {}
    X, Y = np.meshgrid(mx, my)
    for nm in ("KAPPA_E_SRCPOS", "KAPPA_E"):
        snr = kmaps[nm] / sig
        lab, _ = label(snr >= opts.min_snr)
        near = (np.hypot(X - O2[0], Y - O2[1]) < 60) & (lab > 0)
        pk = np.unravel_index(np.argmax(np.where(near, snr, -np.inf)), snr.shape)
        reg = lab == lab[pk]
        m_raw = kmaps[nm][reg].sum() * pix_kpc2 * sc
        cx, cy = (np.sum(kmaps[nm][reg] * v[reg]) / kmaps[nm][reg].sum() for v in (X, Y))
        masses[nm] = dil * m_raw
        print(f"{nm}: S/N >= {opts.min_snr:g} region of {reg.sum()} pixels ({reg.sum() * pix_kpc2:.0f} kpc^2), peak S/N "
              f"{snr[pk]:.1f}; mass-weighted centre ({KPC_PER_ARCSEC * (cx - O2[0]):+.0f}, {KPC_PER_ARCSEC * (cy - O2[1]):+.0f}) kpc "
              f"from O2 (x west, y north); mass {m_raw:.2e} Msun as mapped, {masses[nm]:.2e} Msun with the dilution")

    with fits.open(SL_KAPPA_MAP) as h:
        from astropy.wcs import WCS
        k_sl, w = h[0].data, WCS(h[0].header)
    jj, ii = np.indices(k_sl.shape)
    sra, sdec = w.pixel_to_world_values(ii.ravel(), jj.ravel())
    sx = -(sra - LT_REF[0]) * np.cos(np.deg2rad(LT_REF[1])) * 3600.
    sy_ = (sdec - LT_REF[1]) * 3600.
    ps = w.proj_plane_pixel_scales()
    pix_sl = abs((ps[0] * ps[1]).to(u.arcsec**2).value) * KPC_PER_ARCSEC**2
    sc228 = sigma_crit_inf() / beta(2.28)[0]
    rr = KPC_PER_ARCSEC * np.hypot(sx - O2[0], sy_ - O2[1])
    m_core = np.nansum(k_sl.ravel()[rr < opts.core_check]) * pix_sl * sc228
    m_ex = masses["KAPPA_E_SRCPOS"]
    print(f"Lenstool mass within {opts.core_check:g} kpc of O2: {m_core:.2e} Msun; excess/core = {m_ex / m_core:.2f}")

    # ── P(SIDM) with the mass added to the core ──
    snr_src = kmaps["KAPPA_E_SRCPOS"] / sig
    defl = []
    ax_, ay_ = (fits.getdata(LT / f).astype(float) for f in ("dx_z2.fits", "dy_z2.fits"))
    xg = np.linspace(LT_FIELD[0], LT_FIELD[1], ax_.shape[1])
    yg = np.linspace(LT_FIELD[2], LT_FIELD[3], ax_.shape[0])
    s = b / beta(Z_MAP)[0]
    for a in (ax_, ay_):
        defl.append(np.nan_to_num(RegularGridInterpolator((yg, xg), a, bounds_error=False, fill_value=0.)(np.c_[y, x])))
    xs, ys = x - s * defl[0], y - s * defl[1]
    excess_g = M @ shear_at(np.where(snr_src >= opts.min_snr, kmaps["KAPPA_E_SRCPOS"], 0.), xs, ys)

    def with_shear(dg1, dg2):
        d = data.copy()
        d["gamma1"] = np.asarray(data["gamma1"]) + dg1
        d["gamma2"] = np.asarray(data["gamma2"]) + dg2
        return d

    variants = {"fiducial": data, f"excess removed (S/N>={opts.min_snr:g})": with_shear(-excess_g[0], -excess_g[1])}
    for s_kpc in opts.widths:
        g1, g2, kap = gaussian_shear(x, y, m_ex, s_kpc, b)
        print(f"Gaussian sigma {s_kpc:g} kpc: kappa at O2 (zs = 2.28) {m_ex / (2 * np.pi * s_kpc**2) / sc228:.2f}, "
              f"added |g| on kept galaxies: mean {np.hypot(g1, g2)[keep].mean():.4f}, max {np.hypot(g1, g2)[keep].max():.3f}")
        variants[f"core boost, {s_kpc:g} kpc"] = with_shear(g1, g2)
        variants[f"relocate to core, {s_kpc:g} kpc"] = with_shear(g1 - excess_g[0], g2 - excess_g[1])
    out = {"excess_mass_msun": m_ex, "excess_mass_image_plane_msun": masses["KAPPA_E"], "core_mass_msun": m_core,
           "core_radius_kpc": opts.core_check, "dilution": dil, "results": {}}
    for name, d in variants.items():
        p_obs = sy.observe(R.models, sy.obs_raw_map(d[keep], ra[keep], dec[keep]), 1, R.device)
        m = R.summarise({dm: (fid[dm][0], p_obs) for dm in sy.DOMAINS})["combined"]
        lo, hi = m["log_sigma_err"]
        out["results"][name] = {"p_sidm": float(p_obs.mean()), "log_sigma": m["log_sigma"], "log_sigma_err": [lo, hi],
                                "z_cdm": m["z_cdm"]}
        print(f"{name:30s} P(SIDM) {p_obs.mean():.3f} ± {p_obs.std() / np.sqrt(len(p_obs)):.3f}  "
              f"log sigma/m {m['log_sigma']:+.2f} (-{lo:.2f} +{hi:.2f})  z_CDM {m['z_cdm']:.2f}")
    path = Path(opts.out or REPO / "sbi" / "results" / "main_halo_profile" / f"excess_boost_{opts.tag}.json")
    json.dump(out, open(path, "w"), indent=2, default=float)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
