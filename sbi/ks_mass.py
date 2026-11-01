#!/usr/bin/env python
'''
Projected mass of the A2744 main halo inside --radii (kpc) of the adopted centre, directly from the
Kaiser-Squires reconstruction of the binned A2744 distortion map (the map the models see: 20 kpc
pixels, strong-lensing cut kappa < KAPPA_THRESH). kappa_KS is converted with Sigma_crit,inf / <beta>,
beta = D_ls/D_s of each galaxy's photo-z (0 in front of the cluster). Noise: the galaxies' distortions
rotated by random angles.

KS loses the mass sheet (a zero-mean field) and the empty pixels count as zero shear, so the same
reconstruction is run on the BAHAMAS CDM clusters at A2744's mass, sampled at the A2744 galaxies (ideal
reduced shear, the same cut on each cluster's own convergence), against their true projected mass:
the median ratio calibrates the A2744 number.

    astro-env/bin/python sbi/ks_mass.py
'''
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import systematics as sy  # noqa: E402  (chdirs to notebooks/)
from astropy import constants as const, units as u  # noqa: E402
from astropy.cosmology import Planck18 as cosmo  # noqa: E402
from lenspack.image.inversion import ks93  # noqa: E402
from mass_weighting import a2744_mass_gaussian  # noqa: E402

RADII = [250, 500, 750]
PIX_KPC = 20.
N_ROT = 300


def sigma_crit_inf():
    '''c^2 / (4 pi G D_l) [Msun / kpc^2]: Sigma_crit for beta = D_ls/D_s = 1.'''
    return (const.c**2 / (4 * np.pi * const.G * cosmo.angular_diameter_distance(sy.MAP_ZL))).to(u.Msun / u.kpc**2).value


def beta(z):
    z = np.asarray(z, float)
    out = np.zeros_like(z)
    ok = z > sy.MAP_ZL + 0.01
    out[ok] = (cosmo.angular_diameter_distance_z1z2(sy.MAP_ZL, z[ok]) / cosmo.angular_diameter_distance(z[ok])).value
    return out


def ks_kappa(raw):
    return ks93(raw[0] / 2., raw[1] / 2.)[0]


def aperture_sums(kappa, radii):
    c = (np.arange(sy.NPIX) - (sy.NPIX / 2 - 0.5)) * PIX_KPC
    r = np.hypot(*np.meshgrid(c, c))
    return np.array([kappa[..., r < R].sum(-1) * PIX_KPC**2 for R in radii])


def main():
    cat = sy.Catalogue(sy.get_obs_data("concat", data_dir=sy.DATA_DIR, photoz=True, cuts=sy.FILTER_CUTS))
    data = cat.full
    ra, dec = cat.positions(data)
    keep = sy.strong_lensing_kappa(data) < sy.KAPPA_THRESH
    d, x, y = data[keep], ra[keep], dec[keep]
    b = beta(d["redshift"])
    scrit = sigma_crit_inf() / b.mean()
    print(f"{keep.sum()} galaxies (strong-lensing cut kappa < {sy.KAPPA_THRESH}); <beta> = {b.mean():.3f}; "
          f"Sigma_crit,eff = {scrit:.3e} Msun/kpc^2")

    m_raw = aperture_sums(ks_kappa(sy.obs_raw_map(d, x, y)), RADII) * scrit
    rng = np.random.default_rng(1)
    g = d["gamma1"] + 1j * d["gamma2"]
    rot = []
    for _ in range(N_ROT):
        gr = g * np.exp(2j * np.pi * rng.random(len(g)))
        dr = d.copy()
        dr["gamma1"], dr["gamma2"] = gr.real, gr.imag
        rot.append(aperture_sums(ks_kappa(sy.obs_raw_map(dr, x, y)), RADII) * scrit)
    m_err = np.std(rot, 0)

    # calibration: BAHAMAS CDM at A2744's mass through the same galaxies, cut and reconstruction
    sy.A2744_NCOMP = (2, 3, 4, 5, 6)
    mu, sig = a2744_mass_gaussian()
    sel = sy.select_clusters(str(REPO / "sbi" / "summaries_spin2cx_final_n5"), "bahamas", mu)
    cdm = sel["cross_section"] == 0
    sel = {k: v[cdm] for k, v in sel.items()}
    sm = sy.SimMaps(sel)
    maps = sm.sample(ra, dec, sy.KAPPA_THRESH)                       # (N, 2, 100, 100) distortion, as the data
    scrit_map = sigma_crit_inf() / beta([sy.MAP_ZS])[0]
    m_ks = np.array([aperture_sums(ks_kappa(m), RADII) for m in maps]).T * scrit_map
    m_true = aperture_sums(sm.k, RADII) * scrit_map
    ratio = m_ks / m_true

    res = {"n_gal": int(keep.sum()), "mean_beta": float(b.mean()), "sigma_crit_eff": float(scrit), "radii_kpc": RADII,
           "a2744": {}, "calibration": {"n_clusters": int(cdm.sum())}}
    print(f"\ncalibration on {cdm.sum()} BAHAMAS CDM clusters within {sy.SUBSET_DEX} dex of the A2744 mass "
          f"(n_comp >= 2): M_KS / M_true")
    for i, R in enumerate(RADII):
        q = np.percentile(ratio[i], [16, 50, 84])
        cal = m_raw[i] / q[1]
        res["a2744"][R] = {"M_ks": float(m_raw[i]), "M_ks_err": float(m_err[i]), "M_calibrated": float(cal),
                           "M_calibrated_err_noise": float(m_err[i] / q[1]),
                           "M_calibrated_16_84": [float(m_raw[i] / q[2]), float(m_raw[i] / q[0])]}
        res["calibration"][R] = {"ratio_16_50_84": q.tolist(), "true_log_mass_median": float(np.log10(np.median(m_true[i])))}
        print(f"  R < {R} kpc: ratio {q[1]:.2f} [16-84% {q[0]:.2f}-{q[2]:.2f}]")
        print(f"     A2744 KS:         M = {m_raw[i]:.3e} ± {m_err[i]:.2e}  (log {np.log10(m_raw[i]):.3f})")
        print(f"     A2744 calibrated: M = {cal:.3e} ± {m_err[i] / q[1]:.2e} (shape noise), "
              f"log {np.log10(cal):.3f} [scatter: {np.log10(m_raw[i] / q[2]):.3f}-{np.log10(m_raw[i] / q[0]):.3f}]")
    out = REPO / "sbi" / "results" / "ks_mass.json"
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
