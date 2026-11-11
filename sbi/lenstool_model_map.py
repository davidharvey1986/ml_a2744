#!/usr/bin/env python
'''
The A2744 CNN map rebuilt from the Lenstool model alone: each catalogue galaxy's shape replaced by the model reduced
shear at its position and photo-z (Jacobian of the deflection maps dx_z2/dy_z2, scaled by Dls/Ds), binned like the data
map (sensitivity.a2744_map) and Kaiser-Squires reconstructed (sensitivity.reconstruct_kappa). Panels: the data, the
model with the strong-lensing cut (the galaxies the CNN sees), the model with every galaxy (no hole), data - model,
and the Lenstool convergence itself. Tells mass the model already has, and edge effects of the kappa >= KAPPA_THRESH
hole, from structure the model lacks.

    astro-env/bin/python sbi/lenstool_model_map.py
'''
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]
import sensitivity as sens  # noqa: E402
from add_shear_to_data import (  # noqa: E402
    KAPPA_THRESH, SHEAR_CUTS, SL_KAPPA_MAP, bin2d, galaxy_distortion, get_obs_data, ra_dec_to_simulation_image_pos,
)
from excess_removed import LT, LT_FIELD, LT_REF, Z_MAP, beta  # noqa: E402

OUT = REPO / "sbi" / "results" / "main_halo_profile" / "lenstool_model_map"


def lenstool_xy(cat):
    return (-(np.asarray(cat["RA"]) - LT_REF[0]) * np.cos(np.deg2rad(LT_REF[1])) * 3600.,
            (np.asarray(cat["DEC"]) - LT_REF[1]) * 3600.)


def model_shear(cat):
    '''Lenstool reduced shear (g1, g2) and kappa at each galaxy, for its own redshift.'''
    ax_, ay_ = (fits.getdata(LT / f).astype(float) for f in ("dx_z2.fits", "dy_z2.fits"))
    xg = np.linspace(LT_FIELD[0], LT_FIELD[1], ax_.shape[1])
    yg = np.linspace(LT_FIELD[2], LT_FIELD[3], ax_.shape[0])
    daxdy, daxdx = np.gradient(ax_, yg, xg)
    daydy, daydx = np.gradient(ay_, yg, xg)
    x, y = lenstool_xy(cat)
    f = {k: np.nan_to_num(RegularGridInterpolator((yg, xg), v, bounds_error=False, fill_value=0.)(np.c_[y, x]))
         for k, v in dict(axx=daxdx, axy=daxdy, ayx=daydx, ayy=daydy).items()}
    z = np.asarray(cat["redshift"]) if "redshift" in getattr(cat, "colnames", getattr(cat.dtype, "names", ())) else np.full(len(x), 2.28)
    s = beta(z) / beta(Z_MAP)[0]
    A11, A12, A21, A22 = 1 - s * f["axx"], -s * f["axy"], -s * f["ayx"], 1 - s * f["ayy"]
    kap = 1 - 0.5 * (A11 + A22)
    g1 = 0.5 * (A22 - A11) / (1 - kap)
    g2 = -0.5 * (A12 + A21) / (1 - kap)
    return g1, g2, kap


def binned(cat, g1, g2, ra, dec):
    c1, c2 = galaxy_distortion({"gamma1": g1, "gamma2": g2})
    return np.array(bin2d(ra, dec, v=(c1, c2), npix=sens.NPIX, extent=[-50, 50, -50, 50]), dtype=np.float32)


def main():
    raw, cat, ra, dec = sens.a2744_map()                                  # kappa < KAPPA_THRESH, as the CNN
    forward, inverse = sens.stamp_to_lenstool(cat, ra, dec)
    g1, g2, _ = model_shear(cat)
    slopes = []
    for comp, gm in (("gamma1", g1), ("gamma2", g2)):
        d = np.asarray(cat[comp])
        slopes.append(np.polyfit(gm, d, 1)[0])
        print(f"catalogue {comp} vs model: slope {slopes[-1]:+.2f}, corr {np.corrcoef(gm, d)[0, 1]:+.2f}")
    raw_model = binned(cat, g1, g2, ra, dec)

    cat_all = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS, drop_strong_lensing=False)
    ra_a, dec_a = ra_dec_to_simulation_image_pos(cat_all)
    ga1, ga2, _ = model_shear(cat_all)
    raw_model_all = binned(cat_all, ga1, ga2, ra_a, dec_a)
    hole = np.histogram2d(dec_a, ra_a, bins=sens.NPIX, range=[[-50, 50], [-50, 50]])[0] > 0
    kept = np.histogram2d(dec, ra, bins=sens.NPIX, range=[[-50, 50], [-50, 50]])[0] > 0
    hole = hole & ~kept                                                  # pixels whose galaxies are all cut

    ke_data, snr_data = sens.reconstruct_kappa(raw)
    ke_model = sens.reconstruct_kappa(raw_model)[0]
    ke_all = sens.reconstruct_kappa(raw_model_all)[0]
    noise = ke_data / snr_data                                            # std of the data B-mode
    noise = float(np.nanmedian(noise[np.isfinite(noise)]))

    with fits.open(SL_KAPPA_MAP) as h:
        k_sl, w = h[0].data.astype(float), WCS(h[0].header)
    pix = np.arange(sens.NPIX) - (sens.NPIX / 2 - 0.5)
    PX, PY = np.meshgrid(pix, pix)
    lx, ly = forward(PX, PY)
    i, j = w.world_to_pixel_values(LT_REF[0] - lx / (np.cos(np.deg2rad(LT_REF[1])) * 3600.), LT_REF[1] + ly / 3600.)
    i, j = np.round(i).astype(int), np.round(j).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < k_sl.shape[1]) & (j < k_sl.shape[0])
    k_stamp = np.full(PX.shape, np.nan)
    k_stamp[ok] = k_sl[j[ok], i[ok]]

    halos = {k: tuple(inverse(*v)) for k, v in sens.LENSTOOL_HALOS.items()}
    np.savez(f"{OUT}.npz", ke_data=ke_data, ke_model=ke_model, ke_all=ke_all, noise=noise, hole=hole,
             slope=float(np.mean(slopes)), kappa_lenstool=k_stamp,
             halo_names=np.array(list(halos)), halo_xy=np.array(list(halos.values())))
    plt = sens.style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    panels = [(ke_data / noise, "data (κ < 0.7 galaxies)"), (ke_model / noise, "Lenstool model, κ < 0.7 galaxies"),
              (ke_all / noise, "Lenstool model, all galaxies (no hole)"),
              ((ke_data - ke_model) / noise, "data − model (κ < 0.7 galaxies)"),
              ((ke_model - ke_all) / noise, "model: hole − no hole (edge effect)")]
    for ax, (img, title) in zip(axes.flat, panels):
        im = ax.imshow(img, origin="lower", extent=[-50, 50, -50, 50], cmap="RdBu_r", vmin=-8, vmax=8)
        ax.contour(img, origin="lower", extent=[-50, 50, -50, 50], levels=[2, 3, 4, 6, 8], colors="k", linewidths=0.8)
        ax.contour(img, origin="lower", extent=[-50, 50, -50, 50], levels=[-4, -3, -2], colors="k", linewidths=0.6,
                   linestyles="--")
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=list(axes.flat[:5]), shrink=0.6,
                 label="KS E-mode / data noise (S/N units)")
    ax = axes.flat[5]
    im2 = ax.imshow(k_stamp, origin="lower", extent=[-50, 50, -50, 50], cmap="magma", vmin=0, vmax=1.5)
    ax.contour(k_stamp, origin="lower", extent=[-50, 50, -50, 50], levels=[KAPPA_THRESH], colors="c", linewidths=1.2)
    ax.set_title(f"Lenstool κ (zs 2.28); cyan: κ = {KAPPA_THRESH:g}", fontsize=10)
    fig.colorbar(im2, ax=ax, shrink=0.6, label="κ")
    for ax in axes.flat:
        ax.contour(hole.astype(float), origin="lower", extent=[-50, 50, -50, 50], levels=[0.5], colors="#2ca02c",
                   linewidths=1)
        for nm, (hx, hy) in halos.items():
            ax.plot(hx, hy, "x", color="#2ca02c", ms=8, mew=2)
            ax.text(hx + 2, hy + 2, nm, color="#2ca02c", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("A2744 Kaiser-Squires maps on the CNN stamp (north up, east left; 20 kpc pixels). Green outline: pixels "
                 f"emptied by the κ ≥ {KAPPA_THRESH:g} cut", fontsize=11)
    for e in ("png", "pdf"):
        fig.savefig(f"{OUT}.{e}", dpi=150)
    r = np.hypot(PX, PY)
    for nm, img in (("data", ke_data), ("model (cut)", ke_model), ("model (no hole)", ke_all)):
        n_ = img[(PY > 5) & (PY < 20) & (np.abs(PX) < 8)].mean() / noise
        s_ = img[(PY < -5) & (PY > -20) & (np.abs(PX) < 8)].mean() / noise
        print(f"{nm:16s} mean S/N 100-400 kpc north of the centre {n_:+.1f}, south {s_:+.1f}")
    print(f"wrote {OUT}.png")


if __name__ == "__main__":
    main()
