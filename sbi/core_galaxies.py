#!/usr/bin/env python
'''
The A2744 galaxies that set the shear in the core: a zoom on the JWST image around the stamp
centre (the adopted centre, ~45 kpc from Lenstool O2), with every catalogue galaxy within
--rmax-kpc drawn as a whisker along its distortion (length proportional to |chi|), coloured by
its tangential distortion about the centre. Galaxies in the data map (the current SHEAR_CUTS,
strong-lensing kappa < KAPPA_THRESH) are solid; those removed by the strong-lensing cut are
dashed grey. Circles at 50 and 100 kpc. A table of the galaxies within 100 kpc is written next
to the figure.

    astro-env/bin/python sbi/core_galaxies.py --image /Users/davidharvey/Work/webbPSF/cats_28_10_26/f115w/abell2744clu-grizli-v5.4-f115w-clear_drc_sci.fits
'''
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as u

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import sensitivity as sens  # noqa: E402
from add_shear_to_data import (  # noqa: E402
    KAPPA_THRESH, SHEAR_CUTS, galaxy_distortion, get_obs_data, ra_dec_to_simulation_image_pos, strong_lensing_kappa,
)

KPC_PER_PX = 20.


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--rmax-kpc", type=float, default=150.)
    p.add_argument("--half-width-kpc", type=float, default=180.)
    p.add_argument("--out", default=str(REPO / "sbi" / "results" / "core_galaxies" / "core_galaxies"))
    return p.parse_args()


def main():
    opts = parse()
    out = Path(opts.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    cat = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS, drop_strong_lensing=False)
    x, y = ra_dec_to_simulation_image_pos(cat)
    kappa = strong_lensing_kappa(cat)
    c1, c2 = galaxy_distortion(cat)
    r = KPC_PER_PX * np.hypot(x, y)
    phi = np.arctan2(y, x)
    chi_t = -(c1 * np.cos(2 * phi) + c2 * np.sin(2 * phi))
    chi_x = c1 * np.sin(2 * phi) - c2 * np.cos(2 * phi)
    kept = kappa < KAPPA_THRESH

    # stamp centre on the sky: invert the linear stamp-pixel -> (RA, Dec) relation of the catalogue
    A = np.c_[x, y, np.ones_like(x)]
    ra0 = np.linalg.lstsq(A, cat["RA"], rcond=None)[0][2]
    dec0 = np.linalg.lstsq(A, cat["DEC"], rcond=None)[0][2]
    kpc_per_arcsec = KPC_PER_PX / 4.302                         # stamp_to_lenstool: 4.302 arcsec per stamp pixel
    centre = SkyCoord(ra0 * u.deg, dec0 * u.deg)
    print(f"stamp centre RA {ra0:.5f} Dec {dec0:.5f}; {kpc_per_arcsec:.3f} kpc per arcsec")

    sel = r < opts.rmax_kpc
    print(f"galaxies within {opts.rmax_kpc:g} kpc: {sel.sum()} in the catalogue, {np.sum(sel & kept)} kept "
          f"(strong-lensing kappa < {KAPPA_THRESH}), {np.sum(sel & ~kept)} removed")
    for lo, hi in ((0, 50), (50, 100), (100, 150)):
        m = kept & (r >= lo) & (r < hi)
        if m.sum():
            print(f"  {lo:3d}-{hi:3d} kpc: {m.sum():3d} kept, mean chi_t {chi_t[m].mean():+.3f} ± "
                  f"{chi_t[m].std() / np.sqrt(m.sum()):.3f}, mean chi_x {chi_x[m].mean():+.3f}")

    with fits.open(opts.image, memmap=True) as h:
        wcs = WCS(h[0].header)
        size = 2 * opts.half_width_kpc / kpc_per_arcsec * u.arcsec
        cut = Cutout2D(h[0].data, centre, (size, size), wcs=wcs)
    img = np.nan_to_num(cut.data)
    lo, hi = np.percentile(img[img != 0], [5, 99.7])

    plt = sens.style()
    fig = plt.figure(figsize=(11, 10))
    ax = fig.add_subplot(projection=cut.wcs)
    ax.imshow(np.arcsinh((img - lo) / (hi - lo) * 20), origin="lower", cmap="gray_r",
              vmin=0, vmax=np.arcsinh(20))
    pix_scale = abs(cut.wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value)
    px_per_kpc = 1 / (kpc_per_arcsec * pix_scale)
    gx, gy = cut.wcs.world_to_pixel(SkyCoord(cat["RA"] * u.deg, cat["DEC"] * u.deg))
    cx, cy = cut.wcs.world_to_pixel(centre)
    theta = 0.5 * np.arctan2(c2, c1)          # pyRRG shapes are measured on the (north-up) image axes
    length = 30 * px_per_kpc * np.hypot(c1, c2)
    norm = plt.Normalize(-0.8, 0.8)
    cmap = plt.get_cmap("coolwarm")
    for i in np.where(sel)[0]:
        dx, dy = 0.5 * length[i] * np.cos(theta[i]), 0.5 * length[i] * np.sin(theta[i])
        if kept[i]:
            ax.plot([gx[i] - dx, gx[i] + dx], [gy[i] - dy, gy[i] + dy], color=cmap(norm(chi_t[i])), lw=2.5,
                    solid_capstyle="round")
        else:
            ax.plot([gx[i] - dx, gx[i] + dx], [gy[i] - dy, gy[i] + dy], color="0.5", lw=1.2, ls=(0, (2, 1)))
    for rr in (50, 100, 150):
        ax.add_patch(plt.Circle((cx, cy), rr * px_per_kpc, fill=False, color="#E69F00", lw=1, ls="--"))
        ax.text(cx + rr * px_per_kpc * 0.71, cy + rr * px_per_kpc * 0.71, f"{rr} kpc", color="#E69F00", fontsize=9)
    ax.plot(cx, cy, "+", color="#E69F00", ms=14, mew=2)
    (ra_ref, dec_ref), (xl, yl) = sens.LENSTOOL_REF, sens.LENSTOOL_HALOS["O2"]
    o2 = SkyCoord((ra_ref - xl / (3600 * np.cos(np.deg2rad(dec_ref)))) * u.deg, (dec_ref + yl / 3600) * u.deg)
    ax.plot(*cut.wcs.world_to_pixel(o2), "x", color="#009E73", ms=12, mew=2)
    ax.text(*(np.array(cut.wcs.world_to_pixel(o2)) + 40), "O2", color="#009E73", fontsize=10)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02).set_label("tangential distortion χt about the centre")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    ax.set_title(f"A2744 core, F115W: galaxies within {opts.rmax_kpc:g} kpc of the stamp centre (+). "
                 f"Solid: in the map ({np.sum(sel & kept)}); dashed grey: removed by the strong-lensing cut "
                 f"({np.sum(sel & ~kept)}). Whisker length ∝ |χ|", fontsize=9)
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=150, bbox_inches="tight")

    t = np.where(sel)[0][np.argsort(r[sel])]
    with open(f"{out}.txt", "w") as f:
        f.write("# RA Dec r_kpc chi1 chi2 chi_t chi_x kappa_SL kept\n")
        for i in t:
            f.write(f"{cat['RA'][i]:.6f} {cat['DEC'][i]:.6f} {r[i]:6.1f} {c1[i]:+.3f} {c2[i]:+.3f} {chi_t[i]:+.3f} "
                    f"{chi_x[i]:+.3f} {kappa[i]:.3f} {int(kept[i])}\n")
    print(f"wrote {out}.png and {out}.txt")


if __name__ == "__main__":
    main()
