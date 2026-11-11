#!/usr/bin/env python
'''
The Lenstool excess-convergence S/N maps of Dark_Matter_DA/lenstool/concat/delens_best/excess_mass.py (raw shapes,
delensed excess E-mode, B-mode, excess E-mode at source-plane positions) redrawn in kpc about a chosen centre, given
as an offset from the adopted stamp centre (default: where the NW and SE halves of the main-halo shear profile agree
inside 100 kpc, sbi/halfplane_profile.py --scan-centre). Marks the stamp centre, the Lenstool halos, the 135 deg split
line and --circle kpc about the centre.

    astro-env/bin/python sbi/excess_recentred.py
'''
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.cosmology import Planck18
from astropy.io import fits
import astropy.units as u

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import sensitivity as sens  # noqa: E402
from add_shear_to_data import SHEAR_CUTS, get_obs_data, ra_dec_to_simulation_image_pos  # noqa: E402

KPC_PER_PX = 20.
KPC_PER_ARCSEC = Planck18.kpc_proper_per_arcmin(0.308).to(u.kpc / u.arcsec).value
PANELS = [("KAPPA_E_RAW", "raw shapes (total mass)"), ("KAPPA_E", "delensed: excess E-mode"),
          ("KAPPA_B", "delensed: B-mode"), ("KAPPA_E_SRCPOS", "excess E-mode, source-plane positions")]


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dx-kpc", type=float, default=5., help="centre offset from the stamp centre, to the west")
    p.add_argument("--dy-kpc", type=float, default=70., help="to the north")
    p.add_argument("--half-width", type=float, default=600., help="kpc shown either side of the centre")
    p.add_argument("--angle", type=float, default=135., help="split line, degrees from west through north")
    p.add_argument("--circle", type=float, default=100., help="kpc")
    p.add_argument("--excess-map", default=str(sens.EXCESS_MAP))
    p.add_argument("--out", default=str(REPO / "sbi" / "results" / "main_halo_profile" / "excess_recentred"))
    return p.parse_args()


def main():
    opts = parse()
    cat = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS, drop_strong_lensing=False)
    forward, _ = sens.stamp_to_lenstool(cat, *ra_dec_to_simulation_image_pos(cat))
    x0, y0 = forward(0., 0.)                                              # stamp centre, Lenstool arcsec
    xc, yc = forward(opts.dx_kpc / KPC_PER_PX, opts.dy_kpc / KPC_PER_PX)
    print(f"stamp centre: Lenstool ({x0:.1f}, {y0:.1f}) arcsec; new centre ({xc:.1f}, {yc:.1f}) arcsec")
    for nm, (hx, hy) in sens.LENSTOOL_HALOS.items():
        print(f"  {nm}: ({KPC_PER_ARCSEC * (hx - xc):+.0f}, {KPC_PER_ARCSEC * (hy - yc):+.0f}) kpc from the new centre")

    with fits.open(opts.excess_map) as h:
        sig = h["SIGMA"].data
        hd = h["KAPPA_E"].header
        maps = {nm: h[nm].data / sig for nm, _ in PANELS}
    ny, nx = sig.shape
    x_lo = hd["CRVAL1"] - hd["CDELT1"] / 2
    y_lo = hd["CRVAL2"] - hd["CDELT2"] / 2
    ext = [KPC_PER_ARCSEC * (v - c) for v, c in ((x_lo, xc), (x_lo + nx * hd["CDELT1"], xc),
                                                  (y_lo, yc), (y_lo + ny * hd["CDELT2"], yc))]

    plt = sens.style()
    fig, axs = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    H = opts.half_width
    a = np.deg2rad(opts.angle)
    for ax, (nm, title) in zip(axs.flat, PANELS):
        im = ax.imshow(maps[nm], origin="lower", extent=ext, cmap="RdBu_r", vmin=-5, vmax=5)
        ax.contour(maps[nm], levels=[2, 3, 4, 5], colors="k", linewidths=0.6, origin="lower", extent=ext)
        for hn, (hx, hy) in sens.LENSTOOL_HALOS.items():
            px, py = KPC_PER_ARCSEC * (hx - xc), KPC_PER_ARCSEC * (hy - yc)
            ax.plot(px, py, "x", c="lime", ms=9, mew=2)
            ax.text(px + 15, py + 15, hn, color="lime", fontsize=9, clip_on=True)
        ax.plot(KPC_PER_ARCSEC * (x0 - xc), KPC_PER_ARCSEC * (y0 - yc), "+", c="k", ms=14, mew=2, label="stamp centre")
        ax.plot(0, 0, "*", c="gold", mec="k", ms=15, label=f"new centre ({opts.dx_kpc:+g}, {opts.dy_kpc:+g}) kpc")
        ax.add_patch(plt.Circle((0, 0), opts.circle, fill=False, color="k", ls=":", lw=1.2))
        ax.plot([-H * np.cos(a), H * np.cos(a)], [-H * np.sin(a), H * np.sin(a)], "k--", lw=0.9)
        ax.set_xlim(-H, H)
        ax.set_ylim(-H, H)
        ax.set_aspect("equal")
        ax.set_title(f"{title}, S/N", fontsize=10)
        ax.set_xlabel("kpc from the new centre  (E ← → W)")
        ax.set_ylabel("kpc  (S ← → N)")
    axs[0, 0].legend(fontsize=8, loc="lower left", frameon=True)
    fig.colorbar(im, ax=axs, shrink=0.8, label=r"$\kappa / \sigma_\kappa$")
    fig.suptitle(f"Lenstool excess convergence about the centre where the NW/SE shear halves agree inside 100 kpc "
                 f"(dotted: {opts.circle:g} kpc; dashed: {opts.angle:g}° split)", fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(f"{opts.out}.{e}", dpi=140)
    print(f"wrote {opts.out}.png")


if __name__ == "__main__":
    main()
