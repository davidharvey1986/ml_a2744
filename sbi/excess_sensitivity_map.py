#!/usr/bin/env python
'''
A2744 occlusion-sensitivity map (sbi/sensitivity.py output, mean over its models) with the Lenstool excess S/N
contours of excess_mass.py: the source-plane-position map (KAPPA_E_SRCPOS) next to the image-plane one (KAPPA_E),
each resampled onto the stamp with the stamp -> Lenstool affine fit of sensitivity.py.

    astro-env/bin/python sbi/excess_sensitivity_map.py --sens-dir sbi/results/sensitivity_wb1_final_w4s1
'''
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

REPO = Path(__file__).resolve().parents[1]
LAUNCH = Path.cwd()
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]
import sensitivity as sens  # noqa: E402


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sens-dir", default=str(REPO / "sbi" / "results" / "sensitivity_wb1_final_w4s1"))
    p.add_argument("--lenstool-dir", default=None,
                   help="Lenstool run dir (best.par + delens_best/excess_kappa.fits); default sensitivity.EXCESS_MAP")
    p.add_argument("--tag", default="", help="suffix of the output name")
    p.add_argument("--out", default=None)
    return p.parse_args()


def map_on_stamp(name, forward):
    '''S/N of one excess_kappa.fits map resampled onto the 100x100 stamp ([dec, ra]); nan outside.'''
    with fits.open(sens.EXCESS_MAP) as h:
        snr = h[name].data / h["SIGMA"].data
        hd = h["KAPPA_E"].header
    xg = hd["CRVAL1"] + hd["CDELT1"] * (np.arange(snr.shape[1]) + 1 - hd["CRPIX1"])
    yg = hd["CRVAL2"] + hd["CDELT2"] * (np.arange(snr.shape[0]) + 1 - hd["CRPIX2"])
    f = RegularGridInterpolator((yg, xg), snr, bounds_error=False, fill_value=np.nan)
    pix = np.arange(sens.NPIX) - (sens.NPIX / 2 - 0.5)
    PX, PY = np.meshgrid(pix, pix)
    x, y = forward(PX, PY)
    return f(np.c_[y.ravel(), x.ravel()]).reshape(sens.NPIX, sens.NPIX)


def halos_from_best(par):
    '''{name: (x, y)} of the potentials in a Lenstool best.par.'''
    out, name = {}, None
    for line in open(par):
        w = line.split()
        if w[:1] == ["potential"]:
            name, out[w[1]] = w[1], [None, None]
        elif name and w[:1] in (["x_centre"], ["y_centre"]):
            out[name][w[0] == "y_centre"] = float(w[1])
        elif w[:1] == ["end"]:
            name = None
    return {k: tuple(v) for k, v in out.items()}


def main():
    opts = parse()
    if opts.lenstool_dir:
        ld = Path(opts.lenstool_dir)
        sens.EXCESS_MAP = ld / "delens_best" / "excess_kappa.fits"
        sens.LENSTOOL_HALOS = halos_from_best(ld / "best.par")
        print(f"excess {sens.EXCESS_MAP}; halos {sens.LENSTOOL_HALOS}")
    sd = (LAUNCH / opts.sens_dir).resolve()
    z = np.load(sd / "sensitivity.npz")
    delta = z["a2744_delta"].mean(0)
    centres = z["centres"]
    stride = centres[1] - centres[0]
    extent = [centres[0] - stride / 2, centres[-1] + stride / 2] * 2

    raw, cat, ra_pix, dec_pix = sens.a2744_map()                       # the map sensitivity.py occluded
    forward, inverse = sens.stamp_to_lenstool(cat, ra_pix, dec_pix)
    kappa_snr = sens.reconstruct_kappa(raw)[1]
    halos = {k: tuple(inverse(*v)) for k, v in sens.LENSTOOL_HALOS.items()}
    maps = {"KAPPA_E_SRCPOS": "source-plane positions", "KAPPA_E": "image-plane positions"}
    snr = {k: map_on_stamp(k, forward) for k in maps}

    plt = sens.style()
    vmax = np.percentile(np.abs(delta), 99)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), constrained_layout=True)
    for ax, (k, lab) in zip(axes, maps.items()):
        im = sens.draw_panel(ax, {"delta": delta, "kappa_snr": kappa_snr, "excess_snr": snr[k], "halos": halos,
                                  "title": f"A2744: Lenstool excess, {lab}"}, extent, vmax)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(r"$\Delta P$(SIDM) when removed (red: pushes to SIDM)")
    fig.suptitle("Occlusion sensitivity (mean of the models), north up, east left. Black: reconstructed κ S/N 2,3,4,6,8; "
                 "green: Lenstool excess κ S/N ±2,3,4 and halos", fontsize=10)
    out = Path(opts.out) if opts.out else sd / f"sensitivity_map_excess_srcpos{opts.tag}"
    for e in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{e}"), dpi=170)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
