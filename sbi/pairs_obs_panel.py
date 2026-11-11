#!/usr/bin/env python
'''
2 x 5 figure: occlusion-sensitivity maps of selected SIDM / CDM-twin pairs (pair_sensitivity.py outputs; CDM top,
SIDM bottom) next to A2744. Black: reconstructed kappa S/N 2,4,8,12 (one noise draw for the sims); dotted: true
kappa (sims). A2744: KS kappa S/N (top), Lenstool model kappa and halos (bottom).

    astro-env/bin/python sbi/pairs_obs_panel.py --pairs 7 10 12 15
'''
import argparse
import json
import re
import pickle as pkl
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]
import sensitivity as sens  # noqa: E402  (chdirs to notebooks/)
import make_summaries as ms  # noqa: E402
from excess_sensitivity_map import halos_from_best  # noqa: E402

RES = REPO / "sbi" / "results"
LT = Path("/Users/davidharvey/Work/Dark_Matter_DA/lenstool/concat_g1v2")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", type=int, nargs="+", default=[7, 10, 12, 15])
    p.add_argument("--pairs-json", default=str(RES / "a2744_analogue_pairs_nc4_all.json"))
    p.add_argument("--tag", default="w4s1")
    p.add_argument("--sens-dir", default=str(RES / "sensitivity_wb1_final_w4s1"))
    p.add_argument("--out", default=str(RES / "pairs_obs_sensitivity"))
    return p.parse_args()


def lenstool_kappa_on_stamp(forward):
    '''Lenstool model kappa for a source at z = 2: half the divergence of the dx_z2 / dy_z2 deflection maps.'''
    ax = fits.getdata(LT / "delens_best" / "dx_z2.fits").astype(float)
    ay = fits.getdata(LT / "delens_best" / "dy_z2.fits").astype(float)
    xg = np.linspace(-550., 170., ax.shape[1])
    yg = np.linspace(-390., 170., ax.shape[0])
    k = 0.5 * (np.gradient(ax, xg, axis=1) + np.gradient(ay, yg, axis=0))
    f = RegularGridInterpolator((yg, xg), k, bounds_error=False, fill_value=np.nan)
    pix = np.arange(sens.NPIX) - (sens.NPIX / 2 - 0.5)
    PX, PY = np.meshgrid(pix, pix)
    x, y = forward(PX, PY)
    return f(np.c_[y.ravel(), x.ravel()]).reshape(sens.NPIX, sens.NPIX)


def main():
    opts = parse()
    J = json.load(open(opts.pairs_json))
    files = {"cdm_image": J["settings"]["cdm_file"], "sidm_image": J["settings"]["sidm_file"]}
    kap = {f: pkl.load(open(f"../data/100/shear/{f}", "rb"))[1] for f in files.values()}
    args = ms.setup_args("concat")
    cols = []
    for n in opts.pairs:
        q = J["pairs"][n - 1]
        Z = np.load(RES / f"pair{n}_sensitivity_{opts.tag}.npz")
        col = []
        for key, lab in (("cdm_image", "CDM"), ("sidm_image", "SIDM σ/m=1")):
            f = files[key]
            raws, _ = sens.simulated_maps({"filename": np.array([f]), "image_idx": np.array([q[key]])}, args, 0)
            nc = q["ncomp_cdm" if key == "cdm_image" else "ncomp_sidm"]
            col.append({"delta": Z[f"{key}_delta"],
                        "kappa_snr": sens.reconstruct_kappa(raws[0].numpy())[1],
                        "kappa_true": kap[f][q[key], 2],
                        "title": f"pair {n} {lab} ({q[key]})\nP(SIDM) = {np.mean(Z[key + '_p']):.2f}, n$_c$ = {nc}"})
        cols.append((col, Z["centres"]))

    z = np.load(Path(opts.sens_dir) / "sensitivity.npz")
    delta = z["a2744_delta"].mean(0)
    raw, cat, ra_pix, dec_pix = sens.a2744_map()
    forward, inverse = sens.stamp_to_lenstool(cat, ra_pix, dec_pix)
    sens.EXCESS_MAP = LT / "delens_best" / "excess_kappa.fits"
    halos = {k: tuple(inverse(*v)) for k, v in halos_from_best(LT / "best.par").items()}
    k_snr = sens.reconstruct_kappa(raw)[1]
    obs_p = np.load(REPO / "sbi" / "summaries_wb1_final_n5" / "obs.npz")["softmax"][:, 0].mean()
    ptxt = f", P(SIDM) = {obs_p:.2f}"
    k_lt = lenstool_kappa_on_stamp(forward)
    cols.append(([{"delta": delta, "kappa_snr": k_snr, "title": f"A2744 (data){ptxt}\nKS κ S/N 2,4,8,12"},
                  {"delta": delta, "kappa_snr": np.zeros_like(k_snr), "lenstool_kappa": k_lt, "halos": halos,
                   "title": "A2744 (data)\nLenstool model κ 0.1–0.8 (z$_s$ = 2)"}], z["centres"]))

    vmax = np.percentile(np.abs(np.concatenate([p["delta"].ravel() for c, _ in cols for p in c])), 99)
    plt = sens.style()
    fig, axes = plt.subplots(2, len(cols), figsize=(3.6 * len(cols), 7.6), constrained_layout=True)
    for c, (col, centres) in enumerate(cols):
        s = centres[1] - centres[0]
        ext = [centres[0] - s / 2, centres[-1] + s / 2] * 2
        for r, p in enumerate(col):
            im = sens.draw_panel(axes[r, c], p, ext, vmax, scalebar=False)
            if r == 1 and c == 0:                         # scale bar top right (the label sits bottom left)
                ax_ = axes[r, c]
                (x0, x1), (y0, y1) = ax_.get_xlim(), ax_.get_ylim()
                xc, yc = x1 - 0.06 * (x1 - x0) - 10, y1 - 0.08 * (y1 - y0)
                ax_.errorbar(xc, yc, xerr=10, capsize=3, color="k", lw=2)       # 20 stamp pixels = 400 kpc
                ax_.text(xc, yc - 0.02 * (y1 - y0), "400 kpc", ha="center", va="top", fontsize=12, color="k")
            if p.get("lenstool_kappa") is not None:
                axes[r, c].contour(p["lenstool_kappa"], origin="lower", extent=[-50, 50, -50, 50],
                                   levels=np.linspace(0.1, 0.8, 8), colors="k", linewidths=1)
            t = axes[r, c].get_title()                    # no titles: the label goes inside the panel
            axes[r, c].set_title("")
            t = re.sub(r" \((\d+)\)\n", r"\n(\1)\n", t)        # the cluster id on its own line
            lab = axes[r, c].text(0.03, 0.03, t, transform=axes[r, c].transAxes, ha="left", va="bottom", fontsize=14,
                                  linespacing=1.15, bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))
            lab.set_in_layout(False)                      # the label must not shrink the maps
    axes[0, 0].set_ylabel("CDM", fontsize=11)
    axes[1, 0].set_ylabel(r"SIDM $\sigma/m = 1\,\mathrm{cm^2\,g^{-1}}$ twin", fontsize=11)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label(r"$\Delta P$(SIDM) when removed (red: pushes to SIDM)")
    for e in ("png", "pdf"):
        fig.savefig(f"{opts.out}.{e}", dpi=170)
    print(f"wrote {opts.out}.png")


if __name__ == "__main__":
    main()
