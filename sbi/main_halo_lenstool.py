#!/usr/bin/env python
'''
Tangential and cross distortion profile of the A2744 main halo (galaxy catalogue, SHEAR_CUTS, kappa < KAPPA_THRESH,
about the adopted stamp centre) with the Lenstool model (concat_g1v2: 3 NFW halos) evaluated at every galaxy for its
own photo-z (Jacobian of the z = 2 deflection maps scaled by Dls/Ds), turned into chi = 2g / (1 + g^2) and averaged
in the same bins. No fitting here.

    astro-env/bin/python sbi/main_halo_lenstool.py
'''
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import sensitivity as sens  # noqa: E402  (chdirs to notebooks/)
import lenstool_model_map as lmm  # noqa: E402
import main_halo_profile as mhp  # noqa: E402
import shear_ratio as sr  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from mass_weighting import a2744_mass_gaussian  # noqa: E402
from main_halo_profile import EDGES, profile  # noqa: E402
from add_shear_to_data import (  # noqa: E402
    KAPPA_THRESH, SHEAR_CUTS, galaxy_distortion, get_obs_data, ra_dec_to_simulation_image_pos, strong_lensing_kappa,
)

LT_DIR = Path("/Users/davidharvey/Work/Dark_Matter_DA/lenstool/concat_g1v2/delens_best")
OUT = REPO / "sbi" / "results" / "main_halo_profile" / "main_halo_lenstool"


def main():
    global EDGES
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lenstool-dir", default=str(LT_DIR))
    p.add_argument("--no-lenstool", action="store_true", help="leave out the Lenstool model")
    p.add_argument("--sims", action="store_true", help="add the median (16-84%% shaded) of mass-matched CDM and "
                   "sigma/m = 1 BAHAMAS clusters (n_comp >= 2, within 0.15 dex of the Jauzac mass, final summaries)")
    p.add_argument("--mass-study", default="jauzac", help="mass_weighting study the clusters are matched to "
                   "(harvey_g1v2: Harvey with the core rescaled to the new Lenstool O2)")
    p.add_argument("--core-only", action="store_true", help="match on the study's main-halo (core) mass alone")
    p.add_argument("--rmax", type=float, default=EDGES[-1], help="outer radius shown [kpc]")
    p.add_argument("--nbins", type=int, default=len(EDGES) - 1, help="log radial bins between 20 and 1500 kpc")
    p.add_argument("--out", default=str(OUT))
    opts = p.parse_args()
    lmm.LT = Path(opts.lenstool_dir)
    EDGES = mhp.EDGES = np.geomspace(20, 1500, opts.nbins + 1)

    cat = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS, drop_strong_lensing=False, photoz=True)
    x, y = ra_dec_to_simulation_image_pos(cat)
    kept = strong_lensing_kappa(cat) < KAPPA_THRESH
    c1, c2 = galaxy_distortion(cat)
    r, (rc, n, t, te, xx, xe, _) = profile(x, y, c1, c2, kept)

    # model in the Lenstool frame (x west, y north) about the stamp centre mapped into that frame
    if opts.sims:
        if opts.core_only:
            from mass_weighting import A2744_MASSES
            core = np.log10(A2744_MASSES[opts.mass_study]["core"] * 1e13)
            print(f"matching on the {opts.mass_study} core mass alone: log M = {core:.3f}")
            mhp.a2744_mass_gaussian = lambda study=None: (core, 0.0)
        else:
            mhp.a2744_mass_gaussian = lambda study=None: a2744_mass_gaussian(opts.mass_study)
        weight = np.histogram2d(y[kept], x[kept], bins=sr.NPIX, range=[[-50, 50], [-50, 50]])[0]
        sims = mhp.sim_profiles(SimpleNamespace(summaries=str(REPO / "sbi" / "summaries_wb1_final_n5"), sims="bahamas",
                                                cross_sections=[0, 1], min_ncomp=2, max_ncomp=6, mass_window=0.15), weight)
    g1, g2, _ = lmm.model_shear(cat)
    lx, ly = lmm.lenstool_xy(cat)
    A = np.c_[x, y, np.ones_like(x)]
    cx = np.linalg.lstsq(A, lx, rcond=None)[0][2]
    cy = np.linalg.lstsq(A, ly, rcond=None)[0][2]
    m1, m2 = galaxy_distortion({"gamma1": g1, "gamma2": g2})
    phi = np.arctan2(ly - cy, lx - cx)
    mt = -(m1 * np.cos(2 * phi) + m2 * np.sin(2 * phi))
    bin_of = np.digitize(r, EDGES) - 1
    centres = np.sqrt(EDGES[:-1] * EDGES[1:])
    model = np.array([mt[kept & (bin_of == np.where(np.isclose(centres, c))[0][0])].mean() for c in rc])
    vis = (rc >= 60) & (rc <= opts.rmax)
    print("r, data, model:", *(f"{a:.0f}:{b:+.3f}/{c:+.3f}" for a, b, c in zip(rc[vis], t[vis], model[vis])))

    save = dict(rc=rc[vis], t=t[vis], te=te[vis], xx=xx[vis], xe=xe[vis], rmax=opts.rmax)
    if opts.sims:
        mid = np.sqrt(EDGES[:-1] * EDGES[1:]); km = mid <= opts.rmax
        save["mid"] = mid[km]
        for c, ((q16, q50, q84), _) in sims.items():
            save[f"sim_{c:g}"] = np.array([q16[km], q50[km], q84[km]])
    np.savez(f"{opts.out}.npz", **save)                # for figure 1 panel 6 (plots_for_paper.figureA1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                   # plain matplotlib, full frame, as cdm_pooled_tail.py
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.errorbar(rc[vis], t[vis], te[vis], fmt="o", color="#0072B2", capsize=2, label=r"tangential $\langle\chi_t\rangle$")
    ax.errorbar(rc[vis] * 1.03, xx[vis], xe[vis], fmt="s", color="#CC79A7", capsize=2, ms=4, mfc="none",
                label=r"cross $\langle\chi_\times\rangle$")
    if not opts.no_lenstool:
        ax.plot(rc[vis], model[vis], "-", color="k", lw=2, label="Lenstool model")
    if opts.sims:
        mid = np.sqrt(EDGES[:-1] * EDGES[1:])
        keep_mid = mid <= opts.rmax
        for c, ((q16, q50, q84), _) in sims.items():
            col = "k" if c == 0 else "#D55E00"
            ax.fill_between(mid[keep_mid], q16[keep_mid], q84[keep_mid], color=col, alpha=0.12, lw=0)
            ax.plot(mid[keep_mid], q50[keep_mid], "-", color=col, lw=2, label="BAHAMAS CDM" if c == 0 else rf"BAHAMAS $\sigma/m={c:g}$ cm$^2$g$^{{-1}}$")
    ax.set_xscale("log")
    ax.set_xlim(60, opts.rmax)
    ticks = [t for t in (60, 100, 200, 400, 800, 1500) if t <= opts.rmax]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Radius [kpc]", fontsize=13)
    ax.set_ylabel(r"Mean galaxy distortion, $\chi$", fontsize=13)
    ax.legend(fontsize=10)
    for e in ("png", "pdf"):
        fig.savefig(f"{opts.out}.{e}", dpi=200)
    print(f"wrote {opts.out}.png")


if __name__ == "__main__":
    main()
