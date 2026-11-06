#!/usr/bin/env python
'''
Shear profile of the A2744 main halo from the galaxy catalogue: the mean tangential and cross
distortion of the galaxies (current SHEAR_CUTS) in log radial bins about the stamp centre used
for the CNN (the adopted centre, ~45 kpc from Lenstool O2), error bars from the scatter in each
bin. By default the profile is centred on the centre draw of sbi/centre_uncertainty.py with the
lowest ensemble P(SIDM) (--centre-json), with the profile about the adopted centre for comparison;
--dx-kpc/--dy-kpc give an offset from the adopted centre instead. Galaxies removed by the
strong-lensing cut (Lenstool kappa >= KAPPA_THRESH) are left out of the profile: the radii of the
radius of every removed galaxy is a tick.

    astro-env/bin/python sbi/main_halo_profile.py
'''
import argparse
import json
import pickle as pkl
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import sensitivity as sens  # noqa: E402
import shear_ratio as sr  # noqa: E402
from mass_weighting import a2744_mass_gaussian  # noqa: E402
from add_shear_to_data import (  # noqa: E402
    KAPPA_THRESH, SHEAR_CUTS, galaxy_distortion, get_obs_data, ra_dec_to_simulation_image_pos, strong_lensing_kappa,
)

KPC_PER_PX = 20.
EDGES = np.geomspace(20, 1500, 29)                     # kpc
R_INNER = 100.                                         # kpc: numerator of the ratio <chi_t>(< R_INNER) / <chi_t>(X)
OUT = REPO / "sbi" / "results" / "main_halo_profile" / "main_halo_profile"


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--centre-json", default=str(REPO / "sbi" / "results" / "centre_uncertainty" /
                                                "tweakedalign_k0p7_ctr_wb1_ba_best_spin2_cx_pgc_ft_.json"))
    p.add_argument("--dx-kpc", type=float, default=None)
    p.add_argument("--dy-kpc", type=float, default=None)
    p.add_argument("--kappa-thresh", type=float, default=KAPPA_THRESH,
                   help="drop galaxies with Lenstool kappa >= this (the maps use KAPPA_THRESH)")
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_wb1_best_n5"))
    p.add_argument("--sims", default="bahamas")
    p.add_argument("--cross-sections", type=float, nargs="+", default=[0, 1])
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--max-ncomp", type=int, default=6)
    p.add_argument("--mass-window", type=float, default=0.15)
    return p.parse_args()


def profile(x, y, c1, c2, kept):
    '''Radii and tangential/cross distortion about (0, 0), and the binned profile of the kept galaxies.'''
    r = KPC_PER_PX * np.hypot(x, y)
    phi = np.arctan2(y, x)
    chi_t = -(c1 * np.cos(2 * phi) + c2 * np.sin(2 * phi))
    chi_x = c1 * np.sin(2 * phi) - c2 * np.cos(2 * phi)
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = kept & (r >= lo) & (r < hi)
        n = m.sum()
        if n >= 2:
            rows.append((np.sqrt(lo * hi), n, chi_t[m].mean(), chi_t[m].std() / np.sqrt(n),
                         chi_x[m].mean(), chi_x[m].std() / np.sqrt(n), np.sum(~kept & (r >= lo) & (r < hi))))
    return r, np.array(rows).T


def ncomp_txt(opts):
    return f"= {opts.min_ncomp}" if opts.min_ncomp == opts.max_ncomp else f"≥ {opts.min_ncomp}"


def sim_profiles(opts, weight):
    '''Median and 16-84% chi_t profile (about the stamp centre) of mass-matched simulated clusters
    (n_comp >= --min-ncomp) per cross-section, from their noise-free maps sampled at the A2744
    galaxies. Pixels are weighted by their number of A2744 galaxies, so the pixel mean in each
    annulus approximates the galaxy mean of the data profile. Also the median and 16-84% of each
    cluster's ratio <chi_t>(< R_INNER) / <chi_t>(bin).'''
    mu, _ = a2744_mass_gaussian()
    S = Path(opts.summaries).resolve()
    P = dict(np.load(S / f"params_{opts.sims}.npz"))
    nc = np.minimum(P["ncomponents"], 6)
    m = (nc >= opts.min_ncomp) & (nc <= opts.max_ncomp) & (np.abs(P["log10_mass_750"] - mu) < opts.mass_window)
    t_pix = (-np.cos(2 * sr.PHI), -np.sin(2 * sr.PHI))
    r_pix = KPC_PER_PX * sr.RG
    masks = [weight * (r_pix >= lo) * (r_pix < hi) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    masks.append(weight * (r_pix < R_INNER))
    out = {}
    for c in opts.cross_sections:
        rows = np.where(m & np.isclose(P["cross_section"], c))[0]
        prof = []
        for f in np.unique(P["filename"][rows]):
            _, maps = pkl.load(open(f"../data/100/obs/concat/{f}", "rb"))
            chi = sr.to_chi(maps[P["image_idx"][rows[P["filename"][rows] == f]], :2])
            t = chi[:, 0] * t_pix[0] + chi[:, 1] * t_pix[1]
            prof.append(np.stack([np.sum(t * w, axis=(-2, -1)) / w.sum() if w.sum() else np.full(len(t), np.nan)
                                  for w in masks], -1))
        prof = np.concatenate(prof)
        out[c] = (np.nanpercentile(prof[:, :-1], [16, 50, 84], axis=0),
                  np.nanpercentile(prof[:, -1:] / prof[:, :-1], [16, 50, 84], axis=0))
        print(f"{opts.sims} σ={c:g}: {len(prof)} mass-matched clusters, n_comp {ncomp_txt(opts)}")
    return out


def main():
    opts = parse()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if opts.dx_kpc is None:
        draws = json.load(open(opts.centre_json))["draws"]
        best = min(draws, key=lambda d: d["p_sidm"])
        dx, dy, label = best["dx_px"], best["dy_px"], f"centre of lowest P(SIDM) = {best['p_sidm']:.3f}"
    else:
        dx, dy, label = opts.dx_kpc / KPC_PER_PX, opts.dy_kpc / KPC_PER_PX, "chosen centre"
    print(f"{label}: offset ({KPC_PER_PX * dx:+.0f}, {KPC_PER_PX * dy:+.0f}) kpc from the adopted centre")

    cat = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS, drop_strong_lensing=False)
    x, y = ra_dec_to_simulation_image_pos(cat)
    kappa = strong_lensing_kappa(cat)
    kept = kappa < opts.kappa_thresh
    c1, c2 = galaxy_distortion(cat)
    r, (rc, n, t, te, xx, xe, nrem) = profile(x - dx, y - dy, c1, c2, kept)
    _, (rc0, _, t0, te0, *_) = profile(x, y, c1, c2, kept)
    adopted = dx == 0 and dy == 0
    r_sl = np.sort(r[~kept])
    gaps = np.where(np.diff(r_sl) > 100)[0]               # the inner group ends at the first gap > 100 kpc
    r_in = r_sl[: gaps[0] + 1] if len(gaps) else r_sl
    print(f"{kept.sum()} galaxies kept, {(~kept).sum()} removed by the strong-lensing cut (kappa >= {opts.kappa_thresh}); "
          f"inner group {len(r_in)} at r = {r_in.min():.0f}-{r_in.max():.0f} kpc")
    for row in zip(rc, n, t, te, xx, xe, nrem):
        print(f"r {row[0]:6.0f} kpc  n {row[1]:5.0f} (+{row[6]:.0f} removed)  chi_t {row[2]:+.3f} ± {row[3]:.3f}"
              f"  chi_x {row[4]:+.3f} ± {row[5]:.3f}")

    weight = np.histogram2d(y[kept], x[kept], bins=sr.NPIX, range=[[-50, 50], [-50, 50]])[0]  # maps are [dec, ra]
    sims = sim_profiles(opts, weight) if adopted else {}
    vis = rc >= 60                                        # the x axis starts at 60 kpc
    rc, t, te, xx, xe = (a[vis] for a in (rc, t, te, xx, xe))
    rc0, t0, te0 = (a[rc0 >= 60] for a in (rc0, t0, te0))
    inner = kept & (r < R_INNER)
    rng = np.random.default_rng(0)
    phi = np.arctan2(y - dy, x - dx)
    chi_t = -(c1 * np.cos(2 * phi) + c2 * np.sin(2 * phi))
    lo_e, hi_e = EDGES[:-1], EDGES[1:]
    use = lo_e >= R_INNER
    bin_of = np.digitize(r, EDGES) - 1

    def ratio(idx):
        num = chi_t[idx][inner[idx]].mean()
        b, tt = bin_of[idx][kept[idx]], chi_t[idx][kept[idx]]
        return np.array([num / tt[b == j].mean() if np.any(b == j) else np.nan for j in np.where(use)[0]])
    all_idx = np.arange(len(r))
    ratio_a = ratio(all_idx)
    ratio_b = np.array([ratio(rng.integers(0, len(r), len(r))) for _ in range(200)])
    ratio_lo, ratio_hi = np.nanpercentile(ratio_b, [16, 84], axis=0)
    print(f"A2744 <chi_t>(< {R_INNER:g} kpc) = {chi_t[inner].mean():+.3f} from {inner.sum()} galaxies")

    plt = sens.style()
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(16, 5.4), constrained_layout=True)
    ax.plot(r_sl, np.full_like(r_sl, -0.12), "|", color="0.4", ms=9,
            label=f"radii of the {len(r_sl)} galaxies with κ ≥ {opts.kappa_thresh:g} (removed)")
    ax.axhline(0, color="0.5", lw=0.8)
    if not adopted:
        ax.errorbar(rc0 * 0.97, t0, te0, fmt="o:", color="0.55", capsize=2, ms=4, label="⟨χt⟩ about the adopted centre")
    if opts.kappa_thresh != KAPPA_THRESH:
        _, (rk, _, tk, tke, *_) = profile(x - dx, y - dy, c1, c2, kappa < KAPPA_THRESH)
        ax.errorbar(rk[rk >= 60] * 0.97, tk[rk >= 60], tke[rk >= 60], fmt="o:", color="0.55", capsize=2, ms=4,
                    label=f"⟨χt⟩ with the maps' cut, κ < {KAPPA_THRESH}")
    mid = np.sqrt(EDGES[:-1] * EDGES[1:])
    for c, ((q16, q50, q84), _) in sims.items():
        col = sr.COLOURS.get(c, "k")
        name = "CDM" if c == 0 else f"σ/m = {c:g}"
        ax.fill_between(mid, q16, q84, color=col, alpha=0.12, lw=0)
        ax.plot(mid, q50, "-", color=col, lw=1.8, label=f"{opts.sims.upper()} {name}, n_comp {ncomp_txt(opts)}: median (16-84% shaded)")
    ax.errorbar(rc, t, te, fmt="o-", color="#0072B2", capsize=2, lw=2, label=f"tangential ⟨χt⟩, κ < {opts.kappa_thresh:g}" + ("" if adopted else f", {label}"))
    ax.errorbar(rc * 1.03, xx, xe, fmt="s--", color="#CC79A7", capsize=2, ms=4, label="cross ⟨χx⟩")
    ax.set_xscale("log")
    ax.set_xlim(60, EDGES[-1])
    ax.set_ylim(-0.25, 0.65)
    ax.set_xlabel("radius [kpc]")
    ax.set_ylabel("mean galaxy distortion χ")
    where = ("about the adopted stamp centre (best-fit NFW)" if adopted else
             f"centred ({KPC_PER_PX * dx:+.0f}, {KPC_PER_PX * dy:+.0f}) kpc from the adopted centre")
    ax.set_title(f"A2744 main halo: catalogue shear profile ({kept.sum()} galaxies, SHEAR_CUTS), {where}", fontsize=9)
    ax.legend(fontsize=7.5)

    xr = np.sqrt(lo_e * hi_e)[use]
    for c, (_, (q16, q50, q84)) in sims.items():
        col = sr.COLOURS.get(c, "k")
        name = "CDM" if c == 0 else f"σ/m = {c:g}"
        axr.fill_between(xr, q16[use], q84[use], color=col, alpha=0.12, lw=0)
        axr.plot(xr, q50[use], "-", color=col, lw=1.8, label=f"{opts.sims.upper()} {name}, n_comp {ncomp_txt(opts)}: median")
    axr.errorbar(xr, ratio_a, [ratio_a - ratio_lo, ratio_hi - ratio_a], fmt="o-", color="#0072B2", capsize=2, lw=2,
                 label=f"A2744, κ < {opts.kappa_thresh:g} ({inner.sum()} galaxies inside {R_INNER:g} kpc; bootstrap 16-84%)")
    axr.axhline(1, color="0.5", lw=0.8)
    axr.set_xscale("log")
    axr.set_xlim(95, 1000)                                # beyond 1 Mpc only the stamp corners, where chi_t -> 0
    axr.set_xticks([100, 200, 300, 500, 1000], ["100", "200", "300", "500", "1000"])
    axr.minorticks_off()
    axr.set_ylim(0, 10)
    axr.set_xlabel("X: radius of the denominator bin [kpc]")
    axr.set_ylabel(f"⟨χt⟩(< {R_INNER:g} kpc) / ⟨χt⟩(X)")
    axr.set_title("ratio of the inner shear to the shear at X", fontsize=9)
    axr.legend(fontsize=7.5)
    out = f"{OUT}_k{opts.kappa_thresh:g}".replace(".", "p") + (f"_nc{opts.min_ncomp}" if opts.min_ncomp == opts.max_ncomp else f"_nc{opts.min_ncomp}plus" if opts.min_ncomp != 2 else "")
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=150)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    main()
