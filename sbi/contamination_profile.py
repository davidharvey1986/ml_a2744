#!/usr/bin/env python
'''
Tangential distortion profile of the main halo under the cluster-member contamination systematic.
The contamination test in systematics.py dilutes the simulated shear pixel by pixel,
g -> g (1 - c kappa) (c < 0 boosts it), and leaves the A2744 data alone; this shows what that does to
the mean <chi_t> profile of the BAHAMAS clusters (CDM and sigma/m = 1, n_comp >= --min-ncomp, within
systematics.SUBSET_DEX of the A2744 mass) sampled on the A2744 galaxies, next to the A2744 profile. As in
the systematics, each cluster carries the mass weight N(log M750; mu, sigma) / KDE density of the sample
(mass_weighting.gaussian_mass_weights, A2744 mass and error), and the profiles are weighted percentiles.

    astro-env/bin/python sbi/contamination_profile.py
'''
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import systematics as sy  # noqa: E402  (chdirs to notebooks/)
import sensitivity as sens  # noqa: E402
import shear_ratio as sr  # noqa: E402

KPC_PER_PX = 20.
EDGES = np.geomspace(40, 1200, 16)                    # kpc
MIN_GAL = 10                                           # A2744 bins with fewer galaxies are not shown
CONTAM = [-0.5, -0.25, 0, 0.25, 0.5]
OUT = REPO / "sbi" / "results" / "contamination_profile" / "contamination_profile"


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summaries", default=str(REPO / "sbi" / "summaries_wb1_best_n5"))
    p.add_argument("--min-ncomp", type=int, default=2)
    p.add_argument("--cross-sections", type=float, nargs="+", default=[0, 1])
    return p.parse_args()


def data_profile(d, ra, dec):
    '''Galaxy-mean tangential distortion of A2744 about the stamp centre, error from the scatter; NaN
    in bins with fewer than MIN_GAL galaxies.'''
    g = np.sqrt(d["gamma1"] ** 2 + d["gamma2"] ** 2)
    c1, c2 = 2 * d["gamma1"] / (1 + g ** 2), 2 * d["gamma2"] / (1 + g ** 2)
    r, phi = KPC_PER_PX * np.hypot(ra, dec), np.arctan2(dec, ra)
    t = -(c1 * np.cos(2 * phi) + c2 * np.sin(2 * phi))
    b = np.digitize(r, EDGES) - 1
    rows = [(t[b == j].mean(), t[b == j].std() / np.sqrt((b == j).sum())) if (b == j).sum() >= MIN_GAL
            else (np.nan, np.nan) for j in range(len(EDGES) - 1)]
    return np.array(rows).T


def weighted_percentiles(x, w, q):
    '''Weighted percentiles q (in %) of x (N, n_bins) along axis 0, ignoring NaNs.'''
    out = np.full((len(q), x.shape[1]), np.nan)
    for j in range(x.shape[1]):
        ok = np.isfinite(x[:, j]) & (w > 0)
        if ok.any():
            o = np.argsort(x[ok, j])
            cw = np.cumsum(w[ok][o])
            out[:, j] = np.interp(np.asarray(q) / 100 * cw[-1], cw - w[ok][o] / 2, x[ok, j][o])
    return out


def sim_profiles(maps, weight, c):
    '''<chi_t> per annulus for each cluster after g -> g (1 - c kappa); pixels weighted by their galaxy count.'''
    g = maps[:, :2] * (1 - np.minimum(c * maps[:, 2:3], 1))
    chi = sr.to_chi(g)
    t = -(chi[:, 0] * np.cos(2 * sr.PHI) + chi[:, 1] * np.sin(2 * sr.PHI))
    r = KPC_PER_PX * sr.RG
    out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        w = weight * (r >= lo) * (r < hi)
        out.append(np.sum(t * w, axis=(-2, -1)) / w.sum() if w.sum() else np.full(len(t), np.nan))
    return np.stack(out, -1)


def main():
    opts = parse()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    mu, sig = sy.a2744_mass_gaussian()
    cat = sy.Catalogue(sy.get_obs_data("concat", data_dir=sy.DATA_DIR, photoz=True, cuts=sy.FILTER_CUTS))
    d = cat.select()
    ra, dec = cat.positions(d)
    dsl = sy.get_obs_data("concat", data_dir=sy.DATA_DIR, photoz=True, cuts=sy.FILTER_CUTS, drop_strong_lensing=True)
    t_obs, te_obs = data_profile(dsl, *cat.positions(dsl))         # as the A2744 map: Lenstool kappa < KAPPA_THRESH
    weight = np.histogram2d(dec, ra, bins=sy.NPIX, range=[[-50, 50], [-50, 50]])[0]   # maps are [dec, ra]
    print(f"A2744: {len(d)} galaxies")

    P = dict(np.load(Path(opts.summaries) / "params_bahamas.npz"))
    mid = np.sqrt(EDGES[:-1] * EDGES[1:])
    plt = sens.style()
    fig, axes = plt.subplots(1, len(opts.cross_sections), figsize=(7 * len(opts.cross_sections), 5.4),
                             sharey=True, constrained_layout=True)
    cmap = plt.get_cmap("coolwarm")
    for ax, cs in zip(np.atleast_1d(axes), opts.cross_sections):
        keep = ((P["ncomponents"] >= opts.min_ncomp) & np.isclose(P["cross_section"], cs)
                & (np.abs(P["log10_mass_750"] - mu) < sy.SUBSET_DEX))
        sel = {k: P[k][keep] for k in ("filename", "image_idx", "cross_section", "log10_mass_750", "ncomponents")}
        wm = sy.gaussian_mass_weights(sel["log10_mass_750"], mu, sig)
        n_eff = sy.effective_sample_size(wm)
        maps = sy.SimMaps(sel).sample(ra, dec, with_kappa=True)
        name = "CDM" if cs == 0 else f"σ/m = {cs:g}"
        print(f"BAHAMAS {name}: {keep.sum()} clusters, N_eff {n_eff:.0f}")
        for c in CONTAM:
            q16, q50, q84 = weighted_percentiles(sim_profiles(maps, weight, c), wm, [16, 50, 84])
            col = "k" if c == 0 else cmap(0.5 + c)
            if c == 0:
                ax.fill_between(mid, q16, q84, color="0.6", alpha=0.25, lw=0, label="c = 0: weighted 16-84% of clusters")
            ax.plot(mid, q50, "-", color=col, lw=2.4 if c == 0 else 1.6,
                    label=f"c = {c:+g}" + (" (fiducial)" if c == 0 else " (boost)" if c < 0 else " (dilution)"))
            print(f"  c {c:+.2f}: " + " ".join(f"{v:+.3f}" for v in q50))
        ax.errorbar(mid, t_obs, te_obs, fmt="o", color="#0072B2", capsize=2, ms=5, lw=1.5, zorder=5,
                    label=f"A2744 (data, unchanged; Lenstool κ < {sy.KAPPA_THRESH:g}, ≥ {MIN_GAL} galaxies per bin)")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlim(EDGES[0], EDGES[-1])
        ax.set_xlabel("radius from the stamp centre [kpc]")
        ax.set_title(f"BAHAMAS {name}: {keep.sum()} clusters (N_eff {n_eff:.0f}), n_comp ≥ {opts.min_ncomp}\n"
                     f"mass-weighted to log M₇₅₀ = {mu:.2f} ± {sig:.2f}; weighted median", fontsize=9)
        ax.legend(fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("mean tangential distortion ⟨χt⟩")
    fig.suptitle("Cluster-member contamination: simulated shear × (1 − c κ)", fontsize=11)
    for e in ("png", "pdf"):
        fig.savefig(f"{OUT}.{e}", dpi=150)
    print(f"wrote {OUT}.png")


if __name__ == "__main__":
    main()
