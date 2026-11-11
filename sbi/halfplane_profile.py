#!/usr/bin/env python
'''
Shear profile of the A2744 main halo split into two halves by a line through the stamp centre at --angle degrees
from the stamp x axis (catalogue pixels: x to the west, y to the north, so 135 deg runs NE-SW and the halves are NW
and SE). Mean tangential and cross distortion of the galaxies (SHEAR_CUTS, Lenstool kappa < KAPPA_THRESH) in log
radial bins in each half, error from the scatter in each bin, and the difference of the two halves.

    astro-env/bin/python sbi/halfplane_profile.py
'''
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "notebooks"), str(REPO / "sbi")]

import sensitivity as sens  # noqa: E402
from add_shear_to_data import (  # noqa: E402
    KAPPA_THRESH, SHEAR_CUTS, SL_KAPPA_MAP, galaxy_distortion, get_obs_data, ra_dec_to_simulation_image_pos,
    strong_lensing_kappa,
)

KPC_PER_PX = 20.
COMPASS = ["W", "NW", "N", "NE", "E", "SE", "S", "SW"]          # direction of the stamp angle, 45 deg steps


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--angle", type=float, default=135., help="dividing line, degrees from the stamp x axis (west)")
    p.add_argument("--dx-kpc", type=float, default=0., help="profile centre offset from the adopted stamp centre")
    p.add_argument("--dy-kpc", type=float, default=0.)
    p.add_argument("--kappa-peak", action="store_true",
                   help="centre on the peak of the Lenstool convergence within --peak-radius of the stamp centre")
    p.add_argument("--peak-radius", type=float, default=300., help="kpc")
    p.add_argument("--scan-centre", action="store_true",
                   help="centre where the two halves' mean chi_t inside --core kpc are closest (grid about the stamp centre)")
    p.add_argument("--core", type=float, default=100., help="kpc")
    p.add_argument("--scan-max", type=float, default=150., help="kpc, half-width of the scanned grid")
    p.add_argument("--scan-step", type=float, default=5., help="kpc")
    p.add_argument("--scan-radius", type=float, default=80.,
                   help="kpc: the closest centre is picked within this of the stamp centre (the map shows all of --scan-max)")
    p.add_argument("--min-core", type=int, default=5, help="galaxies needed inside --core in each half")
    p.add_argument("--edges", type=float, nargs="+", default=list(np.geomspace(40, 1000, 12)))
    p.add_argument("--out", default=str(REPO / "sbi" / "results" / "main_halo_profile" / "halfplane_profile"))
    return p.parse_args()


def binned(r, v, edges):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        n = m.sum()
        rows.append((np.sqrt(lo * hi), n, v[m].mean() if n >= 2 else np.nan, v[m].std() / np.sqrt(n) if n >= 2 else np.nan))
    return np.array(rows).T


def kappa_peak(cat, x, y, radius):
    '''Stamp position (x, y) of the Lenstool convergence peak within radius kpc of the stamp centre: map pixels to
    stamp coordinates with the affine fit of the catalogue's (RA, Dec) to its stamp positions.'''
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(SL_KAPPA_MAP) as h:
        kappa, wcs = h[0].data, WCS(h[0].header)
    A = np.column_stack([np.asarray(cat["RA"]), np.asarray(cat["DEC"]), np.ones(len(x))])
    coef = np.linalg.lstsq(A, np.column_stack([x, y]), rcond=None)[0]
    jj, ii = np.indices(kappa.shape)
    ra, dec = wcs.pixel_to_world_values(ii.ravel(), jj.ravel())
    sx, sy = (np.column_stack([ra, dec, np.ones(ra.size)]) @ coef).T
    k = np.where(KPC_PER_PX * np.hypot(sx, sy) < radius, np.nan_to_num(kappa.ravel()), -np.inf)
    b = np.argmax(k)
    print(f"Lenstool kappa peak {k[b]:.2f} at RA {ra[b]:.5f}, Dec {dec[b]:.5f}: "
          f"({KPC_PER_PX * sx[b]:+.0f}, {KPC_PER_PX * sy[b]:+.0f}) kpc from the stamp centre (x west, y north)")
    return sx[b], sy[b]


def core_difference(x, y, c1, c2, kept, angle, core, min_n):
    '''Mean chi_t inside core kpc of each half about (0, 0): (left - right, its error, n_left, n_right).'''
    r = KPC_PER_PX * np.hypot(x, y)
    phi = np.arctan2(y, x)
    chi_t = -(c1 * np.cos(2 * phi) + c2 * np.sin(2 * phi))
    a = np.deg2rad(angle)
    side = (x * -np.sin(a) + y * np.cos(a)) > 0
    m = [kept & (r < core) & h for h in (side, ~side)]
    n = [k.sum() for k in m]
    if min(n) < max(min_n, 2):
        return np.nan, np.nan, *n
    d = chi_t[m[0]].mean() - chi_t[m[1]].mean()
    return d, np.hypot(*(chi_t[k].std() / np.sqrt(k.sum()) for k in m)), *n


def scan_centre(x, y, c1, c2, kept, opts, out):
    '''Grid of centres; the one with the smallest |difference| of the two halves inside opts.core.'''
    g = np.arange(-opts.scan_max, opts.scan_max + 1e-6, opts.scan_step)
    D = np.full((len(g), len(g)), np.nan)
    E, N = np.full_like(D, np.nan), np.zeros(D.shape + (2,), int)
    for i, dy in enumerate(g):
        for j, dx in enumerate(g):
            d, e, n1, n2 = core_difference(x - dx / KPC_PER_PX, y - dy / KPC_PER_PX, c1, c2, kept, opts.angle,
                                           opts.core, opts.min_core)
            D[i, j], E[i, j], N[i, j] = d, e, (n1, n2)
    gx, gy = np.meshgrid(g, g)
    i, j = np.unravel_index(np.nanargmin(np.where(np.hypot(gx, gy) <= opts.scan_radius, np.abs(D), np.nan)), D.shape)
    i0, j0 = np.unravel_index(np.nanargmin(np.abs(D)), D.shape)
    print(f"  over the whole ±{opts.scan_max:g} kpc grid: ({g[j0]:+.0f}, {g[i0]:+.0f}) kpc, difference {D[i0, j0]:+.3f}")
    print(f"closest core (< {opts.core:g} kpc): centre ({g[j]:+.0f}, {g[i]:+.0f}) kpc from the stamp centre, "
          f"difference {D[i, j]:+.3f} ± {E[i, j]:.3f}, n {N[i, j, 0]} / {N[i, j, 1]}")
    k0 = np.argmin(np.abs(g))
    print(f"  at the stamp centre: difference {D[k0, k0]:+.3f} ± {E[k0, k0]:.3f}, n {N[k0, k0, 0]} / {N[k0, k0, 1]}")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ext = [g[0] - opts.scan_step / 2, g[-1] + opts.scan_step / 2] * 2
    for ax, img, lab, kw in ((axes[0], D, "Δ⟨χt⟩ (< core)", dict(cmap="RdBu_r", vmin=-0.5, vmax=0.5)),
                             (axes[1], D / E, "Δ⟨χt⟩ / error", dict(cmap="RdBu_r", vmin=-4, vmax=4))):
        im = ax.imshow(img, origin="lower", extent=ext, **kw)
        ax.contour(g, g, img, [0], colors="k", linewidths=0.8)
        ax.plot(0, 0, "+", color="k", ms=12, label="stamp centre")
        ax.add_patch(plt.Circle((0, 0), opts.scan_radius, fill=False, color="k", ls=":", lw=1))
        ax.plot(g[j], g[i], "*", color="gold", mec="k", ms=16, label=f"closest ({g[j]:+.0f}, {g[i]:+.0f}) kpc")
        L = opts.scan_max
        a = np.deg2rad(opts.angle)
        ax.plot([-L * np.cos(a), L * np.cos(a)], [-L * np.sin(a), L * np.sin(a)], "k--", lw=0.8)
        ax.set_xlabel("centre offset x [kpc]  (E ← → W)")
        ax.set_ylabel("centre offset y [kpc]  (S ← → N)")
        fig.colorbar(im, ax=ax, label=lab)
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Inner (< {opts.core:g} kpc) tangential shear, half left minus half right of the {opts.angle:g}° line, "
                 f"against the profile centre (blank: < {opts.min_core} galaxies in a half)", fontsize=10)
    fig.savefig(f"{out}_scan.png", dpi=150)
    fig.savefig(f"{out}_scan.pdf")
    return g[j], g[i]


def main():
    opts = parse()
    cat = get_obs_data("concat", data_dir="../data/100/a2744/", cuts=SHEAR_CUTS, drop_strong_lensing=False)
    x, y = ra_dec_to_simulation_image_pos(cat)
    if opts.kappa_peak:
        px, py = kappa_peak(cat, x, y, opts.peak_radius)
        opts.dx_kpc, opts.dy_kpc = KPC_PER_PX * px, KPC_PER_PX * py
    kept = strong_lensing_kappa(cat) < KAPPA_THRESH
    c1, c2 = galaxy_distortion(cat)
    if opts.scan_centre:
        opts.dx_kpc, opts.dy_kpc = scan_centre(x, y, c1, c2, kept, opts, opts.out)
    x, y = x - opts.dx_kpc / KPC_PER_PX, y - opts.dy_kpc / KPC_PER_PX
    r = KPC_PER_PX * np.hypot(x, y)
    phi = np.arctan2(y, x)
    chi_t = -(c1 * np.cos(2 * phi) + c2 * np.sin(2 * phi))
    chi_x = c1 * np.sin(2 * phi) - c2 * np.cos(2 * phi)
    edges = np.array(opts.edges)

    a = np.deg2rad(opts.angle)
    side = (x * -np.sin(a) + y * np.cos(a)) > 0                  # left of the line direction
    names = [COMPASS[int(round(((opts.angle + s) % 360) / 45)) % 8] for s in (90, -90)]   # left half, right half
    halves = [(side, names[0], "#D55E00"), (~side, names[1], "#0072B2")]

    plt = sens.style()
    fig, (ax, axd, axm) = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True,
                                       gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
    prof = {}
    for m, name, col in halves:
        sel = m & kept
        rc, n, t, te = binned(r[sel], chi_t[sel], edges)
        _, _, xx, xe = binned(r[sel], chi_x[sel], edges)
        prof[name] = (rc, n, t, te)
        print(f"{name} half: {sel.sum()} galaxies")
        for row in zip(rc, n, t, te, xx, xe):
            print(f"  r {row[0]:6.0f} kpc  n {row[1]:5.0f}  chi_t {row[2]:+.3f} ± {row[3]:.3f}  chi_x {row[4]:+.3f} ± {row[5]:.3f}")
        ax.errorbar(rc * (1.02 if col == "#0072B2" else 0.98), t, te, fmt="o-", color=col, capsize=2, lw=2,
                    label=f"{name} half: tangential ⟨χt⟩ ({sel.sum()} galaxies)")
        ax.errorbar(rc * (1.04 if col == "#0072B2" else 0.96), xx, xe, fmt="s--", color=col, capsize=2, ms=3.5, lw=1,
                    alpha=0.6, label=f"{name} half: cross ⟨χx⟩")
        axm.scatter(x[sel], y[sel], s=2, color=col, alpha=0.5, lw=0)
    axm.scatter(x[~kept], y[~kept], s=3, color="0.6", lw=0, label=f"κ ≥ {KAPPA_THRESH:g} (removed)")

    (rc, _, t1, e1), (_, _, t2, e2) = prof[names[0]], prof[names[1]]
    d, de = t1 - t2, np.hypot(e1, e2)
    ok = np.isfinite(d)
    chisq = np.sum((d[ok] / de[ok]) ** 2)
    p = chi2.sf(chisq, ok.sum())
    print(f"{names[0]} - {names[1]}: chi^2 {chisq:.1f} for {ok.sum()} bins, p = {p:.3g} ({norm.isf(p):.2f} sigma)")
    for lo_r, hi_r in ((40, 200), (200, 500), (500, 1000)):
        w = [(r >= lo_r) & (r < hi_r) & kept & h for h in (side, ~side)]
        m1, m2 = chi_t[w[0]].mean(), chi_t[w[1]].mean()
        s = np.hypot(chi_t[w[0]].std() / np.sqrt(w[0].sum()), chi_t[w[1]].std() / np.sqrt(w[1].sum()))
        print(f"  {lo_r}-{hi_r} kpc: {names[0]} {m1:+.3f}, {names[1]} {m2:+.3f}, difference {m1 - m2:+.3f} ± {s:.3f} ({(m1 - m2) / s:+.1f} sigma)")
    axd.axhline(0, color="0.5", lw=0.8)
    axd.errorbar(rc, d, de, fmt="o-", color="k", capsize=2, lw=2,
                 label=f"⟨χt⟩ {names[0]} − {names[1]}: χ² = {chisq:.1f} / {ok.sum()} bins ({norm.isf(p):.1f}σ)")

    for a_ in (ax, axd):
        a_.set_xscale("log")
        a_.set_xlim(edges[0] * 0.9, edges[-1] * 1.1)
        a_.set_xlabel("radius [kpc]")
        a_.legend(fontsize=8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_ylabel("mean galaxy distortion χ")
    ax.set_title(f"A2744 main halo split along {opts.angle:g}° ({'-'.join(sorted(names, key=lambda s: s != 'NW'))} halves), κ < {KAPPA_THRESH:g}\n"
                 + ("centre: Lenstool κ peak, " if opts.kappa_peak else f"centre: halves closest inside {opts.core:g} kpc, " if opts.scan_centre else "centre: ")
                 + f"({opts.dx_kpc:+.0f}, {opts.dy_kpc:+.0f}) kpc from the stamp centre", fontsize=10)
    axd.set_ylabel("Δ⟨χt⟩")
    axd.set_title(f"difference, {names[0]} − {names[1]}", fontsize=10)
    L = edges[-1] / KPC_PER_PX
    axm.plot([-L * np.cos(a), L * np.cos(a)], [-L * np.sin(a), L * np.sin(a)], "k-", lw=1)
    for rr in edges[[0, -1]] / KPC_PER_PX:
        axm.add_patch(plt.Circle((0, 0), rr, fill=False, color="k", lw=0.6, ls=":"))
    axm.set_xlim(-L, L)                                        # x to the west: east on the left
    axm.set_ylim(-L, L)
    axm.set_aspect("equal")
    axm.set_xlabel("stamp x [20 kpc px]  (E ← → W)")
    axm.set_title("galaxies used (east left, north up)", fontsize=10)
    axm.legend(fontsize=8, loc="lower left", markerscale=4)
    for e in ("png", "pdf"):
        fig.savefig(f"{opts.out}.{e}", dpi=150)
    print(f"wrote {opts.out}.png")


if __name__ == "__main__":
    main()
