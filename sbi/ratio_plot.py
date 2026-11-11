#!/usr/bin/env python
'''
P(SIDM) against the shear ratio R = <chi_t>(inner) / <chi_t>(outer annulus) for BAHAMAS mass-matched clusters,
one panel per outer annulus, in the paper style. Reads the npz ratio_scan.py writes, so it needs no catalogue:

    astro-env/bin/python sbi/ratio_plot.py <dir>/ratio_scan_scatter.npz
'''
import sys

import numpy as np

COLOURS = {0: "#000000", 0.1: "#0072B2", 0.3: "#009E73", 1: "#D55E00"}


def plot_grid(npz, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scienceplots  # noqa: F401
    from matplotlib.lines import Line2D
    plt.style.use(["science", "grid"])
    z = np.load(npz)
    R_all, p, cs, matched, R_a, p_a = z["R"], z["p"], z["cs"], z["matched"], z["R_a"], float(z["p_a"])
    den, num = [tuple(d) for d in z["den"]], tuple(z["num"])
    nc = 2 if len(den) == 4 else 3
    nr = int(np.ceil(len(den) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 4.2 * nr), constrained_layout=True, sharey=True, squeeze=False)
    for ax in axes.ravel()[len(den):]:
        ax.axis("off")
    for j, ax in zip(range(len(den)), axes.ravel()):
        R = R_all[:, j]
        for c in np.unique(cs):
            m = matched & (cs == c) & np.isfinite(R)
            ax.scatter(R[m], p[m], s=6, color=COLOURS.get(float(c)), alpha=0.5, lw=0)
        ax.plot(R_a[j], p_a, "*", color="#E69F00", mec="k", mew=1.2, ms=16, zorder=5)
        lo, hi = np.nanpercentile(R[matched], [1, 99])
        ax.set_xlim(min(lo, R_a[j]) - 0.1 * abs(hi - lo), max(hi, R_a[j]) + 0.1 * abs(hi - lo))
        ax.set_xlabel(rf"$\langle\chi_t\rangle_{{{num[0]}-{num[1]}}} / \langle\chi_t\rangle_{{{den[j][0]}-{den[j][1]}\,\mathrm{{kpc}}}}$",
                      fontsize=15)
    handles = [Line2D([], [], ls="", marker="o", ms=6, color=COLOURS.get(float(c)),
                      label=rf"$\sigma/m = {c:g}$" if c else "CDM") for c in np.unique(cs)]
    handles.append(Line2D([], [], ls="", marker="*", ms=12, color="#E69F00", mec="k", mew=1, label="A2744"))
    axes[0, 0].legend(handles=handles, fontsize=11, loc="lower left", frameon=True)
    for a in axes[:, 0]:
        a.set_ylabel(r"$P(\mathrm{SIDM})$", fontsize=15)
    for e in ("png", "pdf"):
        fig.savefig(f"{out}.{e}", dpi=150)
    print(f"wrote {out}.png")


if __name__ == "__main__":
    f = sys.argv[1]
    plot_grid(f, f.removesuffix(".npz"))
